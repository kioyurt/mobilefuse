"""
seg_condition_via_dcae.py
分割条件注入: SAM分割图 → DC-AE编码 → 与融合条件拼接 → SANA

核心设计:
    分割图不经过任何自定义投影层
    直接通过预训练 DC-AE 编码器进入 SANA 潜空间
    保证分割条件与生成目标在同一表示空间

数据流:
    seg_vis (B,3,448,448) → DC-AE encode → (B,32,14,14) → flatten → (B,196,32) → Linear → (B,196,2304)
    seg_ir  (B,3,448,448) → DC-AE encode → (B,32,14,14) → flatten → (B,196,32) → Linear → (B,196,2304)
    fusion_cond (B,256,2304)
    → cat → (B, 648, 2304) → SANA encoder_hidden_states
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SegConditionViaDCAE(nn.Module):
    """
    分割条件注入器 (DC-AE 版)

    将 SAM 语义分割图通过预训练 DC-AE 编码器映射到 SANA 潜空间，
    再线性投影到 SANA 条件维度，最后与三级融合条件拼接。

    参数:
        dc_ae_encoder: 预训练 DC-AE 编码器 (冻结)
        latent_ch: DC-AE 潜空间通道数 (通常 32)
        caption_dim: SANA 条件维度 (2304)
        spatial_size: DC-AE 输出空间尺寸 (14, 即 14×14=196 tokens)
    """

    def __init__(
        self,
        dc_ae_encoder: nn.Module,
        latent_ch: int = 32,
        caption_dim: int = 2304,
        spatial_size: int = 14,
    ):
        super().__init__()
        self.latent_ch = latent_ch
        self.caption_dim = caption_dim
        self.spatial_size = spatial_size
        self.num_tokens = spatial_size * spatial_size  # 196

        # ---- DC-AE 编码器 (冻结) ----
        self.dc_ae_encoder = dc_ae_encoder
        for param in self.dc_ae_encoder.parameters():
            param.requires_grad = False
        self.dc_ae_encoder.eval()

        # ---- 潜空间 → 条件空间投影 ----
        # DC-AE 输出 (B, 32, 14, 14) → flatten → (B, 196, 32) → Linear → (B, 196, 2304)
        # 注意: 这里只需要一个简单的线性层，因为 DC-AE 已经在正确的潜空间中
        self.latent_to_cond = nn.Sequential(
            nn.LayerNorm(latent_ch),
            nn.Linear(latent_ch, caption_dim // 4),   # 32 → 576
            nn.GELU(),
            nn.Linear(caption_dim // 4, caption_dim),  # 576 → 2304
            nn.LayerNorm(caption_dim),
        )

        # ---- 模态类型嵌入 ----
        self.type_embed_vis = nn.Parameter(torch.zeros(1, 1, caption_dim))
        self.type_embed_ir = nn.Parameter(torch.zeros(1, 1, caption_dim))
        nn.init.normal_(self.type_embed_vis, std=0.02)
        nn.init.normal_(self.type_embed_ir, std=0.02)

        logger.info(f"SegConditionViaDCAE 初始化完成")
        logger.info(f"  DC-AE latent: {latent_ch}ch, {spatial_size}×{spatial_size}")
        logger.info(f"  条件维度: {caption_dim}")
        logger.info(f"  可训练参数: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

    @torch.no_grad()
    def encode_seg_with_dcae(self, seg_map: torch.Tensor) -> torch.Tensor:
        """
        用 DC-AE 编码分割图

        输入: seg_map (B, 3, 448, 448), 范围 [0, 1]
        输出: latent (B, latent_ch, spatial_size, spatial_size)
        """
        # 确保 [0, 1]
        if seg_map.min() < 0:
            seg_map = (seg_map + 1.0) / 2.0
        seg_map = seg_map.clamp(0, 1)

        # DC-AE 编码
        latent = self.dc_ae_encoder(seg_map)
        # (B, latent_ch, H', W')

        # 确保空间尺寸匹配
        if latent.shape[-1] != self.spatial_size or latent.shape[-2] != self.spatial_size:
            latent = F.interpolate(
                latent, size=(self.spatial_size, self.spatial_size),
                mode='bilinear', align_corners=False,
            )

        return latent  # (B, latent_ch, 14, 14)

    def forward(
        self,
        seg_vis: torch.Tensor,
        seg_ir: torch.Tensor,
        fusion_cond: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        完整前向: 分割图 → DC-AE → 投影 → 拼接

        输入:
            seg_vis: (B, 3, 448, 448) — 可见光语义分割图
            seg_ir: (B, 3, 448, 448) — 红外语义分割图
            fusion_cond: (B, 256, 2304) — 三级融合条件
        输出:
            combined_cond: (B, 648, 2304) — 拼接后的完整条件
            info: dict — 中间结果
        """
        info = {}

        # ---- Step 1: DC-AE 编码分割图 ----
        with torch.no_grad():
            latent_vis = self.encode_seg_with_dcae(seg_vis)  # (B, 32, 14, 14)
            latent_ir = self.encode_seg_with_dcae(seg_ir)    # (B, 32, 14, 14)

        info['latent_vis'] = latent_vis
        info['latent_ir'] = latent_ir

        # ---- Step 2: Flatten → 投影到条件空间 ----
        B = latent_vis.shape[0]

        # (B, 32, 14, 14) → (B, 196, 32)
        tokens_vis = latent_vis.flatten(2).transpose(1, 2)
        tokens_ir = latent_ir.flatten(2).transpose(1, 2)

        # (B, 196, 32) → (B, 196, 2304)
        cond_vis = self.latent_to_cond(tokens_vis)
        cond_ir = self.latent_to_cond(tokens_ir)

        info['cond_vis_shape'] = cond_vis.shape
        info['cond_ir_shape'] = cond_ir.shape

        # ---- Step 3: 添加模态类型嵌入 ----
        cond_vis = cond_vis + self.type_embed_vis
        cond_ir = cond_ir + self.type_embed_ir

        # ---- Step 4: 拼接 ----
        # fusion_cond: (B, 256, 2304)
        # cond_vis:    (B, 196, 2304)
        # cond_ir:     (B, 196, 2304)
        combined_cond = torch.cat([fusion_cond, cond_vis, cond_ir], dim=1)
        # (B, 648, 2304)

        info['combined_shape'] = combined_cond.shape

        return combined_cond, info

    def forward_seg_only(
        self,
        seg_vis: torch.Tensor,
        seg_ir: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        仅处理分割条件 (不拼接融合条件)，用于调试

        输入:
            seg_vis: (B, 3, 448, 448)
            seg_ir: (B, 3, 448, 448)
        输出:
            cond_vis: (B, 196, 2304)
            cond_ir: (B, 196, 2304)
        """
        with torch.no_grad():
            latent_vis = self.encode_seg_with_dcae(seg_vis)
            latent_ir = self.encode_seg_with_dcae(seg_ir)

        tokens_vis = latent_vis.flatten(2).transpose(1, 2)
        tokens_ir = latent_ir.flatten(2).transpose(1, 2)

        cond_vis = self.latent_to_cond(tokens_vis) + self.type_embed_vis
        cond_ir = self.latent_to_cond(tokens_ir) + self.type_embed_ir

        return cond_vis, cond_ir

    def get_trainable_params(self):
        """获取可训练参数 (仅投影层和类型嵌入)"""
        return [p for p in self.parameters() if p.requires_grad]