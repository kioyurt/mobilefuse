"""
full_pipeline_with_seg.py
完整融合管线: Qwen + 三级融合 + SAM分割 + DC-AE编码 + SANA

最终架构:
    vis(ir) ──→ SAM ──→ seg_map ──→ DC-AE ──→ latent ──→ Linear ──┐
                                                                    ├── cat → (B,648,2304) → SANA
    vis+ir+txt ──→ Qwen ──→ hidden_states ──→ ThreeLevelMCP ──────┘
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional
import logging

from sam_semantic_segmentor import SAMSemanticSegmentor
from seg_condition_via_dcae import SegConditionViaDCAE
from three_level_fusion_official import ThreeLevelFusionMCP

logger = logging.getLogger(__name__)


class FullFusionPipelineWithSeg(nn.Module):
    """
    完整融合管线

    组件:
        1. SAMSemanticSegmentor — SAM 语义分割 (冻结)
        2. ThreeLevelFusionMCP — 三级分层融合 (可训练)
        3. SegConditionViaDCAE — 分割条件注入 (投影层可训练, DC-AE冻结)

    输入:
        hidden_states: Qwen2.5-VL-7B 所有层输出
        vis_span, ir_span, txt_span: token 位置
        vis_image: (B, 3, 448, 448)
        ir_image: (B, 3, 448, 448)
    输出:
        combined_cond: (B, 648, 2304)
        info: dict
    """

    def __init__(
        self,
        # SAM 配置
        sam_model_path: str = r"E:\ai for science\omnifuse\models\AI-ModelScope\sam-vit-base",
        input_size: int = 448,
        points_per_side: int = 8,
        max_masks: int = 16,
        # DC-AE
        dc_ae_encoder: nn.Module = None,
        latent_ch: int = 32,
        # SANA
        caption_dim: int = 2304,
        spatial_size: int = 14,
        # VLM
        llm_dim: int = 3584,
    ):
        super().__init__()
        self.caption_dim = caption_dim

        # ---- 1. SAM 语义分割器 (冻结) ----
        self.sam_segmentor = SAMSemanticSegmentor(
            model_path=sam_model_path,
            input_size=input_size,
            points_per_side=points_per_side,
            max_masks=max_masks,
            freeze=True,
        )

        # ---- 2. 三级分层融合 MCP (可训练) ----
        self.three_level_mcp = ThreeLevelFusionMCP(
            llm_dim=llm_dim,
            caption_dim=caption_dim,
            num_tokens_per_modality=256,
            spatial_size=16,
            num_heads=8,
            kan_bases=5,
        )

        # ---- 3. 分割条件注入 (DC-AE 冻结, 投影可训练) ----
        assert dc_ae_encoder is not None, "必须提供预训练 DC-AE 编码器"
        self.seg_injector = SegConditionViaDCAE(
            dc_ae_encoder=dc_ae_encoder,
            latent_ch=latent_ch,
            caption_dim=caption_dim,
            spatial_size=spatial_size,
        )

    def forward(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        vis_span: Tuple[int, int],
        ir_span: Tuple[int, int],
        txt_span: Tuple[int, int],
        vis_image: torch.Tensor,
        ir_image: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        完整前向传播

        输入:
            hidden_states: Qwen 所有层输出 (29 tensors, each (B, L, 3584))
            vis_span: 可见光 token 位置 (start, end)
            ir_span: 红外 token 位置 (start, end)
            txt_span: 文本 token 位置 (start, end)
            vis_image: (B, 3, 448, 448) — 可见光原图 [0,1]
            ir_image: (B, 3, 448, 448) — 红外原图 [0,1]
        输出:
            combined_cond: (B, 648, 2304)
            info: dict
        """
        info = {}

        # ==== Step 1: SAM 语义分割 (冻结, no_grad) ====
        seg_vis = self.sam_segmentor(vis_image)  # (B, 3, 448, 448)
        seg_ir = self.sam_segmentor(ir_image)    # (B, 3, 448, 448)
        info['seg_vis'] = seg_vis
        info['seg_ir'] = seg_ir

        # ==== Step 2: 三级分层融合 (可训练) ====
        fusion_cond, weight_info = self.three_level_mcp(
            hidden_states=hidden_states,
            vis_span=vis_span,
            ir_span=ir_span,
            txt_span=txt_span,
        )  # (B, 256, 2304)
        info['fusion_cond'] = fusion_cond
        info['weight_info'] = weight_info

        # ==== Step 3: 分割条件注入 (DC-AE冻结, 投影可训练) ====
        combined_cond, seg_info = self.seg_injector(
            seg_vis=seg_vis,
            seg_ir=seg_ir,
            fusion_cond=fusion_cond,
        )  # (B, 648, 2304)
        info['seg_info'] = seg_info

        return combined_cond, info

    def forward_without_sam(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        vis_span: Tuple[int, int],
        ir_span: Tuple[int, int],
        txt_span: Tuple[int, int],
        vis_image: torch.Tensor,
        ir_image: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        消融模式: 不使用 SAM，分割条件用零填充

        用于验证 SAM 分割条件的贡献
        """
        info = {}

        # 三级融合
        fusion_cond, weight_info = self.three_level_mcp(
            hidden_states=hidden_states,
            vis_span=vis_span,
            ir_span=ir_span,
            txt_span=txt_span,
        )
        info['fusion_cond'] = fusion_cond
        info['weight_info'] = weight_info

        # 零填充分割条件
        B = fusion_cond.shape[0]
        device = fusion_cond.device
        zero_seg = torch.zeros(B, 3, vis_image.shape[2], vis_image.shape[3], device=device)

        combined_cond, seg_info = self.seg_injector(
            seg_vis=zero_seg,
            seg_ir=zero_seg,
            fusion_cond=fusion_cond,
        )
        info['seg_info'] = seg_info
        info['ablation'] = 'no_sam'

        return combined_cond, info

    def get_trainable_param_groups(self) -> Dict[str, list]:
        """
        获取可训练参数组 (分组设置不同学习率)

        返回:
            {
                'three_level_mcp': params,      # 主模块, lr=1e-4
                'seg_projection': params,       # 分割投影, lr=1e-4
            }
        """
        groups = {
            'three_level_mcp': [p for p in self.three_level_mcp.parameters() if p.requires_grad],
            'seg_projection': self.seg_injector.get_trainable_params(),
        }
        return groups

    def print_summary(self):
        """打印模型摘要"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable

        sam_frozen = self.sam_segmentor.get_frozen_param_count()
        dcae_frozen = sum(p.numel() for p in self.seg_injector.dc_ae_encoder.parameters())

        logger.info("=" * 60)
        logger.info("FullFusionPipelineWithSeg 模型摘要")
        logger.info("=" * 60)
        logger.info(f"总参数: {total:,} ({total/1e6:.1f}M)")
        logger.info(f"可训练: {trainable:,} ({trainable/1e6:.1f}M)")
        logger.info(f"冻结:   {frozen:,} ({frozen/1e6:.1f}M)")
        logger.info(f"  ├─ SAM: {sam_frozen:,} ({sam_frozen/1e6:.1f}M)")
        logger.info(f"  ├─ DC-AE: {dcae_frozen:,} ({dcae_frozen/1e6:.1f}M)")
        logger.info(f"  └─ 其他: {frozen-sam_frozen-dcae_frozen:,}")
        logger.info("-" * 40)
        for name, params in self.get_trainable_param_groups().items():
            count = sum(p.numel() for p in params)
            logger.info(f"  {name}: {count:,} ({count/1e6:.1f}M)")
        logger.info("=" * 60)