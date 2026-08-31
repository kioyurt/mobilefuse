# -*- coding: utf-8 -*-
"""
Multimodal Spatial Condition Space v2
======================================

针对 RGB/IR 图像融合重新设计的条件空间，不使用 Mobile-O 原始 MCP。

核心约束：
1. RGB / IR 输入统一为 512x512。
2. DC-AE latent 统一为 16x16。
3. Qwen visual tokens 无论原始网格是 16x16/18x18 等，都重采样到 16x16。
4. 最终空间条件固定为 256 = 16x16 个 token，保留二维位置拓扑。
5. 文本为独立的全局条件 token，不再和空间 token 简单混杂。
6. SAM 不进入 forward；SAM 只作为训练阶段 teacher。
7. DC-AE latent -> condition 的映射从零训练，并和跨模态融合联合优化。
8. 输出包含：condition、attention mask、IR gate、structure map。

该文件不依赖原 three_level_fusion_official.py / seg_condition_via_dcae.py。
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 基础层
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(dtype=x.dtype)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x.float()).to(x.dtype)
        y = self.fc2(self.dropout(self.act(self.fc1(y))))
        return x + y


class TokenSelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.norm1(x)
        y, _ = self.attn(q, q, q, need_weights=False)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


class BidirectionalCrossModalBlock(nn.Module):
    """
    双向 RGB <-> IR 跨模态注意力。
    输入保持 16x16 空间顺序，避免把两种模态在 token 维直接 concat。
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.rgb_norm_q = nn.LayerNorm(dim)
        self.ir_norm_q = nn.LayerNorm(dim)
        self.rgb_norm_kv = nn.LayerNorm(dim)
        self.ir_norm_kv = nn.LayerNorm(dim)

        self.rgb_to_ir = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ir_to_rgb = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.rgb_ff = FeedForward(dim, int(dim * mlp_ratio))
        self.ir_ff = FeedForward(dim, int(dim * mlp_ratio))

        self.alpha = nn.Parameter(
            torch.tensor(0.1)
        )

    def forward(
        self,
        rgb: torch.Tensor,
        ir: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rgb_q = self.rgb_norm_q(rgb)
        ir_kv = self.ir_norm_kv(ir)
        rgb_msg, _ = self.rgb_to_ir(rgb_q, ir_kv, ir_kv, need_weights=False)
        rgb = rgb + self.alpha*rgb_msg
        rgb = self.rgb_ff(rgb)

        ir_q = self.ir_norm_q(ir)
        rgb_kv = self.rgb_norm_kv(rgb)
        ir_msg, _ = self.ir_to_rgb(ir_q, rgb_kv, rgb_kv, need_weights=False)
        ir = ir + self.alpha* ir_msg
        ir = self.ir_ff(ir)
        return rgb, ir


# ============================================================
# Qwen 多层视觉特征处理
# ============================================================

class QwenLayerBank(nn.Module):
    """
    将 Qwen 不同深度 hidden states 映射到统一的 16x16 空间。

    输入：
        layers: Mapping[str/int, Tensor(B,N,2048)]
        each layer can have a different square token grid.

    输出：
        feature: (B,256,dim)
    """

    def __init__(
        self,
        llm_dim: int,
        out_dim: int,
        selected_layers: Sequence[int],
        target_hw: Tuple[int, int] = (16, 16),
    ):
        super().__init__()
        self.selected_layers = tuple(int(x) for x in selected_layers)
        self.target_h, self.target_w = target_hw
        self.out_dim = out_dim

        self.norms = nn.ModuleDict({str(i): RMSNorm(llm_dim) for i in self.selected_layers})
        self.proj = nn.ModuleDict({str(i): nn.Linear(llm_dim, out_dim) for i in self.selected_layers})

        # 先验分层权重：浅层/中层/深层。每组再由 layer logits 学习细分。
        self.level_logits = nn.Parameter(torch.zeros(3))
        self.layer_logits = nn.Parameter(torch.zeros(len(self.selected_layers)))

    @staticmethod
    def _square_hw(n: int) -> Tuple[int, int]:
        side = int(round(math.sqrt(n)))
        if side * side != n:
            raise ValueError(f"Qwen visual token 数 {n} 不是平方数，无法恢复二维网格")
        return side, side

    def _one_layer(self, x: torch.Tensor, key: str) -> torch.Tensor:
        # x: B,N,D
        if x.ndim != 3:
            raise ValueError(f"Qwen layer 必须是 [B,N,D]，得到 {tuple(x.shape)}")
        b, n, _ = x.shape
        hw = self._square_hw(n)
        y = self.proj[key](self.norms[key](x))
        y = y.transpose(1, 2).reshape(b, self.out_dim, hw[0], hw[1])
        y = F.interpolate(
            y,
            size=(self.target_h, self.target_w),
            mode="bilinear",
            align_corners=False,
        )
        return y.flatten(2).transpose(1, 2)

    def forward(self, layers: Mapping[int | str, torch.Tensor]) -> torch.Tensor:
        features = []
        logits = self.layer_logits
        for i, idx in enumerate(self.selected_layers):
            if idx not in layers and str(idx) not in layers:
                raise KeyError(f"缓存中缺少 Qwen layer {idx}")
            key = str(idx)
            x = layers[idx] if idx in layers else layers[key]
            features.append(self._one_layer(x.float(), key))

        weights = torch.softmax(logits.float(), dim=0)
        stacked = torch.stack(features, dim=0)
        out = (stacked * weights[:, None, None, None]).sum(dim=0)
        return out


# ============================================================
# DC-AE latent -> 统一条件空间
# ============================================================

class SpatialLatentAdapter(nn.Module):
    """
    从 DC-AE 32-channel latent 学习空间条件特征。

    使用独立 RGB/IR 参数，不共享模态投影。
    """

    def __init__(self, latent_ch: int, out_dim: int):
        super().__init__()
        self.in_norm = nn.GroupNorm(8, latent_ch)
        self.proj = nn.Sequential(
            nn.Conv2d(latent_ch, out_dim, kernel_size=1, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, groups=out_dim, bias=False),
            nn.Conv2d(out_dim, out_dim, kernel_size=1, bias=True),
        )
        self._init_weights()
        self.out_norm = nn.GroupNorm(32, out_dim)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z.float()
        y = self.proj(self.in_norm(z))
        y = self.out_norm(y)
        return y.flatten(2).transpose(1, 2)


# ============================================================
# 文本全局条件
# ============================================================

class GlobalTextCondition(nn.Module):
    """
    使用少量 learnable query 从 Qwen text hidden 中提取全局任务条件。
    输出固定 num_text_tokens 个 token，不承担空间定位职责。
    """

    def __init__(self, llm_dim: int, hidden_dim: int, num_text_tokens: int = 4, num_heads: int = 8):
        super().__init__()
        self.text_norm = RMSNorm(llm_dim)
        self.text_proj = nn.Linear(llm_dim, hidden_dim)
        self.query = nn.Parameter(torch.randn(1, num_text_tokens, hidden_dim) * 0.02)
        self.cross = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, text_hidden: Optional[torch.Tensor]) -> torch.Tensor:
        if text_hidden is None:
            # text 没提供时仍返回 zero global tokens；推理可以关闭文本。
            b = 1
            if hasattr(self, "_last_batch"):
                b = self._last_batch
            q = self.query.expand(b, -1, -1)
            return q * 0.0

        if text_hidden.ndim != 3:
            raise ValueError(f"text_hidden 必须为 [B,T,D]，得到 {tuple(text_hidden.shape)}")
        self._last_batch = text_hidden.shape[0]
        t = self.text_proj(self.text_norm(text_hidden.float()))
        q = self.query.expand(t.shape[0], -1, -1)
        y, _ = self.cross(q, t, t, need_weights=False)
        return self.out_norm(q + y)


# ============================================================
# 主模型
# ============================================================

class MultimodalSpatialConditionEncoder(nn.Module):
    """
    全新 RGB/IR 多模态条件空间。

    最终：
        spatial_condition = [B,256,2304]
        text_condition    = [B,4,2304]
        condition         = [B,260,2304]
    """

    def __init__(
        self,
        llm_dim: int = 2048,
        latent_ch: int = 32,
        fusion_dim: int = 512,
        caption_dim: int = 2304,
        target_hw: Tuple[int, int] = (16, 16),
        selected_layers: Sequence[int] = (4, 5, 6, 13, 14, 15, 26, 27, 28),
        num_heads: int = 8,
        num_cross_blocks: int = 2,
        num_spatial_blocks: int = 2,
        num_text_tokens: int = 4,
    ):
        super().__init__()
        self.target_h, self.target_w = target_hw
        self.num_spatial_tokens = self.target_h * self.target_w
        self.caption_dim = caption_dim
        self.num_text_tokens = num_text_tokens

        self.rgb_qwen = QwenLayerBank(llm_dim, fusion_dim, selected_layers, target_hw)
        self.ir_qwen = QwenLayerBank(llm_dim, fusion_dim, selected_layers, target_hw)

        self.rgb_latent = SpatialLatentAdapter(latent_ch, fusion_dim)
        self.ir_latent = SpatialLatentAdapter(latent_ch, fusion_dim)

        self.rgb_sem_norm = nn.LayerNorm(fusion_dim)
        self.ir_sem_norm = nn.LayerNorm(fusion_dim)
        self.rgb_lat_norm = nn.LayerNorm(fusion_dim)
        self.ir_lat_norm = nn.LayerNorm(fusion_dim)

        self.cross_blocks = nn.ModuleList(
            [BidirectionalCrossModalBlock(fusion_dim, num_heads) for _ in range(num_cross_blocks)]
        )

        self.modality_gate = nn.Sequential(
            nn.LayerNorm(fusion_dim * 4),
            nn.Linear(fusion_dim * 4, fusion_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(fusion_dim, 1),
        )

        self.semantic_fuse = nn.Sequential(
            nn.LayerNorm(fusion_dim * 4),
            nn.Linear(fusion_dim * 4, fusion_dim * 2),
            nn.GELU(approximate="tanh"),
            nn.Linear(fusion_dim * 2, fusion_dim),
        )

        self.spatial_pos = nn.Parameter(torch.zeros(1, self.num_spatial_tokens, fusion_dim))
        self.spatial_type = nn.Parameter(torch.zeros(1, 1, fusion_dim))
        self.rgb_type = nn.Parameter(torch.zeros(1, 1, fusion_dim))
        self.ir_type = nn.Parameter(torch.zeros(1, 1, fusion_dim))
        nn.init.normal_(self.spatial_pos, std=0.01)
        nn.init.normal_(self.rgb_type, std=0.02)
        nn.init.normal_(self.ir_type, std=0.02)

        self.spatial_blocks = nn.ModuleList(
            [TokenSelfAttentionBlock(fusion_dim, num_heads) for _ in range(num_spatial_blocks)]
        )

        self.ir_gate_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(approximate="tanh"),
            nn.Linear(fusion_dim // 2, 1),
        )
        self.structure_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(approximate="tanh"),
            nn.Linear(fusion_dim // 2, 1),
        )

        self.text_condition = GlobalTextCondition(
            llm_dim=llm_dim,
            hidden_dim=fusion_dim,
            num_text_tokens=num_text_tokens,
            num_heads=num_heads,
        )

        self.spatial_to_caption = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, caption_dim),
        )
        self.text_to_caption = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, caption_dim),
        )

        # 零门控：新条件空间初期不会以超大幅度污染 SANA。
        self.output_gate = nn.Parameter(torch.tensor(0.0))

        self.condition_adapter = nn.Sequential(
            nn.Linear(caption_dim, caption_dim),
            nn.GELU(),
            nn.LayerNorm(caption_dim)
        )

        # condition 内部的 modality/status embedding
        # self.output_norm = nn.LayerNorm(caption_dim)
        self.spatial_out_norm = nn.LayerNorm(caption_dim)

        self.text_out_norm = nn.LayerNorm(caption_dim)

    @property
    def output_scale(self) -> torch.Tensor:
        return torch.sigmoid(self.output_gate)

    def forward(
        self,
        rgb_layers: Mapping[int | str, torch.Tensor],
        ir_layers: Mapping[int | str, torch.Tensor],
        rgb_latent: torch.Tensor,
        ir_latent: torch.Tensor,
        text_hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        rgb_sem = self.rgb_qwen(rgb_layers)
        ir_sem = self.ir_qwen(ir_layers)

        rgb_lat = self.rgb_latent(rgb_latent)
        ir_lat = self.ir_latent(ir_latent)

        rgb = self.rgb_sem_norm(rgb_sem) + self.rgb_lat_norm(rgb_lat) + self.rgb_type
        ir = self.ir_sem_norm(ir_sem) + self.ir_lat_norm(ir_lat) + self.ir_type

        for block in self.cross_blocks:
            rgb, ir = block(rgb, ir)

        cross_rgb = rgb
        cross_ir = ir
        base_cat = torch.cat([rgb_sem, ir_sem, rgb_lat, ir_lat], dim=-1)
        fused_base = self.semantic_fuse(base_cat)

        gate_in = torch.cat([cross_rgb, cross_ir, cross_rgb - cross_ir, fused_base], dim=-1)
        ir_gate_logits = self.modality_gate(gate_in)
        ir_gate = torch.sigmoid(ir_gate_logits)

        # 双向 cross feature + 原始语义/latent residual，保持源信息。
        fused = (
            ir_gate * cross_ir
            + (1.0 - ir_gate) * cross_rgb
            + 0.5 * fused_base
            + 0.25 * (rgb_lat + ir_lat)
        )
        fused = fused + self.spatial_pos + self.spatial_type

        for block in self.spatial_blocks:
            fused = block(fused)

        # head 用最终融合特征预测训练 teacher 对应的 gate / structure。
        gate = torch.sigmoid(self.ir_gate_head(fused))
        structure = torch.sigmoid(self.structure_head(fused))

        text_tokens = self.text_condition(text_hidden)

        spatial_cond = self.spatial_to_caption(fused)
        text_cond = self.text_to_caption(text_tokens)

        # # condition 的最终零门控在 concat + norm 之后执行。
        # condition = torch.cat([spatial_cond, text_cond], dim=1)
        # condition = self.output_norm(condition.float())
        # # scale 在最外层，避免后续 LayerNorm 把零门控效果再次归一化掉。
        # scale = self.output_scale.to(dtype=condition.dtype, device=condition.device)
        # condition = condition * scale
        spatial_cond = self.spatial_out_norm(
            spatial_cond.float()
        )

        text_cond = self.text_out_norm(
            text_cond.float()
        )

        condition = torch.cat(
            [
                spatial_cond,
                text_cond
            ],
            dim=1
        )

        condition=self.condition_adapter(condition)

        scale = self.output_scale.to(
            dtype=condition.dtype,
            device=condition.device
        )

        condition = condition * scale

        mask = torch.ones(
            condition.shape[0], condition.shape[1],
            device=condition.device,
            dtype=torch.bool,
        )

        info = {
            "spatial_condition": spatial_cond,
            "text_condition": text_cond,
            "attention_mask": mask,
            "ir_gate": gate.view(gate.shape[0], self.target_h, self.target_w),
            "structure": structure.view(structure.shape[0], self.target_h, self.target_w),
            "output_scale": scale.detach().float(),
        }
        return condition, info

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


# ============================================================
# checkpoint 工具
# ============================================================

def load_conditioner_checkpoint(
    module: nn.Module,
    path: str,
    device: str = "cpu",
    strict: bool = True,
) -> Dict[str, object]:
    state = torch.load(path, map_location=device)
    if "conditioner" in state:
        state_dict = state["conditioner"]
        meta = state
    else:
        state_dict = state
        meta = {}
    result = module.load_state_dict(state_dict, strict=strict)
    return {
        "missing_keys": result.missing_keys,
        "unexpected_keys": result.unexpected_keys,
        "meta": meta,
    }


def count_parameters(module: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable
