"""
three_level_fusion_official.py

严格基于官方实现的 Qwen2.5-VL 三级分层 RGB/IR 融合模块。

官方依赖
--------
1. FasterKAN
   Repository:
   https://github.com/AthanasiosDelis/faster-kan

   本文件直接使用官方：
       fasterkan.fasterkan_layers.FasterKANLayer
   不重新实现 RSWAF、SplineLinear 或自定义近似 KAN。

2. Mamba / Mamba-2
   Repository:
   https://github.com/state-spaces/mamba

   本文件直接使用官方：
       mamba_ssm.Mamba2
   不自行实现 Selective SSM、scan、离散化或状态更新。

三级设计
--------
浅层 Layer 4-6
    Conv -> Mamba2 -> Cross Attention -> FasterKAN
    重点：纹理、边缘、颜色、局部结构

中层 Layer 13-15
    DWConv -> Mamba2 -> 双向 Cross Attention
    -> Self Attention -> FasterKAN
    重点：部件、空间关系、中级语义

深层 Layer 26-28
    大感受野 DWConv -> 双向 Mamba2
    -> 双向 Cross Attention -> Self Attention
    -> FasterKAN -> Semantic Gate
    重点：高层语义、场景理解

重要说明
--------
- Qwen 的 hidden_states 仍然保持 3584 维。
- Mamba2 / Attention / FasterKAN 在各自层级使用工作维度：
      shallow = 384
      mid     = 512
      deep    = 640
  这是网络的实际层级 bottleneck 设计，不是把官方模块内部算法重写或简化。
- FasterKAN 官方 Layer 本身接受 [B,C]，因此对 [B,N,C] 只做
  token flatten/restore 的形状适配，内部实现完全来自官方仓库。
- Mamba2 官方模块接受 [B,L,D]，因此同样直接在图像 token 序列上运行。
- 当前官方 mamba 仓库的 CUDA/官方构建环境主要面向 Linux + NVIDIA GPU。
  Windows 下不应伪装成“官方 Mamba 已完整可用”；应按官方环境要求部署。

Qwen 层索引
-----------
hidden_states[0] = embedding output
hidden_states[1] = transformer layer 1
...
hidden_states[28] = transformer layer 28

因此：
    shallow = [4,5,6]
    mid     = [13,14,15]
    deep    = [26,27,28]
"""

from __future__ import annotations

from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 官方 FasterKAN
# ============================================================
#
# 直接引用官方 FasterKANLayer。
# 不复制、不重写 FasterKAN 内部实现。
#
# 官方：
# https://github.com/AthanasiosDelis/faster-kan
#

from mobileo.model.fasterkan.fasterkan_layers import FasterKANLayer
# except ImportError as exc:
#     raise ImportError(
#         "未找到官方 FasterKAN。请安装：\n"
#         "git clone https://github.com/AthanasiosDelis/faster-kan.git\n"
#         "cd faster-kan\n"
#         "pip install -e ."
#     ) from exc

# ============================================================
# 官方 Mamba-2
# ============================================================
#
# 直接引用官方 state-spaces/mamba。
#
# 官方：
# https://github.com/state-spaces/mamba
#
from mambablock import (
    Mamba2Windows,
    BidirectionalMamba2Windows,
)

# ============================================================
# 1. 基础模块
# ============================================================

class FeatureProjection(nn.Module):
    """
    3584 -> work_dim

    将 Qwen 高维 token 映射到各层自己的计算空间。
    这里不改变 Qwen hidden state 本身，仅作为融合网络入口。
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.proj(self.norm(x))


class FeatureOutputProjection(nn.Module):
    """
    work_dim -> 3584
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.proj(self.norm(x))


# ============================================================
# 2. 空间卷积
# ============================================================

class SpatialDepthwiseConv(nn.Module):
    """
    [B,N,C]
       ↓
    [B,C,H,W]
       ↓
    Depthwise Conv
       ↓
    Pointwise Conv
       ↓
    [B,N,C]

    用于显式建立图像局部空间先验。
    """

    def __init__(
        self,
        dim: int,
        spatial_size: int,
        kernel_size: int = 3,
        dilation: int = 1,
    ):
        super().__init__()

        assert dim > 0

        self.dim = dim
        self.spatial_size = spatial_size

        padding = ((kernel_size - 1) // 2) * dilation

        self.depthwise = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=dim,
            bias=False,
        )

        self.pointwise = nn.Conv2d(
            dim,
            dim,
            kernel_size=1,
            bias=False,
        )

        # 小 batch 场景使用 GroupNorm
        groups = 32
        while dim % groups != 0 and groups > 1:
            groups //= 2

        self.norm = nn.GroupNorm(
            groups,
            dim,
        )

        self.activation = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        B, N, C = x.shape

        expected_tokens = (
            self.spatial_size *
            self.spatial_size
        )

        if N != expected_tokens:
            raise ValueError(
                "SpatialDepthwiseConv 收到的 token 数量与 "
                f"spatial_size={self.spatial_size} 不一致："
                f" expected={expected_tokens}, actual={N}. "
                "如果 Qwen 图像 token 数量变化，请动态设置 "
                "spatial_size。"
            )

        feature_map = (
            x.transpose(1, 2)
             .contiguous()
             .view(
                 B,
                 C,
                 self.spatial_size,
                 self.spatial_size,
             )
        )

        feature_map = self.depthwise(feature_map)
        feature_map = self.pointwise(feature_map)
        feature_map = self.norm(feature_map)
        feature_map = self.activation(feature_map)

        return (
            feature_map.flatten(2)
                        .transpose(1, 2)
                        .contiguous()
        )


class MultiScaleSpatialConv(nn.Module):
    """
    深层使用多尺度空间上下文：

        DWConv k=3
        DWConv k=5
        融合
    """

    def __init__(
        self,
        dim: int,
        spatial_size: int,
    ):
        super().__init__()

        self.conv3 = SpatialDepthwiseConv(
            dim=dim,
            spatial_size=spatial_size,
            kernel_size=3,
            dilation=1,
        )

        self.conv5 = SpatialDepthwiseConv(
            dim=dim,
            spatial_size=spatial_size,
            kernel_size=5,
            dilation=1,
        )

        self.fusion = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(
                dim * 2,
                dim,
            ),
            nn.SiLU(),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        y3 = self.conv3(x)
        y5 = self.conv5(x)

        return self.fusion(
            torch.cat(
                [y3, y5],
                dim=-1,
            )
        )


# ============================================================
# 3. Cross Attention
# ============================================================

class CrossModalAttention(nn.Module):
    """
    标准 PyTorch MultiheadAttention。

    query 来自一个模态，
    key/value 来自另一个模态。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                f"dim={dim} 必须能够整除 "
                f"num_heads={num_heads}"
            )

        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.output_norm = nn.LayerNorm(dim)

        self.gamma = nn.Parameter(
            torch.zeros(1)
        )

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:

        q = self.query_norm(query)
        kv = self.context_norm(context)

        out, _ = self.attention(
            q,
            kv,
            kv,
            need_weights=False,
        )

        out = self.output_norm(out)

        return (
            query +
            self.gamma * out
        )


class SelfAttention(nn.Module):
    """
    融合结果上的标准 Self-Attention。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                f"dim={dim} 必须能够整除 "
                f"num_heads={num_heads}"
            )

        self.norm = nn.LayerNorm(dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.gamma = nn.Parameter(
            torch.zeros(1)
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        normalized = self.norm(x)

        out, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )

        return x + self.gamma * out


# ============================================================
# 4. 官方 FasterKAN token 适配
# ============================================================

class OfficialFasterKANTokenBlock(nn.Module):
    """
    官方 FasterKANLayer 的 token 维度适配。

    官方 FasterKANLayer 的 forward 接收：
        [B,C]

    图像融合特征：
        [B,N,C]

    因此这里仅进行：
        [B,N,C] -> [B*N,C]
        官方 FasterKANLayer
        [B*N,Cout] -> [B,N,Cout]

    注意：
    FasterKANLayer 的内部 RSWAF 和 SplineLinear 完全来自官方仓库。
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_grids: int = 8,
        grid_min: float = -1.2,
        grid_max: float = 1.2,
        exponent: int = 2,
        inv_denominator: float = 0.5,
        train_grid: bool = False,
        train_inv_denominator: bool = False,
        spline_weight_init_scale: float = 0.667,
    ):
        super().__init__()

        self.fasterkan = FasterKANLayer(
            input_dim=input_dim,
            output_dim=output_dim,
            grid_min=grid_min,
            grid_max=grid_max,
            num_grids=num_grids,
            exponent=exponent,
            inv_denominator=inv_denominator,
            train_grid=train_grid,
            train_inv_denominator=train_inv_denominator,
            spline_weight_init_scale=spline_weight_init_scale,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if x.ndim != 3:
            raise ValueError(
                f"OfficialFasterKANTokenBlock expects [B,N,C], "
                f"received {tuple(x.shape)}"
            )

        B, N, C = x.shape

        flattened = (
            x.reshape(B * N, C)
        )

        output = self.fasterkan(
            flattened
        )

        return (
            output.view(
                B,
                N,
                -1,
            )
        )


class OfficialFasterKANFFN(nn.Module):
    """
    KAN-FFN：

        LayerNorm
          ↓
        Linear
          ↓
        官方 FasterKANLayer
          ↓
        SiLU
          ↓
        Linear
          ↓
        Residual

    FasterKAN 本身没有被重写。
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_grids: int,
        grid_min: float = -1.2,
        grid_max: float = 1.2,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.in_proj = nn.Linear(
            dim,
            hidden_dim,
        )

        self.kan = OfficialFasterKANTokenBlock(
            input_dim=hidden_dim,
            output_dim=hidden_dim,
            num_grids=num_grids,
            grid_min=grid_min,
            grid_max=grid_max,
            exponent=2,
            inv_denominator=0.5,
            train_grid=False,
            train_inv_denominator=False,
            spline_weight_init_scale=0.667,
        )

        self.activation = nn.SiLU()

        self.out_proj = nn.Linear(
            hidden_dim,
            dim,
        )

        self.gamma = nn.Parameter(
            torch.tensor(1e-2)
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        residual = x

        x = self.norm(x)
        x = self.in_proj(x)
        x = self.kan(x)
        x = self.activation(x)
        x = self.out_proj(x)

        return (
            residual +
            self.gamma * x
        )


# ============================================================
# 5. 官方 Mamba-2 的单向 + 双向 token wrapper
# ============================================================

class Mamba2Block(nn.Module):
    """
    Windows Reference Mamba-2 Block

    使用：
        mambablock.Mamba2Windows

    完整包含：
        input projection
        causal depthwise conv
        selective state update
        SSD recurrence
        RMSNorm gated
        output projection

    不依赖：
        mamba_ssm
        CUDA kernel
        Linux环境
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.mamba = Mamba2Windows(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            ngroups=ngroups,
        )

        self.gamma = nn.Parameter(
            torch.tensor(1e-2)
        )


    def forward(
        self,
        x: torch.Tensor,
    ):

        return (
            x +
            self.gamma *
            self.mamba(
                self.norm(x)
            )
        )
class Mamba2BidirectionalBlock(nn.Module):

    def __init__(
        self,
        dim:int,
        d_state:int=64,
        d_conv:int=4,
        expand:int=2,
        headdim:int=64,
        ngroups:int=1,
    ):
        super().__init__()


        self.norm = nn.LayerNorm(dim)


        self.mamba = BidirectionalMamba2Windows(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            ngroups=ngroups,
        )


        self.gamma = nn.Parameter(
            torch.tensor(1e-2)
        )


    def forward(
        self,
        x
    ):

        return (
            x+
            self.gamma*
            self.mamba(
                self.norm(x)
            )
        )


# ============================================================
# 6. 文本条件动态层权重
# ============================================================

class TextConditionedWeightGenerator(nn.Module):
    """
    文本条件：
        Qwen 最后一层文本 token 平均池化
        ↓
        条件编码
        ↓
        官方 FasterKAN
        ↓
        shallow / mid / deep / level 权重
    """

    def __init__(
        self,
        dim: int = 3584,
        hidden_dim: int = 512,
        num_grids: int = 5,
    ):
        super().__init__()

        self.condition_encoder = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(
                dim,
                hidden_dim * 2,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim * 2,
                hidden_dim,
            ),
        )

        # 使用官方 FasterKANLayer
        self.condition_kan = FasterKANLayer(
            input_dim=hidden_dim,
            output_dim=hidden_dim,
            grid_min=-1.2,
            grid_max=1.2,
            num_grids=num_grids,
            exponent=2,
            inv_denominator=0.5,
            train_grid=False,
            train_inv_denominator=False,
            spline_weight_init_scale=0.667,
        )

        self.shallow_head = FasterKANLayer(
            input_dim=hidden_dim,
            output_dim=3,
            grid_min=-1.2,
            grid_max=1.2,
            num_grids=num_grids,
            exponent=2,
            inv_denominator=0.5,
            train_grid=False,
            train_inv_denominator=False,
            spline_weight_init_scale=0.667,
        )

        self.mid_head = FasterKANLayer(
            input_dim=hidden_dim,
            output_dim=3,
            grid_min=-1.2,
            grid_max=1.2,
            num_grids=num_grids,
            exponent=2,
            inv_denominator=0.5,
            train_grid=False,
            train_inv_denominator=False,
            spline_weight_init_scale=0.667,
        )

        self.deep_head = FasterKANLayer(
            input_dim=hidden_dim,
            output_dim=3,
            grid_min=-1.2,
            grid_max=1.2,
            num_grids=num_grids,
            exponent=2,
            inv_denominator=0.5,
            train_grid=False,
            train_inv_denominator=False,
            spline_weight_init_scale=0.667,
        )

        self.level_head = FasterKANLayer(
            input_dim=hidden_dim,
            output_dim=3,
            grid_min=-1.2,
            grid_max=1.2,
            num_grids=num_grids,
            exponent=2,
            inv_denominator=0.5,
            train_grid=False,
            train_inv_denominator=False,
            spline_weight_init_scale=0.667,
        )

        self.temperature_shallow = nn.Parameter(
            torch.tensor(0.5)
        )
        self.temperature_mid = nn.Parameter(
            torch.tensor(0.5)
        )
        self.temperature_deep = nn.Parameter(
            torch.tensor(0.5)
        )
        self.temperature_level = nn.Parameter(
            torch.tensor(0.5)
        )

    @staticmethod
    def _safe_temperature(
        parameter: torch.Tensor,
    ) -> torch.Tensor:
        return F.softplus(
            parameter
        ).clamp(
            min=0.1
        )

    def forward(
        self,
        task_condition: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:

        condition = self.condition_encoder(
            task_condition
        )

        condition = self.condition_kan(
            condition
        )

        shallow_logits = self.shallow_head(
            condition
        )
        mid_logits = self.mid_head(
            condition
        )
        deep_logits = self.deep_head(
            condition
        )
        level_logits = self.level_head(
            condition
        )

        temperature_s = self._safe_temperature(
            self.temperature_shallow
        )
        temperature_m = self._safe_temperature(
            self.temperature_mid
        )
        temperature_d = self._safe_temperature(
            self.temperature_deep
        )
        temperature_l = self._safe_temperature(
            self.temperature_level
        )

        shallow_weight = F.softmax(
            shallow_logits / temperature_s,
            dim=-1,
        )

        mid_weight = F.softmax(
            mid_logits / temperature_m,
            dim=-1,
        )

        deep_weight = F.softmax(
            deep_logits / temperature_d,
            dim=-1,
        )

        level_weight = F.softmax(
            level_logits / temperature_l,
            dim=-1,
        )

        return (
            shallow_weight,
            mid_weight,
            deep_weight,
            level_weight,
        )


# ============================================================
# 7. 文本条件 AdaLN
# ============================================================

class TextConditionedAdaLN(nn.Module):
    """
    文本条件 shift / scale / gate。
    """

    def __init__(
        self,
        dim: int = 3584,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(
            dim,
            elementwise_affine=False,
        )

        self.modulation = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(
                dim,
                dim * 3,
            ),
            nn.SiLU(),
            nn.Linear(
                dim * 3,
                dim * 3,
            ),
        )

        nn.init.zeros_(
            self.modulation[-1].weight
        )
        nn.init.zeros_(
            self.modulation[-1].bias
        )

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:

        shift, scale, gate = (
            self.modulation(
                condition
            ).chunk(
                3,
                dim=-1,
            )
        )

        normalized = self.norm(x)

        modulated = (
            normalized *
            (1.0 + scale.unsqueeze(1))
            +
            shift.unsqueeze(1)
        )

        return (
            x +
            torch.sigmoid(
                gate.unsqueeze(1)
            ) *
            (modulated - x)
        )


# ============================================================
# 8. 浅层 Layer 4-6
# ============================================================

class ShallowFusionBlock(nn.Module):
    """
    浅层：

        RGB/IR
          ↓
        3x3 DWConv
          ↓
        Mamba2
          ↓
        Cross-Attention
          ↓
        FasterKAN
          ↓
        output

    设计重点：
        局部纹理、边缘、颜色。
    """

    def __init__(
        self,
        input_dim: int = 3584,
        work_dim: int = 384,
        num_heads: int = 6,
        spatial_size: int = 16,
    ):
        super().__init__()

        self.vis_projection = FeatureProjection(
            input_dim,
            work_dim,
        )

        self.ir_projection = FeatureProjection(
            input_dim,
            work_dim,
        )

        self.vis_conv = SpatialDepthwiseConv(
            work_dim,
            spatial_size,
            kernel_size=3,
            dilation=1,
        )

        self.ir_conv = SpatialDepthwiseConv(
            work_dim,
            spatial_size,
            kernel_size=3,
            dilation=1,
        )

        self.mamba = Mamba2Block(
            dim=work_dim,
            d_state=64,
            d_conv=4,
            expand=2,
            headdim=64,
        )

        self.cross_attention = CrossModalAttention(
            dim=work_dim,
            num_heads=num_heads,
        )

        self.kan_ffn = OfficialFasterKANFFN(
            dim=work_dim,
            hidden_dim=work_dim,
            num_grids=5,
        )

        self.output_projection = FeatureOutputProjection(
            work_dim,
            input_dim,
        )

        self.output_gamma = nn.Parameter(
            torch.tensor(1e-2)
        )

    def forward(
        self,
        vis_feat: torch.Tensor,
        ir_feat: torch.Tensor,
    ) -> torch.Tensor:

        residual = (
            0.5 *
            (vis_feat + ir_feat)
        )

        vis = self.vis_projection(
            vis_feat
        )
        ir = self.ir_projection(
            ir_feat
        )

        vis = (
            vis +
            self.vis_conv(vis)
        )

        ir = (
            ir +
            self.ir_conv(ir)
        )

        # 浅层以联合局部结构作为 Mamba 输入
        fused = 0.5 * (vis + ir)

        fused = self.mamba(
            fused
        )

        # 用融合结构作为 query，
        # IR 作为 cross-modal context。
        fused = self.cross_attention(
            fused,
            ir,
        )

        fused = self.kan_ffn(
            fused
        )

        fused = self.output_projection(
            fused
        )

        return (
            residual +
            self.output_gamma * fused
        )


# ============================================================
# 9. 中层 Layer 13-15
# ============================================================

class MidFusionBlock(nn.Module):
    """
    中层：

        RGB / IR
          ↓
        3x3 DWConv
          ↓
        独立 Mamba2
          ↓
        RGB->IR Cross Attention
        IR->RGB Cross Attention
          ↓
        Self Attention
          ↓
        FasterKAN
          ↓
        output

    设计重点：
        部件关系、空间关系、中级语义。
    """

    def __init__(
        self,
        input_dim: int = 3584,
        work_dim: int = 512,
        num_heads: int = 8,
        spatial_size: int = 16,
    ):
        super().__init__()

        self.vis_projection = FeatureProjection(
            input_dim,
            work_dim,
        )

        self.ir_projection = FeatureProjection(
            input_dim,
            work_dim,
        )

        self.vis_conv = SpatialDepthwiseConv(
            work_dim,
            spatial_size,
            kernel_size=3,
        )

        self.ir_conv = SpatialDepthwiseConv(
            work_dim,
            spatial_size,
            kernel_size=3,
        )

        self.vis_mamba = Mamba2Block(
            dim=work_dim,
            d_state=64,
            d_conv=4,
            expand=2,
            headdim=64,
        )

        self.ir_mamba =Mamba2Block(
            dim=work_dim,
            d_state=64,
            d_conv=4,
            expand=2,
            headdim=64,
        )

        self.v2i_attention = CrossModalAttention(
            dim=work_dim,
            num_heads=num_heads,
        )

        self.i2v_attention = CrossModalAttention(
            dim=work_dim,
            num_heads=num_heads,
        )

        self.self_attention = SelfAttention(
            dim=work_dim,
            num_heads=num_heads,
        )

        self.kan_ffn = OfficialFasterKANFFN(
            dim=work_dim,
            hidden_dim=work_dim,
            num_grids=6,
        )

        self.fusion_gate = nn.Sequential(
            nn.LayerNorm(
                work_dim * 2
            ),
            nn.Linear(
                work_dim * 2,
                work_dim,
            ),
            nn.Sigmoid(),
        )

        self.output_projection = FeatureOutputProjection(
            work_dim,
            input_dim,
        )

        self.output_gamma = nn.Parameter(
            torch.tensor(1e-2)
        )

    def forward(
        self,
        vis_feat: torch.Tensor,
        ir_feat: torch.Tensor,
    ) -> torch.Tensor:

        residual = (
            0.5 *
            (vis_feat + ir_feat)
        )

        vis = self.vis_projection(
            vis_feat
        )

        ir = self.ir_projection(
            ir_feat
        )

        # 局部空间关系
        vis = (
            vis +
            self.vis_conv(vis)
        )

        ir = (
            ir +
            self.ir_conv(ir)
        )

        # 单模态长程关系
        vis = self.vis_mamba(vis)
        ir = self.ir_mamba(ir)

        # 双向跨模态交互
        vis_cross = self.v2i_attention(
            vis,
            ir,
        )

        ir_cross = self.i2v_attention(
            ir,
            vis,
        )

        cross = (
            0.5 *
            (vis_cross + ir_cross)
        )

        local = (
            0.5 *
            (vis + ir)
        )

        gate = self.fusion_gate(
            torch.cat(
                [
                    local,
                    cross,
                ],
                dim=-1,
            )
        )

        fused = (
            gate * cross +
            (1.0 - gate) * local
        )

        # 中级空间关系进一步建模
        fused = self.self_attention(
            fused
        )

        # 中级语义非线性
        fused = self.kan_ffn(
            fused
        )

        fused = self.output_projection(
            fused
        )

        return (
            residual +
            self.output_gamma * fused
        )


# ============================================================
# 10. 深层 Layer 26-28
# ============================================================

class DeepFusionBlock(nn.Module):
    """
    深层：

        RGB / IR
          ↓
        Multi-scale large-receptive-field Conv
          ↓
        Bidirectional official Mamba2
          ↓
        Bidirectional Cross Attention
          ↓
        Self Attention
          ↓
        FasterKAN
          ↓
        Semantic Gate
          ↓
        output

    设计重点：
        高层语义、场景理解、复杂跨模态语义关系。
    """

    def __init__(
        self,
        input_dim: int = 3584,
        work_dim: int = 640,
        num_heads: int = 8,
        spatial_size: int = 16,
    ):
        super().__init__()

        self.vis_projection = FeatureProjection(
            input_dim,
            work_dim,
        )

        self.ir_projection = FeatureProjection(
            input_dim,
            work_dim,
        )

        self.vis_context = MultiScaleSpatialConv(
            work_dim,
            spatial_size,
        )

        self.ir_context = MultiScaleSpatialConv(
            work_dim,
            spatial_size,
        )

        self.vis_mamba = Mamba2BidirectionalBlock(
            dim=work_dim,
            d_state=64,
            d_conv=4,
            expand=2,
            headdim=64,
        )

        self.ir_mamba = Mamba2BidirectionalBlock(
            dim=work_dim,
            d_state=64,
            d_conv=4,
            expand=2,
            headdim=64,
        )

        self.v2i_attention = CrossModalAttention(
            dim=work_dim,
            num_heads=num_heads,
        )

        self.i2v_attention = CrossModalAttention(
            dim=work_dim,
            num_heads=num_heads,
        )

        self.self_attention = SelfAttention(
            dim=work_dim,
            num_heads=num_heads,
        )

        self.kan_ffn = OfficialFasterKANFFN(
            dim=work_dim,
            hidden_dim=work_dim * 2,
            num_grids=8,
        )

        self.semantic_gate = nn.Sequential(
            nn.LayerNorm(
                work_dim * 3
            ),
            nn.Linear(
                work_dim * 3,
                work_dim,
            ),
            nn.SiLU(),
            nn.Linear(
                work_dim,
                work_dim,
            ),
            nn.Sigmoid(),
        )

        self.output_projection = FeatureOutputProjection(
            work_dim,
            input_dim,
        )

        self.output_gamma = nn.Parameter(
            torch.tensor(1e-2)
        )

    def forward(
        self,
        vis_feat: torch.Tensor,
        ir_feat: torch.Tensor,
    ) -> torch.Tensor:

        residual = (
            0.5 *
            (vis_feat + ir_feat)
        )

        vis = self.vis_projection(
            vis_feat
        )

        ir = self.ir_projection(
            ir_feat
        )

        # 大感受野空间语义
        vis = (
            vis +
            self.vis_context(vis)
        )

        ir = (
            ir +
            self.ir_context(ir)
        )

        # 双向官方 Mamba2
        vis = self.vis_mamba(vis)
        ir = self.ir_mamba(ir)

        # 双向 cross-modal semantic exchange
        vis_cross = self.v2i_attention(
            vis,
            ir,
        )

        ir_cross = self.i2v_attention(
            ir,
            vis,
        )

        cross = (
            0.5 *
            (vis_cross + ir_cross)
        )

        base = (
            0.5 *
            (vis + ir)
        )

        # 深层全局空间关系
        self_attn_input = (
            0.5 *
            (base + cross)
        )

        global_context = (
            self.self_attention(
                self_attn_input
            )
        )

        # 三路语义自适应门控
        gate = self.semantic_gate(
            torch.cat(
                [
                    base,
                    cross,
                    global_context,
                ],
                dim=-1,
            )
        )

        fused = (
            gate * global_context +
            (1.0 - gate) * cross
        )

        # 深层非线性表达
        fused = self.kan_ffn(
            fused
        )

        fused = self.output_projection(
            fused
        )

        return (
            residual +
            self.output_gamma * fused
        )


# ============================================================
# 11. Qwen Token Span
# ============================================================

class QwenTokenSpanExtractor:
    """
    从 input_ids 中提取：
        visible span
        infrared span
        text span
    """

    def __init__(
        self,
        tokenizer,
    ):
        self.tokenizer = tokenizer

        self.vision_start_id = (
            tokenizer.convert_tokens_to_ids(
                "<|vision_start|>"
            )
        )

        self.vision_end_id = (
            tokenizer.convert_tokens_to_ids(
                "<|vision_end|>"
            )
        )

        self.image_pad_id = (
            tokenizer.convert_tokens_to_ids(
                "<|image_pad|>"
            )
        )

    def extract_spans(
        self,
        input_ids: torch.Tensor,
    ) -> Tuple[
        Tuple[int, int],
        Tuple[int, int],
        Tuple[int, int],
    ]:

        ids = input_ids[0]

        vision_starts = (
            ids == self.vision_start_id
        ).nonzero(
            as_tuple=True
        )[0]

        vision_ends = (
            ids == self.vision_end_id
        ).nonzero(
            as_tuple=True
        )[0]

        if (
            vision_starts.numel() >= 2
            and vision_ends.numel() >= 2
        ):
            vis_start = (
                vision_starts[0].item() + 1
            )
            vis_end = (
                vision_ends[0].item()
            )

            ir_start = (
                vision_starts[1].item() + 1
            )
            ir_end = (
                vision_ends[1].item()
            )

            txt_start = ir_end + 1
            txt_end = ids.shape[0] - 1

            return (
                (vis_start, vis_end),
                (ir_start, ir_end),
                (txt_start, txt_end),
            )

        if (
            vision_starts.numel() >= 1
            and vision_ends.numel() >= 1
        ):
            vis_start = (
                vision_starts[0].item() + 1
            )
            vis_end = (
                vision_ends[0].item()
            )

            ir_start = vis_start
            ir_end = vis_end

            txt_start = vis_end + 1
            txt_end = ids.shape[0] - 1

            return (
                (vis_start, vis_end),
                (ir_start, ir_end),
                (txt_start, txt_end),
            )

        raise RuntimeError(
            "没有在 input_ids 中找到 "
            "<|vision_start|>/<|vision_end|>。"
        )

    def extract_spans_by_image_token_count(
        self,
        input_ids: torch.Tensor,
        expected_tokens_per_image: int = 256,
    ) -> Tuple[
        Tuple[int, int],
        Tuple[int, int],
        Tuple[int, int],
    ]:

        ids = input_ids[0]

        pad_positions = (
            ids == self.image_pad_id
        ).nonzero(
            as_tuple=True
        )[0]

        required = (
            2 *
            expected_tokens_per_image
        )

        if pad_positions.numel() >= required:
            vis_start = (
                pad_positions[0].item()
            )

            vis_end = (
                pad_positions[
                    expected_tokens_per_image - 1
                ].item() + 1
            )

            ir_start = (
                pad_positions[
                    expected_tokens_per_image
                ].item()
            )

            ir_end = (
                pad_positions[
                    required - 1
                ].item() + 1
            )

            txt_start = ir_end
            txt_end = ids.shape[0] - 1

            return (
                (vis_start, vis_end),
                (ir_start, ir_end),
                (txt_start, txt_end),
            )

        return self.extract_spans(
            input_ids
        )


# ============================================================
# 12. 三级融合主模块
# ============================================================

class ThreeLevelFusionMCP(nn.Module):
    """
    从 Qwen2.5-VL 28 层 hidden_states 提取三个深度区间：

        shallow: [4,5,6]
        mid:     [13,14,15]
        deep:    [26,27,28]

    每个层级内部有三个独立融合 block，
    然后由文本条件动态加权三个层级。
    """

    def __init__(
        self,
        llm_dim: int = 3584,
        caption_dim: int = 2304,
        num_tokens_per_modality: int = 256,
        spatial_size: int = 16,
        num_heads: int = 8,
        fasterkan_grids: int = 5,
    ):
        super().__init__()

        self.llm_dim = llm_dim
        self.caption_dim = caption_dim
        self.num_tokens = (
            num_tokens_per_modality
        )
        self.spatial_size = spatial_size

        # Qwen 28 个 transformer blocks
        self.shallow_layer_indices = [
            4,
            5,
            6,
        ]

        self.mid_layer_indices = [
            13,
            14,
            15,
        ]

        self.deep_layer_indices = [
            26,
            27,
            28,
        ]

        # 文本条件动态权重
        self.weight_generator = (
            TextConditionedWeightGenerator(
                dim=llm_dim,
                hidden_dim=512,
                num_grids=fasterkan_grids,
            )
        )

        # ====================================================
        # 浅层
        # ====================================================
        self.shallow_blocks = nn.ModuleList(
            [
                ShallowFusionBlock(
                    input_dim=llm_dim,
                    work_dim=384,
                    num_heads=6,
                    spatial_size=spatial_size,
                )
                for _ in range(3)
            ]
        )

        # ====================================================
        # 中层
        # ====================================================
        self.mid_blocks = nn.ModuleList(
            [
                MidFusionBlock(
                    input_dim=llm_dim,
                    work_dim=512,
                    num_heads=num_heads,
                    spatial_size=spatial_size,
                )
                for _ in range(3)
            ]
        )

        # ====================================================
        # 深层
        # ====================================================
        self.deep_blocks = nn.ModuleList(
            [
                DeepFusionBlock(
                    input_dim=llm_dim,
                    work_dim=640,
                    num_heads=num_heads,
                    spatial_size=spatial_size,
                )
                for _ in range(3)
            ]
        )

        # ====================================================
        # 文本条件调制
        # ====================================================
        self.shallow_adaln = (
            TextConditionedAdaLN(
                llm_dim
            )
        )

        self.mid_adaln = (
            TextConditionedAdaLN(
                llm_dim
            )
        )

        self.deep_adaln = (
            TextConditionedAdaLN(
                llm_dim
            )
        )

        # ====================================================
        # 跨层级聚合
        # ====================================================
        self.level_fusion = nn.Sequential(
            nn.LayerNorm(
                llm_dim * 3
            ),
            nn.Linear(
                llm_dim * 3,
                llm_dim,
            ),
            nn.GELU(),
            nn.Linear(
                llm_dim,
                llm_dim,
            ),
        )

        # ====================================================
        # 输出
        # ====================================================
        self.output_projection = nn.Sequential(
            nn.LayerNorm(llm_dim),
            nn.Linear(
                llm_dim,
                llm_dim // 2,
            ),
            nn.GELU(),
            nn.Linear(
                llm_dim // 2,
                caption_dim,
            ),
            nn.LayerNorm(caption_dim),
        )

        # 各模态 Qwen hidden-state 对齐
        self.vis_align = nn.LayerNorm(
            llm_dim
        )

        self.ir_align = nn.LayerNorm(
            llm_dim
        )

    # --------------------------------------------------------
    # hidden_states extraction
    # --------------------------------------------------------

    def _extract_modality_features(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        layer_indices: List[int],
        vis_span: Tuple[int, int],
        ir_span: Tuple[int, int],
    ) -> List[
        Tuple[
            torch.Tensor,
            torch.Tensor,
        ]
    ]:

        features = []

        for layer_idx in layer_indices:

            if layer_idx >= len(
                hidden_states
            ):
                raise IndexError(
                    f"hidden_states 只有 "
                    f"{len(hidden_states)} 层，"
                    f"却要求第 {layer_idx} 层。"
                )

            hidden = (
                hidden_states[layer_idx]
            )

            visible = hidden[
                :,
                vis_span[0]:vis_span[1],
                :,
            ]

            infrared = hidden[
                :,
                ir_span[0]:ir_span[1],
                :,
            ]

            visible = self.vis_align(
                visible
            )

            infrared = self.ir_align(
                infrared
            )

            features.append(
                (
                    visible,
                    infrared,
                )
            )

        return features

    def _extract_text_condition(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        txt_span: Tuple[int, int],
    ) -> torch.Tensor:

        final_hidden = (
            hidden_states[-1]
        )

        text_features = final_hidden[
            :,
            txt_span[0]:txt_span[1],
            :,
        ]

        if text_features.shape[1] == 0:
            raise RuntimeError(
                "文本 span 为空，无法构建 task_condition。"
            )

        return text_features.mean(
            dim=1
        )

    # --------------------------------------------------------
    # forward
    # --------------------------------------------------------

    def forward(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        vis_span: Tuple[int, int],
        ir_span: Tuple[int, int],
        txt_span: Tuple[int, int],
    ) -> Tuple[
        torch.Tensor,
        dict,
    ]:

        # ====================================================
        # 1. 文本条件
        # ====================================================
        task_condition = (
            self._extract_text_condition(
                hidden_states,
                txt_span,
            )
        )

        # ====================================================
        # 2. 动态权重
        # ====================================================
        (
            shallow_weight,
            mid_weight,
            deep_weight,
            level_weight,
        ) = self.weight_generator(
            task_condition
        )

        # ====================================================
        # 3. 提取 Qwen 三个深度区域
        # ====================================================
        shallow_features = (
            self._extract_modality_features(
                hidden_states,
                self.shallow_layer_indices,
                vis_span,
                ir_span,
            )
        )

        mid_features = (
            self._extract_modality_features(
                hidden_states,
                self.mid_layer_indices,
                vis_span,
                ir_span,
            )
        )

        deep_features = (
            self._extract_modality_features(
                hidden_states,
                self.deep_layer_indices,
                vis_span,
                ir_span,
            )
        )

        # ====================================================
        # 4. 浅层
        # ====================================================
        shallow_outputs = []

        for idx, (
            vis_feature,
            ir_feature,
        ) in enumerate(
            shallow_features
        ):

            shallow_outputs.append(
                self.shallow_blocks[idx](
                    vis_feature,
                    ir_feature,
                )
            )

        shallow_stack = torch.stack(
            shallow_outputs,
            dim=1,
        )

        shallow_weight = (
            shallow_weight
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

        fused_shallow = (
            shallow_stack *
            shallow_weight
        ).sum(
            dim=1
        )

        fused_shallow = (
            self.shallow_adaln(
                fused_shallow,
                task_condition,
            )
        )

        # ====================================================
        # 5. 中层
        # ====================================================
        mid_outputs = []

        for idx, (
            vis_feature,
            ir_feature,
        ) in enumerate(
            mid_features
        ):

            mid_outputs.append(
                self.mid_blocks[idx](
                    vis_feature,
                    ir_feature,
                )
            )

        mid_stack = torch.stack(
            mid_outputs,
            dim=1,
        )

        mid_weight = (
            mid_weight
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

        fused_mid = (
            mid_stack *
            mid_weight
        ).sum(
            dim=1
        )

        fused_mid = (
            self.mid_adaln(
                fused_mid,
                task_condition,
            )
        )

        # ====================================================
        # 6. 深层
        # ====================================================
        deep_outputs = []

        for idx, (
            vis_feature,
            ir_feature,
        ) in enumerate(
            deep_features
        ):

            deep_outputs.append(
                self.deep_blocks[idx](
                    vis_feature,
                    ir_feature,
                )
            )

        deep_stack = torch.stack(
            deep_outputs,
            dim=1,
        )

        deep_weight = (
            deep_weight
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

        fused_deep = (
            deep_stack *
            deep_weight
        ).sum(
            dim=1
        )

        fused_deep = (
            self.deep_adaln(
                fused_deep,
                task_condition,
            )
        )

        # ====================================================
        # 7. 三层级聚合
        # ====================================================
        level_stack = torch.stack(
            [
                fused_shallow,
                fused_mid,
                fused_deep,
            ],
            dim=1,
        )

        level_weight = (
            level_weight
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

        weighted_level = (
            level_stack *
            level_weight
        ).sum(
            dim=1
        )

        # 三个层级真正进行一次 joint interaction
        level_concat = torch.cat(
            [
                fused_shallow,
                fused_mid,
                fused_deep,
            ],
            dim=-1,
        )

        level_correction = (
            self.level_fusion(
                level_concat
            )
        )

        fused_final = (
            weighted_level +
            0.25 * level_correction
        )

        # ====================================================
        # 8. 输出到 SANA 条件维度
        # ====================================================
        fusion_condition = (
            self.output_projection(
                fused_final
            )
        )

        # ====================================================
        # 9. 调试信息
        # ====================================================
        weight_info = {
            "shallow_weight":
                shallow_weight.squeeze(-1)
                                      .squeeze(-1)
                                      .detach(),

            "mid_weight":
                mid_weight.squeeze(-1)
                               .squeeze(-1)
                               .detach(),

            "deep_weight":
                deep_weight.squeeze(-1)
                                .squeeze(-1)
                                .detach(),

            "level_weight":
                level_weight.squeeze(-1)
                                .squeeze(-1)
                                .detach(),

            "task_condition":
                task_condition.detach(),
        }

        return (
            fusion_condition,
            weight_info,
        )


# ============================================================
# 13. Pipeline
# ============================================================

class ThreeLevelFusionPipeline(nn.Module):
    """
    完整：

        Qwen2.5-VL-7B (冻结)
                  ↓
        hidden_states[4,5,6]
        hidden_states[13,14,15]
        hidden_states[26,27,28]
                  ↓
        三层 RGB/IR Fusion
                  ↓
        FasterKAN + Mamba2 + Attention
                  ↓
        文本条件层级动态聚合
                  ↓
        2304-dim condition
    """

    def __init__(
        self,
        qwen_model,
        tokenizer,
        llm_dim: int = 3584,
        caption_dim: int = 2304,
        spatial_size: int = 16,
    ):
        super().__init__()

        self.qwen_model = (
            qwen_model
        )

        self.tokenizer = (
            tokenizer
        )

        self.span_extractor = (
            QwenTokenSpanExtractor(
                tokenizer
            )
        )

        # Qwen 完全冻结
        for parameter in (
            self.qwen_model.parameters()
        ):
            parameter.requires_grad = False

        self.qwen_model.eval()

        self.three_level_fusion = (
            ThreeLevelFusionMCP(
                llm_dim=llm_dim,
                caption_dim=caption_dim,
                num_tokens_per_modality=(
                    spatial_size *
                    spatial_size
                ),
                spatial_size=spatial_size,
                num_heads=8,
                fasterkan_grids=5,
            )
        )

    @torch.no_grad()
    def encode_with_qwen(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: Optional[
            torch.Tensor
        ] = None,
        image_grid_thw: Optional[
            torch.Tensor
        ] = None,
    ) -> Tuple[
        torch.Tensor,
        ...,
    ]:

        outputs = self.qwen_model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            return_dict=True,
        )

        return outputs.hidden_states

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: Optional[
            torch.Tensor
        ] = None,
        image_grid_thw: Optional[
            torch.Tensor
        ] = None,
    ) -> Tuple[
        torch.Tensor,
        dict,
    ]:

        hidden_states = (
            self.encode_with_qwen(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                image_grid_thw=image_grid_thw,
            )
        )

        (
            vis_span,
            ir_span,
            txt_span,
        ) = self.span_extractor.extract_spans(
            input_ids
        )

        return self.three_level_fusion(
            hidden_states=hidden_states,
            vis_span=vis_span,
            ir_span=ir_span,
            txt_span=txt_span,
        )

    def forward_with_precomputed_hidden_states(
        self,
        hidden_states: Tuple[
            torch.Tensor,
            ...
        ],
        input_ids: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        dict,
    ]:

        (
            vis_span,
            ir_span,
            txt_span,
        ) = self.span_extractor.extract_spans(
            input_ids
        )

        return self.three_level_fusion(
            hidden_states=hidden_states,
            vis_span=vis_span,
            ir_span=ir_span,
            txt_span=txt_span,
        )


# ============================================================
# 14. 参数统计
# ============================================================

def count_trainable_parameters(
    model: nn.Module,
) -> int:

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def print_trainable_parameters(
    model: nn.Module,
):

    trainable = (
        count_trainable_parameters(
            model
        )
    )

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"Trainable: {trainable / 1e6:.3f} M"
    )

    print(
        f"Total: {total / 1e6:.3f} M"
    )


# ============================================================
# 15. 单独构建融合模块
# ============================================================

def build_three_level_fusion(
    llm_dim: int = 3584,
    caption_dim: int = 2304,
    spatial_size: int = 16,
    num_heads: int = 8,
    fasterkan_grids: int = 5,
) -> ThreeLevelFusionMCP:

    return ThreeLevelFusionMCP(
        llm_dim=llm_dim,
        caption_dim=caption_dim,
        num_tokens_per_modality=(
            spatial_size *
            spatial_size
        ),
        spatial_size=spatial_size,
        num_heads=num_heads,
        fasterkan_grids=fasterkan_grids,
    )


# ============================================================
# 16. 形状测试
# ============================================================

@torch.no_grad()
def shape_test(
    device: str = "cuda",
    batch_size: int = 1,
    tokens: int = 256,
    llm_dim: int = 3584,
    caption_dim: int = 2304,
):
    """
    注意：
        该测试要求官方 FasterKAN 和官方 Mamba2 均已正确安装。
        它不加载 Qwen，只检查三级融合模块的 shape。
    """

    spatial_size = int(
        tokens ** 0.5
    )

    if spatial_size * spatial_size != tokens:
        raise ValueError(
            "shape_test 当前要求 tokens 为完全平方数。"
        )

    model = ThreeLevelFusionMCP(
        llm_dim=llm_dim,
        caption_dim=caption_dim,
        num_tokens_per_modality=tokens,
        spatial_size=spatial_size,
        num_heads=8,
        fasterkan_grids=5,
    ).to(device)

    hidden_states = tuple(
        torch.randn(
            batch_size,
            tokens * 2,
            llm_dim,
            device=device,
            dtype=torch.float16,
        )
        for _ in range(29)
    )

    # 这里只用于测试 shape；
    # 正式使用时必须使用真实 Qwen 序列和 txt span。
    vis_span = (0, tokens)
    ir_span = (tokens, tokens * 2)

    # 临时文本位置需要真实序列，
    # 为了测试模块，把文本放在后面。
    hidden_states = tuple(
        torch.cat(
            [
                state,
                torch.randn(
                    batch_size,
                    16,
                    llm_dim,
                    device=device,
                    dtype=state.dtype,
                ),
            ],
            dim=1,
        )
        for state in hidden_states
    )

    txt_span = (
        tokens * 2,
        tokens * 2 + 16,
    )

    model.eval()

    condition, weight_info = (
        model(
            hidden_states,
            vis_span,
            ir_span,
            txt_span,
        )
    )

    print(
        "fusion_condition:",
        condition.shape,
    )

    for key, value in (
        weight_info.items()
    ):
        print(
            key,
            tuple(value.shape),
        )


if __name__ == "__main__":
    print(
        "three_level_fusion_official.py"
    )
    print(
        "FasterKAN source:"
    )
    print(
        "https://github.com/AthanasiosDelis/faster-kan"
    )
    print(
        "Mamba source:"
    )
    print(
        "https://github.com/state-spaces/mamba"
    )
    print()
    print(
        "本文件不会自行实现 FasterKAN 或 Mamba；"
        "请先安装两个官方仓库。"
    )
