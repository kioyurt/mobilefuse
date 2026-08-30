# -*- coding: utf-8 -*-

"""
QSCFMCP
============================================================

Qwen Semantic Cross-modal Frequency Mamba
Conditioning Projector

For:
    Visible + Infrared Image Fusion

Main pipeline:

    Qwen2.5-VL hidden states
                |
        Visible / Infrared split
                |
                v
    Cross-Modal Semantic Alignment
                |
                v
    Shared / Complementary Decomposition
                |
                v
    Frequency-aware Adaptive Fusion
                |
                v
    Bidirectional Selective State Space Mixer
                |
                v
    Mobile-O Layer-wise Fusion
                |
                v
        2048 -> 512 bottleneck
                |
                v
       Residual Refinement
                |
                v
        512 -> 2304
                |
                v
        SANA DiT condition

Important:
    - No mamba-ssm dependency.
    - No CUDA extension required.
    - Qwen remains frozen externally.
    - Projector is trainable.
    - SANA condition dimension remains 2304.

The SSM implementation is inspired by the structural principles
of Vision Mamba / selective state-space modeling:

    x
    ↓
    norm
    ↓
    input-dependent Δt / B / C
    ↓
    forward selective scan
    +
    backward selective scan
    ↓
    bidirectional fusion
    ↓
    residual

It is intentionally implemented in pure PyTorch so that the
module works without mamba-ssm.
"""


from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Utility
# ============================================================


def inverse_softplus(
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Numerically stable inverse softplus.

        softplus(y) = x
    """

    return x + torch.log(
        -torch.expm1(-x)
    )


# ============================================================
# RMSNorm
# ============================================================


class RMSNorm(nn.Module):
    """
    Pure PyTorch RMSNorm.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
    ) -> None:

        super().__init__()

        self.eps = eps

        self.weight = nn.Parameter(
            torch.ones(dim)
        )


    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        rms = torch.rsqrt(
            x.pow(2).mean(
                dim=-1,
                keepdim=True,
            )
            + self.eps
        )

        return (
            x
            * rms
            * self.weight
        )


# ============================================================
# Cross-modal Semantic Alignment
# ============================================================


class CrossModalSemanticAlignment(
    nn.Module
):
    """
    Bidirectional semantic interaction.

    Visible queries Infrared.
    Infrared queries Visible.

    Unlike simple concatenation, each modality obtains
    complementary information from the other modality.

    Input:

        visible:
            [B, Nv, C]

        infrared:
            [B, Ni, C]

    Output:

        visible_aligned:
            [B, Nv, C]

        infrared_aligned:
            [B, Ni, C]

        shared:
            [B, N, C]

        complementary:
            [B, N, C]
    """

    def __init__(
        self,
        dim: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:

        super().__init__()

        if dim % num_heads != 0:

            raise ValueError(
                f"dim={dim} must be divisible by "
                f"num_heads={num_heads}."
            )


        self.dim = dim

        self.num_heads = num_heads

        self.head_dim = (
            dim // num_heads
        )

        self.scale = (
            self.head_dim ** -0.5
        )


        # --------------------------------------------------------
        # Visible -> IR
        # --------------------------------------------------------

        self.v_q = nn.Linear(
            dim,
            dim,
            bias=False,
        )

        self.i_k = nn.Linear(
            dim,
            dim,
            bias=False,
        )

        self.i_v = nn.Linear(
            dim,
            dim,
            bias=False,
        )


        # --------------------------------------------------------
        # IR -> Visible
        # --------------------------------------------------------

        self.i_q = nn.Linear(
            dim,
            dim,
            bias=False,
        )

        self.v_k = nn.Linear(
            dim,
            dim,
            bias=False,
        )

        self.v_v = nn.Linear(
            dim,
            dim,
            bias=False,
        )


        self.v_out = nn.Linear(
            dim,
            dim,
        )

        self.i_out = nn.Linear(
            dim,
            dim,
        )


        self.v_norm = nn.LayerNorm(
            dim
        )

        self.i_norm = nn.LayerNorm(
            dim
        )


        self.dropout = nn.Dropout(
            dropout
        )


        # --------------------------------------------------------
        # Shared / complementary gates
        # --------------------------------------------------------

        self.shared_gate = nn.Sequential(

            nn.Linear(
                dim * 2,
                dim,
            ),

            nn.GELU(),

            nn.Linear(
                dim,
                dim,
            ),

            nn.Sigmoid(),
        )


        self.comp_gate = nn.Sequential(

            nn.Linear(
                dim * 2,
                dim,
            ),

            nn.GELU(),

            nn.Linear(
                dim,
                dim,
            ),

            nn.Sigmoid(),
        )


    def _reshape_heads(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        b, n, c = x.shape

        x = x.reshape(
            b,
            n,
            self.num_heads,
            self.head_dim,
        )

        return x.permute(
            0,
            2,
            1,
            3,
        )


    def _restore_heads(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        b, h, n, d = x.shape

        x = x.permute(
            0,
            2,
            1,
            3,
        ).contiguous()

        return x.reshape(
            b,
            n,
            h * d,
        )


    def _cross_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:

        q = self._reshape_heads(q)

        k = self._reshape_heads(k)

        v = self._reshape_heads(v)


        scores = torch.matmul(
            q,
            k.transpose(
                -2,
                -1,
            ),
        )

        scores = (
            scores
            * self.scale
        )


        attn = torch.softmax(
            scores,
            dim=-1,
        )


        attn = self.dropout(
            attn
        )


        output = torch.matmul(
            attn,
            v,
        )


        return self._restore_heads(
            output
        )


    def forward(
        self,
        visible: torch.Tensor,
        infrared: torch.Tensor,
    ):

        if visible.ndim != 3:

            raise ValueError(
                "visible must be [B,N,C], "
                f"got {tuple(visible.shape)}"
            )

        if infrared.ndim != 3:

            raise ValueError(
                "infrared must be [B,N,C], "
                f"got {tuple(infrared.shape)}"
            )


        # --------------------------------------------------------
        # Normalize
        # --------------------------------------------------------

        v = self.v_norm(
            visible
        )

        i = self.i_norm(
            infrared
        )


        # --------------------------------------------------------
        # V -> IR
        # --------------------------------------------------------

        q_v = self.v_q(v)

        k_i = self.i_k(i)

        val_i = self.i_v(i)


        v_cross = (
            self._cross_attention(
                q_v,
                k_i,
                val_i,
            )
        )


        v_cross = self.v_out(
            v_cross
        )


        # --------------------------------------------------------
        # IR -> V
        # --------------------------------------------------------

        q_i = self.i_q(i)

        k_v = self.v_k(v)

        val_v = self.v_v(v)


        i_cross = (
            self._cross_attention(
                q_i,
                k_v,
                val_v,
            )
        )


        i_cross = self.i_out(
            i_cross
        )


        # --------------------------------------------------------
        # Residual alignment
        # --------------------------------------------------------

        visible_aligned = (
            visible
            +
            v_cross
        )


        infrared_aligned = (
            infrared
            +
            i_cross
        )


        # --------------------------------------------------------
        # Build modality pair representation
        #
        # Both modalities have potentially different N.
        #
        # Their global statistics provide a common semantic
        # reference without assuming Nv == Nir.
        # --------------------------------------------------------

        v_global = (
            visible_aligned.mean(
                dim=1,
                keepdim=True,
            )
        )

        i_global = (
            infrared_aligned.mean(
                dim=1,
                keepdim=True,
            )
        )


        pair_global = torch.cat(
            [
                v_global,
                i_global,
            ],
            dim=-1,
        )


        shared_gate = self.shared_gate(
            pair_global
        )


        comp_gate = self.comp_gate(
            pair_global
        )


        # --------------------------------------------------------
        # Shared semantics
        # --------------------------------------------------------

        shared_v = (
            shared_gate
            * 0.5
            * (
                visible_aligned
                +
                i_global
            )
        )


        shared_i = (
            shared_gate
            * 0.5
            * (
                infrared_aligned
                +
                v_global
            )
        )


        # --------------------------------------------------------
        # Complementary information
        #
        # Difference is explicitly retained instead of forcing
        # both modalities to collapse into one representation.
        # --------------------------------------------------------

        comp_v = (
            comp_gate
            * (
                visible_aligned
                -
                i_global
            )
        )


        comp_i = (
            comp_gate
            * (
                infrared_aligned
                -
                v_global
            )
        )


        shared = torch.cat(
            [
                shared_v,
                shared_i,
            ],
            dim=1,
        )


        complementary = torch.cat(
            [
                comp_v,
                comp_i,
            ],
            dim=1,
        )


        return (
            visible_aligned,
            infrared_aligned,
            shared,
            complementary,
        )


# ============================================================
# Frequency-aware decomposition
# ============================================================


class FrequencyAwareDecomposition(
    nn.Module
):
    """
    Frequency decomposition along token sequence.

    Instead of treating all Qwen semantic tokens identically,
    we explicitly model low-frequency semantic continuity and
    high-frequency modality-specific variation.

    Uses rFFT for an exact real-valued representation.

    Output:

        low:
            [B,N,C]

        high:
            [B,N,C]

        high_ratio:
            [B,N,C]

    Notes:

        The mask is learned as a smooth spectral gate rather than
        a hard fixed cutoff.

        This is more suitable for semantic token sequences than
        a hand-designed hard split.
    """

    def __init__(
        self,
        dim: int = 2048,
        init_low_ratio: float = 0.35,
    ) -> None:

        super().__init__()

        if not (
            0.0
            <
            init_low_ratio
            <
            1.0
        ):

            raise ValueError(
                "init_low_ratio must be in (0,1)."
            )


        self.dim = dim

        # Learnable frequency preference.
        self.logit_low = nn.Parameter(
            torch.tensor(
                math.log(
                    init_low_ratio
                    /
                    (1.0 - init_low_ratio)
                )
            )
        )


        # Channel-dependent frequency gate.
        self.channel_gate = nn.Sequential(

            nn.LayerNorm(
                dim
            ),

            nn.Linear(
                dim,
                dim,
            ),

            nn.GELU(),

            nn.Linear(
                dim,
                dim,
            ),

            nn.Sigmoid(),
        )


    def forward(
        self,
        x: torch.Tensor,
    ):

        if x.ndim != 3:

            raise ValueError(
                "x must be [B,N,C]."
            )


        b, n, c = x.shape


        # --------------------------------------------------------
        # rFFT along sequence dimension
        # --------------------------------------------------------

        x_fft = torch.fft.rfft(
            x,
            dim=1,
        )


        num_freq = (
            x_fft.shape[1]
        )


        # --------------------------------------------------------
        # Normalized frequency coordinate
        # --------------------------------------------------------

        freq = torch.linspace(
            0.0,
            1.0,
            num_freq,
            device=x.device,
            dtype=x.real.dtype,
        )


        freq = freq.view(
            1,
            num_freq,
            1,
        )


        # --------------------------------------------------------
        # Learnable soft low-pass gate
        #
        # cutoff ∈ (0,1)
        # --------------------------------------------------------

        cutoff = torch.sigmoid(
            self.logit_low
        )


        temperature = 0.12


        low_gate = torch.sigmoid(
            (
                cutoff
                - freq
            )
            /
            temperature
        )


        high_gate = (
            1.0
            -
            low_gate
        )


        # --------------------------------------------------------
        # Low / high spectral components
        # --------------------------------------------------------

        low_fft = (
            x_fft
            * low_gate
        )


        high_fft = (
            x_fft
            * high_gate
        )


        low = torch.fft.irfft(
            low_fft,
            n=n,
            dim=1,
        )


        high = torch.fft.irfft(
            high_fft,
            n=n,
            dim=1,
        )


        # --------------------------------------------------------
        # Channel-dependent modulation
        # --------------------------------------------------------

        gate = self.channel_gate(
            x
        )


        low = (
            low
            * (
                0.5
                +
                0.5 * gate
            )
        )


        high = (
            high
            * (
                1.5
                -
                0.5 * gate
            )
        )


        return (
            low,
            high,
        )


# ============================================================
# Adaptive shared/complementary frequency fusion
# ============================================================


class AdaptiveComplementaryFusion(
    nn.Module
):
    """
    Learns how much shared semantic information and modality
    specific high-frequency information should be retained.

    Inputs:

        shared:
            [B,N,C]

        complementary:
            [B,N,C]

        low:
            [B,N,C]

        high:
            [B,N,C]

    Output:

        [B,N,C]
    """

    def __init__(
        self,
        dim: int = 2048,
    ) -> None:

        super().__init__()


        self.gate = nn.Sequential(

            nn.LayerNorm(
                dim * 4
            ),

            nn.Linear(
                dim * 4,
                dim,
            ),

            nn.GELU(),

            nn.Linear(
                dim,
                dim * 3,
            ),
        )


        self.output = nn.Sequential(

            nn.Linear(
                dim,
                dim,
            ),

            nn.GELU(),

            nn.Linear(
                dim,
                dim,
            ),
        )


        self.norm = nn.LayerNorm(
            dim
        )


    def forward(
        self,
        shared: torch.Tensor,
        complementary: torch.Tensor,
        low: torch.Tensor,
        high: torch.Tensor,
    ) -> torch.Tensor:

        if not (
            shared.shape
            == complementary.shape
            == low.shape
            == high.shape
        ):

            raise ValueError(
                "All frequency fusion tensors "
                "must have identical shapes. "
                f"shared={shared.shape}, "
                f"complementary={complementary.shape}, "
                f"low={low.shape}, "
                f"high={high.shape}"
            )


        statistics = torch.cat(
            [
                shared,
                complementary,
                low,
                high,
            ],
            dim=-1,
        )


        logits = self.gate(
            statistics
        )


        b, n, _ = logits.shape

        logits = logits.reshape(
            b,
            n,
            3,
            -1,
        )


        weights = torch.softmax(
            logits,
            dim=2,
        )


        shared_weight = (
            weights[:, :, 0]
        )


        complementary_weight = (
            weights[:, :, 1]
        )


        high_weight = (
            weights[:, :, 2]
        )


        fused = (
            shared_weight
            * shared

            +
            complementary_weight
            * complementary

            +
            high_weight
            * (
                low + high
            )
        )


        fused = self.output(
            self.norm(
                fused
            )
        )


        return fused


# ============================================================
# Vision-Mamba-style selective SSM
# ============================================================


class BidirectionalSelectiveSSM(
    nn.Module
):
    """
    Pure PyTorch Vision-Mamba-style bidirectional selective SSM.

    This is deliberately implemented without mamba-ssm.

    The module uses:

        x
        ↓
        dt(x)
        B(x)
        C(x)
        ↓
        continuous-time diagonal A
        ↓
        selective discretization
        ↓
        forward scan
        +
        reverse scan
        ↓
        learned directional fusion

    Continuous state equation:

        dh(t)
        -------
         dt

        =
        A h(t)
        +
        B(t) x(t)

    Discretized:

        h_t
        =
        exp(dt_t A) h_{t-1}
        +
        dt_t B_t x_t

    Output:

        y_t = C_t h_t + D x_t


    The forward and backward states are then combined.

    This preserves the central idea of Vision Mamba's
    bidirectional sequence modeling while avoiding the external
    CUDA mamba_ssm dependency.
    """

    def __init__(
        self,
        dim: int = 512,
        state_dim: int = 16,
        expand: int = 2,
        dt_rank: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:

        super().__init__()


        self.dim = dim

        self.state_dim = state_dim

        self.inner_dim = (
            dim * expand
        )


        if dt_rank is None:

            dt_rank = max(
                1,
                math.ceil(
                    dim / 16
                ),
            )


        self.dt_rank = dt_rank


        # --------------------------------------------------------
        # Input projection
        #
        # x -> u + z
        # --------------------------------------------------------

        self.in_proj = nn.Linear(
            dim,
            self.inner_dim * 2,
        )


        # --------------------------------------------------------
        # Local depthwise convolution
        #
        # Vision Mamba also uses local convolution around the
        # SSM mixer.
        # --------------------------------------------------------

        self.conv1d = nn.Conv1d(
            self.inner_dim,
            self.inner_dim,
            kernel_size=3,
            padding=1,
            groups=self.inner_dim,
        )


        # --------------------------------------------------------
        # Selective parameters
        # --------------------------------------------------------

        self.x_proj_fwd = nn.Linear(
            self.inner_dim,
            dt_rank
            +
            2 * state_dim,
            bias=False,
        )


        self.dt_proj_fwd = nn.Linear(
            dt_rank,
            self.inner_dim,
            bias=True,
        )


        self.x_proj_bwd = nn.Linear(
            self.inner_dim,
            dt_rank
            +
            2 * state_dim,
            bias=False,
        )


        self.dt_proj_bwd = nn.Linear(
            dt_rank,
            self.inner_dim,
            bias=True,
        )


        # --------------------------------------------------------
        # Diagonal continuous A
        #
        # A = -exp(A_log)
        # --------------------------------------------------------

        self.A_log_fwd = nn.Parameter(
            torch.log(
                torch.arange(
                    1,
                    state_dim + 1,
                    dtype=torch.float32,
                )
                .repeat(
                    self.inner_dim,
                    1,
                )
            )
        )


        self.A_log_bwd = nn.Parameter(
            torch.log(
                torch.arange(
                    1,
                    state_dim + 1,
                    dtype=torch.float32,
                )
                .repeat(
                    self.inner_dim,
                    1,
                )
            )
        )


        self.D_fwd = nn.Parameter(
            torch.ones(
                self.inner_dim
            )
        )


        self.D_bwd = nn.Parameter(
            torch.ones(
                self.inner_dim
            )
        )


        # --------------------------------------------------------
        # Bidirectional fusion
        # --------------------------------------------------------

        self.direction_gate = nn.Sequential(

            nn.Linear(
                self.inner_dim * 2,
                self.inner_dim,
            ),

            nn.GELU(),

            nn.Linear(
                self.inner_dim,
                self.inner_dim,
            ),

            nn.Sigmoid(),
        )


        self.out_proj = nn.Linear(
            self.inner_dim,
            dim,
        )


        self.dropout = nn.Dropout(
            dropout
        )


        self.norm = nn.LayerNorm(
            dim
        )


        # --------------------------------------------------------
        # dt initialization
        # --------------------------------------------------------

        dt_min = 1e-3

        dt_max = 1e-1

        dt_init = torch.exp(
            torch.rand(
                self.inner_dim,
                dtype=self.dt_proj_fwd.bias.dtype,
                device=self.dt_proj_fwd.bias.device,
            )
            * (
                    math.log(
                        dt_max
                    )
                    -
                    math.log(
                        dt_min
                    )
            )
            +
            math.log(
                dt_min
            )
        )

        dt_bias = inverse_softplus(
            dt_init
        )

        with torch.no_grad():
            self.dt_proj_fwd.bias.copy_(
                dt_bias
            )

            self.dt_proj_bwd.bias.copy_(
                dt_bias
            )


    def _selective_parameters(
        self,
        x: torch.Tensor,
        x_proj: nn.Linear,
        dt_proj: nn.Linear,
    ):

        params = x_proj(
            x
        )


        dt_part = params[
            ...,
            :self.dt_rank
        ]


        state_part = params[
            ...,
            self.dt_rank:
        ]


        B = state_part[
            ...,
            :self.state_dim
        ]


        C = state_part[
            ...,
            self.state_dim:
        ]


        dt = dt_proj(
            dt_part
        )


        dt = F.softplus(
            dt
        )


        # Bound dt to avoid numerical explosion
        dt = dt.clamp(
            min=1e-4,
            max=1.0,
        )


        return (
            dt,
            B,
            C,
        )


    def _scan_direction(
        self,
        x: torch.Tensor,
        A_log: torch.Tensor,
        D: torch.Tensor,
        x_proj: nn.Linear,
        dt_proj: nn.Linear,
        reverse: bool = False,
    ) -> torch.Tensor:
        """
        Selective recurrent scan.

        x:
            [B,N,D]
        """

        if reverse:

            x_work = torch.flip(
                x,
                dims=[1],
            )

        else:

            x_work = x


        b, n, d = x_work.shape


        dt, B, C = (
            self._selective_parameters(
                x_work,
                x_proj,
                dt_proj,
            )
        )


        # --------------------------------------------------------
        # A < 0 guarantees stable decay.
        # --------------------------------------------------------

        A = -torch.exp(
            A_log
        )


        # --------------------------------------------------------
        # State:
        #
        # [B,D,state_dim]
        # --------------------------------------------------------

        state = torch.zeros(
            b,
            d,
            self.state_dim,
            dtype=x.dtype,
            device=x.device,
        )


        outputs = []


        for t in range(n):

            xt = x_work[
                :,
                t,
                :
            ]


            dt_t = dt[
                :,
                t,
                :
            ]


            B_t = B[
                :,
                t,
                :
            ]


            C_t = C[
                :,
                t,
                :
            ]


            # ----------------------------------------------------
            # exp(dt*A)
            # ----------------------------------------------------

            decay = torch.exp(
                dt_t.unsqueeze(-1)
                * A.unsqueeze(0)
            )


            # ----------------------------------------------------
            # Selective input injection
            #
            # u_t = dt_t * B_t * x_t
            # ----------------------------------------------------

            injection = (
                dt_t.unsqueeze(-1)
                *
                B_t.unsqueeze(1)
                *
                xt.unsqueeze(-1)
            )


            state = (
                decay
                * state
                +
                injection
            )


            # ----------------------------------------------------
            # C_t h_t
            # ----------------------------------------------------

            yt = torch.sum(
                C_t.unsqueeze(1)
                * state,
                dim=-1,
            )


            yt = (
                yt
                +
                D.unsqueeze(0)
                * xt
            )


            outputs.append(
                yt.unsqueeze(1)
            )


        y = torch.cat(
            outputs,
            dim=1,
        )


        if reverse:

            y = torch.flip(
                y,
                dims=[1],
            )


        return y


    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        residual = x


        x = self.norm(
            x
        )


        # --------------------------------------------------------
        # Input expansion
        # --------------------------------------------------------

        projected = self.in_proj(
            x
        )


        u, z = torch.chunk(
            projected,
            2,
            dim=-1,
        )


        # --------------------------------------------------------
        # Local positional mixing
        # --------------------------------------------------------

        u = u.transpose(
            1,
            2,
        )


        u = self.conv1d(
            u
        )


        u = u.transpose(
            1,
            2,
        )


        u = F.silu(
            u
        )


        # --------------------------------------------------------
        # Forward SSM
        # --------------------------------------------------------

        y_forward = (
            self._scan_direction(
                u,
                self.A_log_fwd,
                self.D_fwd,
                self.x_proj_fwd,
                self.dt_proj_fwd,
                reverse=False,
            )
        )


        # --------------------------------------------------------
        # Backward SSM
        # --------------------------------------------------------

        y_backward = (
            self._scan_direction(
                u,
                self.A_log_bwd,
                self.D_bwd,
                self.x_proj_bwd,
                self.dt_proj_bwd,
                reverse=True,
            )
        )


        # --------------------------------------------------------
        # Vision-Mamba-style bidirectional fusion
        # --------------------------------------------------------

        direction_input = torch.cat(
            [
                y_forward,
                y_backward,
            ],
            dim=-1,
        )


        gate = self.direction_gate(
            direction_input
        )


        y = (
            gate
            * y_forward
            +
            (
                1.0 - gate
            )
            * y_backward
        )


        # --------------------------------------------------------
        # Gating with z
        # --------------------------------------------------------

        y = y * F.silu(
            z
        )


        y = self.out_proj(
            y
        )


        y = self.dropout(
            y
        )


        # --------------------------------------------------------
        # Residual
        # --------------------------------------------------------

        return (
            residual
            +
            y
        )


# ============================================================
# Mobile-O style sequence refinement
# ============================================================


class MobileSequenceRefinement(
    nn.Module
):
    """
    Lightweight refinement after SSM.

    Preserves local/global interaction without relying on a
    heavyweight transformer stack.
    """

    def __init__(
        self,
        dim: int = 512,
        kernel_size: int = 3,
        expansion: int = 4,
        dropout: float = 0.0,
    ) -> None:

        super().__init__()


        self.norm1 = nn.LayerNorm(
            dim
        )


        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
        )


        self.pwconv = nn.Sequential(

            nn.Linear(
                dim,
                dim * expansion,
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                dim * expansion,
                dim,
            ),
        )


        self.norm2 = nn.LayerNorm(
            dim
        )


        self.channel_gate = nn.Sequential(

            nn.Linear(
                dim,
                dim,
            ),

            nn.Sigmoid(),
        )


    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        residual = x


        x = self.norm1(
            x
        )


        # Local token mixing

        y = x.transpose(
            1,
            2,
        )


        y = self.dwconv(
            y
        )


        y = y.transpose(
            1,
            2,
        )


        y = F.gelu(
            y
        )


        x = (
            residual
            +
            y
        )


        # Channel refinement

        residual = x

        x = self.norm2(
            x
        )


        gate = self.channel_gate(
            x
        )


        x = x * gate


        x = self.pwconv(
            x
        )


        return (
            residual
            +
            x
        )


# ============================================================
# QSCFMCP
# ============================================================


class QSCFMCP(nn.Module):
    """
    Full Qwen Semantic Cross-modal Frequency Mamba
    Conditioning Projector.


    Input:

        visible_hidden_states:
            List[
                Tensor[B,Nv,2048]
            ]

        infrared_hidden_states:
            List[
                Tensor[B,Ni,2048]
            ]


    Output:

        Tensor[
            B,
            Nv + Ni,
            2304
        ]


    Compatibility:

        Mobile-O / SANA condition dimension = 2304
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 512,
        output_dim: int = 2304,
        num_layers: int = 8,
        num_heads: int = 8,
        state_dim: int = 16,
        ssm_expand: int = 2,
        dropout: float = 0.0,
    ) -> None:

        super().__init__()


        self.input_dim = input_dim

        self.hidden_dim = hidden_dim

        self.output_dim = output_dim

        self.num_layers = num_layers


        # ========================================================
        # Shared input normalization
        # ========================================================

        self.visible_input_norm = nn.LayerNorm(
            input_dim
        )


        self.infrared_input_norm = nn.LayerNorm(
            input_dim
        )


        # ========================================================
        # Semantic alignment
        # ========================================================

        self.semantic_align = (
            CrossModalSemanticAlignment(
                dim=input_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
        )


        # ========================================================
        # Frequency decomposition
        # ========================================================

        self.frequency = (
            FrequencyAwareDecomposition(
                dim=input_dim
            )
        )


        # ========================================================
        # Complementary fusion
        # ========================================================

        self.frequency_fusion = (
            AdaptiveComplementaryFusion(
                dim=input_dim
            )
        )


        # ========================================================
        # Mobile-O layer-wise fusion
        #
        # Temperature-scaled learnable weights.
        # ========================================================

        self.layer_weights = nn.Parameter(
            torch.ones(
                num_layers
            )
        )


        self.temperature = nn.Parameter(
            torch.tensor(
                math.log(
                    math.exp(0.5) - 1.0
                )
            )
        )


        # ========================================================
        # 2048 -> 512
        # ========================================================

        self.input_projection = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim,
            ),

            nn.GELU(),

            nn.LayerNorm(
                hidden_dim
            ),
        )


        # ========================================================
        # Vision-Mamba-style bidirectional SSM
        # ========================================================

        self.ssm1 = (
            BidirectionalSelectiveSSM(
                dim=hidden_dim,
                state_dim=state_dim,
                expand=ssm_expand,
                dropout=dropout,
            )
        )


        self.ssm2 = (
            BidirectionalSelectiveSSM(
                dim=hidden_dim,
                state_dim=state_dim,
                expand=ssm_expand,
                dropout=dropout,
            )
        )


        # ========================================================
        # Refinement
        # ========================================================

        self.refine1 = (
            MobileSequenceRefinement(
                dim=hidden_dim,
                dropout=dropout,
            )
        )


        self.refine2 = (
            MobileSequenceRefinement(
                dim=hidden_dim,
                dropout=dropout,
            )
        )


        # ========================================================
        # Output
        #
        # 512 -> 2304
        # ========================================================

        self.output_projection = nn.Sequential(

            nn.LayerNorm(
                hidden_dim
            ),

            nn.Linear(
                hidden_dim,
                output_dim,
            ),
        )


        # ========================================================
        # Final normalization
        # ========================================================

        self.output_norm = nn.LayerNorm(
            output_dim
        )


        # ========================================================
        # Initialization
        # ========================================================

        self._init_weights()


    def _init_weights(
        self,
    ) -> None:

        for module in self.modules():

            if isinstance(
                module,
                nn.Linear,
            ):

                nn.init.xavier_uniform_(
                    module.weight
                )

                if module.bias is not None:

                    nn.init.zeros_(
                        module.bias
                    )


            elif isinstance(
                module,
                nn.Conv1d,
            ):

                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

                if module.bias is not None:

                    nn.init.zeros_(
                        module.bias
                    )


    # ============================================================
    # Layer-wise fusion
    # ============================================================

    def _layerwise_fusion(
        self,
        hidden_states: List[
            torch.Tensor
        ],
    ) -> torch.Tensor:

        if len(
            hidden_states
        ) != self.num_layers:

            raise ValueError(
                "Layer count mismatch. "
                f"Expected {self.num_layers}, "
                f"got {len(hidden_states)}."
            )


        # --------------------------------------------------------
        # Temperature > 0
        # --------------------------------------------------------

        temperature = (
            F.softplus(
                self.temperature
            )
            +
            1e-4
        )


        weights = torch.softmax(
            self.layer_weights
            /
            temperature,
            dim=0,
        )


        x = (
            hidden_states[0]
            * weights[0]
        )


        for idx in range(
            1,
            len(hidden_states),
        ):

            x = (
                x
                +
                hidden_states[idx]
                * weights[idx]
            )


        return x


    # ============================================================
    # Validate layers
    # ============================================================

    def _validate_inputs(
        self,
        visible_hidden_states,
        infrared_hidden_states,
    ):

        if len(
            visible_hidden_states
        ) != self.num_layers:

            raise ValueError(
                "Visible hidden-state count "
                "does not match num_layers."
            )


        if len(
            infrared_hidden_states
        ) != self.num_layers:

            raise ValueError(
                "Infrared hidden-state count "
                "does not match num_layers."
            )


        for idx in range(
            self.num_layers
        ):

            v = visible_hidden_states[
                idx
            ]

            i = infrared_hidden_states[
                idx
            ]


            if v.ndim != 3:

                raise ValueError(
                    "Visible hidden states must "
                    "be [B,N,C]. "
                    f"Layer {idx}: {v.shape}"
                )


            if i.ndim != 3:

                raise ValueError(
                    "Infrared hidden states must "
                    "be [B,N,C]. "
                    f"Layer {idx}: {i.shape}"
                )


            if v.shape[-1] != self.input_dim:

                raise ValueError(
                    f"Visible hidden dimension "
                    f"must be {self.input_dim}, "
                    f"got {v.shape[-1]}."
                )


            if i.shape[-1] != self.input_dim:

                raise ValueError(
                    f"Infrared hidden dimension "
                    f"must be {self.input_dim}, "
                    f"got {i.shape[-1]}."
                )


            if v.shape[0] != i.shape[0]:

                raise ValueError(
                    "Visible / infrared batch "
                    "sizes must match."
                )


    # ============================================================
    # Forward
    # ============================================================

    def forward(
        self,
        visible_hidden_states: List[
            torch.Tensor
        ],
        infrared_hidden_states: List[
            torch.Tensor
        ],
    ) -> torch.Tensor:

        # ========================================================
        # Validation
        # ========================================================

        self._validate_inputs(
            visible_hidden_states,
            infrared_hidden_states,
        )


        processed_layers = []


        # ========================================================
        # Process each Qwen layer
        # ========================================================

        for layer_idx in range(
            self.num_layers
        ):

            v = (
                visible_hidden_states[
                    layer_idx
                ]
            )

            i = (
                infrared_hidden_states[
                    layer_idx
                ]
            )


            # ----------------------------------------------------
            # Normalize each modality
            # ----------------------------------------------------

            v = self.visible_input_norm(
                v
            )

            i = self.infrared_input_norm(
                i
            )


            # ----------------------------------------------------
            # Cross-modal semantic alignment
            # ----------------------------------------------------

            (
                v_aligned,
                i_aligned,
                shared,
                complementary,
            ) = self.semantic_align(
                v,
                i,
            )


            # ----------------------------------------------------
            # Concatenate aligned modality representations
            #
            # This preserves both modalities instead of reducing
            # them to one mean vector.
            # ----------------------------------------------------

            aligned = torch.cat(
                [
                    v_aligned,
                    i_aligned,
                ],
                dim=1,
            )


            # ----------------------------------------------------
            # Frequency decomposition
            # ----------------------------------------------------

            low, high = (
                self.frequency(
                    aligned
                )
            )


            # ----------------------------------------------------
            # Adaptive shared/complementary fusion
            #
            # Note:
            #
            # shared/complementary have the same sequence shape
            # as aligned.
            # ----------------------------------------------------

            fused = (
                self.frequency_fusion(
                    shared=shared,
                    complementary=complementary,
                    low=low,
                    high=high,
                )
            )


            # ----------------------------------------------------
            # Residual path from aligned semantic tokens
            # ----------------------------------------------------

            fused = (
                fused
                +
                aligned
            )


            processed_layers.append(
                fused
            )


        # ========================================================
        # Mobile-O layer-wise semantic fusion
        # ========================================================

        x = self._layerwise_fusion(
            processed_layers
        )


        # ========================================================
        # 2048 -> 512
        # ========================================================

        x = self.input_projection(
            x
        )


        # ========================================================
        # Bidirectional Vision-Mamba-style SSM
        # ========================================================

        x = self.ssm1(
            x
        )

        x = self.ssm2(
            x
        )


        # ========================================================
        # Lightweight refinement
        # ========================================================

        x = self.refine1(
            x
        )

        x = self.refine2(
            x
        )


        # ========================================================
        # 512 -> 2304
        # ========================================================

        x = self.output_projection(
            x
        )


        x = self.output_norm(
            x
        )


        return x


# ============================================================
# Debug utility
# ============================================================


@torch.no_grad()
def inspect_qscfmcp(
    model: QSCFMCP,
    visible_hidden_states,
    infrared_hidden_states,
) -> None:

    print(
        "\n"
        + "=" * 70
    )

    print(
        "QSCFMCP Inspection"
    )

    print(
        "=" * 70
    )


    for idx in range(
        len(visible_hidden_states)
    ):

        print(
            f"Layer {idx}:"
        )

        print(
            "  Visible :",
            tuple(
                visible_hidden_states[
                    idx
                ].shape
            ),
        )

        print(
            "  IR      :",
            tuple(
                infrared_hidden_states[
                    idx
                ].shape
            ),
        )


    output = model(
        visible_hidden_states,
        infrared_hidden_states,
    )


    print(
        "\nOutput:"
    )

    print(
        tuple(
            output.shape
        )
    )


    print(
        "=" * 70
    )