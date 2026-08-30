
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Gated RMSNorm
# ============================================================

class RMSNormGated(nn.Module):
    """
    对应官方 Mamba2 中的 RMSNormGated。

    官方调用：
        self.norm(y, z)

    当 norm_before_gate=False：
        RMSNorm(y) * SiLU(z)

    当 norm_before_gate=True：
        RMSNorm(y * SiLU(z))
    """

    def __init__(
        self,
        hidden_dim: int,
        eps: float = 1e-5,
        norm_before_gate: bool = False,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.eps = eps
        self.norm_before_gate = norm_before_gate

        self.weight = nn.Parameter(
            torch.ones(hidden_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:

        if self.norm_before_gate:
            x = x * F.silu(z)

        variance = (
            x.float()
             .pow(2)
             .mean(
                 dim=-1,
                 keepdim=True,
             )
        )

        x = x * torch.rsqrt(
            variance + self.eps
        )

        x = x * self.weight

        if not self.norm_before_gate:
            x = x * F.silu(z)

        return x


# ============================================================
# 2. Causal Depthwise Conv1d
# ============================================================

class CausalDepthwiseConv1d(nn.Module):
    """
    对应官方 Mamba2 的：

        nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1
        )

    官方 fallback 后续取：
        [:, :-(d_conv - 1)]

    输入：
        [B, L, C]

    输出：
        [B, L, C]
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 4,
        bias: bool = True,
    ):
        super().__init__()

        self.channels = channels
        self.kernel_size = kernel_size

        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            groups=channels,
            padding=kernel_size - 1,
            bias=bias,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if x.ndim != 3:
            raise ValueError(
                f"CausalDepthwiseConv1d 输入必须为 "
                f"[B,L,C]，得到 {tuple(x.shape)}"
            )

        B, L, C = x.shape

        if C != self.channels:
            raise ValueError(
                f"channels mismatch: "
                f"expected {self.channels}, got {C}"
            )

        x = x.transpose(
            1,
            2,
        )

        y = self.conv(x)

        # 与官方 fallback 路径一致：
        # conv 输出长度 L + d_conv - 1
        # 保留前 L 个位置。
        y = y[
            ...,
            :L,
        ]

        y = y.transpose(
            1,
            2,
        )

        return F.silu(y)


# ============================================================
# 3. Windows SSD Reference
# ============================================================

class Mamba2SSDReference(nn.Module):
    """
    Mamba-2 的 PyTorch SSD reference backend。

    对应官方 Mamba2 中：

        mamba_chunk_scan_combined(...)

    这里只实现 reference recurrence。

    状态：

        h_t ∈ R^[H, P, N]

    其中：

        H = number of heads
        P = head dimension
        N = d_state

    参数：

        x:
            [B,L,H,P]

        dt:
            [B,L,H]

        A:
            [H]

        B:
            [B,L,G,N]

        C:
            [B,L,G,N]

        D:
            [H]

    输出：

        y:
            [B,L,H,P]
    """

    def __init__(
        self,
        d_state: int,
        nheads: int,
        headdim: int,
        ngroups: int,
    ):
        super().__init__()

        self.d_state = d_state
        self.nheads = nheads
        self.headdim = headdim
        self.ngroups = ngroups

        if nheads % ngroups != 0:
            raise ValueError(
                "nheads 必须能够整除 ngroups"
            )

        self.heads_per_group = (
            nheads // ngroups
        )

    def forward(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
    ) -> torch.Tensor:

        if x.ndim != 4:
            raise ValueError(
                "x 必须为 [B,L,H,P]"
            )

        batch, seqlen, nheads, headdim = (
            x.shape
        )

        if nheads != self.nheads:
            raise ValueError(
                f"nheads mismatch: "
                f"{nheads} vs {self.nheads}"
            )

        if headdim != self.headdim:
            raise ValueError(
                f"headdim mismatch: "
                f"{headdim} vs {self.headdim}"
            )

        # ----------------------------------------------------
        # 初始状态
        # ----------------------------------------------------
        state = torch.zeros(
            batch,
            nheads,
            headdim,
            self.d_state,
            dtype=x.dtype,
            device=x.device,
        )

        outputs = []

        # ----------------------------------------------------
        # group B/C -> head B/C
        # ----------------------------------------------------
        if self.ngroups == 1:
            B_heads = B.expand(
                batch,
                seqlen,
                nheads,
                self.d_state,
            )

            C_heads = C.expand(
                batch,
                seqlen,
                nheads,
                self.d_state,
            )

        else:
            B_heads = B.repeat_interleave(
                self.heads_per_group,
                dim=2,
            )

            C_heads = C.repeat_interleave(
                self.heads_per_group,
                dim=2,
            )

        A = A.to(
            device=x.device,
            dtype=x.dtype,
        )

        D = D.to(
            device=x.device,
            dtype=x.dtype,
        )

        # ----------------------------------------------------
        # sequential SSD recurrence
        # ----------------------------------------------------
        for t in range(seqlen):

            xt = x[:, t]
            # [B,H,P]

            dtt = dt[:, t]
            # [B,H]

            Bt = B_heads[:, t]
            # [B,H,N]

            Ct = C_heads[:, t]
            # [B,H,N]

            # -----------------------------------------------
            # Discretize A
            # A_bar = exp(dt * A)
            # -----------------------------------------------
            A_bar = torch.exp(
                dtt
                *
                A.unsqueeze(0)
            )
            # [B,H]

            # -----------------------------------------------
            # Discretize B
            # B_bar = dt * B
            # -----------------------------------------------
            B_bar = (
                dtt.unsqueeze(-1)
                *
                Bt
            )
            # [B,H,N]

            # -----------------------------------------------
            # State update
            #
            # h_t =
            #     A_bar * h_(t-1)
            #     +
            #     x_t * B_bar
            # -----------------------------------------------
            state = (
                A_bar.unsqueeze(-1).unsqueeze(-1)
                *
                state
                +
                xt.unsqueeze(-1)
                *
                B_bar.unsqueeze(2)
            )

            # -----------------------------------------------
            # output
            #
            # y_t = C_t h_t + D x_t
            # -----------------------------------------------
            yt = (
                state
                *
                Ct.unsqueeze(2)
            ).sum(
                dim=-1
            )

            # D skip
            yt = yt + (
                xt
                *
                D.view(
                    1,
                    nheads,
                    1,
                )
            )

            outputs.append(
                yt
            )

        return torch.stack(
            outputs,
            dim=1,
        )


# ============================================================
# 4. Mamba-2 Windows
# ============================================================

class Mamba2Windows(nn.Module):
    """
    Windows 原生 Mamba-2。

    参数和组织形式对应官方 Mamba2。

    官方 projection layout：

        [z, x, B, C, dt]

    forward：

        u
        │
        ├── in_proj
        │
        ├── z
        │
        ├── x
        ├── B
        ├── C
        └── dt
             │
             ↓
        causal depthwise conv
             │
             ├── x
             ├── B
             └── C
             │
             ↓
        SiLU(x)
             │
             ↓
        dt + dt_bias
             │
             ↓
        softplus(dt)
             │
             ↓
        A = -exp(A_log)
             │
             ↓
        SSD recurrence
             │
             ↓
        D skip
             │
             ↓
        gated RMSNorm
             │
             ↓
        out_proj

    当前版本：
        d_ssm = d_inner

    因此：
        d_mlp = 0

    也就是说当前版本完整实现的是纯 SSM Mamba-2
    路径，不启用官方 Mamba2 中可选的
    gated-MLP remainder branch。
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        d_ssm: Optional[int] = None,
        ngroups: int = 1,
        A_init_range: Tuple[
            float,
            float,
        ] = (1.0, 16.0),
        D_has_hdim: bool = False,
        rmsnorm: bool = True,
        norm_before_gate: bool = False,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        dt_limit: Tuple[
            float,
            float,
        ] = (
            0.0,
            float("inf"),
        ),
        bias: bool = False,
        conv_bias: bool = True,
        conv_init: Optional[
            float
        ] = None,
        device=None,
        dtype=None,
    ):
        super().__init__()

        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        # ----------------------------------------------------
        # 官方：
        # d_inner = expand * d_model
        # ----------------------------------------------------
        self.d_inner = (
            expand * d_model
        )

        self.headdim = headdim

        self.d_ssm = (
            self.d_inner
            if d_ssm is None
            else d_ssm
        )

        if (
            self.d_ssm %
            self.headdim
            != 0
        ):
            raise ValueError(
                "d_ssm 必须可以被 headdim 整除。"
            )

        self.nheads = (
            self.d_ssm //
            self.headdim
        )

        if (
            self.nheads %
            ngroups
            != 0
        ):
            raise ValueError(
                "nheads 必须可以被 ngroups 整除。"
            )

        self.ngroups = ngroups

        self.D_has_hdim = (
            D_has_hdim
        )

        self.rmsnorm = rmsnorm

        self.norm_before_gate = (
            norm_before_gate
        )

        self.dt_limit = dt_limit

        self.activation = "silu"

        # ----------------------------------------------------
        # 当前实现默认 d_ssm == d_inner
        #
        # 与官方 Mamba2 中 d_mlp = 0 对齐。
        # ----------------------------------------------------
        if self.d_ssm != self.d_inner:
            raise NotImplementedError(
                "当前 Windows reference 版本仅实现 "
                "d_ssm == d_inner 的完整纯 SSM Mamba-2 路径。"
                "这对应官方 Mamba2 的 d_mlp=0 情况。"
            )

        # ====================================================
        # 1. Input Projection
        # ====================================================
        #
        # 官方：
        # [z, x, B, C, dt]
        #
        # total =
        #   2*d_inner
        #   + 2*ngroups*d_state
        #   + nheads
        # ====================================================
        d_in_proj = (
            2 * self.d_inner
            +
            2 *
            self.ngroups
            *
            self.d_state
            +
            self.nheads
        )

        self.in_proj = nn.Linear(
            self.d_model,
            d_in_proj,
            bias=bias,
            **factory_kwargs,
        )

        # ====================================================
        # 2. Conv1d
        # ====================================================
        #
        # 官方：
        # conv_dim =
        #   d_ssm + 2*ngroups*d_state
        # ====================================================
        conv_dim = (
            self.d_ssm
            +
            2 *
            self.ngroups
            *
            self.d_state
        )

        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            bias=conv_bias,
            **factory_kwargs,
        )

        if conv_init is not None:
            nn.init.uniform_(
                self.conv1d.weight,
                -conv_init,
                conv_init,
            )

        self.act = nn.SiLU()

        # ====================================================
        # 3. dt bias initialization
        # ====================================================
        #
        # 完整采用官方 inverse-softplus 初始化。
        # ====================================================
        dt = torch.exp(
            torch.rand(
                self.nheads,
                **factory_kwargs,
            )
            *
            (
                math.log(dt_max)
                -
                math.log(dt_min)
            )
            +
            math.log(dt_min)
        )

        dt = torch.clamp(
            dt,
            min=dt_init_floor,
        )

        inv_dt = (
            dt
            +
            torch.log(
                -torch.expm1(-dt)
            )
        )

        self.dt_bias = nn.Parameter(
            inv_dt
        )

        self.dt_bias._no_weight_decay = True

        # ====================================================
        # 4. A_log
        # ====================================================
        assert (
            A_init_range[0] > 0
            and
            A_init_range[1]
            >=
            A_init_range[0]
        )

        A = torch.empty(
            self.nheads,
            dtype=torch.float32,
            device=device,
        ).uniform_(
            *A_init_range
        )

        self.A_log = nn.Parameter(
            torch.log(A).to(
                dtype=dtype
            )
            if dtype is not None
            else torch.log(A)
        )

        self.A_log._no_weight_decay = True

        # ====================================================
        # 5. D skip parameter
        # ====================================================
        if D_has_hdim:
            D_shape = (
                self.d_ssm,
            )
        else:
            D_shape = (
                self.nheads,
            )

        self.D = nn.Parameter(
            torch.ones(
                D_shape,
                **factory_kwargs,
            )
        )

        self.D._no_weight_decay = True

        # ====================================================
        # 6. RMSNormGated
        # ====================================================
        if rmsnorm:
            self.norm = (
                RMSNormGated(
                    hidden_dim=self.d_ssm,
                    eps=1e-5,
                    norm_before_gate=(
                        norm_before_gate
                    ),
                )
            )
        else:
            self.norm = None

        # ====================================================
        # 7. Output projection
        # ====================================================
        self.out_proj = nn.Linear(
            self.d_inner,
            self.d_model,
            bias=bias,
            **factory_kwargs,
        )

        # ====================================================
        # 8. SSD reference
        # ====================================================
        self.ssd = (
            Mamba2SSDReference(
                d_state=self.d_state,
                nheads=self.nheads,
                headdim=self.headdim,
                ngroups=self.ngroups,
            )
        )

    # ========================================================
    # forward
    # ========================================================

    def forward(
        self,
        u: torch.Tensor,
    ) -> torch.Tensor:
        """
        参数：
            u:
                [B,L,d_model]

        返回：
            [B,L,d_model]
        """

        if u.ndim != 3:
            raise ValueError(
                "Mamba2Windows 输入必须是 [B,L,D]"
            )

        batch, seqlen, dim = (
            u.shape
        )

        if dim != self.d_model:
            raise ValueError(
                f"输入最后一维={dim}，"
                f"但 d_model={self.d_model}"
            )

        # ====================================================
        # 1. projection
        # ====================================================
        zxbcdt = self.in_proj(
            u
        )

        # 官方 order:
        #
        # [z, x, B, C, dt]
        #
        # 当前：
        # d_mlp = 0
        # ====================================================
        z, x, B, C, dt = torch.split(
            zxbcdt,
            [
                self.d_ssm,
                self.d_ssm,
                self.ngroups *
                self.d_state,
                self.ngroups *
                self.d_state,
                self.nheads,
            ],
            dim=-1,
        )

        # ====================================================
        # 2. Conv branch
        #
        # xBC 合并后进入 depthwise causal Conv1d
        # ====================================================
        xBC = torch.cat(
            [
                x,
                B,
                C,
            ],
            dim=-1,
        )

        xBC = (
            self.conv1d(
                xBC.transpose(
                    1,
                    2,
                )
            )
            .transpose(
                1,
                2,
            )
        )

        # 与官方 fallback：
        #
        # self.conv1d(...)
        # [:, :-(d_conv - 1)]
        #
        # 等价的 causal 截断。
        if self.d_conv > 1:
            xBC = xBC[
                :,
                :seqlen,
                :,
            ]
        else:
            xBC = xBC[
                :,
                :seqlen,
                :,
            ]

        xBC = F.silu(
            xBC
        )

        # ====================================================
        # 3. 再次拆分 x / B / C
        # ====================================================
        x, B, C = torch.split(
            xBC,
            [
                self.d_ssm,
                self.ngroups *
                self.d_state,
                self.ngroups *
                self.d_state,
            ],
            dim=-1,
        )

        # ====================================================
        # 4. dt
        # ====================================================
        #
        # 官方：
        #
        # dt = softplus(
        #     dt + dt_bias
        # )
        # ====================================================
        dt = (
            dt
            +
            self.dt_bias
        )

        dt = F.softplus(
            dt
        )

        if self.dt_limit != (
            0.0,
            float("inf"),
        ):
            dt = torch.clamp(
                dt,
                min=self.dt_limit[0],
                max=self.dt_limit[1],
            )

        # ====================================================
        # 5. A
        # ====================================================
        #
        # 官方：
        #
        # A = -exp(A_log.float())
        # ====================================================
        A = -torch.exp(
            self.A_log.float()
        )

        # ====================================================
        # 6. x -> [B,L,H,P]
        # ====================================================
        x = x.view(
            batch,
            seqlen,
            self.nheads,
            self.headdim,
        )

        # ====================================================
        # 7. B/C -> [B,L,G,N]
        # ====================================================
        B = B.view(
            batch,
            seqlen,
            self.ngroups,
            self.d_state,
        )

        C = C.view(
            batch,
            seqlen,
            self.ngroups,
            self.d_state,
        )

        # ====================================================
        # 8. D
        # ====================================================
        #
        # D_has_hdim=False：
        #     [H]
        #
        # D_has_hdim=True：
        #     [H,P]
        # ====================================================
        if self.D_has_hdim:
            D = self.D.view(
                self.nheads,
                self.headdim,
            )
        else:
            D = self.D

        # ====================================================
        # 9. SSD
        # ====================================================
        y = self.ssd(
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            D=D
            if D.ndim == 1
            else D.mean(dim=-1),
        )

        # ====================================================
        # 10. [B,L,H,P] -> [B,L,D]
        # ====================================================
        y = y.reshape(
            batch,
            seqlen,
            self.d_ssm,
        )

        # ====================================================
        # 11. gated RMSNorm
        # ====================================================
        if self.norm is not None:
            y = self.norm(
                y,
                z,
            )
        else:
            y = (
                y *
                F.silu(z)
            )

        # ====================================================
        # 12. Output projection
        # ====================================================
        return self.out_proj(
            y
        )


# ============================================================
# 5. Bidirectional Mamba-2
# ============================================================

class BidirectionalMamba2Windows(
    nn.Module
):
    """
    两个独立的完整 Mamba-2：

        forward:
            1 -> L

        backward:
            L -> 1

    两个方向都使用 Mamba2Windows。
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
    ):
        super().__init__()

        self.forward_mamba = (
            Mamba2Windows(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                headdim=headdim,
                ngroups=ngroups,
            )
        )

        self.backward_mamba = (
            Mamba2Windows(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                headdim=headdim,
                ngroups=ngroups,
            )
        )

        self.fusion = nn.Linear(
            d_model * 2,
            d_model,
        )

        self.norm = nn.LayerNorm(
            d_model * 2
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        forward_output = (
            self.forward_mamba(x)
        )

        reverse_input = torch.flip(
            x,
            dims=[1],
        )

        backward_output = (
            self.backward_mamba(
                reverse_input
            )
        )

        backward_output = torch.flip(
            backward_output,
            dims=[1],
        )

        output = torch.cat(
            [
                forward_output,
                backward_output,
            ],
            dim=-1,
        )

        output = self.norm(
            output
        )

        output = self.fusion(
            output
        )

        return output


# ============================================================
# 6. 参数统计
# ============================================================

def count_parameters(
    model: nn.Module,
) -> int:

    return sum(
        p.numel()
        for p in model.parameters()
    )


def count_trainable_parameters(
    model: nn.Module,
) -> int:

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ============================================================
# 7. 基本 Shape Test
# ============================================================

def test_mamba2_shape(
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
):
    """
    基本前向测试。

    注意：
        这里不要用 float16 做 reference correctness test。
        先用 float32 验证结构。
    """

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = Mamba2Windows(
        d_model=384,
        d_state=16,
        d_conv=4,
        expand=2,
        headdim=64,
        ngroups=1,
        rmsnorm=True,
    ).to(
        device=device,
        dtype=dtype,
    )

    model.eval()

    x = torch.randn(
        1,
        256,
        384,
        device=device,
        dtype=dtype,
    )

    with torch.no_grad():
        y = model(x)

    assert y.shape == x.shape, (
        f"shape mismatch: "
        f"{tuple(y.shape)} vs "
        f"{tuple(x.shape)}"
    )

    print(
        "Mamba2Windows shape test PASS"
    )

    print(
        "input :",
        tuple(x.shape),
    )

    print(
        "output:",
        tuple(y.shape),
    )

    print(
        "parameters:",
        count_parameters(model),
    )


# ============================================================
# 8. Gradient Test
# ============================================================

def test_mamba2_backward(
    device: str = "cuda",
):
    """
    检查 reference implementation 是否可反向传播。
    """

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = Mamba2Windows(
        d_model=128,
        d_state=16,
        d_conv=4,
        expand=2,
        headdim=32,
        ngroups=1,
        rmsnorm=True,
    ).to(device)

    model.train()

    x = torch.randn(
        2,
        32,
        128,
        device=device,
        requires_grad=True,
    )

    y = model(x)

    loss = (
        y.pow(2)
        .mean()
    )

    loss.backward()

    if x.grad is None:
        raise RuntimeError(
            "输入没有获得 gradient。"
        )

    print(
        "Mamba2Windows backward test PASS"
    )

    print(
        "loss:",
        float(loss.detach().cpu()),
    )


# ============================================================
# 9. 双向 Mamba Test
# ============================================================

def test_bidirectional_mamba(
    device: str = "cuda",
):
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = (
        BidirectionalMamba2Windows(
            d_model=128,
            d_state=16,
            d_conv=4,
            expand=2,
            headdim=32,
            ngroups=1,
        )
        .to(device)
        .eval()
    )

    x = torch.randn(
        1,
        32,
        128,
        device=device,
    )

    with torch.no_grad():
        y = model(x)

    assert y.shape == x.shape

    print(
        "Bidirectional Mamba2 test PASS"
    )

    print(
        "input :",
        tuple(x.shape),
    )

    print(
        "output:",
        tuple(y.shape),
    )


# ============================================================
# 10. Main
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print(
        "Mamba-2 Windows Reference Backend"
    )
    print("=" * 70)

    print(
        "\n[1] Shape Test"
    )

    test_mamba2_shape(
        device="cuda"
        if torch.cuda.is_available()
        else "cpu",
        dtype=torch.float32,
    )

    print(
        "\n[2] Backward Test"
    )

    test_mamba2_backward(
        device="cuda"
        if torch.cuda.is_available()
        else "cpu",
    )

    print(
        "\n[3] Bidirectional Test"
    )

    test_bidirectional_mamba(
        device="cuda"
        if torch.cuda.is_available()
        else "cpu",
    )

    print(
        "\nAll tests finished."
    )