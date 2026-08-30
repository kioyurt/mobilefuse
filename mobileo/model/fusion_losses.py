# -*- coding: utf-8 -*-

"""
fusion_losses.py
============================================================
无监督可见光 / 红外图像融合损失函数（不需要 GT 融合图）

设计思路
------------------------------------------------------------
图像融合任务天然没有真实 ground-truth，
因此采用“源图像自监督”的经典范式（DeepFuse 一系）：
融合结果应当同时保留两张源图的互补信息。

本模块提供 4 项像素空间损失（全部可微，图像范围 [0,1]）：

    1. L_ssim_vis : 1 - SSIM(fused, visible)
       融合图应保留可见光的纹理 / 结构 / 色彩信息。

    2. L_ssim_ir  : 1 - SSIM(fused_gray, infrared)
       融合图的灰度结构应保留红外热辐射分布。

    3. L_grad     : Sobel 梯度保持损失
       融合图梯度幅值应覆盖两张源图梯度的逐像素最大值，
       即“两边的边缘细节谁强保谁”（DeepFuse / RFNest 常用）。

    4. L_int      : 强度损失
       融合图亮度应接近两张源图亮度的逐像素最大值，
       防止融合后整体变暗、信息被平均掉。

方法依据（联网调研）：
    - DeepFuse (ICCV 2017)：首个无监督融合网络，
      确立 SSIM + gradient + intensity 的损失组合范式。
    - Dif-Fusion (TIP 2023)：红外-可见光扩散融合，
      强调可见光色彩保真（对应这里的 ssim_vis 彩色项）。
    - FusionFM (2025, arXiv 2511.13794)：flow matching 融合，
      证明“源耦合”显著优于纯噪声轨迹 —— 本训练脚本因此
      使用可见光锚点 latent 作为 flow 目标，而非任意噪声。

注意：
    - 扩散侧的 flow matching 损失（target = noise - latent，
      与 Mobile-O / SANA 原生一致）在训练脚本里实现，
      本模块只负责像素空间的无监督融合损失。
    - 所有损失要求输入同尺寸；训练脚本负责把源图
      resize 到与解码输出一致。

用法：
    criterion = FusionPixelLoss(
        w_ssim_vis=1.0, w_ssim_ir=1.0, w_grad=0.2, w_int=0.1,
    )
    total, parts = criterion(fused, visible, infrared)
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. 色彩空间
# ============================================================

def rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    """
    (B, 3, H, W) [0,1] RGB -> (B, 1, H, W) 灰度

    使用 ITU-R BT.601 亮度权重。
    """
    weight = torch.tensor(
        [0.299, 0.587, 0.114],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 3, 1, 1)

    return (x * weight).sum(dim=1, keepdim=True)


# ============================================================
# 2. 可微 SSIM
# ============================================================

def _gaussian_1d(window_size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


def _make_window(window_size: int, sigma: float, channels: int) -> torch.Tensor:
    w1d = _gaussian_1d(window_size, sigma)
    w2d = w1d[:, None] * w1d[None, :]
    window = w2d.unsqueeze(0).unsqueeze(0)  # (1, 1, k, k)
    return window.expand(channels, 1, window_size, window_size).contiguous()


def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """
    结构相似度 (均值标量)。

    输入:
        x, y: (B, C, H, W)，值域相同即可（建议 [0,1]）
    输出:
        标量 SSIM（越大越相似）
    """
    assert x.shape == y.shape, (
        f"SSIM 输入形状不一致: {x.shape} vs {y.shape}"
    )

    channels = x.shape[1]
    window = _make_window(window_size, sigma, channels).to(
        device=x.device, dtype=x.dtype
    )
    pad = window_size // 2

    mu_x = F.conv2d(x, window, padding=pad, groups=channels)
    mu_y = F.conv2d(y, window, padding=pad, groups=channels)

    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x * x, window, padding=pad, groups=channels) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, window, padding=pad, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(x * y, window, padding=pad, groups=channels) - mu_xy

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)) / (
        (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    )

    return ssim_map.mean()


# ============================================================
# 3. Sobel 梯度
# ============================================================

class SobelGradient(nn.Module):
    """
    灰度图 -> 梯度幅值 (B,1,H,W)

    用于“融合图梯度应覆盖源图最大梯度”的保持损失。
    """

    def __init__(self):
        super().__init__()

        kx = torch.tensor(
            [[-1.0, 0.0, 1.0],
             [-2.0, 0.0, 2.0],
             [-1.0, 0.0, 1.0]]
        )
        ky = kx.t().contiguous()

        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", ky.view(1, 1, 3, 3))

    def forward(self, gray: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(gray, self.kx.to(gray.dtype), padding=1)
        gy = F.conv2d(gray, self.ky.to(gray.dtype), padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-8)


# ============================================================
# 4. 无监督融合像素损失
# ============================================================

class FusionPixelLoss(nn.Module):
    """
    无监督融合像素损失（不需要 GT）

    输入（全部 [0,1]，同尺寸）:
        fused:    (B, 3, H, W) 模型生成的融合图
        visible:  (B, 3, H, W) 可见光源图
        infrared: (B, 1, H, W) 或 (B, 3, H, W) 红外源图

    输出:
        total: 加权和
        parts: 各分项字典（用于日志）
    """

    def __init__(
        self,
        w_ssim_vis: float = 1.0,
        w_ssim_ir: float = 1.0,
        w_grad: float = 0.2,
        w_int: float = 0.1,
        ssim_window: int = 11,
    ):
        super().__init__()

        self.w_ssim_vis = w_ssim_vis
        self.w_ssim_ir = w_ssim_ir
        self.w_grad = w_grad
        self.w_int = w_int
        self.ssim_window = ssim_window

        self.sobel = SobelGradient()

    def forward(
        self,
        fused: torch.Tensor,
        visible: torch.Tensor,
        infrared: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        # ---- 灰度化 ----
        fused_gray = rgb_to_gray(fused)
        vis_gray = rgb_to_gray(visible)

        if infrared.shape[1] == 3:
            ir_gray = rgb_to_gray(infrared)
        else:
            ir_gray = infrared

        # ---- 1. 可见光 SSIM（彩色，保纹理 + 色彩保真）----
        l_ssim_vis = 1.0 - ssim(fused, visible, window_size=self.ssim_window)

        # ---- 2. 红外 SSIM（灰度，保热目标结构）----
        l_ssim_ir = 1.0 - ssim(fused_gray, ir_gray, window_size=self.ssim_window)

        # ---- 3. 梯度保持：覆盖两张源图的最大梯度 ----
        g_fused = self.sobel(fused_gray)
        g_target = torch.max(
            self.sobel(vis_gray),
            self.sobel(ir_gray),
        )
        l_grad = F.l1_loss(g_fused, g_target)

        # ---- 4. 强度保持：亮度对齐源图逐像素最大值 ----
        l_int = F.l1_loss(fused_gray, torch.max(vis_gray, ir_gray))

        # ---- 加权汇总 ----
        total = (
            self.w_ssim_vis * l_ssim_vis
            + self.w_ssim_ir * l_ssim_ir
            + self.w_grad * l_grad
            + self.w_int * l_int
        )

        parts = {
            "ssim_vis": l_ssim_vis.detach(),
            "ssim_ir": l_ssim_ir.detach(),
            "grad": l_grad.detach(),
            "intensity": l_int.detach(),
        }

        return total, parts
