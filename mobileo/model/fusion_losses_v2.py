# -*- coding: utf-8 -*-
"""
Fusion losses v2
================

512x512 full-resolution source-supervised fusion objectives.
No fused GT is required.

The loss is explicitly asymmetric by modality:
- visible light: color / texture / background anchor
- infrared: thermal saliency / structural response
- SAM: training-only structural teacher
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 1:
        return x
    w = torch.tensor([0.299, 0.587, 0.114], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * w).sum(dim=1, keepdim=True)


def normalize01(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mn = x.amin(dim=(-2, -1), keepdim=True)
    mx = x.amax(dim=(-2, -1), keepdim=True)
    return (x - mn) / (mx - mn + eps)


def charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(x * x + eps * eps)


def _gaussian_window(window_size: int, sigma: float, channels: int, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    g = g / g.sum()
    w = (g[:, None] * g[None, :]).view(1, 1, window_size, window_size)
    return w.to(dtype=dtype).expand(channels, 1, window_size, window_size).contiguous()


def ssim_map(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    assert x.shape == y.shape
    c = x.shape[1]
    w = _gaussian_window(window_size, sigma, c, x.device, x.dtype)
    p = window_size // 2
    mu_x = F.conv2d(x, w, padding=p, groups=c)
    mu_y = F.conv2d(y, w, padding=p, groups=c)
    ex2 = F.conv2d(x * x, w, padding=p, groups=c)
    ey2 = F.conv2d(y * y, w, padding=p, groups=c)
    exy = F.conv2d(x * y, w, padding=p, groups=c)
    vx = (ex2 - mu_x * mu_x).clamp_min(0.0)
    vy = (ey2 - mu_y * mu_y).clamp_min(0.0)
    cxy = exy - mu_x * mu_y
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    num = (2 * mu_x * mu_y + c1) * (2 * cxy + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (vx + vy + c2)
    return (num / den.clamp_min(1e-8)).clamp(-1.0, 1.0)


def ssim_loss(x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    return 1.0 - ssim_map(x, y, window_size=window_size).mean()


class GradientMagnitude(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        ky = kx.transpose(-2, -1).contiguous()
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def forward(self, gray: torch.Tensor) -> torch.Tensor:
        if gray.shape[1] != 1:
            gray = rgb_to_gray(gray)
        kx = self.kx.to(dtype=gray.dtype)
        ky = self.ky.to(dtype=gray.dtype)
        gx = F.conv2d(gray, kx, padding=1)
        gy = F.conv2d(gray, ky, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-8)


class FusionPixelLossV2(nn.Module):
    """
    全分辨率 512x512 融合损失。

    Inputs:
        fused:        Bx3xH xW [0,1]
        visible:      Bx3xH xW [0,1]
        infrared:     Bx1xH xW or Bx3xH xW [0,1]
        ir_gate_16:   BxHxW, predicted spatial IR gate at 16x16
        teacher_gate: Bx1xHxW, SAM/IR teacher gate at full resolution
    """

    def __init__(
        self,
        w_ssim_vis: float = 0.5,
        w_ssim_ir: float = 0.7,
        w_grad: float = 0.5,
        w_int: float = 0.15,
        w_color: float = 0.8,
        w_anchor: float = 0.8,
        w_thermal: float = 0.7,
        window_size: int = 11,
    ):
        super().__init__()
        self.w_ssim_vis = w_ssim_vis
        self.w_ssim_ir = w_ssim_ir
        self.w_grad = w_grad
        self.w_int = w_int
        self.w_color = w_color
        self.w_anchor = w_anchor
        self.w_thermal = w_thermal
        self.window_size = window_size
        self.grad = GradientMagnitude()

    def forward(
        self,
        fused: torch.Tensor,
        visible: torch.Tensor,
        infrared: torch.Tensor,
        ir_gate_16: torch.Tensor,
        teacher_gate: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        fused = fused.float().clamp(0, 1)
        visible = visible.float().clamp(0, 1)
        infrared = infrared.float().clamp(0, 1)

        if infrared.shape[1] == 3:
            ir = rgb_to_gray(infrared)
        else:
            ir = infrared

        fused_gray = rgb_to_gray(fused)
        vis_gray = rgb_to_gray(visible)
        ir_norm = normalize01(ir)

        if teacher_gate.ndim == 3:
            teacher_gate = teacher_gate.unsqueeze(1)
        teacher_gate = teacher_gate.float().clamp(0.02, 0.98)
        if teacher_gate.shape[-2:] != fused.shape[-2:]:
            teacher_gate = F.interpolate(teacher_gate, size=fused.shape[-2:], mode="bilinear", align_corners=False)
        bg = 1.0 - teacher_gate

        # ---------------- visible SSIM ----------------
        l_ssim_vis = ssim_loss(fused, visible, window_size=self.window_size)

        # 红外 SSIM 比较归一化灰度，避免 RGB/IR 动态范围直接冲突。
        l_ssim_ir = ssim_loss(fused_gray, ir_norm, window_size=self.window_size)

        # ---------------- gradients ----------------
        g_fused = self.grad(fused_gray)
        g_vis = self.grad(vis_gray)
        g_ir = self.grad(ir_norm)
        g_target = torch.maximum(g_vis, g_ir)
        l_grad_map = charbonnier(g_fused - g_target)
        # 热区域强化 IR gradient，背景强化 visible gradient。
        l_grad = (l_grad_map * (0.45 * bg + 0.55 * teacher_gate)).mean()

        # ---------------- intensity ----------------
        intensity_target = torch.maximum(vis_gray, ir_norm)
        l_int = charbonnier(fused_gray - intensity_target).mean()

        # ---------------- visible color/background anchor ----------------
        l_color_map = charbonnier(fused - visible).mean(dim=1, keepdim=True)
        l_color = (l_color_map * (0.25 + 0.75 * bg)).mean()

        # 这是防止生成模型在非热区域自由“重绘”的关键项。
        l_anchor = (charbonnier(fused - visible).mean(dim=1, keepdim=True) * bg).sum() / bg.sum().clamp_min(1.0)

        # ---------------- thermal preservation ----------------
        l_thermal = (charbonnier(fused_gray - ir_norm) * teacher_gate).sum() / teacher_gate.sum().clamp_min(1.0)

        # ---------------- gate consistency ----------------
        pred_gate = ir_gate_16.unsqueeze(1) if ir_gate_16.ndim == 3 else ir_gate_16
        pred_gate_up = F.interpolate(pred_gate.float(), size=fused.shape[-2:], mode="bilinear", align_corners=False)
        l_gate_full = charbonnier(pred_gate_up - teacher_gate).mean()

        total = (
            self.w_ssim_vis * l_ssim_vis
            + self.w_ssim_ir * l_ssim_ir
            + self.w_grad * l_grad
            + self.w_int * l_int
            + self.w_color * l_color
            + self.w_anchor * l_anchor
            + self.w_thermal * l_thermal
        )

        parts = {
            "ssim_vis": l_ssim_vis.detach(),
            "ssim_ir": l_ssim_ir.detach(),
            "grad": l_grad.detach(),
            "int": l_int.detach(),
            "color": l_color.detach(),
            "anchor": l_anchor.detach(),
            "thermal": l_thermal.detach(),
            "gate_full": l_gate_full.detach(),
        }
        return total, parts


class SpatialTeacherLoss(nn.Module):
    def __init__(self, w_gate: float = 1.0, w_structure: float = 1.0):
        super().__init__()
        self.w_gate = w_gate
        self.w_structure = w_structure

    def forward(
        self,
        pred_gate: torch.Tensor,
        pred_structure: torch.Tensor,
        teacher_gate: torch.Tensor,
        teacher_structure: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if teacher_gate.ndim == 4:
            teacher_gate = teacher_gate[:, 0]
        if teacher_structure.ndim == 4:
            teacher_structure = teacher_structure[:, 0]
        lg = F.binary_cross_entropy(pred_gate.clamp(1e-4, 1 - 1e-4), teacher_gate)
        ls = charbonnier(pred_structure - teacher_structure).mean()
        total = self.w_gate * lg + self.w_structure * ls
        return total, {"gate_bce": lg.detach(), "structure": ls.detach()}
