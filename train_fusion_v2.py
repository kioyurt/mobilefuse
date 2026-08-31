# -*- coding: utf-8 -*-
"""
MobileFuse v2 training
=======================

重新建立 RGB/IR condition space，并训练 SANA 的条件相关通路。

训练对象：
- MultimodalSpatialConditionEncoder：全新、从零初始化
- SANA caption_projection / caption_norm
- SANA 每个 transformer block 的 attn2 条件注意力参数

冻结：
- Qwen cache（离线）
- SAM（离线，仅 teacher）
- DC-AE/VAE
- SANA self-attention / FF / patch embedding / output head

训练目标：
1) spatial adaptive dual-flow
2) latent visible/infrared anchor
3) 512x512 full-resolution pixel fusion loss
4) SAM/IR teacher gate
5) SAM-derived structure teacher

重要：
- 不使用原 ThreeLevelFusionMCP
- 不使用 648 token concat
- 不使用 visible-only target latent
- 不使用 4x4 latent pixel loss
- 推理可不使用 SAM
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from diffusers import (
    AutoencoderDC,
    DPMSolverMultistepScheduler,
    FlowMatchEulerDiscreteScheduler,
    SanaTransformer2DModel,
)
from diffusers.training_utils import compute_density_for_timestep_sampling

from mobileo.model.multimodal_spatial_condition_v2 import (
    MultimodalSpatialConditionEncoder,
)
from mobileo.model.fusion_losses_v2 import (
    FusionPixelLossV2,
    SpatialTeacherLoss,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mobilefuse_v2_train")


# ============================================================
# 常量
# ============================================================

DEFAULT_SANA_PATH = r"D:\Mobile-O-main\mobileo\model\Sana\Sana_600M_512px_diffusers"
DEFAULT_CACHE = r"D:\Mobile-O-main\fusion_cache_v2"
DEFAULT_OUT = r"D:\Mobile-O-main\fusion_runs\v2"

IMAGE_SIZE = 512
LATENT_SIZE = 16
LLM_DIM = 2048
CAPTION_DIM = 2304
LAYERS = (4, 5, 6, 13, 14, 15, 26, 27, 28)
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ============================================================
# dataset
# ============================================================

class FusionV2Dataset(Dataset):
    def __init__(self, cache_dir: Path, vis_root: Path, ir_root: Path):
        self.cache = Path(cache_dir)
        self.vis_root = Path(vis_root)
        self.ir_root = Path(ir_root)
        self.qwen_dir = self.cache / "qwen_v2"
        self.latent_dir = self.cache / "latent_v2"
        self.teacher_dir = self.cache / "teacher_v2"

        stems = []
        for q in sorted(self.qwen_dir.glob("*.pt")):
            stem = q.stem
            if (self.latent_dir / f"{stem}.pt").exists() and (self.teacher_dir / f"{stem}.pt").exists():
                stems.append(stem)
        if not stems:
            raise RuntimeError(f"缓存不完整：{self.cache}")
        self.stems = stems
        logger.info("v2 dataset: %d samples", len(self.stems))

    @staticmethod
    def _find(root: Path, stem: str) -> Path:
        for ext in EXTS:
            p = root / f"{stem}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(f"{root} 中不存在 {stem}.*")

    @staticmethod
    def _rgb(path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        if img.size != (IMAGE_SIZE, IMAGE_SIZE):
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
        a = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)

    @staticmethod
    def _ir(path: Path) -> torch.Tensor:
        img = Image.open(path).convert("L")
        if img.size != (IMAGE_SIZE, IMAGE_SIZE):
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
        a = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(a).unsqueeze(0).unsqueeze(0)

    def __len__(self):
        return len(self.stems)



    def __getitem__(self, idx: int) -> Dict:
        stem = self.stems[idx]
        qwen = torch.load(self.qwen_dir / f"{stem}.pt", map_location="cpu")
        latent = torch.load(self.latent_dir / f"{stem}.pt", map_location="cpu")
        teacher = torch.load(self.teacher_dir / f"{stem}.pt", map_location="cpu")

        vp = self._find(self.vis_root, stem)
        ip = self._find(self.ir_root, stem)
        return {
            "stem": stem,
            "qwen": qwen,
            "rgb_latent": latent["rgb_latent"],
            "ir_latent": latent["ir_latent"],
            "teacher_gate": teacher["ir_gate"],
            "teacher_structure": teacher["structure"],
            "visible": self._rgb(vp),
            "infrared": self._ir(ip),
        }


def collate_one(batch):
    return batch[0]


class StableSanaCrossAttnProcessor:
    """
    SANA 600M 的 attn2 使用 FP32 attention 计算。

    目的：
        - SANA 主体仍保持 FP16，控制显存。
        - cross-attention 的 Q/K/V、SDPA、output projection
          在 FP32 中计算。
        - 输出再恢复为原 hidden_states dtype。
        - K/V/O 参数本身建议保持 FP32，因为它们参与训练。

    适配：
        diffusers 0.32.x / SANA 600M 的标准 Attention API。
    """

    def __init__(self):
        pass

    @staticmethod
    def _linear_fp32(
        x: torch.Tensor,
        layer: nn.Linear,
    ) -> torch.Tensor:
        """
        明确使用 FP32 权重和 FP32 输入做 Linear。
        避免外部 autocast 将它重新降成 FP16。
        """
        weight = layer.weight.float()
        bias = layer.bias.float() if layer.bias is not None else None

        return F.linear(
            x.float(),
            weight,
            bias,
        )

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:

        # --------------------------------------------------------
        # 保存最终输出 dtype
        # --------------------------------------------------------
        output_dtype = hidden_states.dtype

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        batch_size = hidden_states.shape[0]
        query_length = hidden_states.shape[1]
        context_length = encoder_hidden_states.shape[1]

        # --------------------------------------------------------
        # attention mask
        # --------------------------------------------------------
        prepared_mask = None

        if attention_mask is not None:
            prepared_mask = attn.prepare_attention_mask(
                attention_mask,
                context_length,
                batch_size,
                out_dim=4,
            )

            if prepared_mask is not None:
                prepared_mask = prepared_mask.float()

        # --------------------------------------------------------
        # 整个 cross-attention 用 FP32
        # --------------------------------------------------------
        with torch.autocast(
            device_type="cuda",
            enabled=False,
        ):
            hidden_fp32 = hidden_states.float()
            context_fp32 = encoder_hidden_states.float()

            # ----------------------------------------------------
            # Q
            #
            # Query 是 SANA latent side，保持原来的 pretrained Q。
            # 这里虽然 Q 权重冻结，但 attention 计算转 FP32。
            # ----------------------------------------------------
            q = self._linear_fp32(
                hidden_fp32,
                attn.to_q,
            )

            # ----------------------------------------------------
            # K / V
            #
            # condition side，当前正在训练。
            # ----------------------------------------------------
            k = self._linear_fp32(
                context_fp32,
                attn.to_k,
            )

            v = self._linear_fp32(
                context_fp32,
                attn.to_v,
            )

            # ----------------------------------------------------
            # 可选 Q/K normalization
            # 当前 SANA 600M qk_norm=None，
            # 保留兼容性。
            # ----------------------------------------------------
            if getattr(attn, "norm_q", None) is not None:
                q = attn.norm_q(q)

            if getattr(attn, "norm_k", None) is not None:
                k = attn.norm_k(k)

            # ----------------------------------------------------
            # [B, L, D] -> [B, heads, L, head_dim]
            # ----------------------------------------------------
            num_heads = int(attn.heads)

            head_dim = q.shape[-1] // num_heads

            q = q.view(
                batch_size,
                query_length,
                num_heads,
                head_dim,
            ).transpose(1, 2)

            k = k.view(
                batch_size,
                context_length,
                num_heads,
                head_dim,
            ).transpose(1, 2)

            v = v.view(
                batch_size,
                context_length,
                num_heads,
                head_dim,
            ).transpose(1, 2)

            # ----------------------------------------------------
            # FP32 SDPA
            #
            # 不在 FP16 中做 QK^T / softmax。
            # ----------------------------------------------------
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=prepared_mask,
                dropout_p=0.0,
                is_causal=False,
            )

            # ----------------------------------------------------
            # [B,H,L,D] -> [B,L,H*D]
            # ----------------------------------------------------
            out = out.transpose(1, 2).reshape(
                batch_size,
                query_length,
                num_heads * head_dim,
            )

            # ----------------------------------------------------
            # Cross Attention output projection
            # ----------------------------------------------------
            out = self._linear_fp32(
                out,
                attn.to_out[0],
            )

            # dropout
            out = attn.to_out[1](out)

        # --------------------------------------------------------
        # 回到 SANA 主体 dtype
        # --------------------------------------------------------
        out = out.to(output_dtype)

        return out

# ============================================================
# SANA condition-path selection
# ============================================================


def configure_sana_condition_training(dit: nn.Module) -> Dict[str, List[str]]:
    """
    只打开 condition-side 参数：
      caption_projection
      caption_norm
      transformer_blocks.*.attn2.to_k / to_v / to_out

    Query 来自 SANA 当前 latent token，属于生成状态侧而不是条件侧，
    因而保持冻结。这样在 6GB 显存上仍能完整训练 condition 接收路径。
    """
    for p in dit.parameters():
        p.requires_grad = False

    groups = {"caption": [], "cross_attention": []}
    for name, p in dit.named_parameters():
        if name.startswith("caption_projection.") or name.startswith("caption_norm."):
            p.requires_grad = True
            groups["caption"].append(name)
        elif ".attn2.to_k." in name or ".attn2.to_v." in name:         #or ".attn2.to_out." in name:
            p.requires_grad = True
            groups["cross_attention"].append(name)

    if not groups["caption"]:
        raise RuntimeError(
            "SANA 没有找到 caption_projection/caption_norm。请检查本机 diffusers 与 SANA checkpoint 版本。"
        )
    if not groups["cross_attention"]:
        raise RuntimeError(
            "SANA 没有找到 transformer_blocks.*.attn2.* 参数，无法建立新的 condition-space 对齐。"
        )
    return groups


def trainable_sana_state_dict(dit: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: p.detach().cpu()
        for name, p in dit.named_parameters()
        if p.requires_grad
    }


def load_sana_trainable_state(dit: nn.Module, state: Mapping[str, torch.Tensor]):
    current = dict(dit.named_parameters())
    missing = []
    for name, tensor in state.items():
        if name not in current:
            missing.append(name)
            continue
        current[name].data.copy_(tensor.to(device=current[name].device, dtype=current[name].dtype))
    if missing:
        logger.warning("checkpoint 中有 %d 个 SANA 参数在当前模型不存在", len(missing))


# ============================================================
# hidden state preparation
# ============================================================


def to_device_layers(payload: Mapping, device: str):
    rgb = {k: v.to(device=device, dtype=torch.float32) for k, v in payload["rgb_layers"].items()}
    ir = {k: v.to(device=device, dtype=torch.float32) for k, v in payload["ir_layers"].items()}
    return rgb, ir


def choose_text(payload: Mapping, prompt_id: Optional[int] = None) -> Optional[torch.Tensor]:
    bank = payload.get("text_by_prompt", {})
    if not bank:
        return None
    keys = sorted(bank.keys(), key=lambda x: int(x))
    if prompt_id is None:
        key = random.choice(keys)
    else:
        key = str(prompt_id)
        if key not in bank:
            key = keys[0]
    return bank[key].float()


# ============================================================
# numerical helpers
# ============================================================


def finite(name: str, x: torch.Tensor):
    if not torch.isfinite(x).all():
        raise FloatingPointError(f"{name} contains NaN/Inf")


def cuda_free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sigma_for_timesteps(scheduler, timesteps: torch.Tensor) -> torch.Tensor:
    all_t = scheduler.timesteps.to(device=timesteps.device)
    all_s = scheduler.sigmas.to(device=timesteps.device, dtype=torch.float32)
    outs = []
    for t in timesteps:
        idx = (all_t == t).nonzero(as_tuple=False)
        if idx.numel() == 0:
            raise RuntimeError(f"找不到 timestep={int(t.item())} 对应 sigma")
        outs.append(all_s[int(idx[0].item())])
    return torch.stack(outs).view(-1, 1, 1, 1)


def safe_gradient_norm(parameters) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        if not torch.isfinite(g).all():
            return float("nan")
        total += float(torch.sum(g * g).item())
    return math.sqrt(total)


# ============================================================
# full-resolution VAE decode
# ============================================================


def decode_fullres(vae: nn.Module, latent: torch.Tensor, scaling: float) -> torch.Tensor:
    """
    直接 16x16 latent -> 512x512。
    VAE 采用 FP32 + 官方 tiling，保持 pixel backward 的数值稳定性。
    """
    inp = (latent / scaling).float()
    finite("vae_input", inp)
    with torch.autocast(device_type="cuda", enabled=False):
        out = vae.decode(inp).sample
    finite("vae_output", out)
    img = ((out.float() + 1.0) / 2.0).clamp(0.0, 1.0)
    if img.shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE):
        raise RuntimeError(
            f"VAE 输出不是 512x512，而是 {tuple(img.shape[-2:])}；当前设计不允许 silently resize。"
        )
    return img


# ============================================================
# sample
# ============================================================

@torch.no_grad()
def sample_image(
    dit,
    vae,
    scheduler,
    conditioner,
    rgb_layers,
    ir_layers,
    z_rgb,
    z_ir,
    text_hidden,
    steps: int,
    device: str,
    sana_dtype: torch.dtype,
    start_fraction: float = 0.55,
):
    conditioner.eval()
    dit.eval()

    condition, info = conditioner(
        rgb_layers=rgb_layers,
        ir_layers=ir_layers,
        rgb_latent=z_rgb,
        ir_latent=z_ir,
        text_hidden=text_hidden,
    )
    mask = info["attention_mask"]
    condition = condition.float().clamp(-8.0, 8.0).to(sana_dtype)

    g = info["ir_gate"].clamp(0.05, 0.95).unsqueeze(1)
    z_mix = (1.0 - g) * z_rgb.float() + g * z_ir.float()

    scheduler.set_timesteps(steps, device=device)
    sigmas = scheduler.sigmas.to(device=device, dtype=torch.float32)
    timesteps = scheduler.timesteps

    # 选一个非 1 的噪声起点，保证输入空间结构不会完全丢失。
    start_idx = int(round((len(timesteps) - 1) * float(start_fraction)))
    start_idx = max(0, min(start_idx, len(timesteps) - 1))
    sigma0 = sigmas[start_idx]
    noise = torch.randn_like(z_mix)
    latents = (1.0 - sigma0) * z_mix + sigma0 * noise
    latents = latents.to(dtype=sana_dtype)

    for i in range(start_idx, len(timesteps)):
        t = timesteps[i].view(1).to(device)
        with torch.autocast(device_type="cuda", dtype=sana_dtype if sana_dtype != torch.float32 else torch.float32, enabled=sana_dtype != torch.float32):
            pred = dit(
                hidden_states=latents,
                timestep=t,
                encoder_hidden_states=condition,
                encoder_attention_mask=mask,
            ).sample
        pred = pred.float()
        latents = scheduler.step(pred, t, latents.float()).prev_sample
        latents = latents.clamp(-20, 20).to(sana_dtype)

    vae.to(device)
    fused = decode_fullres(vae, latents.float(), float(getattr(vae.config, "scaling_factor", 1.0)))
    vae.to("cpu")
    cuda_free()
    return fused, info


def save_panel(path: Path, visible: torch.Tensor, infrared: torch.Tensor, fused: torch.Tensor):
    def pil_rgb(x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        arr = (x[0].detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    a = pil_rgb(visible)
    b = pil_rgb(infrared)
    c = pil_rgb(fused)
    w, h = a.size
    out = Image.new("RGB", (w * 3 + 8, h), "black")
    out.paste(a, (0, 0))
    out.paste(b, (w + 4, 0))
    out.paste(c, (2 * w + 8, 0))
    out.save(path)


# ============================================================
# training
# ============================================================


def train(args):
    device = args.device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    out = Path(args.output_dir)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "samples").mkdir(parents=True, exist_ok=True)

    dataset = FusionV2Dataset(Path(args.cache_dir), Path(args.vis_root), Path(args.ir_root))
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        collate_fn=collate_one,
    )

    logger.info("加载全新 MultimodalSpatialConditionEncoder ...")
    conditioner = MultimodalSpatialConditionEncoder(
        llm_dim=LLM_DIM,
        latent_ch=32,
        fusion_dim=args.fusion_dim,
        caption_dim=CAPTION_DIM,
        target_hw=(16, 16),
        selected_layers=LAYERS,
        num_heads=args.condition_heads,
        num_cross_blocks=args.cross_blocks,
        num_spatial_blocks=args.spatial_blocks,
        num_text_tokens=args.text_tokens,
    ).to(device=device, dtype=torch.float32)

    # SANA
    logger.info("加载 SANA: %s", args.sana_path)
    sana_dtype = torch.float32 if args.sana_dtype == "fp16" else torch.float32
    dit = SanaTransformer2DModel.from_pretrained(
        args.sana_path,
        subfolder="transformer",
        variant="fp16",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device=device, dtype=sana_dtype)
    if int(getattr(dit.config, "caption_channels", CAPTION_DIM)) != CAPTION_DIM:
        raise RuntimeError(
            f"SANA caption_channels={dit.config.caption_channels}, 但当前 condition space 输出 {CAPTION_DIM}。"
        )
    sana_groups = configure_sana_condition_training(dit)
    if args.gradient_checkpointing:
        dit.enable_gradient_checkpointing()
        logger.info("SANA gradient checkpointing enabled")
    dit.train()

    n_cond = sum(p.numel() for p in conditioner.parameters() if p.requires_grad)
    n_sana = sum(p.numel() for p in dit.parameters() if p.requires_grad)
    logger.info("conditioner trainable: %.2fM", n_cond / 1e6)
    logger.info("SANA condition trainable: %.2fM", n_sana / 1e6)
    logger.info("SANA groups: caption=%d, cross_attn=%d tensors", len(sana_groups["caption"]), len(sana_groups["cross_attention"]))

    # VAE：训练阶段只短暂驻留 GPU
    vae = AutoencoderDC.from_pretrained(
        args.sana_path,
        subfolder="vae",
        variant="fp16",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to("cpu")
    vae.requires_grad_(False)
    vae.eval()
    if hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
    scaling = float(getattr(vae.config, "scaling_factor", 1.0))

    # Scheduler
    flow_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.sana_path, subfolder="scheduler"
    )
    num_train_timesteps = int(flow_scheduler.config.num_train_timesteps)
    flow_scheduler.set_timesteps(num_train_timesteps, device="cpu")
    sample_scheduler = DPMSolverMultistepScheduler.from_pretrained(
        args.sana_path, subfolder="scheduler"
    )

    # losses
    pixel_loss_fn = FusionPixelLossV2(
        w_ssim_vis=args.w_ssim_vis,
        w_ssim_ir=args.w_ssim_ir,
        w_grad=args.w_grad,
        w_int=args.w_int,
        w_color=args.w_color,
        w_anchor=args.w_pixel_anchor,
        w_thermal=args.w_thermal,
    ).to(device)
    teacher_loss_fn = SpatialTeacherLoss(
        w_gate=args.w_teacher_gate,
        w_structure=args.w_teacher_structure,
    ).to(device)

    # optimizer
    cond_params = [p for p in conditioner.parameters() if p.requires_grad]
    sana_params = [p for p in dit.parameters() if p.requires_grad]

    if args.optimizer == "adamw8bit":
        try:
            import bitsandbytes as bnb
        except Exception as exc:
            raise RuntimeError(
                "6GB 配置默认必须使用 bitsandbytes AdamW8bit；当前导入失败。"
                f"请安装/修复 bitsandbytes，或显式使用 --optimizer adamw。原始错误: {exc}"
            )
        optimizer = bnb.optim.AdamW8bit(
            [
                {"params": cond_params, "lr": args.lr_conditioner},
                {"params": sana_params, "lr": args.lr_sana},
            ],
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
        )
    else:
        optimizer = torch.optim.AdamW(
            [
                {"params": cond_params, "lr": args.lr_conditioner},
                {"params": sana_params, "lr": args.lr_sana},
            ],
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
            foreach=False,
        )
    logger.info("optimizer=%s", args.optimizer)

    total_steps = args.max_steps
    warmup = args.warmup_steps

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        p = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    global_step = 0
    epoch = 0
    accum = 0
    ema: Dict[str, float] = {}
    optimizer.zero_grad(set_to_none=True)

    # resume
    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        conditioner.load_state_dict(state["conditioner"], strict=True)
        load_sana_trainable_state(dit, state["sana_condition"])
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        if "lr_scheduler" in state:
            lr_sched.load_state_dict(state["lr_scheduler"])
        global_step = int(state.get("step", 0))
        logger.info("resume step=%d", global_step)

    conditioner.train()
    dit.train()

    logger.info("开始 v2 训练：steps=%d accum=%d", total_steps, args.grad_accum)

    while global_step < total_steps:
        epoch += 1
        for sample in loader:
            if global_step >= total_steps:
                break
            tic = time.time()

            stem = sample["stem"]
            qwen = sample["qwen"]
            rgb_layers, ir_layers = to_device_layers(qwen, device)
            text_hidden = choose_text(qwen, None)
            if text_hidden is not None:
                text_hidden = text_hidden.to(device=device, dtype=torch.float32)

            z_rgb = sample["rgb_latent"].to(device=device, dtype=torch.float32)
            z_ir = sample["ir_latent"].to(device=device, dtype=torch.float32)
            teacher_gate = sample["teacher_gate"].to(device=device, dtype=torch.float32)
            teacher_structure = sample["teacher_structure"].to(device=device, dtype=torch.float32)
            visible = sample["visible"].to(device=device, dtype=torch.float32)
            infrared = sample["infrared"].to(device=device, dtype=torch.float32)

            # ============================================================
            # DEBUG: RGB / IR latent numerical statistics
            # 只在第一个训练样本打印一次
            # ============================================================
            if global_step == 0:
                rgb_diag = z_rgb.detach().float()
                ir_diag = z_ir.detach().float()

                logger.info(
                    "[DIAG][RGB latent] "
                    "shape=%s dtype=%s "
                    "mean=%.8e std=%.8e min=%.8e max=%.8e",
                    tuple(rgb_diag.shape),
                    str(rgb_diag.dtype),
                    rgb_diag.mean().item(),
                    rgb_diag.std(unbiased=False).item(),
                    rgb_diag.min().item(),
                    rgb_diag.max().item(),
                )

                logger.info(
                    "[DIAG][IR latent] "
                    "shape=%s dtype=%s "
                    "mean=%.8e std=%.8e min=%.8e max=%.8e",
                    tuple(ir_diag.shape),
                    str(ir_diag.dtype),
                    ir_diag.mean().item(),
                    ir_diag.std(unbiased=False).item(),
                    ir_diag.min().item(),
                    ir_diag.max().item(),
                )

                del rgb_diag, ir_diag

            # 预计算 teacher gate 在 16x16，形状 Bx16x16。
            if teacher_gate.ndim == 3:
                pass
            elif teacher_gate.ndim == 4:
                teacher_gate = teacher_gate[:, 0]
            if teacher_structure.ndim == 3:
                pass
            elif teacher_structure.ndim == 4:
                teacher_structure = teacher_structure[:, 0]

            # ---------------- condition ----------------
            condition, info = conditioner(
                rgb_layers=rgb_layers,
                ir_layers=ir_layers,
                rgb_latent=z_rgb,
                ir_latent=z_ir,
                text_hidden=text_hidden,
            )
            cond_mask = info["attention_mask"]
            pred_gate = info["ir_gate"].clamp(0.02, 0.98)
            pred_structure = info["structure"].clamp(1e-4, 1 - 1e-4)

            # ============================================================
            # DEBUG: condition numerical statistics
            # 必须放在 finite("condition", ...) 之前
            # ============================================================
            if global_step == 0:
                cond_diag = condition.detach().float()

                logger.info(
                    "[DIAG][Condition] "
                    "shape=%s dtype=%s "
                    "mean=%.8e std=%.8e min=%.8e max=%.8e",
                    tuple(cond_diag.shape),
                    str(cond_diag.dtype),
                    cond_diag.mean().item(),
                    cond_diag.std(unbiased=False).item(),
                    cond_diag.min().item(),
                    cond_diag.max().item(),
                )

                del cond_diag

            finite("condition", condition)
            finite("pred_gate", pred_gate)

            # ---------------- adaptive source anchor ----------------
            # 训练中使用 teacher gate 构造起始 latent；不是固定 z_fusion，
            # 每个样本/空间位置都不同。
            gate_anchor = teacher_gate.unsqueeze(1).clamp(0.05, 0.95)
            z_anchor = (1.0 - gate_anchor) * z_rgb + gate_anchor * z_ir

            # ---------------- flow matching ----------------
            u = compute_density_for_timestep_sampling(
                weighting_scheme="uniform",
                batch_size=1,
                logit_mean=0.0,
                logit_std=1.0,
                mode_scale=1.29,
            )
            index = int((u[0].item() * num_train_timesteps))
            index = min(max(index, 0), num_train_timesteps - 1)
            t = flow_scheduler.timesteps[index].view(1).to(device)
            sigma = flow_scheduler.sigmas.to(device=device, dtype=torch.float32)[index].view(1, 1, 1, 1)

            noise = torch.randn_like(z_anchor)
            noisy = ((1.0 - sigma) * z_anchor + sigma * noise).to(sana_dtype)

            # SANA expects 2304 caption dim. Keep normalization conservative.
            c = F.layer_norm(
                condition.float(),
                (2304,)
            )

            c = c.clamp(-5, 5)

            if sana_dtype == torch.float16:
                c = c.half()

            if global_step == 0:
                condition_diag = condition.detach().float()

                rms = torch.sqrt(
                    torch.mean(condition_diag * condition_diag)
                )

                logger.info(
                    "[DIAG][Condition RMS] %.8e",
                    rms.item(),
                )

                del condition_diag

            autocast_enabled = sana_dtype != torch.float32
            with torch.autocast(device_type="cuda", dtype=sana_dtype, enabled=autocast_enabled):
                pred = dit(
                    hidden_states=noisy,
                    timestep=t,
                    encoder_hidden_states=c,
                    encoder_attention_mask=cond_mask,
                ).sample
            pred = pred.float()

            logger.info(
                "SANA parameter dtype: %s",
                next(dit.parameters()).dtype,
            )

            for idx in [0, 13, 27]:
                block = dit.transformer_blocks[idx]

                logger.info(
                    "attn2[%d] q=%s k=%s v=%s out=%s",
                    idx,
                    block.attn2.to_q.weight.dtype,
                    block.attn2.to_k.weight.dtype,
                    block.attn2.to_v.weight.dtype,
                    block.attn2.to_out[0].weight.dtype,
                )

            if global_step == 0:
                pred_diag = pred.detach().float()

                logger.info(
                    "[DIAG][Flow pred] "
                    "shape=%s dtype=%s "
                    "mean=%.8e std=%.8e min=%.8e max=%.8e",
                    tuple(pred_diag.shape),
                    str(pred_diag.dtype),
                    pred_diag.mean().item(),
                    pred_diag.std(unbiased=False).item(),
                    pred_diag.min().item(),
                    pred_diag.max().item(),
                )

                logger.info(
                    "[DIAG][Flow pred finite] %s",
                    bool(torch.isfinite(pred_diag).all().item()),
                )

                del pred_diag

            finite("flow_pred", pred)

            target_rgb = noise - z_rgb
            target_ir = noise - z_ir
            flow_gate = (0.5 * pred_gate + 0.5 * teacher_gate).unsqueeze(1)
            flow_weight_rgb = 1.0 - flow_gate
            flow_weight_ir = flow_gate

            l_flow_rgb = ((pred - target_rgb) ** 2 * flow_weight_rgb).mean()
            l_flow_ir = ((pred - target_ir) ** 2 * flow_weight_ir).mean()
            l_flow = 0.5 * (l_flow_rgb + l_flow_ir)

            # 由速度恢复 x1，并在 latent space 直接压制几何漂移。
            x1_hat = noisy.float() - sigma * pred
            l_latent_anchor = (
                F.smooth_l1_loss(x1_hat, z_rgb, reduction="none") * flow_weight_rgb
                + F.smooth_l1_loss(x1_hat, z_ir, reduction="none") * flow_weight_ir
            ).mean()

            l_teacher, teacher_parts = teacher_loss_fn(
                pred_gate,
                pred_structure,
                teacher_gate,
                teacher_structure,
            )

            finite("flow_loss", l_flow)
            finite("latent_anchor_loss", l_latent_anchor)

            # ---------------- full-res VAE pixel branch ----------------
            vae.to(device)
            if hasattr(vae, "enable_tiling"):
                vae.enable_tiling()
            fused = decode_fullres(vae, x1_hat, scaling)
            l_pixel, pixel_parts = pixel_loss_fn(
                fused,
                visible,
                infrared,
                pred_gate,
                teacher_gate.unsqueeze(1),
            )
            finite("pixel_loss", l_pixel)

            # pixel loss warm-up：先学会条件与几何，再逐步强化 512x512 图像约束。
            ramp = 1.0 if args.pixel_ramp_steps <= 0 else min(1.0, (global_step + 1) / float(args.pixel_ramp_steps))
            lambda_pixel = args.lambda_pixel * ramp

            base_loss = (
                args.lambda_flow * l_flow
                + args.lambda_latent * l_latent_anchor
                + args.lambda_teacher * l_teacher
            )
            pixel_total = lambda_pixel * l_pixel

            # ---------------- backward：pixel 先行，无需复制数百 MB 梯度 ----------------
            # 6GB 显存下不能为了 rollback 克隆整个 SANA 条件梯度。
            # 这里让 pixel branch 保留图；若它出现非有限梯度，则清空梯度后仅重新计算 base branch。
            scale = float(args.grad_accum)
            pixel_ok = True
            try:
                pixel_total.div(scale).backward(retain_graph=True)
                for p in list(conditioner.parameters()) + list(dit.parameters()):
                    if p.requires_grad and p.grad is not None and not torch.isfinite(p.grad).all():
                        pixel_ok = False
                        break
            except RuntimeError as exc:
                pixel_ok = False
                logger.warning("pixel backward failed at step=%d: %s", global_step, exc)

            if not pixel_ok:
                logger.warning("pixel branch invalid -> clear gradients and recompute base branch only, step=%d", global_step)
                optimizer.zero_grad(set_to_none=True)
                base_loss.div(scale).backward()
            else:
                base_loss.div(scale).backward()

            # VAE no longer needed.
            l_pixel_value = float(l_pixel.item())
            vae.to("cpu")
            del fused, l_pixel
            cuda_free()

            accum += 1

            # ---------------- optimizer ----------------
            if accum >= args.grad_accum or global_step + 1 == total_steps:
                all_trainable = [p for p in list(conditioner.parameters()) + list(dit.parameters()) if p.requires_grad]
                grad_norm = safe_gradient_norm(all_trainable)
                if not math.isfinite(grad_norm):
                    logger.warning("non-finite accumulated gradient -> skip optimizer step")
                    optimizer.zero_grad(set_to_none=True)
                else:
                    torch.nn.utils.clip_grad_norm_(all_trainable, args.max_grad_norm, error_if_nonfinite=True)
                    optimizer.step()
                    lr_sched.step()
                    optimizer.zero_grad(set_to_none=True)
                accum = 0
            else:
                grad_norm = safe_gradient_norm([p for p in conditioner.parameters() if p.requires_grad])

            # ---------------- logs ----------------
            values = {
                "flow": float(l_flow.item()),
                "flow_rgb": float(l_flow_rgb.item()),
                "flow_ir": float(l_flow_ir.item()),
                "latent": float(l_latent_anchor.item()),
                "teacher": float(l_teacher.item()),
                "pixel": l_pixel_value,
                "total": float((base_loss + pixel_total).item()),
                "gate": float(pred_gate.mean().item()),
                "gate_t": float(teacher_gate.mean().item()),
                "scale": float(info["output_scale"].mean().item()),
            }
            values.update({f"pix_{k}": float(v.item()) for k, v in pixel_parts.items()})
            values.update({f"teacher_{k}": float(v.item()) for k, v in teacher_parts.items()})
            for k, v in values.items():
                ema[k] = 0.9 * ema.get(k, v) + 0.1 * v

            if global_step % args.log_every == 0:
                logger.info(
                    "step=%d | total=%.4f flow=%.4f rgb=%.4f ir=%.4f latent=%.4f pixel=%.4f teacher=%.4f | gate=%.3f/t=%.3f | out=%.3f | grad=%.3e | pixRamp=%.2f | %.1fs",
                    global_step,
                    ema["total"], ema["flow"], ema["flow_rgb"], ema["flow_ir"], ema["latent"], ema["pixel"], ema["teacher"],
                    ema["gate"], ema["gate_t"], ema["scale"], grad_norm, ramp, time.time() - tic,
                )

            # ---------------- sample ----------------
            if args.sample_every > 0 and (global_step % args.sample_every == 0 or global_step == total_steps - 1):
                try:
                    conditioner.eval()
                    dit.eval()
                    # Use cached tensors directly, no SAM.
                    fused_sample, s_info = sample_image(
                        dit=dit,
                        vae=vae,
                        scheduler=sample_scheduler,
                        conditioner=conditioner,
                        rgb_layers=rgb_layers,
                        ir_layers=ir_layers,
                        z_rgb=z_rgb,
                        z_ir=z_ir,
                        text_hidden=text_hidden,
                        steps=args.sample_steps,
                        device=device,
                        sana_dtype=sana_dtype,
                        start_fraction=args.sample_start_fraction,
                    )
                    save_panel(
                        out / "samples" / f"step{global_step:07d}_{stem}.png",
                        visible,
                        infrared,
                        fused_sample,
                    )
                    logger.info("sample saved: %s", stem)
                finally:
                    conditioner.train()
                    dit.train()

            # ---------------- checkpoint ----------------
            if args.save_every > 0 and (global_step + 1) % args.save_every == 0:
                ckpt = out / "checkpoints" / f"step{global_step + 1:07d}.pt"
                torch.save(
                    {
                        "step": global_step + 1,
                        "conditioner": conditioner.state_dict(),
                        "sana_condition": trainable_sana_state_dict(dit),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_sched.state_dict(),
                        "args": vars(args),
                    },
                    ckpt,
                )
                logger.info("checkpoint saved: %s", ckpt)

            # release per-sample GPU tensors
            del rgb_layers, ir_layers, qwen, condition, c, pred, noisy, noise
            del z_anchor, x1_hat, base_loss, pixel_total
            del visible, infrared, z_rgb, z_ir, teacher_gate, teacher_structure
            cuda_free()
            global_step += 1

    final = out / "checkpoints" / "final.pt"
    torch.save(
        {
            "step": global_step,
            "conditioner": conditioner.state_dict(),
            "sana_condition": trainable_sana_state_dict(dit),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_sched.state_dict(),
            "args": vars(args),
        },
        final,
    )
    logger.info("训练完成：%s", final)


# ============================================================
# args
# ============================================================


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default=DEFAULT_CACHE)
    ap.add_argument("--vis_root", required=True)
    ap.add_argument("--ir_root", required=True)
    ap.add_argument("--sana_path", default=DEFAULT_SANA_PATH)
    ap.add_argument("--output_dir", default=DEFAULT_OUT)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--max_steps", type=int, default=6000)
    ap.add_argument("--warmup_steps", type=int, default=300)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr_conditioner", type=float, default=1e-4)
    ap.add_argument("--lr_sana", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--optimizer", choices=["adamw8bit", "adamw"], default="adamw8bit")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--sana_dtype", choices=["fp16", "fp32"], default="fp16")

    # condition space
    ap.add_argument("--fusion_dim", type=int, default=512)
    ap.add_argument("--condition_heads", type=int, default=8)
    ap.add_argument("--cross_blocks", type=int, default=2)
    ap.add_argument("--spatial_blocks", type=int, default=2)
    ap.add_argument("--text_tokens", type=int, default=4)

    # loss
    ap.add_argument("--lambda_flow", type=float, default=1.0)
    ap.add_argument("--lambda_latent", type=float, default=0.5)
    ap.add_argument("--lambda_teacher", type=float, default=0.15)
    ap.add_argument("--lambda_pixel", type=float, default=0.5)
    ap.add_argument("--pixel_ramp_steps", type=int, default=1200)

    ap.add_argument("--w_ssim_vis", type=float, default=0.35)
    ap.add_argument("--w_ssim_ir", type=float, default=0.55)
    ap.add_argument("--w_grad", type=float, default=0.45)
    ap.add_argument("--w_int", type=float, default=0.10)
    ap.add_argument("--w_color", type=float, default=0.75)
    ap.add_argument("--w_pixel_anchor", type=float, default=0.90)
    ap.add_argument("--w_thermal", type=float, default=0.75)
    ap.add_argument("--w_teacher_gate", type=float, default=1.0)
    ap.add_argument("--w_teacher_structure", type=float, default=0.5)

    # memory
    ap.add_argument("--gradient_checkpointing", action="store_true", default=True)
    ap.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")

    # logging
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--sample_every", type=int, default=500)
    ap.add_argument("--sample_steps", type=int, default=20)
    ap.add_argument("--sample_start_fraction", type=float, default=0.55)
    ap.add_argument("--save_every", type=int, default=500)
    return ap.parse_args()


if __name__ == "__main__":
    train(parse_args())
