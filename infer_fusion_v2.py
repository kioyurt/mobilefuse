# -*- coding: utf-8 -*-
"""
MobileFuse v2 inference
=======================

最终推理不使用 SAM。
默认使用预计算的 Qwen visual/text cache + RGB/IR DC-AE latent，
因此 6GB GPU 不需要同时加载 Qwen + SAM + SANA + VAE。

输入：
  --cache_dir : v2 precompute cache
  --checkpoint : v2 train checkpoint/final.pt
  --stem : 图像 stem

输出：
  RGB | IR | fused 三联图，以及可选 IR gate 可视化。
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from diffusers import AutoencoderDC, FlowMatchEulerDiscreteScheduler, SanaTransformer2DModel

from mobileo.model.multimodal_spatial_condition_v2 import MultimodalSpatialConditionEncoder, load_conditioner_checkpoint

IMAGE_SIZE = 512
LLM_DIM = 2048
CAPTION_DIM = 2304
SANA_DEFAULT = r"D:\Mobile-O-main\mobileo\model\Sana\Sana_600M_512px_diffusers"
CACHE_DEFAULT = r"D:\Mobile-O-main\fusion_cache_v2"


def cuda_free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_image(root: Path, stem: str, infrared: bool):
    exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
    for e in exts:
        p = root / f"{stem}{e}"
        if p.exists():
            if infrared:
                img = Image.open(p).convert("L")
                if img.size != (IMAGE_SIZE, IMAGE_SIZE):
                    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
                return img
            img = Image.open(p).convert("RGB")
            if img.size != (IMAGE_SIZE, IMAGE_SIZE):
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
            return img
    raise FileNotFoundError(f"找不到 {root}/{stem}.*")


def to_tensor(img: Image.Image, device: str, infrared: bool):
    a = np.asarray(img, dtype=np.float32) / 255.0
    if infrared:
        return torch.from_numpy(a).unsqueeze(0).unsqueeze(0).to(device)
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device)


def save_panel(path: Path, vis: torch.Tensor, ir: torch.Tensor, fused: torch.Tensor):
    def pil(x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        a = (x[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(a)
    a, b, c = pil(vis), pil(ir), pil(fused)
    w, h = a.size
    panel = Image.new("RGB", (w * 3 + 8, h))
    panel.paste(a, (0, 0))
    panel.paste(b, (w + 4, 0))
    panel.paste(c, (2 * w + 8, 0))
    panel.save(path)


def apply_checkpoint(conditioner, dit, checkpoint: str):
    state = torch.load(checkpoint, map_location="cpu")
    conditioner.load_state_dict(state["conditioner"], strict=True)
    if "sana_condition" in state:
        named = dict(dit.named_parameters())
        for k, v in state["sana_condition"].items():
            if k in named:
                named[k].data.copy_(v.to(device=named[k].device, dtype=named[k].dtype))
            else:
                raise KeyError(f"checkpoint 中 SANA trainable 参数不存在于当前 SANA: {k}")
    else:
        raise RuntimeError("checkpoint 不包含 sana_condition")


@torch.no_grad()
def infer(args):
    device = args.device
    sana_dtype = torch.float16 if args.sana_dtype == "fp16" else torch.float32

    cache = Path(args.cache_dir)
    qwen_file = cache / "qwen_v2" / f"{args.stem}.pt"
    latent_file = cache / "latent_v2" / f"{args.stem}.pt"
    if not qwen_file.exists() or not latent_file.exists():
        raise FileNotFoundError(
            f"缺少 v2 cache：{qwen_file} 或 {latent_file}。先运行 precompute_fusion_features_v2.py。"
        )

    qwen = torch.load(qwen_file, map_location="cpu")
    lat = torch.load(latent_file, map_location="cpu")
    rgb_layers = {k: v.to(device=device, dtype=torch.float32) for k, v in qwen["rgb_layers"].items()}
    ir_layers = {k: v.to(device=device, dtype=torch.float32) for k, v in qwen["ir_layers"].items()}
    z_rgb = lat["rgb_latent"].to(device=device, dtype=torch.float32)
    z_ir = lat["ir_latent"].to(device=device, dtype=torch.float32)

    bank = qwen.get("text_by_prompt", {})
    if bank:
        key = str(args.prompt_id)
        if key not in bank:
            key = sorted(bank.keys(), key=lambda x: int(x))[0]
        text_hidden = bank[key].to(device=device, dtype=torch.float32)
    else:
        text_hidden = None

    conditioner = MultimodalSpatialConditionEncoder(
        llm_dim=LLM_DIM,
        latent_ch=32,
        fusion_dim=args.fusion_dim,
        caption_dim=CAPTION_DIM,
        target_hw=(16, 16),
        selected_layers=(4, 5, 6, 13, 14, 15, 26, 27, 28),
        num_heads=args.condition_heads,
        num_cross_blocks=args.cross_blocks,
        num_spatial_blocks=args.spatial_blocks,
        num_text_tokens=args.text_tokens,
    ).to(device=device, dtype=torch.float32)

    dit = SanaTransformer2DModel.from_pretrained(
        args.sana_path,
        subfolder="transformer",
        variant="fp16",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device=device, dtype=sana_dtype)
    for p in dit.parameters():
        p.requires_grad = False
    conditioner.eval()
    dit.eval()
    apply_checkpoint(conditioner, dit, args.checkpoint)

    condition, info = conditioner(
        rgb_layers=rgb_layers,
        ir_layers=ir_layers,
        rgb_latent=z_rgb,
        ir_latent=z_ir,
        text_hidden=text_hidden,
    )
    condition = condition.clamp(-8, 8).to(sana_dtype)
    mask = info["attention_mask"]
    g = info["ir_gate"].clamp(0.05, 0.95).unsqueeze(1)
    z_anchor = (1 - g) * z_rgb + g * z_ir

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.sana_path, subfolder="scheduler")
    scheduler.set_timesteps(args.steps, device=device)
    timesteps = scheduler.timesteps
    sigmas = scheduler.sigmas.to(device=device, dtype=torch.float32)

    start_idx = int(round((len(timesteps) - 1) * float(args.start_fraction)))
    start_idx = max(0, min(start_idx, len(timesteps) - 1))
    sigma0 = sigmas[start_idx]
    noise = torch.randn_like(z_anchor)
    latents = ((1 - sigma0) * z_anchor + sigma0 * noise).to(sana_dtype)

    for i in range(start_idx, len(timesteps)):
        t = timesteps[i].view(1)
        use_amp = sana_dtype != torch.float32
        with torch.autocast(device_type="cuda", dtype=sana_dtype, enabled=use_amp):
            pred = dit(
                hidden_states=latents,
                timestep=t,
                encoder_hidden_states=condition,
                encoder_attention_mask=mask,
            ).sample
        latents = scheduler.step(pred.float(), t, latents.float()).prev_sample.to(sana_dtype).clamp(-20, 20)

    vae = AutoencoderDC.from_pretrained(
        args.sana_path,
        subfolder="vae",
        variant="fp16",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device)
    vae.requires_grad_(False)
    vae.eval()
    if hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
    scale = float(getattr(vae.config, "scaling_factor", 1.0))
    with torch.autocast(device_type="cuda", enabled=False):
        decoded = vae.decode(latents.float() / scale).sample
    fused = ((decoded.float() + 1.0) / 2.0).clamp(0, 1)

    if args.vis_root and args.ir_root:
        vis_img = to_tensor(load_image(Path(args.vis_root), args.stem, False), device, False)
        ir_img = to_tensor(load_image(Path(args.ir_root), args.stem, True), device, True)
        save_panel(Path(args.output), vis_img, ir_img, fused)
    else:
        img = fused[0].permute(1, 2, 0).cpu().numpy()
        Image.fromarray((img * 255).astype(np.uint8)).save(args.output)

    gate = info["ir_gate"][0].cpu().numpy()
    gate_img = Image.fromarray((gate / max(gate.max(), 1e-6) * 255).astype(np.uint8)).resize((512, 512), Image.Resampling.BILINEAR)
    gate_img.save(str(Path(args.output).with_name(Path(args.output).stem + "_ir_gate.png")))
    print("saved:", args.output)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default=CACHE_DEFAULT)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--sana_path", default=SANA_DEFAULT)
    ap.add_argument("--vis_root", default=None)
    ap.add_argument("--ir_root", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sana_dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--start_fraction", type=float, default=0.35)
    ap.add_argument("--prompt_id", type=int, default=0)
    ap.add_argument("--fusion_dim", type=int, default=512)
    ap.add_argument("--condition_heads", type=int, default=8)
    ap.add_argument("--cross_blocks", type=int, default=2)
    ap.add_argument("--spatial_blocks", type=int, default=2)
    ap.add_argument("--text_tokens", type=int, default=4)
    return ap.parse_args()


if __name__ == "__main__":
    infer(parse_args())
