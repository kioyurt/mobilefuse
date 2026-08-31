# -*- coding: utf-8 -*-
"""
Precompute features v2
======================

为新 MobileFuse v2 训练准备缓存。

变化：
- 全部源图像统一读取/预处理到 512x512。
- Qwen 不再硬编码 256 visual tokens；实际 visual token 网格记录到缓存，训练时统一重采样到 16x16。
- 不再缓存 14x14 SAM/DC-AE 条件。
- DC-AE 编码真实 RGB / IR 图像，得到 16x16 latent。
- SAM 只生成训练 teacher：IR gate + structure teacher，最终模型推理不需要 SAM。
- Qwen 保存视觉多层特征 + 多个文本 prompt 的全局 hidden。

输出：
cache_dir/
  qwen_v2/{stem}.pt
      rgb_layers[str(layer)] : [1,N_rgb,2048] fp16
      ir_layers[str(layer)]  : [1,N_ir,2048] fp16
      rgb_hw / ir_hw
      text_by_prompt[str(i)] : [1,T,2048] fp16
      prompts : list[str]
  latent_v2/{stem}.pt
      rgb_latent : [1,32,16,16] fp16
      ir_latent  : [1,32,16,16] fp16
  teacher_v2/{stem}.pt
      ir_gate      : [1,16,16] fp16
      structure    : [1,16,16] fp16
      sam_vis / sam_ir 不落盘（默认只保留老师结果）
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("precompute_v2")

QWEN_PATH = r"E:\ai for science\omnifuse\models\Qwen\Qwen2___5-VL-3B-Instruct"
SANA_PATH = r"D:\Mobile-O-main\mobileo\model\Sana\Sana_600M_512px_diffusers"
SAM_PATH = r"E:\ai for science\omnifuse\models\AI-ModelScope\sam-vit-base"
CACHE_DIR = r"D:\Mobile-O-main\fusion_cache_v2"

IMAGE_SIZE = 512
LATENT_SIZE = 16
LLM_DIM = 2048
LAYERS = (4, 5, 6, 13, 14, 15, 26, 27, 28)
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

DEFAULT_PROMPTS = [
    "Fuse the visible and infrared images while preserving visible colors and textures and highlighting thermal targets.",
    "Preserve visible-light background details and introduce complementary infrared salient targets.",
    "Prioritize thermal targets and boundaries while keeping the visible image geometry and colors stable.",
    "Create a geometrically faithful RGB-infrared fusion with visible color and infrared thermal structure.",
]


def ensure_model_path():
    model_dir = Path(__file__).resolve().parent / "mobileo" / "model"
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    return model_dir


def cuda_free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_rgb_512(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if img.size != (IMAGE_SIZE, IMAGE_SIZE):
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def load_ir_512(path: Path) -> torch.Tensor:
    # 原始 IR 如果为单通道，则复制到 3 channel 给 Qwen/DC-AE；训练 pixel loss 仍从源图读取单通道。
    img = Image.open(path).convert("L")
    if img.size != (IMAGE_SIZE, IMAGE_SIZE):
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
    return t


def collect_pairs(vis_root: Path, ir_root: Path):
    ir_map = {p.name: p for p in ir_root.iterdir() if p.suffix.lower() in EXTS}
    out = []
    for p in sorted(vis_root.iterdir()):
        if p.suffix.lower() in EXTS and p.name in ir_map:
            out.append((p, ir_map[p.name]))
    return out


def gray(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 1:
        return x
    return (x * torch.tensor([0.299, 0.587, 0.114], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)).sum(1, keepdim=True)


def norm01(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mn = x.amin((-2, -1), keepdim=True)
    mx = x.amax((-2, -1), keepdim=True)
    return (x - mn) / (mx - mn + eps)


def sobel_mag(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] != 1:
        x = gray(x)
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    ky = kx.transpose(-1, -2).contiguous()
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx*gx + gy*gy + 1e-8)


def boundary_from_sam(seg: torch.Tensor) -> torch.Tensor:
    """
    从 SAM 的彩色分割图提取单通道边界图。

    输入必须是 [B,C,H,W]。每个通道的 Sobel 梯度都是 [B,1,H,W]；
    这里必须沿通道维 concat 成 [B,C,H,W]，不能使用 stack。
    如果使用 stack，会得到 [B,C,1,H,W]，后续 sum/插值会多出一个
    空间维度，最终触发 F.interpolate 的:
        input with spatial dimensions of [1,H,W]
    错误。
    """
    if seg.ndim != 4:
        raise ValueError(
            f"SAM segmentation must be [B,C,H,W], got shape={tuple(seg.shape)}"
        )

    gx_list = []
    gy_list = []
    kx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=seg.dtype,
        device=seg.device,
    ).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2).contiguous()

    for c in range(seg.shape[1]):
        x = seg[:, c:c + 1]
        gx_list.append(F.conv2d(x, kx, padding=1))
        gy_list.append(F.conv2d(x, ky, padding=1))

    # [B,C,H,W]，保持四维，不引入额外的 singleton spatial dimension。
    gx = torch.cat(gx_list, dim=1)
    gy = torch.cat(gy_list, dim=1)

    boundary = torch.sqrt(
        (gx * gx + gy * gy + 1e-8).sum(dim=1, keepdim=True)
    )
    return boundary


def make_teacher_gate(ir: torch.Tensor, sam_ir: torch.Tensor, sam_vis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    ir_g = gray(ir)
    ir_contrast = norm01(ir_g)
    ir_grad = norm01(sobel_mag(ir_g))
    b_ir = norm01(boundary_from_sam(sam_ir))
    b_vis = norm01(boundary_from_sam(sam_vis))

    # 红外响应 + IR 边界为主；SAM boundary 作为训练阶段的结构先验。
    sal = 0.55 * ir_contrast + 0.30 * ir_grad + 0.15 * b_ir
    sal = sal.clamp(0, 1)
    gate = torch.sigmoid(4.0 * (sal - 0.50))
    gate = 0.10 + 0.80 * gate

    structure = torch.maximum(b_ir, b_vis).clamp(0, 1)

    gate16 = F.interpolate(gate, size=(LATENT_SIZE, LATENT_SIZE), mode="area").squeeze(1)
    structure16 = F.interpolate(structure, size=(LATENT_SIZE, LATENT_SIZE), mode="area").squeeze(1)
    return gate16, structure16


def precompute_sam(pairs, sam_path, cache_dir: Path, device: str):
    ensure_model_path()
    from mobileo.model.sam_semantic_segmentor import SAMSemanticSegmentor

    out_dir = cache_dir / "sam_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    segmentor = SAMSemanticSegmentor(
        model_path=sam_path,
        input_size=IMAGE_SIZE,
        points_per_side=8,
        max_masks=16,
        freeze=True,
    ).to(device)
    segmentor.eval()

    for vp, ip in tqdm(pairs, desc="SAM teacher"):
        out = out_dir / f"{vp.stem}.pt"
        if out.exists():
            continue
        rgb = load_rgb_512(vp).to(device)
        ir = load_ir_512(ip).to(device)
        with torch.no_grad():
            s_rgb = segmentor(rgb).detach().cpu()
            s_ir = segmentor(ir).detach().cpu()
        torch.save({"sam_vis": s_rgb, "sam_ir": s_ir}, out)
        del rgb, ir, s_rgb, s_ir
    del segmentor
    cuda_free()


def precompute_qwen(pairs, qwen_path, cache_dir: Path, device: str, prompts: Sequence[str]):
    ensure_model_path()
    from mobileo.model.qwen25vl_wrapper import Qwen25VLWrapper

    out_dir = cache_dir / "qwen_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    wrapper = Qwen25VLWrapper(
        model_path=qwen_path,
        torch_dtype=torch.bfloat16,
        device=device,
        freeze=True,
        min_pixels=IMAGE_SIZE * IMAGE_SIZE,
        max_pixels=IMAGE_SIZE * IMAGE_SIZE,
    )

    base_prompt = prompts[0]

    for vp, ip in tqdm(pairs, desc="Qwen v2"):
        out = out_dir / f"{vp.stem}.pt"
        if out.exists():
            continue

        vis_pil = Image.open(vp).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
        ir_pil = Image.open(ip).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC).convert("RGB")

        result_base = None
        try:
            with torch.no_grad():
                result_base = wrapper(
                    visible=vis_pil,
                    infrared=ir_pil,
                    prompt=base_prompt,
                    return_all_hidden_states=True,
                )
        except Exception as exc:
            logger.warning("Qwen base failed %s: %s", vp.name, exc)
            continue

        hs = result_base["hidden_states"]
        vm = result_base["visible_token_mask"][0].bool().cpu()
        im = result_base["infrared_token_mask"][0].bool().cpu()
        attn = result_base.get("attention_mask")
        if attn is not None:
            attn = attn[0].bool().cpu()
        else:
            attn = torch.ones(hs[0].shape[1], dtype=torch.bool)
        text_mask = attn & ~vm & ~im

        n_rgb = int(vm.sum())
        n_ir = int(im.sum())
        side_rgb = int(round(math.sqrt(n_rgb)))
        side_ir = int(round(math.sqrt(n_ir)))
        if side_rgb * side_rgb != n_rgb or side_ir * side_ir != n_ir:
            logger.warning("%s visual tokens are not square: rgb=%s ir=%s; skip", vp.name, n_rgb, n_ir)
            continue

        rgb_layers = {}
        ir_layers = {}
        for idx in LAYERS:
            h = hs[idx][0].float().cpu()
            rgb_layers[str(idx)] = h[vm].to(torch.float16).unsqueeze(0)
            ir_layers[str(idx)] = h[im].to(torch.float16).unsqueeze(0)

        text_bank = {}
        text_bank["0"] = hs[-1][0].float().cpu()[text_mask].to(torch.float16).unsqueeze(0)

        # 额外 prompt 只缓存文本 hidden；视觉层固定取 base prompt，避免视觉空间因文本变化而漂移。
        for pi, prompt in enumerate(prompts[1:], start=1):
            try:
                with torch.no_grad():
                    res = wrapper(
                        visible=vis_pil,
                        infrared=ir_pil,
                        prompt=prompt,
                        return_all_hidden_states=True,
                    )
                hs_p = res["hidden_states"]
                attn_p = res.get("attention_mask")
                if attn_p is not None:
                    am = attn_p[0].bool().cpu()
                else:
                    am = torch.ones(hs_p[0].shape[1], dtype=torch.bool)
                # 重新基于 base 的视觉 token 位置不可靠，因此仅取尾部文本区：先排除同数量 visual tokens。
                vm_p = res["visible_token_mask"][0].bool().cpu()
                im_p = res["infrared_token_mask"][0].bool().cpu()
                tm_p = am & ~vm_p & ~im_p
                text_bank[str(pi)] = hs_p[-1][0].float().cpu()[tm_p].to(torch.float16).unsqueeze(0)
                del res, hs_p
            except Exception as exc:
                logger.warning("text prompt %d failed for %s: %s", pi, vp.name, exc)

        payload = {
            "rgb_layers": rgb_layers,
            "ir_layers": ir_layers,
            "rgb_hw": (side_rgb, side_rgb),
            "ir_hw": (side_ir, side_ir),
            "text_by_prompt": text_bank,
            "prompts": list(prompts),
        }
        torch.save(payload, out)
        del result_base, hs, rgb_layers, ir_layers, text_bank
        cuda_free()

    del wrapper
    cuda_free()


def precompute_latents_and_teacher(pairs, sana_path, cache_dir: Path, device: str):
    from diffusers import AutoencoderDC

    latent_dir = cache_dir / "latent_v2"
    teacher_dir = cache_dir / "teacher_v2"
    latent_dir.mkdir(parents=True, exist_ok=True)
    teacher_dir.mkdir(parents=True, exist_ok=True)
    sam_dir = cache_dir / "sam_v2"

    vae = AutoencoderDC.from_pretrained(
        sana_path,
        subfolder="vae",
        variant="fp16",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device)
    vae.eval()
    scaling = float(getattr(vae.config, "scaling_factor", 1.0))

    for vp, ip in tqdm(pairs, desc="DC-AE + teacher"):
        latent_out = latent_dir / f"{vp.stem}.pt"
        teacher_out = teacher_dir / f"{vp.stem}.pt"

        rgb = load_rgb_512(vp).to(device)
        ir3 = load_ir_512(ip).to(device)

        with torch.no_grad():
            z_rgb = vae.encode(rgb.float()).latent * scaling
            z_ir = vae.encode(ir3.float()).latent * scaling

        if not latent_out.exists():
            torch.save(
                {
                    "rgb_latent": z_rgb.detach().cpu().to(torch.float16),
                    "ir_latent": z_ir.detach().cpu().to(torch.float16),
                },
                latent_out,
            )

        if not teacher_out.exists():
            sam_file = sam_dir / f"{vp.stem}.pt"
            if not sam_file.exists():
                raise FileNotFoundError(f"missing SAM cache: {sam_file}")
            try:
                sam = torch.load(sam_file, map_location="cpu", weights_only=True)
            except TypeError:
                # 兼容旧版 PyTorch：不支持 weights_only 参数时回退。
                sam = torch.load(sam_file, map_location="cpu")
            sam_vis = sam["sam_vis"].to(device=device, dtype=torch.float32)
            sam_ir = sam["sam_ir"].to(device=device, dtype=torch.float32)

            if sam_vis.ndim == 3:
                sam_vis = sam_vis.unsqueeze(0)
            if sam_ir.ndim == 3:
                sam_ir = sam_ir.unsqueeze(0)
            if ir3.ndim == 3:
                ir3 = ir3.unsqueeze(0)

            if sam_vis.ndim != 4 or sam_ir.ndim != 4 or ir3.ndim != 4:
                raise RuntimeError(
                    f"teacher input shape error for {vp.name}: "
                    f"ir={tuple(ir3.shape)}, sam_vis={tuple(sam_vis.shape)}, "
                    f"sam_ir={tuple(sam_ir.shape)}"
                )

            gate16, structure16 = make_teacher_gate(ir3, sam_ir, sam_vis)
            torch.save(
                {
                    "ir_gate": gate16.detach().cpu().to(torch.float16),
                    "structure": structure16.detach().cpu().to(torch.float16),
                },
                teacher_out,
            )
            del sam, sam_vis, sam_ir, gate16, structure16

        del rgb, ir3, z_rgb, z_ir
        cuda_free()

    del vae
    cuda_free()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vis_root", required=True)
    ap.add_argument("--ir_root", required=True)
    ap.add_argument("--cache_dir", default=CACHE_DIR)
    ap.add_argument("--qwen_path", default=QWEN_PATH)
    ap.add_argument("--sana_path", default=SANA_PATH)
    ap.add_argument("--sam_path", default=SAM_PATH)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", choices=["sam", "qwen", "latent"], default=None)
    ap.add_argument("--prompts_json", default=None)
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    pairs = collect_pairs(Path(args.vis_root), Path(args.ir_root))
    if not pairs:
        raise RuntimeError("没有找到严格按文件名配对的 RGB/IR 图像")

    prompts = DEFAULT_PROMPTS
    if args.prompts_json:
        prompts = json.loads(Path(args.prompts_json).read_text(encoding="utf-8"))
        if not isinstance(prompts, list) or not prompts:
            raise ValueError("prompts_json 必须是非空字符串列表")

    stages = [args.only] if args.only else ["sam", "qwen", "latent"]
    logger.info("pairs=%d stages=%s prompts=%d", len(pairs), stages, len(prompts))

    if "sam" in stages:
        precompute_sam(pairs, args.sam_path, cache_dir, args.device)
    if "qwen" in stages:
        precompute_qwen(pairs, args.qwen_path, cache_dir, args.device, prompts)
    if "latent" in stages:
        precompute_latents_and_teacher(pairs, args.sana_path, cache_dir, args.device)

    logger.info("v2 预计算完成: %s", cache_dir)


if __name__ == "__main__":
    main()
