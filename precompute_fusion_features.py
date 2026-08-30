# -*- coding: utf-8 -*-

r"""
precompute_fusion_features.py
============================================================
离线预计算脚本：为融合训练缓存所有与可训练参数无关的重型前向

针对 6GB 显存笔记本的"极限顺序卸载"设计：
Qwen(3B) + SAM + DC-AE + SANA DiT 无法同时驻留显存，
训练时只保留 条件模块 + SANA DiT + VAE，其余全部离线预计算。

产出 4 类缓存（均在 --cache_dir 下）：

    sam_seg/{stem}.pt          阶段 A  SAM 彩色分割图 (1,3,H,W)
    qwen_hidden/{stem}.pt      阶段 B  Qwen 关键层 hidden（只存
                                 ThreeLevelFusionMCP 用到的 9 层，
                                 且只保留视/文本 token，磁盘占用小）
    seg_latent/{stem}.pt       阶段 C  分割图的 DC-AE latent
                                 latent_vis / latent_ir (1,32,14,14)
    target_latent/{stem}.pt    阶段 D  可见光锚点 latent (1,32,16,16)
                                 （flow matching 的目标，保纹理色彩）

关键约定（与训练脚本严格对齐）：
    - Qwen 输入统一 resize 到 448×448 → 每张图恰好 256 个视觉
      token（448/28=16, 16×16=256），满足 ThreeLevelFusionMCP 的
      num_tokens_per_modality=256 硬性要求。
    - 分割图用 448×448（对应 DC-AE latent 14×14 = 196 token）。
    - 可见光锚点用 512×512（对应 SANA latent 16×16）。
    - 红外若为单通道，复制为 3 通道。

用法：
    python precompute_fusion_features.py \
        --vis_root "E:\ai for science\SeAFusion-main\MSRS\Visible\train\MSRS" \
        --ir_root  "E:\ai for science\SeAFusion-main\MSRS\Infrared\train\MSRS" \
        --cache_dir D:\Mobile-O-main\fusion_cache

    # 分阶段跑（每阶段跑完自动释放显存，可分开执行）：
    #   --only sam | qwen | seg_latent | target_latent
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
from pathlib import Path
from typing import List, Tuple

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
logger = logging.getLogger("precompute")


# ============================================================
# 路径配置（与项目现有脚本一致，可用命令行覆盖）
# ============================================================

DEFAULT_QWEN_PATH = (
    r"E:\ai for science\omnifuse\models\Qwen\Qwen2___5-VL-3B-Instruct"
)

DEFAULT_SANA_PATH = (
    r"D:\Mobile-O-main\mobileo\model\Sana\Sana_600M_512px_diffusers"
)

DEFAULT_SAM_PATH = (
    r"E:\ai for science\omnifuse\models\AI-ModelScope\sam-vit-base"
)

DEFAULT_CACHE_DIR = r"D:\Mobile-O-main\fusion_cache"

# ThreeLevelFusionMCP 硬编码使用的 Qwen 层索引
MCP_LAYER_INDICES = [4, 5, 6, 13, 14, 15, 26, 27, 28]

# 448×448 -> Qwen2.5-VL 每图 256 个视觉 token（patch 14, merge 2）
QWEN_IMAGE_SIZE = 448
EXPECTED_TOKENS_PER_IMAGE = 256

# SANA / DC-AE 空间
SANA_IMAGE_SIZE = 512

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


# ============================================================
# 工具
# ============================================================

def cuda_free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_model_path():
    """把 mobileo/model 加入 sys.path（模块间为扁平导入）"""
    model_dir = Path(__file__).resolve().parent / "mobileo" / "model"
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    return model_dir


def load_image_01(path: Path, size: int) -> torch.Tensor:
    """读取图像 -> (1,3,size,size) [0,1]；灰度图复制为 3 通道"""
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    t = F.interpolate(
        t, size=(size, size), mode="bilinear", align_corners=False
    )
    return t


def collect_image_pairs(
    vis_root: Path, ir_root: Path
) -> List[Tuple[Path, Path]]:
    """按文件名配对可见光 / 红外图像"""
    ir_map = {
        p.name: p
        for p in ir_root.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    }
    pairs = []
    for p in sorted(vis_root.iterdir()):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        if p.name in ir_map:
            pairs.append((p, ir_map[p.name]))
    return pairs


# ============================================================
# 阶段 A：SAM 分割图
# ============================================================

def precompute_sam(
    image_paths: List[Path],
    sam_model_path: str,
    cache_dir: Path,
    device: str,
):
    """
    每张图 -> SAM 彩色语义分割图 -> sam_seg/{stem}.pt

    分割图统一为 448×448，下游 DC-AE 编码得到 14×14 latent。
    """
    ensure_model_path()
    from mobileo.model.sam_semantic_segmentor import SAMSemanticSegmentor

    seg_dir = cache_dir / "sam_seg"
    seg_dir.mkdir(parents=True, exist_ok=True)

    logger.info("加载 SAM 分割器 ...")
    segmentor = SAMSemanticSegmentor(
        model_path=sam_model_path,
        input_size=QWEN_IMAGE_SIZE,
        freeze=True,
    ).to(device)
    segmentor.eval()

    for path in tqdm(image_paths, desc="阶段A SAM分割"):
        out_path = seg_dir / f"{path.stem}.pt"
        if out_path.exists():
            continue

        img = load_image_01(path, QWEN_IMAGE_SIZE).to(device)
        with torch.no_grad():
            seg = segmentor(img)  # (1,3,448,448) [0,1]
        torch.save({"seg": seg.detach().cpu()}, out_path)

    del segmentor
    cuda_free()
    logger.info("阶段 A 完成")


# ============================================================
# 阶段 B：Qwen 关键层 hidden states
# ============================================================

def precompute_qwen(
    pairs: List[Tuple[Path, Path]],
    qwen_path: str,
    cache_dir: Path,
    device: str,
):
    """
    (vis, ir) 对 -> Qwen2.5-VL-3B -> qwen_hidden/{vis_stem}.pt

    只保存 ThreeLevelFusionMCP 实际使用的内容：
        layers[str(idx)]: (1, N_vis+N_ir, 2048)  fp16
            idx ∈ [4,5,6,13,14,15,26,27,28]，
            按序列顺序拼接可见光 + 红外视觉 token
        text_last: (1, N_txt, 2048) fp16   最后一层文本 token
        vis_span: (0, N_vis)               相对拼接序列的索引
        ir_span:  (N_vis, N_vis+N_ir)
        txt_span: (0, N_txt)
    """
    ensure_model_path()
    from mobileo.model.qwen25vl_wrapper import Qwen25VLWrapper

    qwen_dir = cache_dir / "qwen_hidden"
    qwen_dir.mkdir(parents=True, exist_ok=True)

    logger.info("加载 Qwen2.5-VL-3B ...")
    wrapper = Qwen25VLWrapper(
        model_path=qwen_path,
        torch_dtype=torch.bfloat16,
        device=device,
        freeze=True,
        # 强制 448×448 -> 每图 256 token
        min_pixels=QWEN_IMAGE_SIZE * QWEN_IMAGE_SIZE,
        max_pixels=QWEN_IMAGE_SIZE * QWEN_IMAGE_SIZE,
    )

    for vis_path, ir_path in tqdm(pairs, desc="阶段B Qwen特征"):
        out_path = qwen_dir / f"{vis_path.stem}.pt"
        if out_path.exists():
            continue

        # 先统一 resize 到 448×448，保证恰好 256 视觉 token
        vis_pil = Image.open(vis_path).convert("RGB").resize(
            (QWEN_IMAGE_SIZE, QWEN_IMAGE_SIZE), Image.BICUBIC
        )
        ir_pil = Image.open(ir_path).convert("RGB").resize(
            (QWEN_IMAGE_SIZE, QWEN_IMAGE_SIZE), Image.BICUBIC
        )

        try:
            with torch.no_grad():
                result = wrapper(
                    visible=vis_pil,
                    infrared=ir_pil,
                    prompt=(
                        "Fuse the visible and infrared images: preserve "
                        "visible textures and colors, and inject infrared "
                        "thermal targets and salient regions."
                    ),
                    return_all_hidden_states=True,
                )
        except Exception as e:
            logger.warning(f"跳过 {vis_path.name}: {e}")
            continue

        # ---- 校验 token 数 ----
        counts = result["image_token_counts"][0]
        if counts[0] != EXPECTED_TOKENS_PER_IMAGE or counts[1] != EXPECTED_TOKENS_PER_IMAGE:
            logger.warning(
                f"跳过 {vis_path.name}: 视觉 token 数 {counts} "
                f"!= [{EXPECTED_TOKENS_PER_IMAGE}, {EXPECTED_TOKENS_PER_IMAGE}]"
            )
            continue

        hs = result["hidden_states"]                       # tuple, 37 x (1,L,2048)
        vis_mask = result["visible_token_mask"][0].bool().cpu()
        ir_mask = result["infrared_token_mask"][0].bool().cpu()
        attn = result["attention_mask"]
        attn = attn[0].bool().cpu() if attn is not None else torch.ones(
            hs[0].shape[1], dtype=torch.bool
        )
        text_mask = attn & ~vis_mask & ~ir_mask

        n_vis = int(vis_mask.sum())
        n_ir = int(ir_mask.sum())
        n_txt = int(text_mask.sum())

        if n_txt == 0:
            logger.warning(f"跳过 {vis_path.name}: 文本 span 为空")
            continue

        layers_payload = {}
        for idx in MCP_LAYER_INDICES:
            h = hs[idx][0].float().cpu()                  # (L, 2048)
            sliced = h[vis_mask | ir_mask]                # 保持序列顺序
            layers_payload[str(idx)] = sliced.to(torch.float16).unsqueeze(0)

        text_last = (
            hs[-1][0].float().cpu()[text_mask]
            .to(torch.float16)
            .unsqueeze(0)
        )

        payload = {
            "layers": layers_payload,
            "text_last": text_last,
            "vis_span": (0, n_vis),
            "ir_span": (n_vis, n_vis + n_ir),
            "txt_span": (0, n_txt),
        }
        torch.save(payload, out_path)

    del wrapper
    cuda_free()
    logger.info("阶段 B 完成")


# ============================================================
# 阶段 C：分割图 -> DC-AE latent
# ============================================================

def precompute_seg_latents(
    pairs: List[Tuple[Path, Path]],
    sana_path: str,
    cache_dir: Path,
    device: str,
):
    """
    sam_seg/{vis_stem}.pt + sam_seg/{ir_stem}.pt
        -> DC-AE encode -> seg_latent/{vis_stem}.pt

    输出:
        latent_vis: (1,32,14,14) fp16   （已乘 scaling_factor）
        latent_ir:  (1,32,14,14) fp16
    """
    from diffusers import AutoencoderDC

    seg_dir = cache_dir / "sam_seg"
    out_dir = cache_dir / "seg_latent"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("加载 DC-AE ...")
    vae = AutoencoderDC.from_pretrained(
        sana_path, subfolder="vae", variant="fp16",
        torch_dtype=torch.float32, low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device)
    vae.eval()
    scaling = float(getattr(vae.config, "scaling_factor", 1.0))

    for vis_path, ir_path in tqdm(pairs, desc="阶段C 分割latent"):
        out_path = out_dir / f"{vis_path.stem}.pt"
        if out_path.exists():
            continue

        seg_vis_file = seg_dir / f"{vis_path.stem}.pt"
        seg_ir_file = seg_dir / f"{ir_path.stem}.pt"
        if not seg_vis_file.exists() or not seg_ir_file.exists():
            logger.warning(f"缺少分割图缓存: {vis_path.stem}，跳过")
            continue

        seg_vis = torch.load(seg_vis_file, map_location="cpu")["seg"].to(device)
        seg_ir = torch.load(seg_ir_file, map_location="cpu")["seg"].to(device)

        with torch.no_grad():
            lat_vis = vae.encode(seg_vis.float()).latent * scaling
            lat_ir = vae.encode(seg_ir.float()).latent * scaling

        torch.save(
            {
                "latent_vis": lat_vis.detach().cpu().to(torch.float16),
                "latent_ir": lat_ir.detach().cpu().to(torch.float16),
            },
            out_path,
        )

    del vae
    cuda_free()
    logger.info("阶段 C 完成")


# ============================================================
# 阶段 D：可见光锚点 -> SANA latent（flow 目标）
# ============================================================

def precompute_target_latents(
    pairs: List[Tuple[Path, Path]],
    sana_path: str,
    cache_dir: Path,
    device: str,
):
    """
    可见光图 512×512 -> DC-AE encode -> target_latent/{vis_stem}.pt
        latent: (1,32,16,16) fp16  （已乘 scaling_factor）

    这是 flow matching 的目标 z：以可见光为锚点，保证融合结果
    保留纹理与色彩（Dif-Fusion 色彩保真思想）；红外信息通过
    条件与像素损失注入。
    """
    from diffusers import AutoencoderDC

    out_dir = cache_dir / "target_latent"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("加载 DC-AE ...")
    vae = AutoencoderDC.from_pretrained(
        sana_path, subfolder="vae", variant="fp16",
        torch_dtype=torch.float32, low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device)
    vae.eval()
    scaling = float(getattr(vae.config, "scaling_factor", 1.0))

    for vis_path, _ in tqdm(pairs, desc="阶段D 锚点latent"):
        out_path = out_dir / f"{vis_path.stem}.pt"
        if out_path.exists():
            continue

        img = load_image_01(vis_path, SANA_IMAGE_SIZE).to(device)
        with torch.no_grad():
            lat = vae.encode(img.float()).latent * scaling  # (1,32,16,16)

        torch.save({"latent": lat.detach().cpu().to(torch.float16)}, out_path)

    del vae
    cuda_free()
    logger.info("阶段 D 完成")


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vis_root", type=str, required=True)
    parser.add_argument("--ir_root", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--qwen_path", type=str, default=DEFAULT_QWEN_PATH)
    parser.add_argument("--sana_path", type=str, default=DEFAULT_SANA_PATH)
    parser.add_argument("--sam_path", type=str, default=DEFAULT_SAM_PATH)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        choices=["sam", "qwen", "seg_latent", "target_latent"],
        help="只运行指定阶段（每个阶段独立加载/释放模型）",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    pairs = collect_image_pairs(Path(args.vis_root), Path(args.ir_root))
    if not pairs:
        raise RuntimeError(
            f"未找到配对图像: {args.vis_root} <-> {args.ir_root}"
        )
    logger.info(f"共 {len(pairs)} 对图像")

    all_paths = [p for pair in pairs for p in pair]
    stages = [args.only] if args.only else ["sam", "qwen", "seg_latent", "target_latent"]

    for stage in stages:
        if stage == "sam":
            precompute_sam(all_paths, args.sam_path, cache_dir, args.device)
        elif stage == "qwen":
            precompute_qwen(pairs, args.qwen_path, cache_dir, args.device)
        elif stage == "seg_latent":
            precompute_seg_latents(pairs, args.sana_path, cache_dir, args.device)
        elif stage == "target_latent":
            precompute_target_latents(pairs, args.sana_path, cache_dir, args.device)

    logger.info("全部预计算完成。训练时用 --cache_dir 指向该目录。")


if __name__ == "__main__":
    main()
