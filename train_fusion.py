# -*- coding: utf-8 -*-

r"""
train_fusion.py
============================================================
Mobile-O baseline 可见光 / 红外图像融合训练脚本（无 GT 监督）

训练对象（用户确认的方案）：
    - 只训练条件融合模块：
        ThreeLevelFusionMCP（三级分层融合）
        + SegConditionViaDCAE 的可训练投影（latent_to_cond + 类型嵌入）
    - 冻结：Qwen2.5-VL-3B / SAM / DC-AE / SANA DiT / VAE

损失设计（无需融合 ground-truth）：
    L_total = L_flow + w_pix * L_pixel

    1. L_flow —— SANA/Mobile-O 原生 flow matching 损失（潜空间）
         noisy = (1-σ)·z + σ·ε
         target = ε - z          （z = 可见光锚点 latent）
         与 mobileo/model/language_model/mobileo.py 完全一致。
         可见光锚点保证生成图保留纹理与色彩（Dif-Fusion 的
         色彩保真思想），红外信息由下面的像素损失注入。

    2. L_pixel —— 无监督融合像素损失（像素空间，可微）
         由 mobileo/model/fusion_losses.py 提供：
         SSIM(fused, vis) + SSIM(fused_gray, ir)
         + Sobel 梯度保持 + 强度保持。
         通过 VAE 解码回像素空间计算，梯度沿
         VAE decoder <- SANA DiT <- 条件 回传到融合模块。

显存策略（6GB 笔记本，极限顺序卸载）：
    Qwen / SAM / DC-AE 的重型前向已由
    precompute_fusion_features.py 离线缓存，训练时显存里只有：
        SANA DiT (fp16, 梯度检查点, 冻结)
        + VAE (fp32, 冻结, 仅解码)
        + 融合条件模块 (fp32, 可训练)
    batch_size 固定为 1，用梯度累积增大等效批量。

用法：
    python train_fusion.py --cache_dir   D:\Mobile-O-main\fusion_cache --vis_root    E:\ai for science\SeAFusion-main\MSRS\Visible\train\MSRS --ir_root     E:\ai for science\SeAFusion-main\MSRS\Infrared\train\MSRS --output_dir  D:\Mobile-O-main\fusion_runs\exp1
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_fusion")


# ============================================================
# 路径 / 常量
# ============================================================

MODEL_DIR = Path(__file__).resolve().parent / "mobileo" / "model"

DEFAULT_SANA_PATH = (
    r"D:\Mobile-O-main\mobileo\model\Sana\Sana_600M_512px_diffusers"
)

# ThreeLevelFusionMCP 使用的 Qwen 层索引（2620 行版官方实现硬编码）
MCP_LAYER_INDICES = (4, 5, 6, 13, 14, 15, 26, 27, 28)
QWEN_NUM_LAYERS_3B = 36          # 3B 共 36 层
HIDDEN_TUPLE_LEN = QWEN_NUM_LAYERS_3B + 1   # +embedding = 37，[-1] = 第 36 层

LLM_DIM_3B = 2048
CAPTION_DIM = 2304
TOKENS_PER_MODALITY = 256        # 448x448 输入 -> 每图恰好 256 个视觉 token
SANA_LATENT_SIZE = 16            # 512px / f32 -> 16x16
SANA_LATENT_CH = 32

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


# ============================================================
# 数据集：读取预计算缓存
# ============================================================

class FusionCacheDataset(Dataset):
    """
    每个样本包含：
        qwen_hidden/{stem}.pt : Qwen 多层 hidden + spans
        seg_latent/{stem}.pt  : vis+ir 分割图的 DC-AE latent
        target_latent/{stem}.pt : 可见光锚点 latent（flow 目标）
        源图像（像素损失用，训练时从磁盘读取）
    """

    def __init__(
        self,
        cache_dir: Path,
        vis_root: Path,
        ir_root: Path,
        image_size: int = 512,
    ):
        self.cache_dir = Path(cache_dir)
        self.vis_root = Path(vis_root)
        self.ir_root = Path(ir_root)
        self.image_size = image_size

        qwen_dir = self.cache_dir / "qwen_hidden"
        seg_dir = self.cache_dir / "seg_latent"
        tgt_dir = self.cache_dir / "target_latent"

        self.stems: List[str] = []
        for f in sorted(qwen_dir.glob("*.pt")):
            stem = f.stem
            if (seg_dir / f"{stem}.pt").exists() and (tgt_dir / f"{stem}.pt").exists():
                self.stems.append(stem)

        if not self.stems:
            raise RuntimeError(
                f"缓存不完整或为空: {cache_dir}\n"
                "请先运行 precompute_fusion_features.py"
            )

        logger.info(f"数据集: {len(self.stems)} 个有效样本")

    def __len__(self):
        return len(self.stems)

    def _find_image(self, root: Path, stem: str) -> Path:
        for ext in IMAGE_EXTS:
            p = root / f"{stem}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(f"{root} 中找不到 {stem}.*")

    def _load_image_01(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        t = F.interpolate(
            t, size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False,
        )
        return t  # (1,3,H,W) [0,1]

    def __getitem__(self, idx: int) -> Dict:
        stem = self.stems[idx]

        qwen = torch.load(
            self.cache_dir / "qwen_hidden" / f"{stem}.pt", map_location="cpu"
        )
        seg = torch.load(
            self.cache_dir / "seg_latent" / f"{stem}.pt", map_location="cpu"
        )
        tgt = torch.load(
            self.cache_dir / "target_latent" / f"{stem}.pt", map_location="cpu"
        )

        vis_img = self._load_image_01(self._find_image(self.vis_root, stem))
        ir_img = self._load_image_01(self._find_image(self.ir_root, stem))

        return {
            "stem": stem,
            "qwen": qwen,
            "seg_latent_vis": seg["latent_vis"],      # (1,32,14,14) fp16
            "seg_latent_ir": seg["latent_ir"],        # (1,32,14,14) fp16
            "target_latent": tgt["latent"],           # (1,32,16,16) fp16
            "vis_img": vis_img,                       # (1,3,512,512) [0,1]
            "ir_img": ir_img,                         # (1,3,512,512) [0,1]
        }


def build_hidden_states_tuple(qwen_payload: Dict, device: str) -> Tuple:
    """
    从缓存重建 ThreeLevelFusionMCP 需要的 hidden_states tuple。

    缓存格式（precompute_fusion_features.py 阶段 B 产出）：
        layers[str(idx)] : (1, N_vis+N_ir, 2048) fp16
            idx ∈ [4,5,6,13,14,15,26,27,28]，仅含视觉 token
        text_last        : (1, N_txt, 2048) fp16，最后一层文本段
        vis_span/ir_span : 相对拼接视觉序列 (0..512) 的索引
        txt_span         : 相对 text_last 的索引 (0, N_txt)

    MCP 只按绝对索引取数（4..28 与 [-1]），因此其余位置填 None，
    短于全长的切片同样合法。tuple 长度 37 = embedding + 36 层。
    """
    layers: Dict[int, torch.Tensor] = qwen_payload["layers"]

    hidden: List[Optional[torch.Tensor]] = [None] * HIDDEN_TUPLE_LEN

    for idx_str, tensor in layers.items():
        hidden[int(idx_str)] = tensor.to(device=device, dtype=torch.float32)

    # hidden_states[-1] = 第 36 层，这里直接放文本段
    # （_extract_text_condition 只取 [:, txt_span] 做均值池化）
    hidden[QWEN_NUM_LAYERS_3B] = qwen_payload["text_last"].to(
        device=device, dtype=torch.float32
    )

    return tuple(hidden)


# ============================================================
# 可训练条件模块封装
# ============================================================

class TrainableFusionConditioner(nn.Module):
    """
    封装两个可训练组件，前向输出 (1, 648, 2304) 条件：

        ThreeLevelFusionMCP(hidden_states, spans) -> (1, 256, 2304)
        SegConditionViaDCAE 投影层(seg latent)     -> (1, 392, 2304)

    SegConditionViaDCAE 的 DC-AE 编码器在训练时不使用（分割图
    latent 已离线预计算），这里用一个哑模块占位以保持原始类
    的构造不变、权重键名与推理管线完全兼容。
    """

    def __init__(self, llm_dim: int, caption_dim: int):
        super().__init__()

        if str(MODEL_DIR) not in sys.path:
            sys.path.insert(0, str(MODEL_DIR))

        from mobileo.model.three_level_fusion_official import ThreeLevelFusionMCP
        from mobileo.model.seg_condition_via_dcae import SegConditionViaDCAE

        self.three_level_mcp = ThreeLevelFusionMCP(
            llm_dim=llm_dim,
            caption_dim=caption_dim,
            num_tokens_per_modality=TOKENS_PER_MODALITY,
            spatial_size=int(math.sqrt(TOKENS_PER_MODALITY)),
            num_heads=8,
        )

        # 哑 DC-AE 占位（真实编码器冻结且已离线使用，不参与训练前向）
        dummy_dcae = nn.Conv2d(3, SANA_LATENT_CH, kernel_size=1)
        self.seg_injector = SegConditionViaDCAE(
            dc_ae_encoder=dummy_dcae,
            latent_ch=SANA_LATENT_CH,
            caption_dim=caption_dim,
            spatial_size=14,
        )

    def forward(
        self,
        hidden_states: Tuple,
        vis_span: Tuple[int, int],
        ir_span: Tuple[int, int],
        txt_span: Tuple[int, int],
        seg_latent_vis: torch.Tensor,
        seg_latent_ir: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:

        fusion_cond, weight_info = self.three_level_mcp(
            hidden_states=hidden_states,
            vis_span=vis_span,
            ir_span=ir_span,
            txt_span=txt_span,
        )  # (1, 256, 2304)

        # 分割条件：预计算 latent -> 投影（与 SegConditionViaDCAE.forward
        # 的 Step2-4 完全一致，仅跳过离线已完成的 DC-AE 编码）
        injector = self.seg_injector
        tokens_vis = seg_latent_vis.flatten(2).transpose(1, 2)
        tokens_ir = seg_latent_ir.flatten(2).transpose(1, 2)
        cond_vis = injector.latent_to_cond(tokens_vis) + injector.type_embed_vis
        cond_ir = injector.latent_to_cond(tokens_ir) + injector.type_embed_ir

        combined = torch.cat([fusion_cond, cond_vis, cond_ir], dim=1)
        # (1, 256 + 196 + 196, 2304) = (1, 648, 2304)

        return combined, weight_info

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


# ============================================================
# 采样可视化
# ============================================================

@torch.no_grad()
def sample_fused_image(
    dit: nn.Module,
    vae: nn.Module,
    scheduler,
    condition: torch.Tensor,
    num_steps: int,
    device: str,
) -> torch.Tensor:
    """用当前条件跑一次完整采样，返回 [0,1] 图像 (1,3,512,512)"""
    latent_ch = int(dit.config.in_channels)
    latent_size = int(dit.config.sample_size)

    latents = torch.randn(
        1, latent_ch, latent_size, latent_size, device=device, dtype=torch.float16
    )

    scheduler.set_timesteps(num_steps, device=device)
    cond_mask = torch.ones(
        condition.shape[0], condition.shape[1], dtype=torch.bool, device=device
    )

    for t in scheduler.timesteps:
        timestep = t.unsqueeze(0).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            pred = dit(
                hidden_states=latents,
                timestep=timestep,
                encoder_hidden_states=condition.to(torch.float16),
                encoder_attention_mask=cond_mask,
            ).sample
        latents = scheduler.step(pred.float(), t, latents.float()).prev_sample.to(
            torch.float16
        )

    latents = (latents.float() / float(getattr(vae.config, "scaling_factor", 1.0)))
    decoded = vae.decode(latents).sample
    return ((decoded + 1.0) / 2.0).clamp(0, 1)


def save_comparison_panel(
    path: Path, vis: torch.Tensor, ir: torch.Tensor, fused: torch.Tensor
):
    """横向拼接 可见光 | 红外 | 融合 保存"""
    def to_pil(t):
        arr = (t[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    panel = Image.new("RGB", (vis.shape[3] * 3 + 8, vis.shape[2]), (0, 0, 0))
    w = vis.shape[3]
    panel.paste(to_pil(vis), (0, 0))
    panel.paste(to_pil(ir), (w + 4, 0))
    panel.paste(to_pil(fused), (2 * w + 8, 0))
    panel.save(path)


# ============================================================
# 训练主流程
# ============================================================

def train(args):
    device = args.device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)

    # ---------------- 数据集 ----------------
    dataset = FusionCacheDataset(
        cache_dir=Path(args.cache_dir),
        vis_root=Path(args.vis_root),
        ir_root=Path(args.ir_root),
        image_size=512,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        # 样本内张量已带 batch 维，直接原样返回，避免再叠一层
        collate_fn=lambda batch: batch[0],
    )

    # ---------------- 条件模块（可训练）----------------
    conditioner = TrainableFusionConditioner(
        llm_dim=LLM_DIM_3B, caption_dim=CAPTION_DIM
    ).to(device)

    # 冻结哑 DC-AE（SegConditionViaDCAE 构造时已冻结，这里双保险）
    for p in conditioner.seg_injector.dc_ae_encoder.parameters():
        p.requires_grad = False

    def trainable_state_dict() -> Dict:
        """只保存可训练权重（跳过占位的哑 DC-AE，推理时会被真实模型覆盖）"""
        dummy_prefix = "seg_injector.dc_ae_encoder."
        return {
            k: v
            for k, v in conditioner.state_dict().items()
            if not k.startswith(dummy_prefix)
        }

    n_train = sum(p.numel() for p in conditioner.trainable_parameters())
    logger.info(f"可训练参数: {n_train:,} ({n_train/1e6:.2f}M)")

    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        conditioner.load_state_dict(state["conditioner"], strict=False)
        logger.info(f"已从 {args.resume} 恢复条件模块权重")

    # ---------------- SANA DiT（冻结 + 梯度检查点）----------------
    logger.info("加载 SANA DiT ...")
    dit = SanaTransformer2DModel.from_pretrained(
        args.sana_path, subfolder="transformer", variant="fp16",
        torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True,
    ).to(device)
    dit.requires_grad_(False)
    dit.eval()
    if args.gradient_checkpointing:
        dit.enable_gradient_checkpointing()
        logger.info("DiT 梯度检查点已启用")

    if int(dit.config.caption_channels) != CAPTION_DIM:
        raise RuntimeError("SANA caption_channels 必须为 2304")

    # ---------------- VAE（冻结，像素损失解码用）----------------
    logger.info("加载 DC-AE VAE ...")
    vae = AutoencoderDC.from_pretrained(
        args.sana_path, subfolder="vae", variant="fp16",
        torch_dtype=torch.float32, low_cpu_mem_usage=True, use_safetensors=True,
    ).to(device)
    vae.requires_grad_(False)
    vae.eval()
    scaling_factor = float(getattr(vae.config, "scaling_factor", 1.0))

    # ---------------- 调度器（与 Mobile-O 原生训练一致）----------------
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.sana_path, subfolder="scheduler"
    )
    num_train_timesteps = int(noise_scheduler.config.num_train_timesteps)
    noise_scheduler.set_timesteps(num_train_timesteps, device="cpu")

    sample_scheduler = DPMSolverMultistepScheduler.from_pretrained(
        args.sana_path, subfolder="scheduler"
    )

    # ---------------- 损失与优化器 ----------------
    from mobileo.model.fusion_losses import FusionPixelLoss

    pixel_loss_fn = FusionPixelLoss(
        w_ssim_vis=args.w_ssim_vis,
        w_ssim_ir=args.w_ssim_ir,
        w_grad=args.w_grad,
        w_int=args.w_int,
    ).to(device)

    # 可训练参数约 351.88M（ThreeLevelFusionMCP 含三个
    # TextConditionedAdaLN 调制网络）。标准 AdamW 的 m/v 状态
    # 需要约 2.8GB，6GB 显卡放不下，因此默认用 bitsandbytes
    # 的 8 位 Adam（m/v 量化，约 0.4GB），不可用时回退 AdamW。
    if args.optimizer == "adamw8bit":
        try:
            import bitsandbytes as bnb

            optimizer = bnb.optim.AdamW8bit(
                conditioner.trainable_parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
            logger.info("优化器: bitsandbytes AdamW8bit（8位状态）")
        except Exception as e:
            logger.warning(f"bitsandbytes 不可用（{e}），回退标准 AdamW")
            args.optimizer = "adamw"

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            conditioner.trainable_parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        logger.info("优化器: torch AdamW（注意：6GB 显存可能不足）")
    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])

    total_steps = args.max_steps
    warmup = args.warmup_steps

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---------------- 训练循环 ----------------
    conditioner.train()
    global_step = 0
    epoch = 0
    log_ema = {}

    logger.info(f"开始训练: {total_steps} 步, 梯度累积 {args.grad_accum}")

    while global_step < total_steps:
        epoch += 1
        for sample in loader:
            if global_step >= total_steps:
                break

            t0 = time.time()
            optimizer.zero_grad(set_to_none=True)

            stem = sample["stem"]
            qwen_payload = sample["qwen"]

            # ---- 1. 条件生成（可训练部分）----
            hidden_states = build_hidden_states_tuple(qwen_payload, device)
            vis_span = tuple(int(x) for x in qwen_payload["vis_span"])
            ir_span = tuple(int(x) for x in qwen_payload["ir_span"])
            txt_span = tuple(int(x) for x in qwen_payload["txt_span"])

            seg_vis = sample["seg_latent_vis"].to(device, dtype=torch.float32)
            seg_ir = sample["seg_latent_ir"].to(device, dtype=torch.float32)

            condition, weight_info = conditioner(
                hidden_states=hidden_states,
                vis_span=vis_span,
                ir_span=ir_span,
                txt_span=txt_span,
                seg_latent_vis=seg_vis,
                seg_latent_ir=seg_ir,
            )  # (1, 648, 2304)

            del hidden_states
            torch.cuda.empty_cache()

            # ---- 2. 可见光锚点 latent（flow 目标 z）----
            latents = sample["target_latent"].to(device, dtype=torch.float32)

            # ---- 3. flow matching 加噪（与 mobileo.py 一致）----
            u = compute_density_for_timestep_sampling(
                weighting_scheme="uniform",
                batch_size=latents.shape[0],
                logit_mean=0.0,
                logit_std=1.0,
                mode_scale=1.29,
            )
            indices = (u * num_train_timesteps).long().clamp(max=num_train_timesteps - 1)
            timesteps = noise_scheduler.timesteps[indices].to(device)

            sigmas_full = noise_scheduler.sigmas.to(device=device, dtype=torch.float32)
            schedule_timesteps = noise_scheduler.timesteps.to(device)
            sigma_list = []
            for t in timesteps:
                match = (schedule_timesteps == t).nonzero(as_tuple=False)
                sigma_list.append(sigmas_full[match[0].item()])
            sigmas = torch.stack(sigma_list).view(-1, 1, 1, 1)

            noise = torch.randn_like(latents)
            noisy_latents = (1.0 - sigmas) * latents + sigmas * noise

            # ---- 4. SANA 前向（冻结，梯度经条件回传）----
            cond_mask = torch.ones(
                condition.shape[0], condition.shape[1],
                dtype=torch.bool, device=device,
            )
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                diffusion_pred = dit(
                    hidden_states=noisy_latents.to(torch.float16),
                    timestep=timesteps,
                    encoder_hidden_states=condition.to(torch.float16),
                    encoder_attention_mask=cond_mask,
                ).sample.float()

            # ---- 5. flow 损失（与 mobileo.py 一致：全在 float32 下计算）----
            target = noise - latents
            weighting = compute_loss_weighting_for_sd3(
                weighting_scheme="uniform", sigmas=sigmas.squeeze()
            )
            diff_loss = torch.mean(
                (
                    weighting.float()
                    * (diffusion_pred.float() - target.float()) ** 2
                ).reshape(target.shape[0], -1),
                dim=1,
            ).mean()

            # ---- 6. 像素空间无监督融合损失 ----
            # 由速度预测恢复 x1: x1_hat = noisy - σ·v
            x1_hat = noisy_latents - sigmas * diffusion_pred
            # VAE 解码加梯度检查点，压缩 6GB 显存峰值
            decoded = torch.utils.checkpoint.checkpoint(
                lambda z: vae.decode(z).sample,
                x1_hat / scaling_factor,
                use_reentrant=False,
            )
            fused_img = ((decoded + 1.0) / 2.0).clamp(0, 1)  # [0,1]

            vis_img = sample["vis_img"].to(device)
            ir_img = sample["ir_img"].to(device)

            pixel_loss, pixel_parts = pixel_loss_fn(fused_img, vis_img, ir_img)

            # ---- 7. 总损失 ----
            total_loss = diff_loss + args.pixel_weight * pixel_loss
            (total_loss / args.grad_accum).backward()

            # ---- 8. 梯度累积步 ----
            if (global_step + 1) % args.grad_accum == 0 or (global_step + 1) == total_steps:
                torch.nn.utils.clip_grad_norm_(
                    conditioner.trainable_parameters(), args.max_grad_norm
                )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # ---- 9. 日志 ----
            parts = {
                "total": total_loss.item(),
                "flow": diff_loss.item(),
                "pixel": pixel_loss.item(),
                **{k: v.item() for k, v in pixel_parts.items()},
            }
            for k, v in parts.items():
                log_ema[k] = 0.9 * log_ema.get(k, v) + 0.1 * v

            if global_step % args.log_every == 0:
                msg = " | ".join(f"{k}={v:.4f}" for k, v in log_ema.items())
                lw = weight_info["level_weight"][0].tolist()
                logger.info(
                    f"[step {global_step}] {msg} | "
                    f"lvl_w=[{lw[0]:.2f},{lw[1]:.2f},{lw[2]:.2f}] | "
                    f"{time.time()-t0:.1f}s/it"
                )

            # ---- 10. 采样可视化 ----
            if args.sample_every > 0 and (
                global_step % args.sample_every == 0 or global_step == total_steps - 1
            ):
                conditioner.eval()
                try:
                    fused = sample_fused_image(
                        dit, vae, sample_scheduler,
                        condition.detach(), args.sample_steps, device,
                    )
                    save_comparison_panel(
                        out_dir / "samples" / f"step{global_step:07d}_{stem}.png",
                        vis_img, ir_img, fused,
                    )
                    logger.info(f"已保存采样对比图: step {global_step}")
                finally:
                    conditioner.train()

            # ---- 11. 检查点 ----
            if args.save_every > 0 and (global_step + 1) % args.save_every == 0:
                ckpt_path = out_dir / "checkpoints" / f"step{global_step+1:07d}.pt"
                torch.save(
                    {
                        "step": global_step + 1,
                        "conditioner": trainable_state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "args": vars(args),
                    },
                    ckpt_path,
                )
                logger.info(f"检查点已保存: {ckpt_path}")

            global_step += 1

    # ---------------- 收尾 ----------------
    final_path = out_dir / "checkpoints" / "final.pt"
    torch.save(
        {
            "step": global_step,
            "conditioner": trainable_state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        },
        final_path,
    )
    logger.info(f"训练完成，最终权重: {final_path}")
    logger.info(
        "推理时将 conditioner 权重载入 FullFusionPipelineWithSeg："
        "three_level_mcp.* 与 seg_injector.* 键名完全兼容。"
    )


# ============================================================
# 参数
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    # 数据
    p.add_argument("--cache_dir", type=str, default=r"D:\Mobile-O-main\fusion_cache")
    p.add_argument("--vis_root", type=str, required=True)
    p.add_argument("--ir_root", type=str, required=True)
    p.add_argument("--sana_path", type=str, default=DEFAULT_SANA_PATH)
    p.add_argument("--output_dir", type=str, default=r"D:\Mobile-O-main\fusion_runs\exp1")

    # 训练
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--optimizer", type=str, default="adamw8bit",
                   choices=["adamw8bit", "adamw"],
                   help="adamw8bit: bitsandbytes 8位状态，6GB 显存必选")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--resume", type=str, default=None)

    # 损失权重
    p.add_argument("--pixel_weight", type=float, default=1.0,
                   help="像素融合损失总权重 L = flow + w*pixel")
    p.add_argument("--w_ssim_vis", type=float, default=1.0)
    p.add_argument("--w_ssim_ir", type=float, default=1.0)
    p.add_argument("--w_grad", type=float, default=0.2)
    p.add_argument("--w_int", type=float, default=0.1)

    # 显存
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing",
                   action="store_false")

    # 日志 / 采样 / 保存
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--sample_every", type=int, default=200)
    p.add_argument("--sample_steps", type=int, default=20)
    p.add_argument("--save_every", type=int, default=500)

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
