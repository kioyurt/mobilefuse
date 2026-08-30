"""
train_magicfuse.py
MagicFuse 红外-可见光融合训练脚本

对齐 Mobile-O (https://github.com/Amshaker/Mobile-O) 的 SANA 训练协议:
  - Flow Matching (velocity target = noise - x0), 不是 DDPM epsilon
  - FlowMatchEulerDiscreteScheduler + DC-AE (AutoencoderDC)
  - 可见光锚点 latent 作为 source-coupled flow 目标 (FusionFM)
  - SFT 像素损失用 FusionPixelLoss (SSIM + 梯度 + 强度, 无需 GT 融合图)

对齐本仓库模块真实 API:
  Qwen25VLWrapper / ThreeLevelFusionMCP / SegConditionViaDCAE
  SAMSemanticSegmentor / FusionPixelLoss / MagicFuseSANAWrapper

管线:
  vis + ir + prompt
      │
      ├─ SAM(vis/ir) → DC-AE → SegProj → seg_tokens (392, 2304)
      │
      └─ Qwen2.5-VL-3B (frozen)
            hidden_states[6,7,8 / 22,23,24 / 34,35,36]
            → ThreeLevelFusionMCP → fusion_cond (256, 2304)
      │
      ConditionCombiner → combined_cond (648, 2304)
      │
      MagicFuse-SANA (IKR / CKG / MKF + frozen SANA DiT)
      → velocity → DC-AE decode → 融合图

Stage 1  SFT   : flow matching + 无监督像素损失
Stage 2  DPO   : 偏好对上的 flow-matching DPO (可选)

用法:
  python train.py --stage sft --data_root ./data/infrared_visible
  python train.py --stage dpo --resume_from ./outputs/magicfuse/ckpt_sft_final.pt
  python train.py --stage all
"""

from __future__ import annotations

import os
import sys
import math
import time
import json
import copy
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms
from PIL import Image

# torch 2.4+ moved autocast; keep both import paths
try:
    from torch.amp import GradScaler, autocast
except ImportError:  # pragma: no cover
    from torch.cuda.amp import GradScaler, autocast  # type: ignore

# ============================================================
# 导入已有模型文件 (mobileo.model.* 或同目录)
# ============================================================
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

# try:
from mobileo.model.qwen25vl_wrapper import Qwen25VLWrapper
from mobileo.model.three_level_fusion_official import ThreeLevelFusionMCP
from mobileo.model.seg_condition_via_dcae import SegConditionViaDCAE
from mobileo.model.sam_semantic_segmentor import SAMSemanticSegmentor
from mobileo.model.fusion_losses import FusionPixelLoss
from mobileo.model.magic import MagicFuseSANAWrapper
# except ImportError:

#     from qwen25vl_wrapper import Qwen25VLWrapper
#     from three_level_fusion_official import ThreeLevelFusionMCP
#     from seg_condition_via_dcae import SegConditionViaDCAE
#     from sam_semantic_segmentor import SAMSemanticSegmentor
#     from fusion_losses import FusionPixelLoss
#     from magic import MagicFuseSANAWrapper

try:
    from diffusers.training_utils import (
        compute_density_for_timestep_sampling,
        compute_loss_weighting_for_sd3,
    )
except ImportError:
    compute_density_for_timestep_sampling = None  # type: ignore
    compute_loss_weighting_for_sd3 = None  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# Flow-matching helpers (Mobile-O / SANA / SD3)
# ============================================================

def _density_for_timestep_sampling(
    weighting_scheme: str,
    batch_size: int,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    mode_scale: float = 1.29,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if compute_density_for_timestep_sampling is not None:
        return compute_density_for_timestep_sampling(
            weighting_scheme=weighting_scheme,
            batch_size=batch_size,
            logit_mean=logit_mean,
            logit_std=logit_std,
            mode_scale=mode_scale,
        )
    if weighting_scheme == "logit_normal":
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,))
        u = torch.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(batch_size)
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(batch_size)
    return u


def _loss_weighting_for_sd3(weighting_scheme: str, sigmas: torch.Tensor) -> torch.Tensor:
    if compute_loss_weighting_for_sd3 is not None:
        return compute_loss_weighting_for_sd3(
            weighting_scheme=weighting_scheme, sigmas=sigmas
        )
    if weighting_scheme == "sigma_sqrt":
        return (sigmas ** -2.0).float()
    if weighting_scheme == "cosmap":
        bot = 1 - 2 * sigmas + 2 * sigmas ** 2
        return 2 / (math.pi * bot)
    return torch.ones_like(sigmas)


def get_sigmas(scheduler, timesteps: torch.Tensor, n_dim: int = 4, dtype=torch.float32):
    """Map discrete scheduler timesteps → sigma, broadcast to latent ndim.

    Copied from Mobile-O `LlavaMetaForCausalLM.get_sigmas`.
    """
    device = timesteps.device
    sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = scheduler.timesteps.to(device=device)
    timesteps = timesteps.to(device)
    step_indices = []
    for t in timesteps:
        matches = (schedule_timesteps == t).nonzero()
        if matches.numel() == 0:
            idx = int((schedule_timesteps - t).abs().argmin().item())
        else:
            idx = int(matches[0].item())
        step_indices.append(idx)
    sigma = sigmas[step_indices].flatten()
    while sigma.ndim < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def denorm_to_unit(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] → [0, 1]"""
    return (x * 0.5 + 0.5).clamp(0.0, 1.0)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """(3, H, W) in [-1, 1] → PIL RGB"""
    x = denorm_to_unit(t.detach().float().cpu())
    arr = (x * 255.0).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


def resize_tokens(tokens: torch.Tensor, target_n: int) -> torch.Tensor:
    """(B, N, C) → (B, target_n, C) via bilinear on a square grid."""
    B, N, C = tokens.shape
    if N == target_n:
        return tokens
    src = int(round(N ** 0.5))
    tgt = int(round(target_n ** 0.5))
    if src * src != N or tgt * tgt != target_n:
        if N > target_n:
            return tokens[:, :target_n]
        pad = tokens[:, -1:, :].expand(B, target_n - N, C)
        return torch.cat([tokens, pad], dim=1)
    x = tokens.transpose(1, 2).reshape(B, C, src, src)
    x = F.interpolate(x, size=(tgt, tgt), mode="bilinear", align_corners=False)
    return x.flatten(2).transpose(1, 2)


# ============================================================
# 配置
# ============================================================
@dataclass
class TrainConfig:
    # --- 路径 ---
    data_root=r"E:\ai for science\SeAFusion-main\MSRS"
    pref_data_root: str = "./data/preferences"
    output_dir: str = "./outputs/magicfuse"
    qwen_model_path =r"E:\ai for science\omnifuse\models\Qwen\Qwen2___5-VL-3B-Instruct"
    sana_model_path =r"D:\Mobile-O-main\mobileo\model\Sana\Sana_600M_512px_diffusers"
    sam_model_path =r"E:\ai for science\omnifuse\models\AI-ModelScope\sam-vit-base"
    resume_from: str = "./output"
    stage: str = "all"

    # --- 模型维度 (方案文档) ---
    image_size: int = 128
    llm_dim: int = 2048
    caption_dim: int = 2304
    latent_ch: int = 32
    spatial_size: int = 14          # 448 / 32
    num_vis_tokens: int = 256       # Qwen visual tokens per image
    num_seg_tokens: int = 196       # 14 * 14
    total_cond_tokens: int = 648    # 256 + 196 + 196
    adapter_hidden: int = 512
    freeze_sana: bool = True

    # Qwen-3B 三级层 (hidden_states 含 embedding, 故 36 = 最后一层)
    shallow_layers: List[int] = field(default_factory=lambda: [6, 7, 8])
    mid_layers: List[int] = field(default_factory=lambda: [22, 23, 24])
    deep_layers: List[int] = field(default_factory=lambda: [34, 35, 36])

    # --- SFT ---
    sft_epochs: int = 2
    sft_batch_size: int = 1
    sft_grad_accum: int = 4
    sft_lr: float = 1e-4
    dit_lr: float = 1e-5
    sft_grad_clip: float = 1.0
    warmup_ratio: float = 0.05

    # --- DPO ---
    dpo_epochs: int = 4
    dpo_batch_size: int = 1
    dpo_grad_accum: int = 4
    dpo_lr: float = 5e-6
    dpo_grad_clip: float = 0.5
    dpo_beta: float = 0.1

    # --- 损失 ---
    lambda_flow: float = 1.0
    lambda_pixel: float = 0.2
    w_ssim_vis: float = 1.0
    w_ssim_ir: float = 1.0
    w_grad: float = 0.2
    w_int: float = 0.1
    weighting_scheme: str = "uniform"   # Mobile-O SFT default
    flow_anchor: str = "visible"        # visible | mix

    # --- 通用 ---
    weight_decay: float = 0.01
    num_workers: int = 4
    log_interval: int = 20
    save_interval: int = 1000
    device: str = "cuda"
    bf16: bool = True
    default_prompt: str = (
        "Fuse the visible and infrared images. Preserve visible color and "
        "texture, and inject infrared thermal targets and complementary structure."
    )


# ============================================================
# 数据集
# ============================================================
class InfraredVisibleDataset(Dataset):
    """(vis, ir, prompt) 配对. 图像以 [-1, 1] 返回, 与 DC-AE 一致."""

    def __init__(self, data_root: str, split: str = "train", image_size: int = 448):
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.samples = self._load(split)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def _load(self, split):

        samples = []

        vis_dir = Path(
            r"E:\ai for science\SeAFusion-main\MSRS\Visible"
        ) / split / "MSRS"

        ir_dir = Path(
            r"E:\ai for science\SeAFusion-main\MSRS\Infrared"
        ) / split / "MSRS"

        for f in sorted(vis_dir.glob("*")):

            if f.suffix.lower() not in [
                ".png", ".jpg", ".jpeg"
            ]:
                continue

            ir_file = ir_dir / f.name

            if ir_file.exists():
                samples.append(
                    {
                        "vis": str(f),
                        "ir": str(ir_file),
                        "text":
                            "Fuse visible and infrared images."
                    }
                )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        vis = self.transform(Image.open(s["vis"]).convert("RGB"))
        ir = self.transform(Image.open(s["ir"]).convert("RGB"))
        return {"vis": vis, "ir": ir, "text": s.get("text", "Fuse the infrared and visible images.")}


class PreferenceDataset(Dataset):
    """DPO 偏好对."""

    def __init__(self, data_root: str, image_size: int = 448):
        self.data_root = Path(data_root)
        ann = self.data_root / "preferences.json"
        with open(ann, "r", encoding="utf-8") as f:
            self.samples = json.load(f)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        return {
            "vis": self.transform(Image.open(s["vis"]).convert("RGB")),
            "ir": self.transform(Image.open(s["ir"]).convert("RGB")),
            "text": s.get("text", "Fuse the infrared and visible images."),
            "preferred": self.transform(Image.open(s["preferred"]).convert("RGB")),
            "rejected": self.transform(Image.open(s["rejected"]).convert("RGB")),
        }


def collate_sft(batch):
    return {
        "vis": torch.stack([b["vis"] for b in batch]),
        "ir": torch.stack([b["ir"] for b in batch]),
        "text": [b["text"] for b in batch],
    }


def collate_dpo(batch):
    return {
        "vis": torch.stack([b["vis"] for b in batch]),
        "ir": torch.stack([b["ir"] for b in batch]),
        "text": [b["text"] for b in batch],
        "preferred": torch.stack([b["preferred"] for b in batch]),
        "rejected": torch.stack([b["rejected"] for b in batch]),
    }


# ============================================================
# 训练器
# ============================================================
class MagicFuseTrainer:

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.global_step = 0
        self.dtype = torch.bfloat16 if cfg.bf16 else torch.float32

        self._build_models()
        self.pixel_loss = FusionPixelLoss(
            w_ssim_vis=cfg.w_ssim_vis,
            w_ssim_ir=cfg.w_ssim_ir,
            w_grad=cfg.w_grad,
            w_int=cfg.w_int,
        ).to(self.device)

        self.trainable_params = self._collect_trainable_params()
        n = sum(p.numel() for p in self.trainable_params)
        logger.info(f"trainable params: {n:,} ({n / 1e6:.2f}M)")

    # --------------------------------------------------------
    # build
    # --------------------------------------------------------
    def _build_models(self):
        cfg = self.cfg
        logger.info("=" * 64)
        logger.info("building MagicFuse")

        # 1. Qwen2.5-VL-3B frozen
        logger.info("  [1/5] Qwen2.5-VL-3B (frozen)")
        qwen_pixels = cfg.image_size * cfg.image_size  # 448*448 = 256 visual tokens
        self.qwen = Qwen25VLWrapper(
            model_path=cfg.qwen_model_path,
            selected_layers=cfg.shallow_layers + cfg.mid_layers + cfg.deep_layers,
            torch_dtype=self.dtype,
            device=str(self.device),
            freeze=True,
            min_pixels=qwen_pixels,
            max_pixels=qwen_pixels,
            strict_visual_token_check=False,
        )
        self.qwen.eval()

        # 2. SAM frozen
        logger.info("  [2/5] SAM (frozen)")
        sam_kwargs: Dict[str, Any] = dict(
            input_size=cfg.image_size,
            freeze=True,
        )
        # support both `model_path=` (this repo) and a raw checkpoint dir
        try:
            self.sam = SAMSemanticSegmentor(
                model_path=cfg.sam_model_path, **sam_kwargs
            )
        except TypeError:
            self.sam = SAMSemanticSegmentor(
                checkpoint_path=cfg.sam_model_path, **sam_kwargs
            )
        self.sam.to(self.device)
        self.sam.eval()
        for p in self.sam.parameters():
            p.requires_grad = False

        # 3. MagicFuse + SANA (loads DC-AE + DiT + scheduler)
        logger.info("  [3/5] MagicFuseSANAWrapper")
        self.magic = MagicFuseSANAWrapper(
            sana_model_path=cfg.sana_model_path,
            latent_ch=cfg.latent_ch,
            cond_dim=cfg.caption_dim,
            hidden_dim=cfg.adapter_hidden,
            freeze_sana=cfg.freeze_sana,
            torch_dtype=self.dtype,
            fusion_tokens=cfg.num_vis_tokens,
            seg_tokens=cfg.num_seg_tokens,
        ).to(self.device)

        # 4. Three-level fusion (Qwen-3B dims + layer indices)
        logger.info("  [4/5] ThreeLevelFusionMCP (trainable)")
        vis_grid = int(cfg.num_vis_tokens ** 0.5)  # 16
        self.fusion = ThreeLevelFusionMCP(
            llm_dim=cfg.llm_dim,
            caption_dim=cfg.caption_dim,
            num_tokens_per_modality=cfg.num_vis_tokens,
            spatial_size=vis_grid,
            num_heads=8,
            fasterkan_grids=5,
            shallow_layers=cfg.shallow_layers,
            mid_layers=cfg.mid_layers,
            deep_layers=cfg.deep_layers,
        ).to(self.device)
        # belt-and-suspenders if constructor ignored the new kwargs
        self.fusion.shallow_layer_indices = list(cfg.shallow_layers)
        self.fusion.mid_layer_indices = list(cfg.mid_layers)
        self.fusion.deep_layer_indices = list(cfg.deep_layers)

        # 5. Seg condition projector (DC-AE frozen, Linear trainable)
        logger.info("  [5/5] SegConditionViaDCAE (trainable proj)")
        self.seg_cond = SegConditionViaDCAE(
            dc_ae_encoder=self.magic.dcae_encoder,
            latent_ch=cfg.latent_ch,
            caption_dim=cfg.caption_dim,
            spatial_size=cfg.spatial_size,
        ).to(self.device)

        logger.info("=" * 64)

    def _collect_trainable_params(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        seen = set()

        def add(module: nn.Module):
            for p in module.parameters():
                if p.requires_grad and id(p) not in seen:
                    seen.add(id(p))
                    params.append(p)

        add(self.fusion)
        add(self.seg_cond)
        add(self.magic.branch_a)
        add(self.magic.branch_b)
        add(self.magic.branch_c)
        add(self.magic.time_embed)
        add(self.magic.missing_mod)
        if not self.cfg.freeze_sana:
            add(self.magic.dit)
        return params

    def _param_groups(self, lr: float) -> List[Dict[str, Any]]:
        """Mobile-O style: adapters/MCP at `lr`, DiT (if unfrozen) at dit_lr."""
        adapter = []
        dit = []
        for n, p in list(self.fusion.named_parameters()) + list(self.seg_cond.named_parameters()):
            if p.requires_grad:
                adapter.append(p)
        for name, module in [
            ("a", self.magic.branch_a),
            ("b", self.magic.branch_b),
            ("c", self.magic.branch_c),
            ("t", self.magic.time_embed),
            ("m", self.magic.missing_mod),
        ]:
            for p in module.parameters():
                if p.requires_grad:
                    adapter.append(p)
        if not self.cfg.freeze_sana:
            for p in self.magic.dit.parameters():
                if p.requires_grad:
                    dit.append(p)
        groups = [{"params": adapter, "lr": lr, "weight_decay": self.cfg.weight_decay}]
        if dit:
            groups.append({"params": dit, "lr": self.cfg.dit_lr, "weight_decay": self.cfg.weight_decay})
        return groups

    # --------------------------------------------------------
    # condition builder
    # --------------------------------------------------------
    @torch.no_grad()
    def _sam_segment(self, images: torch.Tensor) -> torch.Tensor:
        """images [-1,1] or [0,1] → seg_map (B, 3, H, W) in [0, 1]."""
        return self.sam(images)

    def _extract_qwen_batch(
        self,
        vis: torch.Tensor,
        ir: torch.Tensor,
        texts: List[str],
    ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """
        Qwen25VLWrapper is B=1 + PIL. Loop the batch, stack hidden states.

        Returns hidden_states tuple (each (B, L, C)) and shared vis/ir/txt spans.
        Visual tokens are resized to num_vis_tokens if the processor drifts.
        """
        cfg = self.cfg
        B = vis.shape[0]
        per_sample: List[Dict[str, Any]] = []

        for i in range(B):
            prompt = texts[i] if texts[i] else cfg.default_prompt
            out = self.qwen(
                visible=tensor_to_pil(vis[i]),
                infrared=tensor_to_pil(ir[i]),
                prompt=prompt,
                return_all_hidden_states=True,
            )
            per_sample.append(out)

        # spans from image_token_id runs
        spans0 = per_sample[0]["image_token_spans"][0]
        vis_span = tuple(spans0[0])
        ir_span = tuple(spans0[1])
        seq0 = per_sample[0]["last_hidden_state"].shape[1]
        txt_span = (int(ir_span[1]), int(seq0))
        if txt_span[1] <= txt_span[0]:
            txt_span = (seq0 - 1, seq0)

        n_layers = len(per_sample[0]["hidden_states"])
        stacked: List[torch.Tensor] = []
        target_vis = cfg.num_vis_tokens

        for layer_i in range(n_layers):
            vis_toks = []
            ir_toks = []
            txt_toks = []
            for s in per_sample:
                h = s["hidden_states"][layer_i]  # (1, L, C)
                sp = s["image_token_spans"][0]
                vs, ve = sp[0]
                irs, ire = sp[1]
                vis_toks.append(resize_tokens(h[:, vs:ve, :], target_vis))
                ir_toks.append(resize_tokens(h[:, irs:ire, :], target_vis))
                txt = h[:, ire:, :]
                if txt.shape[1] == 0:
                    txt = h[:, -1:, :]
                txt_toks.append(txt)

            max_txt = max(t.shape[1] for t in txt_toks)
            txt_pad = []
            for t in txt_toks:
                if t.shape[1] < max_txt:
                    t = F.pad(t, (0, 0, 0, max_txt - t.shape[1]))
                txt_pad.append(t)

            vis_b = torch.cat(vis_toks, dim=0)
            ir_b = torch.cat(ir_toks, dim=0)
            txt_b = torch.cat(txt_pad, dim=0)
            stacked.append(torch.cat([vis_b, ir_b, txt_b], dim=1))

        vis_span = (0, target_vis)
        ir_span = (target_vis, target_vis * 2)
        txt_span = (target_vis * 2, stacked[0].shape[1])
        return tuple(stacked), vis_span, ir_span, txt_span

    def build_condition(
        self,
        vis: torch.Tensor,
        ir: torch.Tensor,
        texts: List[str],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        SAM → DC-AE → SegProj
        Qwen → ThreeLevelFusionMCP
        → combined_cond (B, 648, 2304)
        """
        vis_01 = denorm_to_unit(vis)
        ir_01 = denorm_to_unit(ir)

        with torch.no_grad():
            seg_vis = self._sam_segment(vis_01)
            seg_ir = self._sam_segment(ir_01)

        hidden_states, vis_span, ir_span, txt_span = self._extract_qwen_batch(vis, ir, texts)

        fusion_cond, weight_info = self.fusion(
            hidden_states=hidden_states,
            vis_span=vis_span,
            ir_span=ir_span,
            txt_span=txt_span,
        )  # (B, 256, 2304)

        combined_cond, seg_info = self.seg_cond(
            seg_vis=seg_vis,
            seg_ir=seg_ir,
            fusion_cond=fusion_cond,
        )  # (B, 648, 2304)

        info = {
            "fusion_cond": fusion_cond,
            "weight_info": weight_info,
            "seg_info": seg_info,
            "seg_vis": seg_vis,
            "seg_ir": seg_ir,
        }
        return combined_cond, info

    # --------------------------------------------------------
    # flow matching
    # --------------------------------------------------------
    def _sample_flow(
        self,
        latents: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Mobile-O SFT:
            noisy = (1 - σ) * x0 + σ * ε
            target = ε - x0
        """
        noise = torch.randn_like(latents)
        u = _density_for_timestep_sampling(
            weighting_scheme=self.cfg.weighting_scheme,
            batch_size=latents.shape[0],
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.29,
        )
        scheduler = self.magic.noise_scheduler
        indices = (u * scheduler.config.num_train_timesteps).long()
        indices = indices.clamp(0, scheduler.config.num_train_timesteps - 1)
        timesteps = scheduler.timesteps[indices].to(device=latents.device)
        sigmas = get_sigmas(scheduler, timesteps, n_dim=latents.ndim, dtype=latents.dtype)
        noisy = (1.0 - sigmas) * latents + sigmas * noise
        target = noise - latents
        weighting = _loss_weighting_for_sd3(self.cfg.weighting_scheme, sigmas)
        return noisy, timesteps, target, weighting, sigmas

    def _flow_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weighting: torch.Tensor,
    ) -> torch.Tensor:
        loss = (weighting.float() * (pred.float() - target.float()) ** 2).reshape(
            target.shape[0], -1
        ).mean(dim=1)
        return loss.mean()

    def _anchor_latent(self, vis: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        if self.cfg.flow_anchor == "mix":
            img = 0.5 * vis + 0.5 * ir
        else:
            img = vis  # source-coupled visible anchor
        return self.magic.encode_image(img)

    def _velocity_to_x0(
        self,
        noisy: torch.Tensor,
        velocity: torch.Tensor,
        sigmas: torch.Tensor,
    ) -> torch.Tensor:
        # noisy = x0 + σ * v  ⇒  x0 = noisy - σ v
        return noisy - sigmas * velocity

    # --------------------------------------------------------
    # SFT step
    # --------------------------------------------------------
    def sft_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        vis = batch["vis"].to(self.device, non_blocking=True)
        ir = batch["ir"].to(self.device, non_blocking=True)
        texts = batch["text"]

        static_cond, _info = self.build_condition(vis, ir, texts)

        with torch.no_grad():
            target_latent = self._anchor_latent(vis, ir)

        noisy, timesteps, target, weighting, sigmas = self._sample_flow(
            target_latent.to(dtype=self.dtype)
        )

        pred = self.magic(
            noisy_latent=noisy.to(dtype=self.dtype),
            static_cond=static_cond.to(dtype=self.dtype),
            timesteps=timesteps,
        )

        l_flow = self._flow_loss(pred, target, weighting)

        losses: Dict[str, torch.Tensor] = {
            "L_flow": l_flow,
            "L_pixel": pred.new_zeros(()),
            "ssim_vis": pred.new_zeros(()),
            "ssim_ir": pred.new_zeros(()),
            "grad": pred.new_zeros(()),
            "intensity": pred.new_zeros(()),
        }

        if self.cfg.lambda_pixel > 0:
            x0_pred = self._velocity_to_x0(noisy, pred, sigmas)
            decoded = self.magic.decode_latent(x0_pred)
            fused_01 = denorm_to_unit(decoded)
            vis_01 = denorm_to_unit(vis)
            ir_01 = denorm_to_unit(ir)
            if fused_01.shape[-2:] != vis_01.shape[-2:]:
                vis_01 = F.interpolate(vis_01, size=fused_01.shape[-2:], mode="bilinear", align_corners=False)
                ir_01 = F.interpolate(ir_01, size=fused_01.shape[-2:], mode="bilinear", align_corners=False)
            l_pixel, parts = self.pixel_loss(fused_01, vis_01, ir_01)
            losses["L_pixel"] = l_pixel
            losses["ssim_vis"] = parts["ssim_vis"]
            losses["ssim_ir"] = parts["ssim_ir"]
            losses["grad"] = parts["grad"]
            losses["intensity"] = parts["intensity"]

        losses["total"] = (
            self.cfg.lambda_flow * losses["L_flow"]
            + self.cfg.lambda_pixel * losses["L_pixel"]
        )
        return losses

    # --------------------------------------------------------
    # DPO step (flow-matching logp proxy)
    # --------------------------------------------------------
    def _flow_logp(
        self,
        image: torch.Tensor,
        static_cond: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
        sigmas: torch.Tensor,
        magic_module: nn.Module,
    ) -> torch.Tensor:
        latent = self.magic.encode_image(image).to(dtype=self.dtype)
        noisy = (1.0 - sigmas) * latent + sigmas * noise
        target = noise - latent
        pred = magic_module(
            noisy_latent=noisy,
            static_cond=static_cond,
            timesteps=timesteps,
        )
        # per-sample mean MSE → logp proxy
        mse = (pred.float() - target.float()).pow(2).reshape(pred.shape[0], -1).mean(dim=1)
        return -0.5 * mse

    def dpo_step(self, batch: Dict[str, Any], ref_magic: nn.Module) -> Dict[str, torch.Tensor]:
        vis = batch["vis"].to(self.device, non_blocking=True)
        ir = batch["ir"].to(self.device, non_blocking=True)
        texts = batch["text"]
        preferred = batch["preferred"].to(self.device, non_blocking=True)
        rejected = batch["rejected"].to(self.device, non_blocking=True)
        B = vis.shape[0]

        static_cond, _ = self.build_condition(vis, ir, texts)

        with torch.no_grad():
            dummy = self.magic.encode_image(preferred)
        noise = torch.randn_like(dummy)
        u = _density_for_timestep_sampling(
            weighting_scheme=self.cfg.weighting_scheme, batch_size=B
        )
        scheduler = self.magic.noise_scheduler
        indices = (u * scheduler.config.num_train_timesteps).long().clamp(
            0, scheduler.config.num_train_timesteps - 1
        )
        timesteps = scheduler.timesteps[indices].to(self.device)
        sigmas = get_sigmas(scheduler, timesteps, n_dim=dummy.ndim, dtype=dummy.dtype)

        logp_w = self._flow_logp(preferred, static_cond, timesteps, noise, sigmas, self.magic)
        logp_l = self._flow_logp(rejected, static_cond, timesteps, noise, sigmas, self.magic)

        with torch.no_grad():
            ref_logp_w = self._flow_logp(preferred, static_cond, timesteps, noise, sigmas, ref_magic)
            ref_logp_l = self._flow_logp(rejected, static_cond, timesteps, noise, sigmas, ref_magic)

        advantage = (logp_w - ref_logp_w) - (logp_l - ref_logp_l)
        loss = -F.logsigmoid(self.cfg.dpo_beta * advantage).mean()
        return {"total": loss, "advantage": advantage.mean().detach()}

    # --------------------------------------------------------
    # loops
    # --------------------------------------------------------
    def _amp_ctx(self):
        if self.cfg.bf16:
            try:
                return autocast("cuda", dtype=torch.bfloat16)
            except TypeError:
                return autocast(dtype=torch.bfloat16)
        from contextlib import nullcontext
        return nullcontext()

    def _set_train_mode(self):
        self.fusion.train()
        self.seg_cond.train()
        self.magic.branch_a.train()
        self.magic.branch_b.train()
        self.magic.branch_c.train()
        self.magic.time_embed.train()
        self.magic.missing_mod.train()
        self.qwen.eval()
        self.sam.eval()
        self.magic.vae.eval()
        if self.cfg.freeze_sana:
            self.magic.dit.eval()
        else:
            self.magic.dit.train()

    def _get_cosine_with_warmup(self, optimizer, total_steps: int, warmup_ratio: float = 0.05):
        warmup_steps = int(total_steps * warmup_ratio)

        def lr_lambda(step: int):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return LambdaLR(optimizer, lr_lambda)

    def train_sft(self):
        cfg = self.cfg
        logger.info("=" * 64)
        logger.info(f"[Stage 1] SFT flow-matching | epochs={cfg.sft_epochs} lr={cfg.sft_lr}")
        logger.info("=" * 64)

        dataset = InfraredVisibleDataset(cfg.data_root, "train", cfg.image_size)
        if len(dataset) == 0:
            raise RuntimeError(
                f"empty dataset at {cfg.data_root}. "
                "Expect train.json or visible/train + infrared/train."
            )
        loader = DataLoader(
            dataset,
            batch_size=cfg.sft_batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=collate_sft,
            pin_memory=True,
            drop_last=True,
        )
        steps_per_epoch = max(len(loader) // max(cfg.sft_grad_accum, 1), 1)
        total_steps = cfg.sft_epochs * steps_per_epoch

        optimizer = AdamW(self._param_groups(cfg.sft_lr), betas=(0.9, 0.95), weight_decay=cfg.weight_decay)
        scheduler = self._get_cosine_with_warmup(optimizer, total_steps, cfg.warmup_ratio)
        self.global_step = 0
        optimizer.zero_grad(set_to_none=True)

        for epoch in range(cfg.sft_epochs):
            self._set_train_mode()
            epoch_loss = 0.0
            t0 = time.time()
            accum = 0

            for batch in loader:
                with self._amp_ctx():
                    losses = self.sft_step(batch)
                    loss = losses["total"] / max(cfg.sft_grad_accum, 1)
                loss.backward()
                accum += 1
                epoch_loss += losses["total"].item()

                if accum % cfg.sft_grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.trainable_params, cfg.sft_grad_clip)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1

                    if self.global_step % cfg.log_interval == 0:
                        lr_now = optimizer.param_groups[0]["lr"]
                        logger.info(
                            f"[SFT][Ep{epoch}][{self.global_step}/{total_steps}] "
                            f"Loss={losses['total']:.4f} "
                            f"flow={losses['L_flow']:.4f} "
                            f"pixel={losses['L_pixel']:.4f} "
                            f"(ssim_v={losses['ssim_vis']:.3f} ssim_ir={losses['ssim_ir']:.3f} "
                            f"grad={losses['grad']:.3f} int={losses['intensity']:.3f}) "
                            f"LR={lr_now:.2e}"
                        )

                    if self.global_step % cfg.save_interval == 0:
                        self._save(f"sft_step{self.global_step}")

            n_batches = max(len(loader), 1)
            logger.info(
                f"Epoch {epoch} done | AvgLoss={epoch_loss / n_batches:.4f} | "
                f"Time={time.time() - t0:.0f}s"
            )

        self._save("sft_final")
        logger.info("[Stage 1] SFT done")

    def train_dpo(self):
        cfg = self.cfg
        logger.info("=" * 64)
        logger.info(f"[Stage 2] DPO flow-matching | epochs={cfg.dpo_epochs} lr={cfg.dpo_lr}")
        logger.info("=" * 64)

        ref_magic = copy.deepcopy(self.magic)
        for p in ref_magic.parameters():
            p.requires_grad = False
        ref_magic.eval()

        dataset = PreferenceDataset(cfg.pref_data_root, cfg.image_size)
        loader = DataLoader(
            dataset,
            batch_size=cfg.dpo_batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=collate_dpo,
            pin_memory=True,
            drop_last=True,
        )
        steps_per_epoch = max(len(loader) // max(cfg.dpo_grad_accum, 1), 1)
        total_steps = cfg.dpo_epochs * steps_per_epoch

        optimizer = AdamW(self._param_groups(cfg.dpo_lr), betas=(0.9, 0.95), weight_decay=cfg.weight_decay)
        scheduler = self._get_cosine_with_warmup(optimizer, total_steps, warmup_ratio=0.0)
        self.global_step = 0
        optimizer.zero_grad(set_to_none=True)
        accum = 0

        for epoch in range(cfg.dpo_epochs):
            self._set_train_mode()
            epoch_loss = 0.0

            for batch in loader:
                with self._amp_ctx():
                    losses = self.dpo_step(batch, ref_magic)
                    loss = losses["total"] / max(cfg.dpo_grad_accum, 1)
                loss.backward()
                accum += 1
                epoch_loss += losses["total"].item()

                if accum % cfg.dpo_grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.trainable_params, cfg.dpo_grad_clip)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1

                    if self.global_step % cfg.log_interval == 0:
                        logger.info(
                            f"[DPO][Ep{epoch}][{self.global_step}/{total_steps}] "
                            f"Loss={losses['total']:.4f} Adv={losses['advantage']:.4f}"
                        )

            logger.info(f"DPO Epoch {epoch} | AvgLoss={epoch_loss / max(len(loader), 1):.4f}")

        self._save("dpo_final")
        logger.info("[Stage 2] DPO done")

    # --------------------------------------------------------
    # ckpt
    # --------------------------------------------------------
    def _save(self, tag: str):
        path = os.path.join(self.cfg.output_dir, f"ckpt_{tag}.pt")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "fusion": self.fusion.state_dict(),
                "seg_cond": self.seg_cond.state_dict(),
                "branch_a": self.magic.branch_a.state_dict(),
                "branch_b": self.magic.branch_b.state_dict(),
                "branch_c": self.magic.branch_c.state_dict(),
                "time_embed": self.magic.time_embed.state_dict(),
                "missing_mod": self.magic.missing_mod.state_dict(),
                "dit": None if self.cfg.freeze_sana else self.magic.dit.state_dict(),
                "global_step": self.global_step,
                "cfg": vars(self.cfg),
            },
            path,
        )
        logger.info(f"saved → {path}")

    def _load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.fusion.load_state_dict(ckpt["fusion"])
        self.seg_cond.load_state_dict(ckpt["seg_cond"])
        self.magic.branch_a.load_state_dict(ckpt["branch_a"])
        self.magic.branch_b.load_state_dict(ckpt["branch_b"])
        self.magic.branch_c.load_state_dict(ckpt["branch_c"])
        if "time_embed" in ckpt:
            self.magic.time_embed.load_state_dict(ckpt["time_embed"])
        if "missing_mod" in ckpt:
            self.magic.missing_mod.load_state_dict(ckpt["missing_mod"])
        if ckpt.get("dit") is not None and not self.cfg.freeze_sana:
            self.magic.dit.load_state_dict(ckpt["dit"])
        self.global_step = ckpt.get("global_step", 0)
        logger.info(f"loaded ← {path} (step={self.global_step})")


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="MagicFuse SFT / DPO trainer (Mobile-O flow matching)")
    parser.add_argument("--stage", type=str, default="all", choices=["sft", "dpo", "all"])
    parser.add_argument("--data_root", type=str, default="./data/infrared_visible")
    parser.add_argument("--pref_data_root", type=str, default="./data/preferences")
    parser.add_argument("--output_dir", type=str, default="./outputs/magicfuse")
    parser.add_argument("--qwen_model_path", type=str, default="./weights/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--sana_model_path", type=str, default="./weights/Sana_600M_512px_diffusers")
    parser.add_argument("--sam_model_path", type=str, default="./weights/sam-vit-base")
    parser.add_argument("--resume_from", type=str, default="")
    parser.add_argument("--sft_epochs", type=int, default=40)
    parser.add_argument("--sft_batch_size", type=int, default=2)
    parser.add_argument("--sft_grad_accum", type=int, default=4)
    parser.add_argument("--dpo_epochs", type=int, default=4)
    parser.add_argument("--dpo_batch_size", type=int, default=1)
    parser.add_argument("--flow_anchor", type=str, default="visible", choices=["visible", "mix"])
    parser.add_argument("--freeze_sana", action="store_true", default=True)
    parser.add_argument("--unfreeze_sana", action="store_true", help="also train SANA DiT (Mobile-O style, lr=dit_lr)")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg = TrainConfig()
    for k, v in vars(args).items():
        if k == "unfreeze_sana":
            continue
        if hasattr(cfg, k) and v is not None:
            setattr(cfg, k, v)
    if args.unfreeze_sana:
        cfg.freeze_sana = False

    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(cfg), f, indent=2, default=str)

    trainer = MagicFuseTrainer(cfg)
    if cfg.resume_from:
        trainer._load(cfg.resume_from)

    if cfg.stage in ("sft", "all"):
        trainer.train_sft()
    if cfg.stage in ("dpo", "all"):
        trainer.train_dpo()

    logger.info("training finished")


if __name__ == "__main__":
    main()
