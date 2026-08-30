"""
sam_semantic_segmentor.py
SAM ViT-Base 语义分割模块

职责:
    输入可见光/红外图像 → SAM 自动分割 → 输出语义分割彩色图 (3通道)
    不做任何投影/编码，仅输出分割图像本身
    后续由 DC-AE 编码器负责编码到潜空间

模型路径: E:\ai for science\omnifuse\models\AI-ModelScope\sam-vit-base
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SAMSemanticSegmentor(nn.Module):
    """
    SAM 语义分割器

    输入: image (B, 3, H, W), 范围 [0, 1]
    输出: seg_map (B, 3, H, W), 范围 [0, 1], 语义分割彩色图

    流程:
        1. 预处理: resize → 1024×1024, ImageNet 归一化
        2. SAM 图像编码器 → image_embedding
        3. 均匀撒点 prompt → mask_decoder → 多掩码
        4. 掩码筛选 + 合并 → 语义分割图
        5. Resize 回原始尺寸
    """

    def __init__(
        self,
        model_path: str = r"E:\ai for science\omnifuse\models\AI-ModelScope\sam-vit-base",
        input_size: int = 448,
        points_per_side: int = 8,
        pred_iou_thresh: float = 0.5,
        stability_score_thresh: float = 0.8,
        min_mask_area: int = 100,
        max_masks: int = 16,
        freeze: bool = True,
    ):
        super().__init__()
        self.model_path = Path(model_path)
        self.input_size = input_size
        self.points_per_side = points_per_side
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.min_mask_area = min_mask_area
        self.max_masks = max_masks
        self.sam_input_size = 1024  # SAM 原生输入尺寸

        # 加载完整 SAM 模型
        self.sam_model = self._load_sam()

        # 冻结
        if freeze:
            for param in self.sam_model.parameters():
                param.requires_grad = False
            self.sam_model.eval()

        # 颜色调色板 (最多 max_masks 种颜色)
        self._build_color_palette(max_masks)

        logger.info(f"SAM Semantic Segmentor 初始化完成")
        logger.info(f"  模型路径: {model_path}")
        logger.info(f"  输入尺寸: {input_size}×{input_size}")
        logger.info(f"  每边点数: {points_per_side}")
        logger.info(f"  最大掩码数: {max_masks}")
        logger.info(f"  冻结: {freeze}")

    def _load_sam(self):
        """加载完整 SAM 模型 (encoder + prompt_encoder + mask_decoder)"""
        import sys
        sys.path.insert(0, str(self.model_path))

        # 方式 1: segment_anything 库
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
            candidates = list(self.model_path.glob("*.pth")) + list(self.model_path.glob("*.pt"))
            if not candidates:
                raise FileNotFoundError(f"未找到 SAM 权重: {self.model_path}")
            checkpoint = candidates[0]
            sam = sam_model_registry["vit_b"](checkpoint=str(checkpoint))
            logger.info(f"从 segment_anything 加载: {checkpoint.name}")
            return sam
        except ImportError:
            pass

        # 方式 2: transformers
        try:
            from transformers import SamModel
            sam = SamModel.from_pretrained(str(self.model_path))
            logger.info(f"从 transformers 加载: {self.model_path}")
            return sam
        except Exception:
            pass

        raise RuntimeError(
            f"无法加载 SAM 模型。请确保:\n"
            f"  1. 安装了 segment_anything 或 transformers\n"
            f"  2. 模型路径正确: {self.model_path}\n"
            f"  3. 权重文件存在 (*.pth / *.pt)"
        )

    def _build_color_palette(self, n_colors: int):
        """构建语义分割颜色调色板"""
        # 使用 HSV 色彩空间生成均匀分布的颜色
        colors = []
        for i in range(n_colors):
            hue = i / n_colors
            # HSV → RGB
            h = hue * 360
            s, v = 0.9, 0.95
            c = v * s
            x = c * (1 - abs((h / 60) % 2 - 1))
            m = v - c
            if h < 60:
                r, g, b = c, x, 0
            elif h < 120:
                r, g, b = x, c, 0
            elif h < 180:
                r, g, b = 0, c, x
            elif h < 240:
                r, g, b = 0, x, c
            elif h < 300:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x
            colors.append([r + m, g + m, b + m])
        self.color_palette = torch.tensor(colors, dtype=torch.float32)  # (N, 3)

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """
        SAM 预处理

        输入: images (B, 3, H, W), 范围 [0, 1] 或 [-1, 1]
        输出: (B, 3, 1024, 1024), ImageNet 归一化
        """
        # 确保 [0, 1]
        if images.min() < 0:
            images = (images + 1.0) / 2.0
        images = images.clamp(0, 1)

        # Resize 到 1024
        if images.shape[-1] != self.sam_input_size or images.shape[-2] != self.sam_input_size:
            images = F.interpolate(
                images, size=(self.sam_input_size, self.sam_input_size),
                mode='bilinear', align_corners=False,
            )

        # ImageNet 归一化
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        images = (images - mean) / std

        return images

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        语义分割前向传播

        输入: images (B, 3, H, W), 范围 [0, 1]
        输出: seg_maps (B, 3, H, W), 范围 [0, 1], 语义分割彩色图
        """
        B, _, H, W = images.shape
        device = images.device

        # 预处理
        processed = self.preprocess(images)  # (B, 3, 1024, 1024)

        # 编码
        image_embeddings = self.sam_model.image_encoder(processed)
        # (B, 256, 64, 64)

        # 获取 dense PE
        dense_pe = self.sam_model.prompt_encoder.get_dense_pe()

        # 生成点网格
        points = self._generate_point_grid(device)  # (N_points, 2)

        all_seg_maps = []

        for b in range(B):
            emb = image_embeddings[b:b+1]  # (1, 256, 64, 64)

            masks_list = []
            scores_list = []

            for point in points:
                # Prompt 编码
                point_coord = point.unsqueeze(0).unsqueeze(0)  # (1, 1, 2)
                point_label = torch.ones(1, 1, device=device, dtype=torch.float32)

                sparse_emb, dense_emb = self.sam_model.prompt_encoder(
                    points=(point_coord, point_label),
                    boxes=None,
                    masks=None,
                )

                # Mask 解码
                mask_pred, iou_pred = self.sam_model.mask_decoder(
                    image_embeddings=emb,
                    image_pe=dense_pe,
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=True,
                )
                # mask_pred: (1, 3, 256, 256)
                # iou_pred: (1, 3)

                # 选最佳掩码
                best_idx = iou_pred[0].argmax()
                best_mask = mask_pred[0, best_idx]  # (256, 256)
                best_score = iou_pred[0, best_idx].item()

                if best_score > self.pred_iou_thresh:
                    # Resize 到原始尺寸
                    mask_resized = F.interpolate(
                        best_mask.unsqueeze(0).unsqueeze(0),
                        size=(H, W),
                        mode='bilinear', align_corners=False,
                    ).squeeze()  # (H, W)

                    binary_mask = (mask_resized > 0.5).float()

                    if binary_mask.sum() > self.min_mask_area:
                        masks_list.append(binary_mask)
                        scores_list.append(best_score)

            # 合并为语义分割图
            seg_map = self._merge_masks_to_seg_map(masks_list, scores_list, H, W, device)
            all_seg_maps.append(seg_map)

        seg_maps = torch.stack(all_seg_maps, dim=0)  # (B, 3, H, W)
        return seg_maps

    def _generate_point_grid(self, device: torch.device) -> torch.Tensor:
        """生成均匀分布的点网格"""
        coords = torch.linspace(0, self.sam_input_size - 1, self.points_per_side, device=device)
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing='ij')
        points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
        return points  # (points_per_side^2, 2)

    def _merge_masks_to_seg_map(
        self,
        masks: List[torch.Tensor],
        scores: List[float],
        H: int, W: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        将多个二值掩码合并为一张语义分割彩色图

        策略:
            1. 按置信度降序排列
            2. 高分掩码优先着色 (覆盖低分)
            3. 未覆盖区域为黑色 (背景)

        输入:
            masks: list of (H, W) 二值掩码
            scores: list of float 置信度
            H, W: 图像尺寸
        输出:
            seg_map: (3, H, W), 范围 [0, 1]
        """
        seg_map = torch.zeros(3, H, W, device=device)

        if len(masks) == 0:
            return seg_map

        # 按分数降序
        sorted_indices = np.argsort(scores)[::-1]

        palette = self.color_palette.to(device)

        for rank, idx in enumerate(sorted_indices):
            mask = masks[idx]  # (H, W)
            color_idx = rank % len(palette)
            color = palette[color_idx]  # (3,)

            # 高分优先: 只在当前未被着色的区域着色
            uncovered = (seg_map.sum(dim=0) == 0)  # (H, W)
            region = (mask > 0.5) & uncovered

            for c in range(3):
                seg_map[c] = torch.where(region, color[c], seg_map[c])

        return seg_map

    def get_frozen_param_count(self) -> int:
        """获取冻结参数量"""
        return sum(p.numel() for p in self.sam_model.parameters() if not p.requires_grad)