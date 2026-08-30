"""
sam_config.py
SAM 模块的配置管理与初始化
"""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class SAMConfig:
    """SAM 分割模块配置"""

    # 模型路径
    model_path: str = r"E:\ai for science\omnifuse\models\AI-ModelScope\sam-vit-base"

    # 输入配置
    input_size: int = 448
    image_channels: int = 3

    # SAM 编码器配置 (ViT-Base)
    sam_embed_dim: int = 256  # SAM 输出通道数
    sam_patch_size: int = 16
    sam_vit_depth: int = 12
    sam_vit_heads: int = 12
    sam_vit_dim: int = 768

    # 分割条件投影配置
    latent_ch: int = 32  # DC-AE 潜空间通道数
    caption_dim: int = 2304  # SANA 条件维度
    spatial_size: int = 14  # 分割 token 空间尺寸 (14×14=196)

    # 功能开关
    use_sam: bool = True
    use_edge: bool = True
    use_auto_mask: bool = True
    use_dc_ae: bool = True
    freeze_sam: bool = True

    # 自动掩码配置
    points_per_side: int = 8
    pred_iou_thresh: float = 0.5
    stability_score_thresh: float = 0.8
    min_mask_area: int = 100
    max_masks: int = 16

    # 训练配置
    use_precomputed_seg: bool = False
    precomputed_seg_dir: str = "./precomputed_seg"

    # 辅助损失权重
    lambda_struct: float = 0.1
    lambda_edge: float = 0.05
    lambda_consist: float = 0.02

    # 梯度监控
    grad_monitor_interval: int = 100

    # 可视化
    vis_interval: int = 500
    vis_max_samples: int = 4
    vis_save_dir: str = "./vis/segmentation"

    def validate(self):
        """验证配置合法性"""
        assert self.input_size % self.sam_patch_size == 0, \
            f"input_size ({self.input_size}) 必须能被 patch_size ({self.sam_patch_size}) 整除"

        assert self.spatial_size ** 2 == (self.input_size // 32) ** 2 or \
               self.spatial_size == 14, \
            f"spatial_size ({self.spatial_size}) 与 input_size ({self.input_size}) 不匹配"

        model_dir = Path(self.model_path)
        assert model_dir.exists(), f"SAM 模型路径不存在: {self.model_path}"

        return True

    def get_spatial_tokens(self) -> int:
        """获取分割 token 数量"""
        return self.spatial_size ** 2  # 196

    def get_total_seg_tokens(self) -> int:
        """获取双模态总分割 token 数"""
        return self.spatial_size ** 2 * 2  # 392

    def get_combined_cond_length(self, fusion_tokens: int = 256) -> int:
        """获取拼接后的总条件长度"""
        return fusion_tokens + self.get_total_seg_tokens()  # 648


def create_sam_pipeline(config: SAMConfig):
    """
    根据配置创建 SAM 分割管线

    参数:
        config: SAM 配置
    输出:
        SAMSegmentationPipeline 实例
    """
    from sam_semantic_segmentor import SAMSegmentationPipeline

    config.validate()

    pipeline = SAMSegmentationPipeline(
        sam_model_path=config.model_path,
        input_size=config.input_size,
        caption_dim=config.caption_dim,
        latent_ch=config.latent_ch,
        freeze_sam=config.freeze_sam,
        use_edge=config.use_edge,
        use_auto_mask=config.use_auto_mask,
    )

    return pipeline


def create_full_fusion_model(config: SAMConfig, qwen_model=None, tokenizer=None):
    """
    创建完整的融合模型 (三级融合 + SAM)

    参数:
        config: SAM 配置
        qwen_model: Qwen2.5-VL-7B 模型
        tokenizer: Qwen tokenizer
    输出:
        FullFusionWithSAM 实例
    """
    from seg_condition_via_dcae import FullFusionWithSAM

    config.validate()

    model = FullFusionWithSAM(
        qwen_model=qwen_model,
        tokenizer=tokenizer,
        llm_dim=3584,
        sam_model_path=config.model_path,
        caption_dim=config.caption_dim,
        latent_ch=config.latent_ch,
        input_size=config.input_size,
        use_sam=config.use_sam,
        use_edge=config.use_edge,
        use_post_fusion_attn=True,
    )

    model.print_model_summary()

    return model