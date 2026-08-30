"""
sam_inference.py
SAM 分割推理脚本

支持:
1. 单张图像分割
2. 批量分割
3. 分割结果保存 (掩码/特征/可视化)
4. 与三级融合联合推理
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Dict, Tuple
import logging
import time

from sam_semantic_segmentor import SAMSegmentationPipeline, SAMImageEncoderWrapper
from sam_config import SAMConfig, create_sam_pipeline
from seg_condition_via_dcae import FullFusionWithSAM

logger = logging.getLogger(__name__)


class SAMInferenceEngine:
    """
    SAM 分割推理引擎

    支持单图/批量推理，输出分割条件
    """

    def __init__(
            self,
            config: SAMConfig,
            device: str = 'cuda',
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.config = config

        # 创建管线
        self.pipeline = create_sam_pipeline(config)
        self.pipeline.to(self.device)
        self.pipeline.eval()

        logger.info(f"SAM 推理引擎初始化完成 (设备: {self.device})")

    @torch.no_grad()
    def segment_single(
            self,
            vis_image: torch.Tensor,
            ir_image: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        单张图像分割

        输入:
            vis_image: (1, 3, 448, 448)
            ir_image: (1, 3, 448, 448)
        输出:
            完整分割结果
        """
        vis_image = vis_image.to(self.device)
        ir_image = ir_image.to(self.device)

        start_time = time.time()

        vis_seg_tokens, ir_seg_tokens, seg_info = self.pipeline(vis_image, ir_image)

        elapsed = time.time() - start_time
        logger.info(f"分割耗时: {elapsed * 1000:.2f} ms")

        return {
            'vis_seg_tokens': vis_seg_tokens,
            'ir_seg_tokens': ir_seg_tokens,
            'seg_info': seg_info,
        }

    @torch.no_grad()
    def segment_batch(
            self,
            vis_images: torch.Tensor,
            ir_images: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        批量分割

        输入:
            vis_images: (B, 3, 448, 448)
            ir_images: (B, 3, 448, 448)
        输出:
            批量分割结果
        """
        vis_images = vis_images.to(self.device)
        ir_images = ir_images.to(self.device)

        vis_seg_tokens, ir_seg_tokens, seg_info = self.pipeline(vis_images, ir_images)

        return {
            'vis_seg_tokens': vis_seg_tokens,
            'ir_seg_tokens': ir_seg_tokens,
            'seg_info': seg_info,
        }

    @torch.no_grad()
    def segment_and_save(
            self,
            vis_image: torch.Tensor,
            ir_image: torch.Tensor,
            save_dir: str,
            sample_name: str = "sample",
    ):
        """
        分割并保存结果

        输入:
            vis_image: (1, 3, 448, 448)
            ir_image: (1, 3, 448, 448)
            save_dir: 保存目录
            sample_name: 样本名称
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        result = self.segment_single(vis_image, ir_image)

        # 保存分割特征
        torch.save({
            'vis_seg_tokens': result['vis_seg_tokens'].cpu(),
            'ir_seg_tokens': result['ir_seg_tokens'].cpu(),
        }, save_path / f"{sample_name}_seg_tokens.pt")

        # 保存分割图
        if result['seg_info'].get('vis_seg_map') is not None:
            torch.save({
                'vis_seg_map': result['seg_info']['vis_seg_map'].cpu(),
                'ir_seg_map': result['seg_info']['ir_seg_map'].cpu(),
            }, save_path / f"{sample_name}_seg_maps.pt")

        logger.info(f"分割结果保存: {save_path / sample_name}")

    def benchmark(
            self,
            num_iterations: int = 100,
            batch_size: int = 4,
    ) -> Dict[str, float]:
        """
        性能基准测试

        输入:
            num_iterations: 迭代次数
            batch_size: batch 大小
        输出:
            {
                'avg_time_ms': 平均耗时(ms),
                'throughput_fps': 吞吐量(FPS),
            }
        """
        # 随机输入
        vis_images = torch.randn(batch_size, 3, 448, 448, device=self.device)
        ir_images = torch.randn(batch_size, 3, 448, 448, device=self.device)

        # 预热
        for _ in range(5):
            self.segment_batch(vis_images, ir_images)

        # 计时
        torch.cuda.synchronize()
        start = time.time()

        for _ in range(num_iterations):
            self.segment_batch(vis_images, ir_images)

        torch.cuda.synchronize()
        elapsed = time.time() - start

        avg_time_ms = elapsed / num_iterations * 1000
        throughput = batch_size * num_iterations / elapsed

        results = {
            'avg_time_ms': avg_time_ms,
            'throughput_fps': throughput,
            'batch_size': batch_size,
            'num_iterations': num_iterations,
        }

        logger.info(f"SAM 推理基准测试:")
        logger.info(f"  平均耗时: {avg_time_ms:.2f} ms/batch")
        logger.info(f"  吞吐量: {throughput:.1f} FPS")

        return results


def main():
    """主函数: SAM 分割推理示例"""
    logging.basicConfig(level=logging.INFO)

    # 配置
    config = SAMConfig(
        model_path=r"E:\ai for science\omnifuse\models\AI-ModelScope\sam-vit-base",
        input_size=448,
        caption_dim=2304,
        latent_ch=32,
        use_edge=True,
        use_auto_mask=True,
    )

    # 创建推理引擎
    engine = SAMInferenceEngine(config, device='cuda')

    # 创建测试输入
    vis_image = torch.randn(1, 3, 448, 448)
    ir_image = torch.randn(1, 3, 448, 448)

    # 执行分割
    result = engine.segment_single(vis_image, ir_image)

    print(f"可见光分割条件: {result['vis_seg_tokens'].shape}")  # (1, 196, 2304)
    print(f"红外分割条件: {result['ir_seg_tokens'].shape}")  # (1, 196, 2304)

    # 性能测试
    engine.benchmark(num_iterations=50, batch_size=4)


if __name__ == "__main__":
    main()