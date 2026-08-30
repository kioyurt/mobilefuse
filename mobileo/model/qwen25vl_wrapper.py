# -*- coding: utf-8 -*-
"""
Qwen2.5-VL-3B wrapper for Mobile-O style conditioning.

用途：
    Visible Image + Infrared Image
                ↓
        Qwen2.5-VL-3B
                ↓
        Multi-layer hidden states
                ↓
    Accurate visual-token extraction
        ┌──────────────┴──────────────┐
        ↓                             ↓
 Visible visual tokens       Infrared visual tokens
        └──────────────┬──────────────┘
                       ↓
             MobileConditioningProjector

第一阶段：
    - Qwen 完全冻结
    - 不调用 generate()
    - 只提取 hidden_states
    - 精确分离 Visible / Infrared visual tokens
    - 保留原始全部 hidden states
    - 保留 selected_hidden_states
    - 保留 attention_mask
    - 额外返回 image token mask / span / grid information

重要：
    Qwen2.5-VL 的 sequence 中：
        image_token_id
    是图像视觉 token 的占位位置。

    每张图片对应一段连续的 image_token_id。
    我们依据：
        1. input_ids
        2. image_token_id
        3. image_grid_thw
    三者联合确定每张图片的 visual-token 范围。

典型输出：
    last_hidden_state:
        [B, N, 2048]

    selected_hidden_states[i]:
        [B, N, 2048]

    visible_hidden_states[i]:
        [B, Nv, 2048]

    infrared_hidden_states[i]:
        [B, Nir, 2048]

    visual_hidden_states[i]:
        [B, Nv + Nir, 2048]
"""

from __future__ import annotations

from pathlib import Path
from typing import (
    List,
    Sequence,
    Union,
    Optional,
    Dict,
    Any,
    Tuple,
)

import torch
import torch.nn as nn
from PIL import Image

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)


ImageInput = Union[str, Path, Image.Image]


class Qwen25VLWrapper(nn.Module):
    """
    Qwen2.5-VL-3B-Instruct wrapper.

    输入：
        visible   : 可见光图像
        infrared  : 红外图像
        prompt    : 文本指令

    输出：
        hidden_states:
            Qwen 所有 layer 的 hidden states。

        selected_hidden_states:
            选中的多层 hidden states，
            默认最后 8 层。

        visible_hidden_states:
            从每层 hidden state 中精确提取出的
            Visible visual tokens。

        infrared_hidden_states:
            从每层 hidden state 中精确提取出的
            Infrared visual tokens。

        visual_hidden_states:
            每层将 Visible + Infrared visual tokens
            按原始 sequence 顺序拼接后的结果。

    典型 shape：

        selected_hidden_states[i]:
            [B, sequence_length, 2048]

        visible_hidden_states[i]:
            [B, visible_tokens, 2048]

        infrared_hidden_states[i]:
            [B, infrared_tokens, 2048]
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        selected_layers: Optional[Sequence[int]] = None,
        last_n_layers: int = 8,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        freeze: bool = True,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1280 * 28 * 28,
        strict_visual_token_check: bool = True,
    ) -> None:
        super().__init__()

        self.model_path = str(model_path)
        self.device_name = device
        self.freeze = freeze
        self.strict_visual_token_check = strict_visual_token_check

        # ---------------------------------------------------------
        # 1. Load processor
        # ---------------------------------------------------------
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            trust_remote_code=True,
        )

        # ---------------------------------------------------------
        # 2. Load Qwen2.5-VL
        # ---------------------------------------------------------
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )



        # ---------------------------------------------------------
        # 3. Freeze Qwen
        # ---------------------------------------------------------
        if self.freeze:
            self.model.eval()

            for param in self.model.parameters():
                param.requires_grad = False

        # ---------------------------------------------------------
        # 4. Qwen configuration
        # ---------------------------------------------------------
        self.hidden_size = int(
            getattr(self.model.config, "hidden_size", 2048)
        )

        self.num_hidden_layers = int(
            getattr(self.model.config, "num_hidden_layers", 36)
        )

        # Qwen2.5-VL configuration中的特殊视觉 token ID
        self.image_token_id = int(
            getattr(
                self.model.config,
                "image_token_id",
                151655,
            )
        )

        self.video_token_id = int(
            getattr(
                self.model.config,
                "video_token_id",
                151656,
            )
        )

        self.vision_start_token_id = int(
            getattr(
                self.model.config,
                "vision_start_token_id",
                151652,
            )
        )

        self.vision_end_token_id = int(
            getattr(
                self.model.config,
                "vision_end_token_id",
                151653,
            )
        )

        # ---------------------------------------------------------
        # 5. Vision configuration
        # ---------------------------------------------------------
        vision_config = getattr(
            self.model.config,
            "vision_config",
            None,
        )

        if vision_config is not None:
            self.spatial_merge_size = int(
                getattr(
                    vision_config,
                    "spatial_merge_size",
                    2,
                )
            )

        else:
            self.spatial_merge_size = 2

        # ---------------------------------------------------------
        # 6. Layer selection
        # ---------------------------------------------------------
        if selected_layers is not None:

            self.selected_layers = list(selected_layers)

            if len(self.selected_layers) == 0:
                raise ValueError(
                    "selected_layers cannot be empty."
                )

            for layer_idx in self.selected_layers:

                if not 0 <= layer_idx <= self.num_hidden_layers:
                    raise ValueError(
                        f"Invalid layer index {layer_idx}. "
                        f"Valid range: "
                        f"[0, {self.num_hidden_layers}]"
                    )

        else:

            if last_n_layers <= 0:
                raise ValueError(
                    "last_n_layers must be > 0."
                )

            if last_n_layers > self.num_hidden_layers:
                raise ValueError(
                    f"last_n_layers={last_n_layers} is larger "
                    f"than num_hidden_layers="
                    f"{self.num_hidden_layers}."
                )

            # hidden_states:
            #
            # [0]  = embedding output
            # [1]  = layer 1
            # ...
            # [36] = layer 36
            #
            # last 8 layers:
            #
            # 29, 30, 31, 32, 33, 34, 35, 36

            self.selected_layers = list(
                range(
                    self.num_hidden_layers
                    - last_n_layers
                    + 1,
                    self.num_hidden_layers + 1,
                )
            )

        print("=" * 70)
        print("Qwen2.5-VL Wrapper")
        print(f"Model path             : {self.model_path}")
        print(f"Hidden size            : {self.hidden_size}")
        print(f"Num layers             : {self.num_hidden_layers}")
        print(
            f"Selected layers        : "
            f"{self.selected_layers}"
        )
        print(f"Image token ID         : {self.image_token_id}")
        print(
            f"Vision start token ID  : "
            f"{self.vision_start_token_id}"
        )
        print(
            f"Vision end token ID    : "
            f"{self.vision_end_token_id}"
        )
        print(f"Spatial merge size     : {self.spatial_merge_size}")
        print(f"Device                 : {self.device_name}")
        print(f"Freeze                 : {self.freeze}")
        print(
            "Strict token check     : "
            f"{self.strict_visual_token_check}"
        )
        print("=" * 70)

    # ==================================================================
    # Utility: load image
    # ==================================================================

    @staticmethod
    def _load_image(
        image: ImageInput,
    ) -> Image.Image:
        """
        将路径/PIL Image统一转换成 RGB PIL Image。
        """

        if isinstance(image, Image.Image):

            return image.convert("RGB")

        image = Path(image)

        if not image.exists():

            raise FileNotFoundError(
                f"Image does not exist: {image}"
            )

        with Image.open(image) as img:

            return img.convert("RGB")

    # ==================================================================
    # Build Qwen messages
    # ==================================================================

    @staticmethod
    def _build_messages(
        visible: ImageInput,
        infrared: ImageInput,
        prompt: str,
    ) -> List[Dict[str, Any]]:
        """
        构造 Qwen2.5-VL 多图输入。

        顺序固定：

            image 1 = visible
            image 2 = infrared
            text     = prompt

        这个顺序非常重要，因为后续：
            image_grid_thw[0]
        对应 Visible，

            image_grid_thw[1]
        对应 Infrared。
        """

        visible_img = Qwen25VLWrapper._load_image(
            visible
        )

        infrared_img = Qwen25VLWrapper._load_image(
            infrared
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": visible_img,
                    },
                    {
                        "type": "image",
                        "image": infrared_img,
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        return messages

    # ==================================================================
    # Forward
    # ==================================================================

    @torch.no_grad()
    def forward(
        self,
        visible: ImageInput,
        infrared: ImageInput,
        prompt: str = (
            "Analyze the visible and infrared images. "
            "Identify salient targets, structural information, "
            "thermal information, boundaries, textures, and "
            "complementary information between the two modalities."
        ),
        return_all_hidden_states: bool = False,
    ) -> Dict[str, Any]:
        """
        执行 Qwen2.5-VL forward。

        除了原有 hidden_states 外，
        额外进行 Visible / Infrared visual token 精确分离。

        Returns:
            dict
                last_hidden_state:
                    [B, N, hidden_size]

                selected_hidden_states:
                    List[[B, N, hidden_size]]

                visible_hidden_states:
                    List[[B, Nv, hidden_size]]

                infrared_hidden_states:
                    List[[B, Nir, hidden_size]]

                visual_hidden_states:
                    List[[B, Nv + Nir, hidden_size]]

                visible_token_mask:
                    [B, N]

                infrared_token_mask:
                    [B, N]

                visual_token_mask:
                    [B, N]

                attention_mask:
                    [B, N]

                image_token_spans:
                    每个 batch/sample 中两张图像的
                    image-token 起止位置。

                image_token_counts:
                    每张图实际 image token 数量。

                image_grid_thw:
                    Qwen processor 生成的图像网格信息。

                hidden_states:
                    可选，全部 hidden states。
        """

        # ---------------------------------------------------------
        # 0. Build messages
        # ---------------------------------------------------------

        messages = self._build_messages(
            visible=visible,
            infrared=infrared,
            prompt=prompt,
        )

        # ---------------------------------------------------------
        # 1. Apply chat template
        # ---------------------------------------------------------

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # ---------------------------------------------------------
        # 2. Extract images
        # ---------------------------------------------------------

        image_inputs = self._extract_image_inputs(
            messages
        )

        if len(image_inputs) != 2:

            raise RuntimeError(
                "Qwen25VLWrapper requires exactly two images: "
                "Visible + Infrared. "
                f"Got {len(image_inputs)} images."
            )

        # ---------------------------------------------------------
        # 3. Processor
        #
        # 这里同时保留 input_ids / attention_mask /
        # image_grid_thw 等原始信息。
        # ---------------------------------------------------------

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )

        # ---------------------------------------------------------
        # 4. 保存 CPU / GPU 前的原始 metadata
        # ---------------------------------------------------------

        if "input_ids" not in inputs:

            raise RuntimeError(
                "Processor output does not contain input_ids."
            )

        input_ids = inputs["input_ids"]

        # ---------------------------------------------------------
        # image_grid_thw
        #
        # shape:
        #     [num_images, 3]
        #
        # 对本任务应该：
        #     [2, 3]
        # ---------------------------------------------------------

        image_grid_thw = inputs.get(
            "image_grid_thw",
            None,
        )

        if image_grid_thw is None:

            raise RuntimeError(
                "Processor output does not contain "
                "'image_grid_thw'. "
                "Qwen2.5-VL visual-token separation "
                "requires image_grid_thw."
            )

        if image_grid_thw.shape[0] != 2:

            raise RuntimeError(
                "Expected exactly two image grids "
                "(Visible + Infrared), but got "
                f"{image_grid_thw.shape[0]}."
            )

        # ---------------------------------------------------------
        # 5. Move inputs to GPU
        # ---------------------------------------------------------

        inputs = {
            key: value.to(self.device_name)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in inputs.items()
        }

        # GPU version
        input_ids = inputs["input_ids"]

        image_grid_thw = inputs[
            "image_grid_thw"
        ]

        attention_mask = inputs.get(
            "attention_mask",
            None,
        )

        # ---------------------------------------------------------
        # 6. Analyze image-token positions BEFORE forward
        #
        # 这是精确分离的核心。
        # ---------------------------------------------------------

        (
            image_token_spans,
            image_token_counts,
            visual_token_mask,
            visible_token_mask,
            infrared_token_mask,
        ) = self._build_visual_token_masks(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )

        # ---------------------------------------------------------
        # 7. Forward
        #
        # 非常关键：
        #
        # output_hidden_states=True
        # ---------------------------------------------------------

        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        # ---------------------------------------------------------
        # 8. Read hidden states
        #
        # hidden_states:
        #
        # tuple length = num_hidden_layers + 1
        #
        # [0]  = embedding output
        # [1]  = layer 1
        # ...
        # [36] = layer 36
        # ---------------------------------------------------------

        hidden_states = outputs.hidden_states

        if hidden_states is None:

            raise RuntimeError(
                "Qwen2.5-VL did not return hidden_states. "
                "Please make sure output_hidden_states=True."
            )

        # ---------------------------------------------------------
        # 9. Select specific layers
        # ---------------------------------------------------------

        selected_hidden_states = [
            hidden_states[layer_idx]
            for layer_idx in self.selected_layers
        ]

        # ---------------------------------------------------------
        # 10. Last hidden state
        # ---------------------------------------------------------

        last_hidden_state = hidden_states[-1]

        # ---------------------------------------------------------
        # 11. Extract visual hidden states layer by layer
        # ---------------------------------------------------------

        (
            visible_hidden_states,
            infrared_hidden_states,
            visual_hidden_states,
        ) = self._extract_visual_hidden_states(
            selected_hidden_states=selected_hidden_states,
            visible_token_mask=visible_token_mask,
            infrared_token_mask=infrared_token_mask,
            visual_token_mask=visual_token_mask,
        )

        # ---------------------------------------------------------
        # 12. Build result
        # ---------------------------------------------------------

        result: Dict[str, Any] = {
            # ---------------------------
            # 原始输出
            # ---------------------------
            "last_hidden_state": last_hidden_state,

            "selected_hidden_states": (
                selected_hidden_states
            ),

            "attention_mask": attention_mask,

            # ---------------------------
            # 精确视觉 token
            # ---------------------------
            "visible_hidden_states": (
                visible_hidden_states
            ),

            "infrared_hidden_states": (
                infrared_hidden_states
            ),

            "visual_hidden_states": (
                visual_hidden_states
            ),

            # ---------------------------
            # token masks
            # ---------------------------
            "visible_token_mask": (
                visible_token_mask
            ),

            "infrared_token_mask": (
                infrared_token_mask
            ),

            "visual_token_mask": (
                visual_token_mask
            ),

            # ---------------------------
            # image-token metadata
            # ---------------------------
            "image_token_spans": (
                image_token_spans
            ),

            "image_token_counts": (
                image_token_counts
            ),

            "image_grid_thw": (
                image_grid_thw
            ),

            # ---------------------------
            # special token IDs
            # ---------------------------
            "image_token_id": (
                self.image_token_id
            ),

            "vision_start_token_id": (
                self.vision_start_token_id
            ),

            "vision_end_token_id": (
                self.vision_end_token_id
            ),
        }

        # ---------------------------------------------------------
        # 13. Optionally return every hidden layer
        # ---------------------------------------------------------

        if return_all_hidden_states:

            result["hidden_states"] = hidden_states

        return result

    # ==================================================================
    # Visual token mask construction
    # ==================================================================

    def _build_visual_token_masks(
        self,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[
        List[List[Tuple[int, int]]],
        List[List[int]],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        根据 input_ids 中真实的 image_token_id
        精确确定两张图像在 sequence 中的位置。

        返回：

            image_token_spans:
                [
                    [
                        (start_v, end_v),
                        (start_ir, end_ir)
                    ]
                ]

            image_token_counts:
                [
                    [
                        N_visible,
                        N_infrared
                    ]
                ]

            visual_token_mask:
                [B, N]

            visible_token_mask:
                [B, N]

            infrared_token_mask:
                [B, N]

        注意：
            end 是 exclusive。

            例如：
                (10, 266)

            表示：
                token positions 10 ... 265
        """

        if input_ids.ndim != 2:

            raise ValueError(
                "input_ids must have shape [B, N], "
                f"but got {tuple(input_ids.shape)}"
            )

        batch_size, sequence_length = (
            input_ids.shape
        )

        if image_grid_thw.ndim != 2:

            raise ValueError(
                "image_grid_thw must have shape "
                "[num_images, 3], "
                f"but got "
                f"{tuple(image_grid_thw.shape)}"
            )

        if image_grid_thw.shape[0] != 2:

            raise ValueError(
                "This wrapper expects exactly two images."
            )

        # ---------------------------------------------------------
        # Expected image token count for each image
        #
        # Qwen2.5-VL:
        #
        # N =
        # T * H * W / spatial_merge_size^2
        #
        # The current Qwen2.5-VL implementation uses
        # spatial_merge_size when constructing image
        # representations.
        # ---------------------------------------------------------

        expected_counts = []

        for image_idx in range(
            image_grid_thw.shape[0]
        ):

            grid = image_grid_thw[
                image_idx
            ]

            t = int(grid[0].item())
            h = int(grid[1].item())
            w = int(grid[2].item())

            expected_count = (
                t
                * h
                * w
                // (
                    self.spatial_merge_size
                    ** 2
                )
            )

            expected_counts.append(
                expected_count
            )

        # ---------------------------------------------------------
        # Initialize masks
        # ---------------------------------------------------------

        visual_token_mask = torch.zeros(
            (
                batch_size,
                sequence_length,
            ),
            dtype=torch.bool,
            device=input_ids.device,
        )

        visible_token_mask = torch.zeros_like(
            visual_token_mask
        )

        infrared_token_mask = torch.zeros_like(
            visual_token_mask
        )

        # ---------------------------------------------------------
        # Find image-token runs for every batch sample
        # ---------------------------------------------------------

        image_token_spans: List[
            List[Tuple[int, int]]
        ] = []

        image_token_counts: List[
            List[int]
        ] = []

        image_token_id = self.image_token_id

        for batch_idx in range(
            batch_size
        ):

            row = input_ids[batch_idx]

            # ---------------------------------------------
            # Find all positions where token == image_token_id
            # ---------------------------------------------

            image_positions = torch.nonzero(
                row == image_token_id,
                as_tuple=False,
            ).flatten()

            if image_positions.numel() == 0:

                raise RuntimeError(
                    "No image tokens were found in "
                    f"sample {batch_idx}. "
                    f"Expected image_token_id="
                    f"{image_token_id}."
                )

            # ---------------------------------------------
            # Remove padding positions if attention_mask exists
            # ---------------------------------------------

            if attention_mask is not None:

                valid_mask = (
                    attention_mask[
                        batch_idx
                    ].bool()
                )

                image_positions = (
                    image_positions[
                        valid_mask[
                            image_positions
                        ]
                    ]
                )

            if image_positions.numel() == 0:

                raise RuntimeError(
                    "No valid image tokens were found "
                    f"in sample {batch_idx}."
                )

            # ---------------------------------------------
            # Convert positions into contiguous runs
            #
            # Example:
            #
            # [10,11,12,13, 50,51,52]
            #
            # →
            #
            # [(10,14), (50,53)]
            # ---------------------------------------------

            spans = self._contiguous_spans(
                image_positions
            )

            actual_counts = [
                end - start
                for start, end in spans
            ]

            # ---------------------------------------------
            # Strictly require exactly two image segments
            # ---------------------------------------------

            if len(spans) != 2:

                raise RuntimeError(
                    "Expected exactly two contiguous "
                    "image-token segments "
                    "(Visible + Infrared), but found "
                    f"{len(spans)} in batch sample "
                    f"{batch_idx}.\n"
                    f"Image token ID: "
                    f"{image_token_id}\n"
                    f"Spans: {spans}\n"
                    f"Counts: {actual_counts}\n"
                    f"Expected counts from "
                    f"image_grid_thw: "
                    f"{expected_counts}"
                )

            # ---------------------------------------------
            # Compare token counts with image_grid_thw
            # ---------------------------------------------

            for image_idx in range(2):

                actual = actual_counts[
                    image_idx
                ]

                expected = expected_counts[
                    image_idx
                ]

                if actual != expected:

                    error_message = (
                        "\n"
                        "Qwen2.5-VL visual-token "
                        "alignment error.\n"
                        f"Batch index      : "
                        f"{batch_idx}\n"
                        f"Image index      : "
                        f"{image_idx}\n"
                        f"Expected tokens  : "
                        f"{expected}\n"
                        f"Actual tokens    : "
                        f"{actual}\n"
                        f"image_grid_thw   : "
                        f"{image_grid_thw[image_idx].tolist()}\n"
                        f"spatial_merge    : "
                        f"{self.spatial_merge_size}\n"
                        f"image_token_id   : "
                        f"{image_token_id}\n"
                    )

                    if (
                        self.strict_visual_token_check
                    ):

                        raise RuntimeError(
                            error_message
                        )

                    else:

                        print(
                            "WARNING:"
                            + error_message
                        )

            # ---------------------------------------------
            # Save metadata
            # ---------------------------------------------

            image_token_spans.append(
                spans
            )

            image_token_counts.append(
                actual_counts
            )

            # ---------------------------------------------
            # Create exact masks
            # ---------------------------------------------

            visible_start, visible_end = (
                spans[0]
            )

            infrared_start, infrared_end = (
                spans[1]
            )

            visible_token_mask[
                batch_idx,
                visible_start:visible_end,
            ] = True

            infrared_token_mask[
                batch_idx,
                infrared_start:infrared_end,
            ] = True

            visual_token_mask[
                batch_idx,
                visible_start:visible_end,
            ] = True

            visual_token_mask[
                batch_idx,
                infrared_start:infrared_end,
            ] = True

        return (
            image_token_spans,
            image_token_counts,
            visual_token_mask,
            visible_token_mask,
            infrared_token_mask,
        )

    # ==================================================================
    # Find contiguous spans
    # ==================================================================

    @staticmethod
    def _contiguous_spans(
        positions: torch.Tensor,
    ) -> List[Tuple[int, int]]:
        """
        将 image token positions 转换成连续区间。

        输入：
            tensor([10,11,12,50,51,52])

        输出：
            [
                (10,13),
                (50,53)
            ]

        注意：
            end 是 exclusive。
        """

        if positions.numel() == 0:

            return []

        positions = positions.detach()

        spans: List[
            Tuple[int, int]
        ] = []

        start = int(
            positions[0].item()
        )

        previous = start

        for idx in range(
            1,
            positions.numel(),
        ):

            current = int(
                positions[idx].item()
            )

            # 连续
            if current == previous + 1:

                previous = current
                continue

            # 当前连续段结束
            spans.append(
                (
                    start,
                    previous + 1,
                )
            )

            start = current
            previous = current

        # 最后一段
        spans.append(
            (
                start,
                previous + 1,
            )
        )

        return spans

    # ==================================================================
    # Extract visual hidden states
    # ==================================================================

    @staticmethod
    def _extract_visual_hidden_states(
        selected_hidden_states: List[torch.Tensor],
        visible_token_mask: torch.Tensor,
        infrared_token_mask: torch.Tensor,
        visual_token_mask: torch.Tensor,
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
    ]:
        """
        从每一个 selected hidden state 中
        精确提取：

            Visible
            Infrared
            Visible + Infrared

        注意：

            目前 wrapper 的设计场景是：
                batch_size = 1

            因为 Visible / IR 动态分辨率不同，
            不同 sample 的 token 数可能不同。

            对 B=1：

                [1, Nv, H]
                [1, Nir, H]

            是最自然的下游接口。

        对 B>1：
            如果不同 batch sample 的视觉 token 数不同，
            不应该静默 stack，而应该明确报错。
        """

        batch_size = (
            visible_token_mask.shape[0]
        )

        if batch_size != 1:

            raise NotImplementedError(
                "Current Visible/Infrared visual-token "
                "extraction is designed for batch_size=1, "
                "because Qwen2.5-VL uses dynamic visual "
                "token counts. For training with B>1, "
                "a padding/packing strategy should be "
                "implemented explicitly rather than "
                "silently mixing variable-length tokens."
            )

        visible_hidden_states: List[
            torch.Tensor
        ] = []

        infrared_hidden_states: List[
            torch.Tensor
        ] = []

        visual_hidden_states: List[
            torch.Tensor
        ] = []

        for hidden in selected_hidden_states:

            if hidden.ndim != 3:

                raise ValueError(
                    "Hidden state must have shape "
                    "[B, N, H], "
                    f"but got {tuple(hidden.shape)}"
                )

            if (
                hidden.shape[0]
                != visible_token_mask.shape[0]
            ):

                raise ValueError(
                    "Batch size mismatch between "
                    "hidden state and token mask."
                )

            if (
                hidden.shape[1]
                != visible_token_mask.shape[1]
            ):

                raise ValueError(
                    "Sequence length mismatch between "
                    "hidden state and token mask.\n"
                    f"Hidden: {tuple(hidden.shape)}\n"
                    f"Mask: {tuple(visible_token_mask.shape)}"
                )

            # -----------------------------------------------------
            # Visible
            # -----------------------------------------------------

            visible = hidden[
                visible_token_mask
            ]

            visible = visible.unsqueeze(0)

            # -----------------------------------------------------
            # Infrared
            # -----------------------------------------------------

            infrared = hidden[
                infrared_token_mask
            ]

            infrared = infrared.unsqueeze(0)

            # -----------------------------------------------------
            # All visual tokens
            # -----------------------------------------------------

            visual = hidden[
                visual_token_mask
            ]

            visual = visual.unsqueeze(0)

            visible_hidden_states.append(
                visible
            )

            infrared_hidden_states.append(
                infrared
            )

            visual_hidden_states.append(
                visual
            )

        return (
            visible_hidden_states,
            infrared_hidden_states,
            visual_hidden_states,
        )

    # ==================================================================
    # Extract image inputs
    # ==================================================================

    @staticmethod
    def _extract_image_inputs(
        messages: List[Dict[str, Any]]
    ) -> List[Image.Image]:
        """
        从 messages 中提取所有 image。

        当前任务固定返回：

            [
                visible_image,
                infrared_image
            ]

        image 顺序必须和 _build_messages()
        完全一致。
        """

        images: List[
            Image.Image
        ] = []

        for message in messages:

            content = message.get(
                "content",
                [],
            )

            for item in content:

                if item.get("type") != "image":
                    continue

                image = item.get(
                    "image",
                    None,
                )

                if image is None:

                    raise ValueError(
                        "Image item does not "
                        "contain 'image'."
                    )

                image = (
                    Qwen25VLWrapper._load_image(
                        image
                    )
                )

                images.append(
                    image
                )

        if len(images) == 0:

            raise ValueError(
                "No image was found in messages."
            )

        return images

    # ==================================================================
    # Debug helper
    # ==================================================================

    def print_hidden_shapes(
        self,
        visible: ImageInput,
        infrared: ImageInput,
        prompt: str = (
            "Analyze the visible and infrared images and "
            "identify complementary visual information."
        ),
    ) -> None:
        """
        打印：

            1. 所有 hidden states shape
            2. selected hidden states
            3. Visible visual tokens
            4. Infrared visual tokens
            5. image token span
            6. image token count
            7. image_grid_thw
        """

        result = self.forward(
            visible=visible,
            infrared=infrared,
            prompt=prompt,
            return_all_hidden_states=True,
        )

        # ---------------------------------------------------------
        # All hidden states
        # ---------------------------------------------------------

        print(
            "\n"
            + "=" * 70
        )

        print(
            "All Hidden States"
        )

        print(
            "=" * 70
        )

        all_hidden_states = (
            result["hidden_states"]
        )

        for idx, hidden in enumerate(
            all_hidden_states
        ):

            print(
                f"Layer {idx:02d}: "
                f"{tuple(hidden.shape)}"
            )

        # ---------------------------------------------------------
        # Selected layers
        # ---------------------------------------------------------

        print(
            "\n"
            + "=" * 70
        )

        print(
            "Selected Layers"
        )

        print(
            "=" * 70
        )

        for layer_idx, hidden in zip(
            self.selected_layers,
            result[
                "selected_hidden_states"
            ],
        ):

            print(
                f"Layer {layer_idx:02d}: "
                f"{tuple(hidden.shape)}"
            )

        # ---------------------------------------------------------
        # Image metadata
        # ---------------------------------------------------------

        print(
            "\n"
            + "=" * 70
        )

        print(
            "Visual Token Metadata"
        )

        print(
            "=" * 70
        )

        print(
            "image_token_id:",
            result["image_token_id"],
        )

        print(
            "image_grid_thw:"
        )

        print(
            result["image_grid_thw"]
            .detach()
            .cpu()
            .tolist()
        )

        print(
            "image_token_spans:"
        )

        print(
            result["image_token_spans"]
        )

        print(
            "image_token_counts:"
        )

        print(
            result["image_token_counts"]
        )

        # ---------------------------------------------------------
        # Visual hidden states
        # ---------------------------------------------------------

        print(
            "\n"
            + "=" * 70
        )

        print(
            "Visible / Infrared Visual "
            "Hidden States"
        )

        print(
            "=" * 70
        )

        for i, layer_idx in enumerate(
            self.selected_layers
        ):

            visible_hidden = (
                result[
                    "visible_hidden_states"
                ][i]
            )

            infrared_hidden = (
                result[
                    "infrared_hidden_states"
                ][i]
            )

            visual_hidden = (
                result[
                    "visual_hidden_states"
                ][i]
            )

            print(
                f"\nLayer {layer_idx:02d}"
            )

            print(
                "  Visible   : ",
                tuple(
                    visible_hidden.shape
                ),
            )

            print(
                "  Infrared  : ",
                tuple(
                    infrared_hidden.shape
                ),
            )

            print(
                "  Visual all: ",
                tuple(
                    visual_hidden.shape
                ),
            )

        # ---------------------------------------------------------
        # Last hidden state
        # ---------------------------------------------------------

        print(
            "\n"
            + "=" * 70
        )

        print(
            "Last Hidden State"
        )

        print(
            "=" * 70
        )

        print(
            tuple(
                result[
                    "last_hidden_state"
                ].shape
            )
        )

        print(
            "=" * 70
        )