# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .qscfmcp import QSCFMCP
from .qwen25vl_wrapper import Qwen25VLWrapper

from .multimodal_llava_encoder.builder import build_vision_tower
from .multimodal_llava_projector.builder import build_vision_projector
from .multimodal_decoder.builder import build_vae, build_sana

from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
)

from diffusers.models.normalization import RMSNorm

from mobileo.constants import (
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)


# ============================================================
# 原 Mobile-O Diffusion Connector
# 保留，供 use_qwen25vl=False 时使用
# ============================================================


class DiffusionConnector(nn.Module):

    def __init__(
        self,
        input_dim=1536,
        hidden_dim=1024,
        output_dim=2304,
        eps=1e-5,
    ):

        super().__init__()

        self.linear1 = nn.Linear(
            input_dim,
            hidden_dim,
        )

        self.act = nn.GELU(
            approximate="tanh"
        )

        self.linear2 = nn.Linear(
            hidden_dim,
            output_dim,
        )

        self.norm = RMSNorm(
            output_dim,
            eps=eps,
            elementwise_affine=True,
        )

        nn.init.xavier_uniform_(
            self.linear1.weight
        )

        nn.init.zeros_(
            self.linear1.bias
        )

        nn.init.xavier_uniform_(
            self.linear2.weight
        )

        nn.init.zeros_(
            self.linear2.bias
        )

        with torch.no_grad():

            self.norm.weight.fill_(
                math.sqrt(5.5)
            )


    def forward(
        self,
        x,
    ):

        x = self.linear1(x)

        x = self.act(x)

        x = self.linear2(x)

        x = self.norm(x)

        return x


# ============================================================
# Llava Meta Model
# ============================================================


class LlavaMetaModel:

    def __init__(
        self,
        config,
    ):

        super(
            LlavaMetaModel,
            self
        ).__init__(
            config
        )

        print(
            "=" * 20,
            "Initializing the model",
            "=" * 20,
        )

        print(config)

        print(
            "=" * 50
        )

        # ========================================================
        # Configuration
        # ========================================================

        self.use_qwen25vl = bool(
            getattr(
                config,
                "use_qwen25vl",
                False,
            )
        )

        print(
            "use_qwen25vl:",
            self.use_qwen25vl,
        )


        # ========================================================
        # 原 Mobile-O Vision Tower
        #
        # 只有在非 Qwen2.5-VL 模式下初始化。
        # ========================================================

        if not self.use_qwen25vl:

            if hasattr(
                config,
                "mm_vision_tower",
            ):

                self.vision_tower = (
                    build_vision_tower(
                        config,
                        delay_load=True,
                    )
                )

                self.mm_projector = (
                    build_vision_projector(
                        config
                    )
                )


        # ========================================================
        # Qwen2.5-VL
        # ========================================================

        if self.use_qwen25vl:

            qwen25vl_path = getattr(
                config,
                "qwen25vl_model_path",
                None,
            )

            if qwen25vl_path is None:

                raise ValueError(
                    "use_qwen25vl=True, but "
                    "qwen25vl_model_path is not configured."
                )


            qwen25vl_device = getattr(
                config,
                "qwen25vl_device",
                "cuda",
            )


            qwen25vl_last_n_layers = int(
                getattr(
                    config,
                    "qwen25vl_last_n_layers",
                    8,
                )
            )


            qwen25vl_dtype_name = getattr(
                config,
                "qwen25vl_dtype",
                "bfloat16",
            )


            if qwen25vl_dtype_name == "float16":

                qwen25vl_dtype = torch.float16

            elif qwen25vl_dtype_name == "float32":

                qwen25vl_dtype = torch.float32

            elif qwen25vl_dtype_name == "bfloat16":

                qwen25vl_dtype = torch.bfloat16

            else:

                raise ValueError(
                    "Unsupported qwen25vl_dtype: "
                    f"{qwen25vl_dtype_name}"
                )


            print(
                "=" * 70
            )

            print(
                "Loading Qwen2.5-VL-3B"
            )

            print(
                "Path:",
                qwen25vl_path
            )

            print(
                "Device:",
                qwen25vl_device
            )

            print(
                "Layers:",
                qwen25vl_last_n_layers
            )

            print(
                "Dtype:",
                qwen25vl_dtype
            )

            print(
                "=" * 70
            )


            self.qwen25vl_wrapper = (
                Qwen25VLWrapper(
                    model_path=qwen25vl_path,
                    last_n_layers=qwen25vl_last_n_layers,
                    torch_dtype=qwen25vl_dtype,
                    device=qwen25vl_device,
                    freeze=True,
                )
            )


        # ========================================================
        # Diffusion / SANA
        # ========================================================

        if hasattr(
            config,
            "diffusion_name_or_path",
        ):

            self.dit = build_sana(
                config
            )

            self.vae = build_vae(
                config
            )


            # ====================================================
            # 新 QSCFMCP
            # ====================================================

            if self.use_qwen25vl:

                self.diffusion_connector = (
                    QSCFMCP(
                        input_dim=2048,
                        hidden_dim=512,
                        output_dim=2304,
                        num_layers=8,
                    )
                )

            # ====================================================
            # 原 Mobile-O connector
            # ====================================================

            else:

                self.diffusion_connector = (
                    DiffusionConnector(
                        input_dim=getattr(
                            config,
                            "hidden_size",
                            896,
                        ),
                        hidden_dim=512,
                        output_dim=2304,
                    )
                )


            # ====================================================
            # Scheduler
            # ====================================================

            if hasattr(
                config,
                "is_train",
            ):

                if config.is_train:

                    print(
                        "FlowMatchEulerDiscreteScheduler is used"
                    )

                    self.noise_scheduler = (
                        FlowMatchEulerDiscreteScheduler.from_pretrained(
                            config.diffusion_name_or_path,
                            subfolder="scheduler",
                        )
                    )

                else:

                    print(
                        "DPMSolverMultistepScheduler is used"
                    )

                    self.noise_scheduler = (
                        DPMSolverMultistepScheduler.from_pretrained(
                            config.diffusion_name_or_path,
                            subfolder="scheduler",
                        )
                    )

            else:

                print(
                    "FlowMatchEulerDiscreteScheduler is used"
                )

                self.noise_scheduler = (
                    FlowMatchEulerDiscreteScheduler.from_pretrained(
                        config.diffusion_name_or_path,
                        subfolder="scheduler",
                    )
                )


    # ============================================================
    # Vision tower
    # ============================================================

    def get_vision_tower(
        self,
    ):

        vision_tower = getattr(
            self,
            "vision_tower",
            None,
        )

        if type(vision_tower) is list:

            vision_tower = (
                vision_tower[0]
            )

        return vision_tower


    # ============================================================
    # SANA
    # ============================================================

    def get_sana(
        self,
    ):

        dit = getattr(
            self,
            "dit",
            None,
        )

        if type(dit) is list:

            dit = dit[0]

        if dit is not None:

            dit.to(
                self.device
            )

        return dit


    # ============================================================
    # VAE
    # ============================================================

    def get_sana_vae(
        self,
    ):

        vae = getattr(
            self,
            "vae",
            None,
        )

        if type(vae) is list:

            vae = vae[0]

        if vae is not None:

            vae.to(
                self.device
            )

        return vae


    # ============================================================
    # Qwen2.5-VL
    # ============================================================

    def get_qwen25vl(
        self,
    ):

        if not self.use_qwen25vl:

            return None

        return getattr(
            self,
            "qwen25vl_wrapper",
            None,
        )


    # ============================================================
    # Initialize modules
    # ============================================================

    def initialize_vision_modules(
        self,
        model_args,
        fsdp=None,
    ):

        mm_vision_select_layer = getattr(
            model_args,
            "mm_vision_select_layer",
            -2,
        )

        mm_vision_select_feature = getattr(
            model_args,
            "mm_vision_select_feature",
            "patch",
        )

        mm_patch_merge_type = getattr(
            model_args,
            "mm_patch_merge_type",
            "flat",
        )


        # ========================================================
        # SANA
        # ========================================================

        if self.get_sana() is None:

            dit = build_sana(
                model_args
            )

            if hasattr(
                model_args,
                "is_train",
            ):

                if model_args.is_train:

                    print(
                        "FlowMatchEulerDiscreteScheduler is used"
                    )

                    self.noise_scheduler = (
                        FlowMatchEulerDiscreteScheduler.from_pretrained(
                            model_args.diffusion_name_or_path,
                            subfolder="scheduler",
                        )
                    )

                else:

                    print(
                        "DPMSolverMultistepScheduler is used"
                    )

                    self.noise_scheduler = (
                        DPMSolverMultistepScheduler.from_pretrained(
                            model_args.diffusion_name_or_path,
                            subfolder="scheduler",
                        )
                    )

            if (
                fsdp is not None
                and len(fsdp) > 0
            ):

                self.dit = [
                    dit
                ]

            else:

                self.dit = dit

        else:

            if (
                fsdp is not None
                and len(fsdp) > 0
            ):

                dit = self.dit[0]

            else:

                dit = self.dit


        # ========================================================
        # SANA trainable
        # ========================================================

        for p in dit.parameters():

            p.requires_grad = True


        # ========================================================
        # VAE
        # ========================================================

        if self.get_sana_vae() is None:

            vae = build_vae(
                model_args
            )

            if (
                fsdp is not None
                and len(fsdp) > 0
            ):

                self.vae = [
                    vae
                ]

            else:

                self.vae = vae

        else:

            if (
                fsdp is not None
                and len(fsdp) > 0
            ):

                vae = self.vae[0]

            else:

                vae = self.vae


        for p in vae.parameters():

            p.requires_grad = False


        # ========================================================
        # Qwen2.5-VL mode
        # ========================================================

        if self.use_qwen25vl:

            print(
                "=" * 70
            )

            print(
                "Qwen2.5-VL mode enabled."
            )

            print(
                "Original Mobile-O vision tower "
                "is not used."
            )

            print(
                "=" * 70
            )


            if getattr(
                self,
                "diffusion_connector",
                None,
            ) is None:

                self.diffusion_connector = (
                    QSCFMCP(
                        input_dim=2048,
                        hidden_dim=512,
                        output_dim=2304,
                        num_layers=8,
                    )
                )


            for p in (
                self.diffusion_connector
                .parameters()
            ):

                p.requires_grad = True


            self.config.use_mm_proj = False

            self.config.mm_vision_select_layer = (
                mm_vision_select_layer
            )

            self.config.mm_vision_select_feature = (
                mm_vision_select_feature
            )

            self.config.mm_patch_merge_type = (
                mm_patch_merge_type
            )

            self.config.diffusion_name_or_path = (
                model_args.diffusion_name_or_path
            )

            self.config.is_train = True

            return


        # ========================================================
        # Original Mobile-O vision mode
        # ========================================================

        if self.get_vision_tower() is None:

            print(
                "=" * 20,
                "Building vision tower",
                "=" * 20,
            )

            vision_tower = build_vision_tower(
                model_args
            )

            if (
                fsdp is not None
                and len(fsdp) > 0
            ):

                self.vision_tower = [
                    vision_tower
                ]

            else:

                self.vision_tower = vision_tower

        else:

            if (
                fsdp is not None
                and len(fsdp) > 0
            ):

                vision_tower = (
                    self.vision_tower[0]
                )

            else:

                vision_tower = (
                    self.vision_tower
                )

            vision_tower.load_model()


        # Original Vision Tower
        for p in vision_tower.parameters():

            p.requires_grad = False


        # Original MM projector
        self.config.use_mm_proj = True

        self.config.mm_projector_type = getattr(
            model_args,
            "mm_projector_type",
            "linear",
        )

        self.config.mm_vision_select_layer = (
            mm_vision_select_layer
        )

        self.config.mm_vision_select_feature = (
            mm_vision_select_feature
        )

        self.config.mm_patch_merge_type = (
            mm_patch_merge_type
        )

        self.config.diffusion_name_or_path = (
            model_args.diffusion_name_or_path
        )

        self.config.is_train = True


    # ============================================================
    # Qwen image tensor → PIL
    # ============================================================

    @staticmethod
    def _tensor_to_pil(
        image: torch.Tensor,
    ) -> Image.Image:

        if image.ndim == 4:

            if image.shape[0] != 1:

                raise ValueError(
                    "_tensor_to_pil expects "
                    "one image at a time."
                )

            image = image[0]


        if image.ndim != 3:

            raise ValueError(
                "Image tensor must have shape "
                "[C,H,W] or [1,C,H,W]. "
                f"Got {tuple(image.shape)}"
            )


        image = (
            image.detach()
            .float()
            .cpu()
        )


        # --------------------------------------------------------
        # Channel handling
        # --------------------------------------------------------

        if image.shape[0] == 1:

            image = image.repeat(
                3,
                1,
                1,
            )

        elif image.shape[0] >= 3:

            image = image[:3]

        else:

            raise ValueError(
                "Image must have 1 or at least "
                f"3 channels. Got {image.shape[0]}"
            )


        # --------------------------------------------------------
        # Normalize
        #
        # 默认兼容：
        #   [0,1]
        #   [-1,1]
        # --------------------------------------------------------

        min_value = image.min().item()
        max_value = image.max().item()


        if (
            min_value < 0.0
            and min_value >= -1.1
            and max_value <= 1.1
        ):

            image = (
                image + 1.0
            ) / 2.0

        else:

            image = image.clamp(
                0.0,
                1.0,
            )


        image = (
            image
            .permute(
                1,
                2,
                0,
            )
            * 255.0
        )


        image = image.byte()


        return Image.fromarray(
            image.numpy()
        )


    # ============================================================
    # Split Visible / Infrared
    # ============================================================

    @classmethod
    def split_visible_infrared(
        cls,
        images,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        # --------------------------------------------------------
        # Case 1:
        # [B, 2, C, H, W]
        # --------------------------------------------------------

        if torch.is_tensor(images):

            if images.ndim == 5:

                if images.shape[1] != 2:

                    raise ValueError(
                        "5D multimodal tensor must "
                        "have shape [B,2,C,H,W]. "
                        f"Got {tuple(images.shape)}"
                    )

                visible = images[:, 0]
                infrared = images[:, 1]

                return (
                    visible,
                    infrared,
                )


            # ----------------------------------------------------
            # Case 2:
            # [B, 6, H, W]
            # ----------------------------------------------------

            if images.ndim == 4:

                channels = images.shape[1]

                if channels == 6:

                    visible = images[
                        :, 0:3
                    ]

                    infrared = images[
                        :, 3:6
                    ]

                    return (
                        visible,
                        infrared,
                    )

                raise ValueError(
                    "A 4D tensor for Visible/IR "
                    "must have 6 channels "
                    "[V_RGB, IR_RGB]. "
                    f"Got {channels} channels."
                )


        # --------------------------------------------------------
        # Case 3:
        #
        # [visible_tensor, infrared_tensor]
        # --------------------------------------------------------

        if isinstance(
            images,
            (list, tuple),
        ):

            if len(images) != 2:

                raise ValueError(
                    "Visible/Infrared input list "
                    "must contain exactly 2 elements."
                )

            visible = images[0]
            infrared = images[1]

            if not torch.is_tensor(
                visible
            ) or not torch.is_tensor(
                infrared
            ):

                raise TypeError(
                    "Visible and Infrared elements "
                    "must both be torch.Tensor."
                )


            if visible.ndim == 3:

                visible = visible.unsqueeze(
                    0
                )

            if infrared.ndim == 3:

                infrared = infrared.unsqueeze(
                    0
                )


            return (
                visible,
                infrared,
            )


        raise TypeError(
            "Unsupported Visible/Infrared input type: "
            f"{type(images)}"
        )


    # ============================================================
    # Qwen2.5-VL condition extraction
    # ============================================================

    def extract_qwen25vl_condition(
        self,
        images,
        prompt,
    ):

        qwen = self.get_qwen25vl()

        if qwen is None:

            raise RuntimeError(
                "Qwen2.5-VL wrapper is not initialized."
            )


        visible, infrared = (
            self.split_visible_infrared(
                images
            )
        )


        if visible.shape[0] != infrared.shape[0]:

            raise ValueError(
                "Visible and Infrared batch size mismatch."
            )


        batch_size = (
            visible.shape[0]
        )


        visible_layers_batch = [
            []
            for _ in qwen.selected_layers
        ]

        infrared_layers_batch = [
            []
            for _ in qwen.selected_layers
        ]


        # --------------------------------------------------------
        # Qwen wrapper 当前为了保证动态 token 精确对齐，
        # 每次对一个 sample 进行处理。
        # --------------------------------------------------------

        for batch_idx in range(
            batch_size
        ):

            visible_pil = (
                self._tensor_to_pil(
                    visible[batch_idx]
                )
            )

            infrared_pil = (
                self._tensor_to_pil(
                    infrared[batch_idx]
                )
            )


            qwen_output = qwen(
                visible=visible_pil,
                infrared=infrared_pil,
                prompt=prompt,
                return_all_hidden_states=False,
            )


            visible_layers = (
                qwen_output[
                    "visible_hidden_states"
                ]
            )

            infrared_layers = (
                qwen_output[
                    "infrared_hidden_states"
                ]
            )


            for layer_idx in range(
                len(qwen.selected_layers)
            ):

                visible_layers_batch[
                    layer_idx
                ].append(
                    visible_layers[
                        layer_idx
                    ]
                )

                infrared_layers_batch[
                    layer_idx
                ].append(
                    infrared_layers[
                        layer_idx
                    ]
                )


        # --------------------------------------------------------
        # Stack / validate dynamic lengths
        # --------------------------------------------------------

        visible_hidden_states = []

        infrared_hidden_states = []


        for layer_idx in range(
            len(qwen.selected_layers)
        ):

            current_visible = (
                visible_layers_batch[
                    layer_idx
                ]
            )

            current_infrared = (
                infrared_layers_batch[
                    layer_idx
                ]
            )


            visible_lengths = [
                x.shape[1]
                for x in current_visible
            ]

            infrared_lengths = [
                x.shape[1]
                for x in current_infrared
            ]


            if len(
                set(visible_lengths)
            ) != 1:

                raise RuntimeError(
                    "Visible visual token counts "
                    "are different inside the "
                    "same batch. "
                    f"Layer={qwen.selected_layers[layer_idx]}, "
                    f"counts={visible_lengths}. "
                    "Use fixed image resolution or "
                    "implement explicit padding/masking."
                )


            if len(
                set(infrared_lengths)
            ) != 1:

                raise RuntimeError(
                    "Infrared visual token counts "
                    "are different inside the "
                    "same batch. "
                    f"Layer={qwen.selected_layers[layer_idx]}, "
                    f"counts={infrared_lengths}. "
                    "Use fixed image resolution or "
                    "implement explicit padding/masking."
                )


            visible_hidden_states.append(
                torch.cat(
                    current_visible,
                    dim=0,
                )
            )

            infrared_hidden_states.append(
                torch.cat(
                    current_infrared,
                    dim=0,
                )
            )


        return (
            visible_hidden_states,
            infrared_hidden_states,
        )


    # ============================================================
    # Visual encoder
    # ============================================================

    def visual(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:

        if self.use_qwen25vl:

            raise RuntimeError(
                "visual() from original Mobile-O "
                "vision tower must not be used when "
                "use_qwen25vl=True."
            )


        image_features = (
            self.get_vision_tower()(
                pixel_values
            )
        )


        image_features = (
            self.mm_projector(
                image_features.to(
                    self.mm_projector[0]
                    .weight.dtype
                )
            )
        )


        return image_features


# ============================================================
# LlavaMetaForCausalLM
# ============================================================


class LlavaMetaForCausalLM(
    ABC
):

    @abstractmethod
    def get_model(
        self
    ):
        pass


    def get_vision_tower(
        self
    ):

        return (
            self.get_model()
            .get_vision_tower()
        )


    def get_qwen25vl(
        self
    ):

        return (
            self.get_model()
            .get_qwen25vl()
        )


    def get_mm_projector(
        self
    ):

        return (
            self.get_model()
            .mm_projector
        )


    # ============================================================
    # Sigma
    # ============================================================

    def get_sigmas(
        self,
        timesteps,
        device,
        n_dim=4,
        dtype=torch.float32,
    ):

        sigmas = (
            self.get_model()
            .noise_scheduler
            .sigmas
            .to(
                device=device,
                dtype=dtype,
            )
        )


        schedule_timesteps = (
            self.get_model()
            .noise_scheduler
            .timesteps
            .to(
                device=device
            )
        )


        timesteps = (
            timesteps.to(device)
        )


        step_indices = []

        for t in timesteps:

            indices = (
                schedule_timesteps == t
            ).nonzero(
                as_tuple=False
            )

            if indices.numel() == 0:

                raise RuntimeError(
                    f"Timestep {t.item()} "
                    "not found in scheduler."
                )

            step_indices.append(
                indices[0].item()
            )


        step_indices = torch.tensor(
            step_indices,
            device=device,
            dtype=torch.long,
        )


        sigma = sigmas[
            step_indices
        ].flatten()


        while len(
            sigma.shape
        ) < n_dim:

            sigma = sigma.unsqueeze(
                -1
            )


        return sigma


    # ============================================================
    # Latent mask drop
    # ============================================================

    def mask_drop(
        self,
        latents,
        drop_prob=0.1,
    ):

        if drop_prob <= 0:

            return latents


        mask = torch.bernoulli(
            torch.zeros(
                latents.shape[0],
                device=latents.device,
                dtype=latents.dtype,
            )
            + drop_prob
        )


        while len(
            mask.shape
        ) < len(
            latents.shape
        ):

            mask = mask.unsqueeze(
                -1
            )


        mask = (
            1 - mask
        )


        return latents * mask


    # ============================================================
    # Prepare original Mobile-O multimodal input
    # ============================================================

    def _prepare_original_mobileo_inputs(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        und_images,
    ):

        images = und_images


        if (
            type(images) is list
            or images.ndim == 5
        ):

            if type(images) is list:

                images = [
                    x.unsqueeze(0)
                    if x.ndim == 3
                    else x
                    for x in images
                ]


            concat_images = torch.cat(
                [
                    image
                    for image in images
                ],
                dim=0,
            )


            image_features = (
                self.visual(
                    concat_images
                )
            )


            split_sizes = [
                image.shape[0]
                for image in images
            ]


            image_features = (
                torch.split(
                    image_features,
                    split_sizes,
                    dim=0,
                )
            )


            image_features = [
                x.flatten(
                    0,
                    1
                )
                for x in image_features
            ]

        else:

            image_features = (
                self.visual(
                    images
                )
            )


        # ========================================================
        # Dummy tensors
        # ========================================================

        _labels = labels

        _position_ids = (
            position_ids
        )

        _attention_mask = (
            attention_mask
        )


        if attention_mask is None:

            attention_mask = torch.ones_like(
                input_ids,
                dtype=torch.bool,
            )

        else:

            attention_mask = (
                attention_mask.bool()
            )


        if position_ids is None:

            position_ids = torch.arange(
                0,
                input_ids.shape[1],
                dtype=torch.long,
                device=input_ids.device,
            )


        if labels is None:

            labels = torch.full_like(
                input_ids,
                IGNORE_INDEX,
            )


        # ========================================================
        # Remove padding
        # ========================================================

        input_ids = [
            cur_input_ids[
                cur_attention_mask
            ]

            for cur_input_ids,
            cur_attention_mask

            in zip(
                input_ids,
                attention_mask,
            )
        ]


        labels = [
            cur_labels[
                cur_attention_mask
            ]

            for cur_labels,
            cur_attention_mask

            in zip(
                labels,
                attention_mask,
            )
        ]


        new_input_embeds = []

        new_labels = []

        new_input_ids = []

        cur_image_idx = 0


        # ========================================================
        # Insert image features
        # ========================================================

        for batch_idx, cur_input_ids in enumerate(
            input_ids
        ):

            num_images = int(
                (
                    cur_input_ids
                    == IMAGE_TOKEN_INDEX
                ).sum().item()
            )


            if num_images == 0:

                cur_image_features = (
                    image_features[
                        cur_image_idx
                    ]
                )


                cur_input_embeds_1 = (
                    self.get_model()
                    .embed_tokens(
                        cur_input_ids
                    )
                )


                cur_input_embeds = (
                    torch.cat(
                        [
                            cur_input_embeds_1,
                            cur_image_features[
                                0:0
                            ],
                        ],
                        dim=0,
                    )
                )


                new_input_embeds.append(
                    cur_input_embeds
                )

                new_labels.append(
                    labels[batch_idx]
                )

                new_input_ids.append(
                    cur_input_ids
                )

                cur_image_idx += 1

                continue


            image_token_indices = (
                [-1]
                + torch.where(
                    cur_input_ids
                    == IMAGE_TOKEN_INDEX
                )[0].tolist()
                + [
                    cur_input_ids.shape[0]
                ]
            )


            cur_input_ids_noim = []

            cur_labels = (
                labels[batch_idx]
            )

            cur_labels_noim = []


            for i in range(
                len(image_token_indices)
                - 1
            ):

                cur_input_ids_noim.append(
                    cur_input_ids[
                        image_token_indices[i]
                        + 1:
                        image_token_indices[i + 1]
                    ]
                )


                cur_labels_noim.append(
                    cur_labels[
                        image_token_indices[i]
                        + 1:
                        image_token_indices[i + 1]
                    ]
                )


            split_sizes = [
                x.shape[0]
                for x in cur_labels_noim
            ]


            cur_input_embeds = (
                self.get_model()
                .embed_tokens(
                    torch.cat(
                        cur_input_ids_noim
                    )
                )
            )


            cur_input_embeds_no_im = (
                torch.split(
                    cur_input_embeds,
                    split_sizes,
                    dim=0,
                )
            )


            cur_new_input_embeds = []

            cur_new_labels = []

            cur_new_input_ids = []


            for i in range(
                num_images + 1
            ):

                cur_new_input_embeds.append(
                    cur_input_embeds_no_im[i]
                )

                cur_new_labels.append(
                    cur_labels_noim[i]
                )

                cur_new_input_ids.append(
                    cur_input_ids_noim[i]
                )


                if i < num_images:

                    if (
                        cur_image_idx
                        < len(image_features)
                    ):

                        cur_image_features = (
                            image_features[
                                cur_image_idx
                            ]
                        )

                    else:

                        cur_image_features = (
                            image_features[-1]
                        )


                    cur_image_idx += 1


                    cur_new_input_embeds.append(
                        cur_image_features
                    )


                    cur_new_labels.append(
                        torch.full(
                            (
                                cur_image_features
                                .shape[0],
                            ),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )


                    cur_new_input_ids.append(
                        torch.full(
                            (
                                cur_image_features
                                .shape[0],
                            ),
                            IMAGE_TOKEN_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )


            cur_new_input_embeds = [
                x.to(self.device)
                for x in cur_new_input_embeds
            ]


            cur_new_input_embeds = (
                torch.cat(
                    cur_new_input_embeds,
                    dim=0,
                )
            )


            cur_new_labels = (
                torch.cat(
                    cur_new_labels,
                    dim=0,
                )
            )


            cur_new_input_ids = (
                torch.cat(
                    cur_new_input_ids,
                    dim=0,
                )
            )


            new_input_embeds.append(
                cur_new_input_embeds
            )

            new_labels.append(
                cur_new_labels
            )

            new_input_ids.append(
                cur_new_input_ids
            )


        # ========================================================
        # Padding
        # ========================================================

        max_len = max(
            x.shape[0]
            for x in new_input_embeds
        )


        batch_size = len(
            new_input_embeds
        )


        new_input_embeds_padded = []


        new_labels_padded = torch.full(
            (
                batch_size,
                max_len,
            ),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )


        attention_mask = torch.zeros(
            (
                batch_size,
                max_len,
            ),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )


        position_ids = torch.zeros(
            (
                batch_size,
                max_len,
            ),
            dtype=position_ids.dtype,
            device=position_ids.device,
        )


        new_input_ids_padded = torch.full(
            (
                batch_size,
                max_len,
            ),
            -300,
            dtype=new_input_ids[0].dtype,
            device=new_input_ids[0].device,
        )


        for i, (
            cur_new_embed,
            cur_new_labels,
            cur_new_input_ids,
        ) in enumerate(
            zip(
                new_input_embeds,
                new_labels,
                new_input_ids,
            )
        ):

            cur_len = (
                cur_new_embed.shape[0]
            )


            new_input_embeds_padded.append(
                torch.cat(
                    (
                        cur_new_embed,

                        torch.zeros(
                            (
                                max_len
                                - cur_len,
                                cur_new_embed.shape[1],
                            ),
                            dtype=cur_new_embed.dtype,
                            device=cur_new_embed.device,
                        ),
                    ),
                    dim=0,
                )
            )


            if cur_len > 0:

                new_labels_padded[
                    i,
                    :cur_len
                ] = cur_new_labels


                attention_mask[
                    i,
                    :cur_len
                ] = True


                position_ids[
                    i,
                    :cur_len
                ] = torch.arange(
                    0,
                    cur_len,
                    dtype=position_ids.dtype,
                    device=position_ids.device,
                )


                new_input_ids_padded[
                    i,
                    :cur_len
                ] = cur_new_input_ids


        new_input_embeds = torch.stack(
            new_input_embeds_padded,
            dim=0,
        )


        if _labels is None:

            new_labels = None

        else:

            new_labels = new_labels_padded


        if _attention_mask is None:

            attention_mask = None

        else:

            attention_mask = attention_mask.to(
                dtype=_attention_mask.dtype
            )


        if _position_ids is None:

            position_ids = None


        return (
            None,
            position_ids,
            attention_mask,
            past_key_values,
            new_input_embeds,
            new_labels,
            None,
            None,
            None,
        )


    # ============================================================
    # Qwen2.5-VL input preparation
    # ============================================================

    def _prepare_qwen25vl_inputs(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        gen_images,
        und_images,
    ):

        # ========================================================
        # 1. Encode target / GT fusion image with VAE
        # ========================================================

        if gen_images is not None:

            vae = (
                self.get_model()
                .get_sana_vae()
            )


            vae_device = vae.device


            prompt_image_embeds = (
                vae.encode(
                    gen_images.to(
                        vae_device
                    )
                ).latent
            )


            prompt_image_embeds = (
                prompt_image_embeds
                * vae.config.scaling_factor
            )


            target_image_embeds = (
                torch.clone(
                    prompt_image_embeds
                ).detach()
            )

        else:

            target_image_embeds = None


        # ========================================================
        # 2. Qwen prompt
        # ========================================================

        qwen_prompt = getattr(
            self.config,
            "qwen25vl_prompt",
            (
                "Analyze the visible and infrared images. "
                "Identify salient targets, structural "
                "information, thermal information, boundaries, "
                "textures, and complementary information "
                "between the two modalities."
            ),
        )


        # ========================================================
        # 3. Visible / Infrared Qwen hidden states
        # ========================================================

        visible_hidden_states, infrared_hidden_states = (
            self.get_model()
            .extract_qwen25vl_condition(
                und_images,
                qwen_prompt,
            )
        )


        # ========================================================
        # 4. Text inputs for original Qwen2 language head
        #
        # Qwen2.5-VL 已经单独负责 image encoding。
        # 因此这里不能再走原 Mobile-O image embedding。
        #
        # 如果 input_ids 中存在 IMAGE_TOKEN_INDEX，
        # 将其删除。
        # ========================================================

        if attention_mask is None:

            attention_mask_bool = torch.ones_like(
                input_ids,
                dtype=torch.bool,
            )

        else:

            attention_mask_bool = (
                attention_mask.bool()
            )


        input_ids_list = []

        labels_list = []


        for b in range(
            input_ids.shape[0]
        ):

            cur_ids = (
                input_ids[
                    b
                ][
                    attention_mask_bool[
                        b
                    ]
                ]
            )


            if labels is not None:

                cur_labels = (
                    labels[
                        b
                    ][
                        attention_mask_bool[
                            b
                        ]
                    ]
                )

            else:

                cur_labels = None


            image_mask = (
                cur_ids
                == IMAGE_TOKEN_INDEX
            )


            keep = ~image_mask


            cur_ids = cur_ids[
                keep
            ]


            if cur_labels is not None:

                cur_labels = cur_labels[
                    keep
                ]


            input_ids_list.append(
                cur_ids
            )

            if cur_labels is not None:

                labels_list.append(
                    cur_labels
                )


        max_len = max(
            x.shape[0]
            for x in input_ids_list
        )


        batch_size = len(
            input_ids_list
        )


        new_input_ids = torch.zeros(
            (
                batch_size,
                max_len,
            ),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )


        new_attention_mask = torch.zeros(
            (
                batch_size,
                max_len,
            ),
            dtype=torch.bool,
            device=input_ids.device,
        )


        new_labels = None


        if labels is not None:

            new_labels = torch.full(
                (
                    batch_size,
                    max_len,
                ),
                IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )


        new_position_ids = torch.zeros(
            (
                batch_size,
                max_len,
            ),
            dtype=torch.long,
            device=input_ids.device,
        )


        for b in range(
            batch_size
        ):

            cur_len = (
                input_ids_list[b].shape[0]
            )


            new_input_ids[
                b,
                :cur_len
            ] = (
                input_ids_list[b]
            )


            new_attention_mask[
                b,
                :cur_len
            ] = True


            new_position_ids[
                b,
                :cur_len
            ] = torch.arange(
                cur_len,
                dtype=torch.long,
                device=input_ids.device,
            )


            if labels is not None:

                new_labels[
                    b,
                    :cur_len
                ] = labels_list[b]


        # ========================================================
        # Return exactly 9 fields
        # ========================================================

        return (
            new_input_ids,
            new_position_ids,
            new_attention_mask,
            past_key_values,
            None,
            new_labels,
            target_image_embeds,
            visible_hidden_states,
            infrared_hidden_states,
        )


    # ============================================================
    # Main multimodal preparation API
    # ============================================================

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        gen_images=None,
        und_images=None,
    ):

        use_qwen25vl = bool(
            getattr(
                self.get_model(),
                "use_qwen25vl",
                False,
            )
        )


        # ========================================================
        # Qwen2.5-VL path
        # ========================================================

        if use_qwen25vl:

            if und_images is None:

                raise ValueError(
                    "use_qwen25vl=True requires "
                    "und_images containing "
                    "Visible + Infrared images."
                )


            return (
                self._prepare_qwen25vl_inputs(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    labels=labels,
                    gen_images=gen_images,
                    und_images=und_images,
                )
            )


        # ========================================================
        # Original Mobile-O path
        # ========================================================

        if (
            gen_images is None
            and und_images is None
        ):

            return (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                None,
                labels,
                None,
                None,
                None,
            )


        if (
            input_ids is None
            or input_ids.shape[1] == 1
            or self.get_vision_tower() is None
        ):

            return (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                None,
                labels,
                None,
                None,
                None,
            )


        return (
            self._prepare_original_mobileo_inputs(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                labels=labels,
                und_images=und_images,
            )
        )


    # ============================================================
    # Vision tokenizer
    # ============================================================

    def initialize_vision_tokenizer(
        self,
        model_args,
        tokenizer,
    ):

        if model_args.mm_use_im_patch_token:

            tokenizer.add_tokens(
                [
                    DEFAULT_IMAGE_PATCH_TOKEN
                ],
                special_tokens=True,
            )

            self.resize_token_embeddings(
                len(tokenizer)
            )


        if model_args.mm_use_im_start_end:

            num_new_tokens = (
                tokenizer.add_tokens(
                    [
                        DEFAULT_IM_START_TOKEN,
                        DEFAULT_IM_END_TOKEN,
                    ],
                    special_tokens=True,
                )
            )


            self.resize_token_embeddings(
                len(tokenizer)
            )


            if num_new_tokens > 0:

                input_embeddings = (
                    self.get_input_embeddings()
                    .weight.data
                )

                output_embeddings = (
                    self.get_output_embeddings()
                    .weight.data
                )


                input_embeddings_avg = (
                    input_embeddings[
                        :-num_new_tokens
                    ]
                    .mean(
                        dim=0,
                        keepdim=True,
                    )
                )


                output_embeddings_avg = (
                    output_embeddings[
                        :-num_new_tokens
                    ]
                    .mean(
                        dim=0,
                        keepdim=True,
                    )
                )


                input_embeddings[
                    -num_new_tokens:
                ] = input_embeddings_avg


                output_embeddings[
                    -num_new_tokens:
                ] = output_embeddings_avg


            if model_args.tune_mm_mlp_adapter:

                for p in (
                    self.get_input_embeddings()
                    .parameters()
                ):

                    p.requires_grad = True


                for p in (
                    self.get_output_embeddings()
                    .parameters()
                ):

                    p.requires_grad = False


            if model_args.pretrain_mm_mlp_adapter:

                mm_projector_weights = torch.load(
                    model_args.pretrain_mm_mlp_adapter,
                    map_location="cpu",
                )


                embed_tokens_weight = (
                    mm_projector_weights[
                        "model.embed_tokens.weight"
                    ]
                )


                assert (
                    num_new_tokens == 2
                )


                if (
                    input_embeddings.shape
                    == embed_tokens_weight.shape
                ):

                    input_embeddings[
                        -num_new_tokens:
                    ] = embed_tokens_weight[
                        -num_new_tokens:
                    ]

                elif (
                    embed_tokens_weight.shape[0]
                    == num_new_tokens
                ):

                    input_embeddings[
                        -num_new_tokens:
                    ] = embed_tokens_weight

                else:

                    raise ValueError(
                        "Unexpected "
                        "embed_tokens_weight shape. "
                        f"Pretrained: "
                        f"{embed_tokens_weight.shape}. "
                        f"Current: "
                        f"{input_embeddings.shape}. "
                        f"Number of new tokens: "
                        f"{num_new_tokens}."
                    )


        elif model_args.mm_use_im_patch_token:

            if (
                model_args.tune_mm_mlp_adapter
            ):

                for p in (
                    self.get_input_embeddings()
                    .parameters()
                ):

                    p.requires_grad = False


                for p in (
                    self.get_output_embeddings()
                    .parameters()
                ):

                    p.requires_grad = False