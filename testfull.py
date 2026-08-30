# -*- coding: utf-8 -*-

"""
testfull.py

RTX 3060 Laptop 6GB 显存友好的完整接口测试。

Pipeline:

    Paired Visible / Infrared
              |
              v
    Synchronized Random Crop
              |
              v
        Qwen2.5-VL-3B
              |
              v
    Visible / Infrared Hidden
              |
       Hidden -> CPU
              |
       Release Qwen GPU
              |
              v
           QSCFMCP
              |
              v
       [B, N, 2304]
              |
              v
         SANA-0.6B
              |
              v
        [B, 32, 16, 16]
              |
              v
          SANA VAE
              |
              v
         decoded image


说明：

1. 当前只是 forward / interface 测试。
2. QSCFMCP 尚未训练，所以生成结果没有融合意义。
3. Visible / Infrared 使用完全相同的 crop box。
4. 当前 batch_size = 1。
5. Qwen / QSCFMCP / SANA 不会同时长期驻留 GPU。
6. Qwen visual token 使用 384x384 输入限制。
7. QSCFMCP 使用 float32 做数值稳定性测试。
8. SANA 使用 float16。
"""

from __future__ import annotations

import gc
import os
import random
from pathlib import Path

import numpy as np
import torch

from PIL import Image

from diffusers import (
    SanaTransformer2DModel,
    AutoencoderDC,
    FlowMatchEulerDiscreteScheduler,
)

from mobileo.model.qwen25vl_wrapper import (
    Qwen25VLWrapper,
)

from mobileo.model.qscfmcp import (
    QSCFMCP,
)


# ============================================================
# 1. Configuration
# ============================================================

QWEN_PATH = (
    r"E:\ai for science\omnifuse\models"
    r"\Qwen\Qwen2___5-VL-3B-Instruct"
)


SANA_PATH = (
    r"D:\Mobile-O-main"
    r"\mobileo\model\Sana\Sana_600M_512px_diffusers"
)


# ------------------------------------------------------------
# MSRS test pair
# ------------------------------------------------------------

VISIBLE_PATH = (
    r"E:\ai for science\SeAFusion-main\MSRS"
    r"\Visible\train\MSRS\00001D.png"
)


INFRARED_PATH = (
    r"E:\ai for science\SeAFusion-main\MSRS"
    r"\Infrared\train\MSRS\00001D.png"
)


# ------------------------------------------------------------
# Output directory
# ------------------------------------------------------------

OUTPUT_DIR = Path(
    r"D:\Mobile-O-main\test_outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


VISIBLE_CROP_PATH = (
    OUTPUT_DIR / "visible_crop.png"
)

INFRARED_CROP_PATH = (
    OUTPUT_DIR / "infrared_crop.png"
)

OUTPUT_PATH = (
    OUTPUT_DIR / "full_pipeline_output.png"
)


# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

DEVICE = "cuda"

GPU_ID = 0


# ------------------------------------------------------------
# Precision
# ------------------------------------------------------------

QWEN_DTYPE = torch.bfloat16

QSCFMCP_DTYPE = torch.float32

SANA_DTYPE = torch.float32


# ------------------------------------------------------------
# Qwen
# ------------------------------------------------------------

QWEN_LAST_N_LAYERS = 8


# ------------------------------------------------------------
# Paired crop
# ------------------------------------------------------------

CROP_SIZE = 512


RANDOM_SEED = 3407


# ------------------------------------------------------------
# Qwen processor token budget
#
# 384 x 384 的限制主要用于降低视觉 token。
# ------------------------------------------------------------

QWEN_MIN_PIXELS = (
    256 * 28 * 28
)

QWEN_MAX_PIXELS = (
    CROP_SIZE * CROP_SIZE
)


# ------------------------------------------------------------
# Prompt
# ------------------------------------------------------------

PROMPT = (
    "Analyze the visible and infrared images. "
    "Identify salient targets, structural information, "
    "thermal information, boundaries, textures, and "
    "complementary information between the two modalities."
)


# ============================================================
# Utility
# ============================================================

def print_title(
    title: str,
) -> None:

    print(
        "\n"
        + "=" * 80
    )

    print(title)

    print(
        "=" * 80
    )


def check_file(
    path: str,
    name: str,
) -> None:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"{name} does not exist:\n"
            f"{path}"
        )


def cuda_memory_report(
    tag: str,
) -> None:

    if not torch.cuda.is_available():

        return

    allocated = (
        torch.cuda.memory_allocated(
            GPU_ID
        )
        / 1024**3
    )

    reserved = (
        torch.cuda.memory_reserved(
            GPU_ID
        )
        / 1024**3
    )

    max_allocated = (
        torch.cuda.max_memory_allocated(
            GPU_ID
        )
        / 1024**3
    )

    print(
        f"[CUDA] {tag} | "
        f"allocated={allocated:.3f} GB | "
        f"reserved={reserved:.3f} GB | "
        f"max={max_allocated:.3f} GB"
    )


def cleanup_cuda(
    *objects,
) -> None:

    """
    删除给定对象，然后进行：

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    """

    for obj in objects:

        try:
            del obj

        except Exception:
            pass


    gc.collect()


    if torch.cuda.is_available():

        torch.cuda.empty_cache()

        try:
            torch.cuda.ipc_collect()

        except Exception:
            pass


def load_image(
    path: str,
) -> Image.Image:

    return Image.open(
        path
    ).convert(
        "RGB"
    )


def resize_and_pad_pair(
    visible: Image.Image,
    infrared: Image.Image,
    target_size: int = 512,
    fill_value: int = 0,
):
    """
    对已经配准的 Visible / Infrared 图像执行完全一致的：

        1. 等比例 resize
        2. 对称 padding

    最终得到：
        [target_size, target_size]

    不改变原始长宽比。
    """

    if visible.size != infrared.size:
        raise ValueError(
            "Visible / Infrared image size mismatch:\n"
            f"Visible  : {visible.size}\n"
            f"Infrared : {infrared.size}"
        )

    width, height = visible.size

    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid image size: {visible.size}"
        )

    # --------------------------------------------------------
    # 保持纵横比
    # --------------------------------------------------------

    scale = min(
        target_size / width,
        target_size / height,
    )

    new_width = max(
        1,
        int(round(width * scale)),
    )

    new_height = max(
        1,
        int(round(height * scale)),
    )

    # --------------------------------------------------------
    # Visible / IR 完全相同 resize
    # --------------------------------------------------------

    visible_resized = visible.resize(
        (new_width, new_height),
        Image.Resampling.BICUBIC,
    )

    infrared_resized = infrared.resize(
        (new_width, new_height),
        Image.Resampling.BICUBIC,
    )

    # --------------------------------------------------------
    # 对称 padding
    # --------------------------------------------------------

    pad_left = (
        target_size - new_width
    ) // 2

    pad_right = (
        target_size
        - new_width
        - pad_left
    )

    pad_top = (
        target_size - new_height
    ) // 2

    pad_bottom = (
        target_size
        - new_height
        - pad_top
    )

    padding = (
        pad_left,
        pad_top,
        pad_right,
        pad_bottom,
    )

    visible_padded = Image.new(
        "RGB",
        (
            target_size,
            target_size,
        ),
        color=(
            fill_value,
            fill_value,
            fill_value,
        ),
    )

    infrared_padded = Image.new(
        "RGB",
        (
            target_size,
            target_size,
        ),
        color=(
            fill_value,
            fill_value,
            fill_value,
        ),
    )

    visible_padded.paste(
        visible_resized,
        (
            pad_left,
            pad_top,
        ),
    )

    infrared_padded.paste(
        infrared_resized,
        (
            pad_left,
            pad_top,
        ),
    )

    return (
        visible_padded,
        infrared_padded,
        {
            "original_size": (
                width,
                height,
            ),
            "resized_size": (
                new_width,
                new_height,
            ),
            "padding": padding,
            "scale": scale,
        },
    )


def pil_to_tensor(
    image: Image.Image,
) -> torch.Tensor:

    """
    PIL RGB
        ->
    [1,3,H,W]
        ->
    [-1,1]
    """

    image = image.convert(
        "RGB"
    )


    array = np.asarray(
        image
    )


    tensor = torch.from_numpy(
        array
    ).float()


    tensor = (
        tensor / 255.0
    )


    tensor = (
        tensor * 2.0
        - 1.0
    )


    tensor = (
        tensor.permute(
            2,
            0,
            1,
        )
        .unsqueeze(0)
    )


    return tensor


def print_tensor(
    name: str,
    tensor: torch.Tensor,
) -> None:

    print(
        f"{name:<40}"
        f"shape={str(tuple(tensor.shape)):<28}"
        f"dtype={str(tensor.dtype):<16}"
        f"device={str(tensor.device)}"
    )


def print_hidden_state_summary(
    visible_hidden_states,
    infrared_hidden_states,
    selected_layers,
) -> None:

    for idx, (
        visible,
        infrared,
    ) in enumerate(
        zip(
            visible_hidden_states,
            infrared_hidden_states,
        )
    ):

        layer = selected_layers[
            idx
        ]

        print_tensor(
            f"Layer {layer} Visible",
            visible,
        )

        print_tensor(
            f"Layer {layer} Infrared",
            infrared,
        )


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # 0. Environment
    # ========================================================

    print_title(
        "0. Environment"
    )


    print(
        "PyTorch:",
        torch.__version__,
    )


    print(
        "CUDA:",
        torch.version.cuda,
    )


    print(
        "CUDA available:",
        torch.cuda.is_available(),
    )


    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available."
        )


    print(
        "GPU:",
        torch.cuda.get_device_name(
            GPU_ID
        ),
    )


    total_memory = (
        torch.cuda.get_device_properties(
            GPU_ID
        ).total_memory
        / 1024**3
    )


    print(
        f"GPU memory: "
        f"{total_memory:.2f} GB"
    )


    cuda_memory_report(
        "startup"
    )


    # ========================================================
    # 1. Check paths
    # ========================================================

    print_title(
        "1. Check paths"
    )


    check_file(
        QWEN_PATH,
        "Qwen2.5-VL",
    )


    check_file(
        SANA_PATH,
        "SANA",
    )


    check_file(
        VISIBLE_PATH,
        "Visible image",
    )


    check_file(
        INFRARED_PATH,
        "Infrared image",
    )


    print(
        "Qwen:",
        QWEN_PATH,
    )


    print(
        "SANA:",
        SANA_PATH,
    )


    print(
        "Visible:",
        VISIBLE_PATH,
    )


    print(
        "Infrared:",
        INFRARED_PATH,
    )


    # ========================================================
    # 2. Load and paired crop images
    # ========================================================

    print_title(
        "2. Paired random crop"
    )


    visible_full = load_image(
        VISIBLE_PATH
    )


    infrared_full = load_image(
        INFRARED_PATH
    )


    print(
        "Original size:",
        visible_full.size,
    )

    (
        visible_crop,
        infrared_crop,
        preprocess_info,
    ) = resize_and_pad_pair(
        visible=visible_full,
        infrared=infrared_full,
        target_size=512,
        fill_value=0,
    )

    print(
        "Original size:",
        preprocess_info["original_size"],
    )

    print(
        "Resized size:",
        preprocess_info["resized_size"],
    )

    print(
        "Padding:",
        preprocess_info["padding"],
    )

    print(
        "Scale:",
        preprocess_info["scale"],
    )

    print(
        "Final size:",
        visible_crop.size,
    )


    # 原图及时释放
    del visible_full
    del infrared_full

    gc.collect()


    # ========================================================
    # 3. Load Qwen
    # ========================================================

    print_title(
        "3. Load Qwen2.5-VL-3B"
    )


    print(
        "Qwen dtype:",
        QWEN_DTYPE,
    )


    print(
        "Qwen crop:",
        CROP_SIZE,
        "x",
        CROP_SIZE,
    )


    print(
        "Qwen min_pixels:",
        QWEN_MIN_PIXELS,
    )


    print(
        "Qwen max_pixels:",
        QWEN_MAX_PIXELS,
    )


    """
    注意：

    qwen25vl_wrapper.py 必须已经修改为：

        device_map="auto"

    并删除：

        self.model.to(device)

    否则 6GB GPU 仍然可能 OOM。
    """

    qwen = Qwen25VLWrapper(

        model_path=QWEN_PATH,

        last_n_layers=QWEN_LAST_N_LAYERS,

        torch_dtype=QWEN_DTYPE,

        device=DEVICE,

        freeze=True,

        min_pixels=QWEN_MIN_PIXELS,

        max_pixels=QWEN_MAX_PIXELS,
    )


    cuda_memory_report(
        "after loading Qwen"
    )


    # ========================================================
    # 4. Qwen forward
    # ========================================================

    print_title(
        "4. Qwen2.5-VL forward"
    )


    with torch.no_grad():

        qwen_output = qwen(

            visible=visible_crop,

            infrared=infrared_crop,

            prompt=PROMPT,

            return_all_hidden_states=False,
        )


    visible_hidden_states_gpu = (
        qwen_output[
            "visible_hidden_states"
        ]
    )


    infrared_hidden_states_gpu = (
        qwen_output[
            "infrared_hidden_states"
        ]
    )


    print_hidden_state_summary(
        visible_hidden_states_gpu,
        infrared_hidden_states_gpu,
        qwen.selected_layers,
    )


    if len(
        visible_hidden_states_gpu
    ) != QWEN_LAST_N_LAYERS:

        raise RuntimeError(
            "Unexpected number of "
            "Visible hidden-state layers."
        )


    if len(
        infrared_hidden_states_gpu
    ) != QWEN_LAST_N_LAYERS:

        raise RuntimeError(
            "Unexpected number of "
            "Infrared hidden-state layers."
        )


    # ========================================================
    # 5. Move hidden states to CPU
    # ========================================================

    print_title(
        "5. Move Qwen hidden states to CPU"
    )


    """
    Qwen3B 本体后面不再需要。

    所以必须：

        GPU hidden
            ↓
        CPU hidden
            ↓
        delete Qwen
            ↓
        empty_cache

    否则 SANA 加载时会再次 OOM。
    """


    visible_hidden_states = [

        x.detach()
        .to(
            device="cpu",
            dtype=torch.float32,
        )

        for x in visible_hidden_states_gpu
    ]


    infrared_hidden_states = [

        x.detach()
        .to(
            device="cpu",
            dtype=torch.float32,
        )

        for x in infrared_hidden_states_gpu
    ]


    del qwen_output
    del visible_hidden_states_gpu
    del infrared_hidden_states_gpu
    del qwen


    gc.collect()


    torch.cuda.empty_cache()


    try:

        torch.cuda.ipc_collect()

    except Exception:

        pass


    cuda_memory_report(
        "after releasing Qwen"
    )


    # ========================================================
    # 6. QSCFMCP
    # ========================================================

    print_title(
        "6. QSCFMCP forward"
    )


    qscfmcp = QSCFMCP(

        input_dim=2048,

        hidden_dim=512,

        output_dim=2304,

        num_layers=QWEN_LAST_N_LAYERS,

        num_heads=8,

        state_dim=16,

        ssm_expand=2,

        dropout=0.0,
    )


    qscfmcp = qscfmcp.to(
        device=DEVICE,
        dtype=QSCFMCP_DTYPE,
    )


    qscfmcp.eval()


    total_params = sum(
        p.numel()
        for p in qscfmcp.parameters()
    )


    print(
        "QSCFMCP total parameters:",
        f"{total_params:,}",
    )


    print(
        "QSCFMCP dtype:",
        QSCFMCP_DTYPE,
    )


    # --------------------------------------------------------
    # Move hidden states to GPU one layer at a time.
    #
    # 这里不用把所有 CPU hidden 一次性复制成独立 GPU
    # list 后长期保存，而是统一在 connector 输入阶段复制。
    # --------------------------------------------------------

    visible_connector_states = [

        x.to(
            device=DEVICE,
            dtype=QSCFMCP_DTYPE,
        )

        for x in visible_hidden_states
    ]


    infrared_connector_states = [

        x.to(
            device=DEVICE,
            dtype=QSCFMCP_DTYPE,
        )

        for x in infrared_hidden_states
    ]


    print_hidden_state_summary(
        visible_connector_states,
        infrared_connector_states,
        list(
            range(
                29,
                29 + QWEN_LAST_N_LAYERS,
            )
        ),
    )


    cuda_memory_report(
        "before QSCFMCP forward"
    )


    with torch.no_grad():

        condition = qscfmcp(

            visible_hidden_states=(
                visible_connector_states
            ),

            infrared_hidden_states=(
                infrared_connector_states
            ),
        )


    print_tensor(
        "QSCFMCP condition",
        condition,
    )


    # --------------------------------------------------------
    # Critical checks
    # --------------------------------------------------------

    if condition.ndim != 3:

        raise RuntimeError(
            "QSCFMCP output must be "
            "[B,N,2304]."
        )


    if condition.shape[0] != 1:

        raise RuntimeError(
            "Current test requires "
            "batch_size=1."
        )


    if condition.shape[-1] != 2304:

        raise RuntimeError(
            "QSCFMCP output dimension "
            "must be 2304."
        )


    expected_condition_tokens = (
        visible_connector_states[0]
        .shape[1]
        +
        infrared_connector_states[0]
        .shape[1]
    )


    if (
        condition.shape[1]
        != expected_condition_tokens
    ):

        raise RuntimeError(
            "QSCFMCP changed the number of "
            "condition tokens unexpectedly.\n"
            f"Expected: "
            f"{expected_condition_tokens}\n"
            f"Got: "
            f"{condition.shape[1]}"
        )


    print(
        "✓ QSCFMCP output dimension = 2304"
    )


    print(
        "✓ QSCFMCP token count preserved"
    )


    # --------------------------------------------------------
    # Release QSCFMCP input copies.
    # --------------------------------------------------------

    del visible_connector_states
    del infrared_connector_states
    del qscfmcp


    gc.collect()
    torch.cuda.empty_cache()


    try:

        torch.cuda.ipc_collect()

    except Exception:

        pass


    cuda_memory_report(
        "after releasing QSCFMCP"
    )


    # --------------------------------------------------------
    # CPU hidden states are no longer needed after condition.
    # --------------------------------------------------------

    del visible_hidden_states
    del infrared_hidden_states

    gc.collect()


    # ========================================================
    # 7. Load SANA Transformer
    # ========================================================

    print_title(
        "7. Load SANA Transformer"
    )


    print(
        "Loading SANA Transformer..."
    )

    dit = (
        SanaTransformer2DModel
        .from_pretrained(
            SANA_PATH,
            subfolder="transformer",
            variant="fp16",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            ignore_mismatched_sizes=False,
        )
    )


    dit = dit.to(
        DEVICE
    )


    dit.eval()


    dit_config = dit.config


    print(
        "SANA in_channels:",
        dit_config.in_channels,
    )


    print(
        "SANA out_channels:",
        dit_config.out_channels,
    )


    print(
        "SANA sample_size:",
        dit_config.sample_size,
    )


    print(
        "SANA caption_channels:",
        dit_config.caption_channels,
    )


    # --------------------------------------------------------
    # Validate condition interface
    # --------------------------------------------------------

    if int(
        dit_config.caption_channels
    ) != 2304:

        raise RuntimeError(
            "SANA caption_channels "
            "must be 2304."
        )


    print(
        "✓ SANA caption_channels = 2304"
    )


    # --------------------------------------------------------
    # Condition to SANA dtype
    # --------------------------------------------------------

    condition_for_sana = (
        condition.to(
            device=DEVICE,
            dtype=SANA_DTYPE,
        )
    )


    del condition


    gc.collect()


    # ========================================================
    # 8. Load VAE
    # ========================================================

    print_title(
        "8. Load SANA VAE"
    )

    vae = (
        AutoencoderDC
        .from_pretrained(
            SANA_PATH,
            subfolder="vae",
            variant="fp16",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
    )


    vae = vae.to(
        DEVICE
    )


    vae.eval()


    cuda_memory_report(
        "after loading SANA + VAE"
    )


    # ========================================================
    # 9. Prepare Visible latent
    # ========================================================

    print_title(
        "9. VAE encode"
    )


    visible_tensor = (
        pil_to_tensor(
            visible_crop
        )
        .to(
            device=DEVICE,
            dtype=SANA_DTYPE,
        )
    )


    infrared_tensor = (
        pil_to_tensor(
            infrared_crop
        )
        .to(
            device=DEVICE,
            dtype=SANA_DTYPE,
        )
    )


    print_tensor(
        "Visible tensor",
        visible_tensor,
    )


    print_tensor(
        "Infrared tensor",
        infrared_tensor,
    )


    # --------------------------------------------------------
    # VAE encode
    #
    # 当前只使用 Visible latent 验证 SANA 输入接口。
    # 真正 fusion training 后续应使用：
    #
    # target fusion image -> VAE -> target latent
    #
    # --------------------------------------------------------

    with torch.no_grad():

        vae_output = vae.encode(
            visible_tensor
        )


    if hasattr(
        vae_output,
        "latent",
    ):

        latent = (
            vae_output.latent
        )

    elif hasattr(
        vae_output,
        "latent_dist",
    ):

        latent = (
            vae_output
            .latent_dist
            .sample()
        )

    else:

        raise RuntimeError(
            "Unable to find VAE latent."
        )


    scaling_factor = getattr(
        vae.config,
        "scaling_factor",
        None,
    )


    if scaling_factor is not None:

        latent = (
            latent
            * scaling_factor
        )


    latent = latent.to(
        dtype=SANA_DTYPE
    )


    print_tensor(
        "VAE latent",
        latent,
    )


    expected_channels = int(
        dit_config.in_channels
    )


    expected_size = int(
        dit_config.sample_size
    )


    if latent.shape[1] != (
        expected_channels
    ):

        raise RuntimeError(
            "VAE latent channels mismatch.\n"
            f"Expected: {expected_channels}\n"
            f"Actual:   {latent.shape[1]}"
        )


    if (
        latent.shape[2]
        != expected_size
        or
        latent.shape[3]
        != expected_size
    ):

        raise RuntimeError(
            "VAE latent spatial size mismatch.\n"
            f"Expected: "
            f"[B,{expected_channels},"
            f"{expected_size},{expected_size}]\n"
            f"Actual: "
            f"{tuple(latent.shape)}"
        )


    print(
        "✓ VAE latent is compatible "
        "with SANA."
    )


    # --------------------------------------------------------
    # VAE no longer needed for SANA forward.
    # --------------------------------------------------------

    del vae
    del visible_tensor
    del infrared_tensor

    gc.collect()


    torch.cuda.empty_cache()


    try:

        torch.cuda.ipc_collect()

    except Exception:

        pass


    cuda_memory_report(
        "after releasing VAE"
    )


    # ========================================================
    # 10. SANA scheduler
    # ========================================================

    print_title(
        "10. FlowMatch scheduler"
    )


    scheduler = (
        FlowMatchEulerDiscreteScheduler
        .from_pretrained(
            SANA_PATH,
            subfolder="scheduler",
        )
    )


    scheduler.set_timesteps(
        num_inference_steps=20,
        device=DEVICE,
    )


    timesteps = (
        scheduler.timesteps[:1]
        .to(DEVICE)
    )


    print(
        "Test timestep:",
        timesteps,
    )


    # ========================================================
    # 11. Build noisy latent
    # ========================================================

    print_title(
        "11. Build noisy latent"
    )


    noise = torch.randn_like(
        latent
    )


    noisy_latent = (
        0.5 * latent
        +
        0.5 * noise
    )


    print_tensor(
        "Latent",
        latent,
    )


    print_tensor(
        "Noise",
        noise,
    )


    print_tensor(
        "Noisy latent",
        noisy_latent,
    )


    # ========================================================
    # 12. Condition mask
    # ========================================================

    print_title(
        "12. SANA condition mask"
    )


    condition_attention_mask = (
        torch.ones(
            condition_for_sana.shape[0],
            condition_for_sana.shape[1],
            dtype=torch.bool,
            device=DEVICE,
        )
    )


    print_tensor(
        "Condition mask",
        condition_attention_mask,
    )


    # ========================================================
    # 13. SANA forward
    # ========================================================

    print_title(
        "13. SANA Transformer forward"
    )


    cuda_memory_report(
        "before SANA forward"
    )


    print(
        "Running SANA..."
    )


    with torch.no_grad():

        with torch.autocast(
            device_type="cuda",
            dtype=SANA_DTYPE,
        ):

            sana_output = dit(

                hidden_states=(
                    noisy_latent
                ),

                timestep=(
                    timesteps
                ),

                encoder_hidden_states=(
                    condition_for_sana
                ),

                encoder_attention_mask=(
                    condition_attention_mask
                ),
            )


    if not hasattr(
        sana_output,
        "sample",
    ):

        raise RuntimeError(
            "SANA output does not contain "
            "'sample'."
        )


    diffusion_pred = (
        sana_output.sample
    )


    print_tensor(
        "SANA diffusion output",
        diffusion_pred,
    )


    if (
        diffusion_pred.shape
        != noisy_latent.shape
    ):

        raise RuntimeError(
            "SANA output shape mismatch.\n"
            f"Input:  "
            f"{tuple(noisy_latent.shape)}\n"
            f"Output: "
            f"{tuple(diffusion_pred.shape)}"
        )


    print(
        "✓ SANA forward succeeded."
    )


    cuda_memory_report(
        "after SANA forward"
    )


    # ========================================================
    # 14. Release SANA transformer
    # ========================================================

    print_title(
        "14. Release SANA Transformer"
    )


    del dit
    del condition_for_sana
    del condition_attention_mask
    del sana_output
    del diffusion_pred
    del noise
    del noisy_latent
    del timesteps
    del scheduler


    gc.collect()

    torch.cuda.empty_cache()


    try:

        torch.cuda.ipc_collect()

    except Exception:

        pass


    cuda_memory_report(
        "after releasing SANA"
    )


    # ========================================================
    # 15. Reload VAE for decode
    # ========================================================

    print_title(
        "15. Reload VAE for decode"
    )

    vae = (
        AutoencoderDC
        .from_pretrained(
            SANA_PATH,
            subfolder="vae",
            variant="fp16",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
    )


    vae = vae.to(
        DEVICE
    )


    vae.eval()


    # ========================================================
    # 16. Decode latent
    # ========================================================

    print_title(
        "16. VAE decode"
    )


    with torch.no_grad():

        decoded = vae.decode(
            latent
        )


    if hasattr(
        decoded,
        "sample",
    ):

        decoded_image = (
            decoded.sample
        )

    else:

        raise RuntimeError(
            "Unable to find decoded image "
            "in VAE output."
        )


    print_tensor(
        "Decoded image",
        decoded_image,
    )


    # ========================================================
    # 17. Save decoded image
    # ========================================================

    print_title(
        "17. Save output"
    )


    output = (
        decoded_image
        .float()
        .cpu()
    )


    output = (
        output.clamp(
            -1.0,
            1.0,
        )
        + 1.0
    ) / 2.0


    output = (
        output[0]
        .permute(
            1,
            2,
            0,
        )
        .numpy()
    )


    output = (
        output * 255.0
    ).clip(
        0,
        255,
    ).astype(
        np.uint8
    )


    output_image = (
        Image.fromarray(
            output
        )
    )


    output_image.save(
        OUTPUT_PATH
    )


    print(
        "Output saved:",
        OUTPUT_PATH,
    )


    # ========================================================
    # 18. Final
    # ========================================================

    print_title(
        "FULL PIPELINE TEST PASSED"
    )


    print(
        "✓ CUDA available"
    )

    print(
        "✓ Paired random crop"
    )

    print(
        "✓ Qwen2.5-VL loaded"
    )

    print(
        "✓ Visible visual tokens extracted"
    )

    print(
        "✓ Infrared visual tokens extracted"
    )

    print(
        "✓ Qwen released before SANA"
    )

    print(
        "✓ QSCFMCP forward succeeded"
    )

    print(
        "✓ QSCFMCP output = 2304 condition"
    )

    print(
        "✓ SANA Transformer loaded"
    )

    print(
        "✓ SANA forward succeeded"
    )

    print(
        "✓ SANA VAE encode succeeded"
    )

    print(
        "✓ SANA VAE decode succeeded"
    )


    print(
        "\n"
        "注意："
    )


    print(
        "当前输出图像只是 VAE reconstruction "
        "interface test，不是融合结果。"
    )


    print(
        "QSCFMCP 仍是随机初始化，"
        "尚未进行 IVIF diffusion training。"
    )


    print(
        "\nOutput:"
    )


    print(
        OUTPUT_PATH
    )


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    main()