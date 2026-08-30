# lmms_eval/models/showo_hf.py
# Show-O wrapper for lmms-eval with config-file support and full device safety

import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
from accelerate import Accelerator, DistributedType
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPImageProcessor

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

warnings.filterwarnings("ignore")
from loguru import logger as eval_logger

# --- Show-O specific imports ---
from .models import Showo, MAGVITv2, CLIPVisionTower
from .training.prompting_utils import (
    UniversalPrompting,
    create_attention_mask_for_mmu,
    create_attention_mask_for_mmu_vit,
)
from .training.utils import image_transform
from omegaconf import OmegaConf

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)
SYSTEM_PROMPT_LEN = 28


def _to_list(x):
    return x if isinstance(x, (list, tuple)) else [x]


@register_model("showo")
class ShowoLMMS(lmms):
    """
    Show-O model wrapper for lmms-eval

    Example:
    accelerate launch --num_processes=8 --main_process_port 12345 -m lmms_eval \
      --model showo \
      --model_args "config_file=/path/to/configs/showo_demo_512x512.yaml" \
      --tasks mmmu_val \
      --batch_size 1 \
      --output_path ./logs/ \
      --log_samples
    """

    def __init__(self, config_file: Optional[str] = None, **kwargs):
        super().__init__()

        # ------------------------------------------------------------------
        # Load Show-O config if provided
        # ------------------------------------------------------------------
        if config_file is not None:
            cfg = OmegaConf.load(config_file)
            cfg_flat = {
                "showo_pretrained_model_path": cfg.model.showo.pretrained_model_path,
                "llm_model_path": cfg.model.showo.llm_model_path,
                "vq_model_type": cfg.model.vq_model.type,
                "vq_model_name": cfg.model.vq_model.vq_model_name,
                "w_clip_vit": cfg.model.showo.w_clip_vit,
                "resolution": cfg.dataset.params.resolution,
                "temperature": 0.8,
                "top_k": 1,
                "max_new_tokens": 512,
                "batch_size": 1,
                "device": "cuda",
            }
            for k, v in kwargs.items():
                cfg_flat[k] = v
            kwargs = cfg_flat

        # ------------------------------------------------------------------
        # Extract arguments
        # ------------------------------------------------------------------
        showo_pretrained_model_path = kwargs.pop("showo_pretrained_model_path")
        llm_model_path = kwargs.pop("llm_model_path")
        vq_model_type = kwargs.pop("vq_model_type", "magvitv2")
        vq_model_name = kwargs.pop("vq_model_name", "showlab/magvitv2")
        w_clip_vit = bool(kwargs.pop("w_clip_vit", True))
        resolution = int(kwargs.pop("resolution", 512))
        temperature = float(kwargs.pop("temperature", 0.8))
        top_k = int(kwargs.pop("top_k", 1))
        max_new_tokens = int(kwargs.pop("max_new_tokens", 512))
        batch_size = int(kwargs.pop("batch_size", 1))
        device = kwargs.pop("device", "cuda")
        dtype = kwargs.pop("dtype", "auto")
        device_map = kwargs.pop("device_map", "")
        trust_remote_code = kwargs.pop("trust_remote_code", False)

        accelerator = Accelerator()
        if accelerator.num_processes > 1 and device_map == "":
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        else:
            self._device = torch.device(device)

        if isinstance(dtype, str) and dtype != "auto":
            dtype = getattr(torch, dtype)
        self._dtype = dtype

        # ------------------------------------------------------------------
        # Tokenizer + prompting
        # ------------------------------------------------------------------
        self._tokenizer = AutoTokenizer.from_pretrained(
            llm_model_path, padding_side="left", trust_remote_code=trust_remote_code
        )
        self.uni_prompting = UniversalPrompting(
            self._tokenizer,
            max_text_len=4096,
            special_tokens=(
                "<|soi|>",
                "<|eoi|>",
                "<|sov|>",
                "<|eov|>",
                "<|t2i|>",
                "<|mmu|>",
                "<|t2v|>",
                "<|v2v|>",
                "<|lvg|>",
            ),
            ignore_id=-100,
            cond_dropout_prob=0.0,
        )

        # ------------------------------------------------------------------
        # Load VQ model (MAGVITv2)
        # ------------------------------------------------------------------
        if vq_model_type.lower() != "magvitv2":
            raise ValueError(f"Unsupported vq_model_type: {vq_model_type}")
        self.vq_model = MAGVITv2.from_pretrained(vq_model_name).to(self._device)
        self.vq_model.eval()
        self.vq_model.requires_grad_(False)

        # ------------------------------------------------------------------
        # Vision tower
        # ------------------------------------------------------------------
        if w_clip_vit:
            self.vision_tower = CLIPVisionTower("openai/clip-vit-large-patch14-336").to(self._device)
            self.vision_tower.eval()
            self.clip_image_processor = CLIPImageProcessor.from_pretrained(
                "openai/clip-vit-large-patch14-336"
            )
        else:
            self.vision_tower = None
            self.clip_image_processor = None

        # ------------------------------------------------------------------
        # Show-O core
        # ------------------------------------------------------------------
        self.model = Showo.from_pretrained(showo_pretrained_model_path).to(self._device)
        self.model.eval()

        # Generation params
        self.w_clip_vit = w_clip_vit
        self.resolution = resolution
        self.batch_size_per_gpu = batch_size
        self.temperature = temperature
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens

        eval_logger.info(f"Using device: {self._device}")
        self.accelerator = accelerator
        self._rank = accelerator.local_process_index
        self._world_size = accelerator.num_processes

    # ------------------------------------------------------------------
    # Base properties
    # ------------------------------------------------------------------
    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def eot_token_id(self):
        return self._tokenizer.eos_token_id

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None):
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self._tokenizer.encode(string, add_special_tokens=add_special_tokens)
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        return self._tokenizer.decode(tokens)

    def _prep_images_for_showo(self, pil_images: List[PIL.Image.Image]):
        device = self._device
        image_tensors = []
        for img in pil_images:
            image = image_transform(img.convert("RGB"), resolution=self.resolution).to(device)
            image_tensors.append(image.unsqueeze(0))
        img_batch = torch.cat(image_tensors, dim=0) if image_tensors else None

        image_tokens, clip_pixel_values, image_embeddings = None, None, None
        if img_batch is not None:
            with torch.no_grad():
                image_tokens = self.vq_model.get_code(img_batch).to(device) + len(self.uni_prompting.text_tokenizer)
        if self.w_clip_vit and pil_images:
            clip_pixel_values = torch.stack(
                [
                    self.clip_image_processor.preprocess(im, return_tensors="pt")["pixel_values"][0]
                    for im in pil_images
                ],
                dim=0,
            ).to(device)
            with torch.no_grad():
                vis_feats = self.vision_tower(clip_pixel_values)
                if hasattr(self.model, "mm_projector") and self.w_clip_vit:
                    image_embeddings = self.model.mm_projector(vis_feats)
                else:
                    eval_logger.debug("Show-O: w_clip_vit=False or mm_projector missing; using CLIP features directly.")
                    image_embeddings = vis_feats
        return image_tokens, clip_pixel_values, image_embeddings

    # ------------------------------------------------------------------
    # loglikelihood stub
    # ------------------------------------------------------------------
    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        eval_logger.warning("loglikelihood not implemented for Show-O; returning neutral values.")
        return [(0.0, False) for _ in requests]

    # ------------------------------------------------------------------
    # generate_until
    # ------------------------------------------------------------------
    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = list(re_ords.get_batched(n=self.batch_size, batch_fn=None))
        num_iters = len(chunks)
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals_nested = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = [v for group in visuals_nested for v in _to_list(group)]
            task_type = "text" if len(visuals) == 0 else "image"

            gen_kwargs = all_gen_kwargs[0] if all_gen_kwargs else {}
            max_new_tokens = int(gen_kwargs.get("max_new_tokens", self.max_new_tokens))
            top_k = int(gen_kwargs.get("top_k", self.top_k))

            assert self.batch_size_per_gpu == 1, "Show-O wrapper supports batch_size_per_gpu == 1"

            context = contexts[0]
            if DEFAULT_IMAGE_TOKEN not in context and task_type != "text":
                context = f"{DEFAULT_IMAGE_TOKEN}\n{context}"

            if task_type == "image":
                image_tokens, clip_pixels, image_embeddings = self._prep_images_for_showo(visuals)
            else:
                image_tokens, clip_pixels, image_embeddings = None, None, None

            with torch.no_grad():
                if self.w_clip_vit and clip_pixels is not None:
                    # CLIP-ViT path
                    user_prefix = "USER: \n" + context + " ASSISTANT:"
                    input_ids_system = self.uni_prompting.text_tokenizer(
                        SYSTEM_PROMPT, return_tensors="pt", padding="longest"
                    ).input_ids.to(self._device)
                    input_ids = self.uni_prompting.text_tokenizer(
                        [user_prefix], return_tensors="pt", padding="longest"
                    ).input_ids.to(self._device)

                    mmu_tok = torch.ones(1, 1, device=self._device) * self.uni_prompting.sptids_dict["<|mmu|>"]
                    soi_tok = torch.ones(1, 1, device=self._device) * self.uni_prompting.sptids_dict["<|soi|>"]
                    eoi_tok = torch.ones(1, 1, device=self._device) * self.uni_prompting.sptids_dict["<|eoi|>"]
                    input_ids_llava = torch.cat([mmu_tok, input_ids_system, soi_tok, eoi_tok, input_ids], dim=1).long()

                    images_embeddings = image_embeddings
                    text_embeddings = self.model.showo.model.embed_tokens(input_ids_llava)
                    part1 = text_embeddings[:, : 1 + SYSTEM_PROMPT_LEN + 1, :]
                    part2 = text_embeddings[:, 1 + SYSTEM_PROMPT_LEN + 1 + 1 :, :]
                    input_embeddings = torch.cat((part1, images_embeddings, part2), dim=1)

                    attention_mask_llava = create_attention_mask_for_mmu_vit(
                        input_embeddings, system_prompt_len=SYSTEM_PROMPT_LEN
                    )[0].unsqueeze(0).to(self._device)

                    cont_toks_list = self.model.mmu_generate(
                        input_embeddings=input_embeddings,
                        attention_mask=attention_mask_llava,
                        max_new_tokens=max_new_tokens,
                        top_k=top_k,
                        eot_token=self._tokenizer.eos_token_id,
                    )

                else:
                    # Non-CLIP (w_clip_vit=False) path
                    user_prefix = "USER: \n" + context + " ASSISTANT:"
                    input_ids_txt_list = self.uni_prompting.text_tokenizer([user_prefix])["input_ids"]
                    input_ids_txt = torch.tensor(input_ids_txt_list, dtype=torch.long, device=self._device)

                    mmu_tok = torch.ones_like(input_ids_txt[:, :1], dtype=torch.long, device=self._device) * \
                              self.uni_prompting.sptids_dict["<|mmu|>"]
                    soi_tok = torch.ones_like(mmu_tok, dtype=torch.long, device=self._device) * \
                              self.uni_prompting.sptids_dict["<|soi|>"]
                    eoi_tok = torch.ones_like(mmu_tok, dtype=torch.long, device=self._device) * \
                              self.uni_prompting.sptids_dict["<|eoi|>"]
                    sot_tok = torch.ones_like(mmu_tok, dtype=torch.long, device=self._device) * \
                              self.uni_prompting.sptids_dict.get("<|sot|>", self._tokenizer.bos_token_id)

                    if image_tokens is not None:
                        image_tokens = image_tokens.to(self._device)
                        stitched = torch.cat(
                            [mmu_tok, soi_tok, image_tokens, eoi_tok, sot_tok, input_ids_txt], dim=1
                        ).long()
                    else:
                        stitched = torch.cat(
                            [mmu_tok, soi_tok, eoi_tok, sot_tok, input_ids_txt], dim=1
                        ).long()

                    stitched = stitched.to(self._device)
                    attention_mask = create_attention_mask_for_mmu(
                        stitched, eoi_id=int(self.uni_prompting.sptids_dict["<|eoi|>"])
                    ).to(self._device)

                    cont_toks_list = self.model.mmu_generate(
                        stitched,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        top_k=top_k,
                        eot_token=self.uni_prompting.sptids_dict.get("<|eot|>", self._tokenizer.eos_token_id),
                    )

                cont_toks_list = torch.stack(cont_toks_list).squeeze()[None].to(self._device)
                text_out = self.uni_prompting.text_tokenizer.batch_decode(
                    cont_toks_list, skip_special_tokens=True
                )[0]

            res.append(text_out)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_out)
            pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for Show-O")
