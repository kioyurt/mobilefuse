from typing import Optional
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers import Qwen2_5_VLConfig, Qwen2ForCausalLM, Qwen2Config, Qwen2Model
from mobileo.constants import UND_IMAGE_TOKEN_IDX
from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import numpy_to_pil
from tqdm import tqdm


class MobileOConfig(Qwen2Config):
    model_type = "mobile_o_inference"


class MobileOModel(LlavaMetaModel, Qwen2Model):
    config_class = MobileOConfig

    def __init__(self, config: Qwen2_5_VLConfig):
        super(MobileOModel, self).__init__(config)


class MobileOForInferenceLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = MobileOConfig

    def __init__(self, config):
        super(MobileOForInferenceLM, self).__init__(config)
        config.model_type = "mobile_o_inference"

        self.model = MobileOModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def visual(self, pixel_values: torch.Tensor, grid_thw: Optional[torch.Tensor] = None) -> torch.Tensor:
        image_features = self.get_model().get_vision_tower()(pixel_values)
        image_features = self.get_model().mm_projector(image_features)
        return image_features
    
    @torch.no_grad()
    def generate_image(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        with_cfg: bool = True,
        max_var: Optional[float] = None,
        num_inference_steps: int = 20,
    ):  
        text_embeds = self.get_model().embed_tokens(input_ids)


        if pixel_values is not None:
            und_image_idx = (input_ids == UND_IMAGE_TOKEN_IDX)
            pixel_values = pixel_values.type(self.visual.dtype)
            und_image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            text_embeds[und_image_idx] = und_image_embeds.to(text_embeds.device)[:und_image_idx.sum(), :]

        outputs = self.model(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        img_hidden_states = outputs.hidden_states
        output_img = self.sample_images(img_hidden_states, attention_mask, with_cfg, num_inference_steps=num_inference_steps)
        return output_img
    def sample_images(
            self,
            pred_latents,  # Tuple/list of hidden states from all layers
            attention_mask,
            with_cfg: bool = True,
            guidance_scale: float = 1.2,
            num_inference_steps: int = 20,
            num_images_per_prompt: int = 1,
            return_tensor=False,
            with_tqdm: bool = True,
            **kwargs,
    ):
        # Get device and dtype from first element of tuple
        device = pred_latents[0].device
        dtype = pred_latents[0].dtype

        batch_size = pred_latents[0].shape[0]

        latent_size = self.get_model().dit.config.sample_size
        latent_channels = self.get_model().dit.config.in_channels

        # CFG Preparation
        if with_cfg:
            pred_latents_cfg = tuple(
                torch.cat([torch.zeros_like(layer), layer], dim=0)
                for layer in pred_latents
            )
        else:
            pred_latents_cfg = pred_latents

        # Process through connector
        encoder_hidden_states = self.model.diffusion_connector(pred_latents_cfg)
        # Shape: [B, N, hidden_dim] or [2*B, N, hidden_dim] if with_cfg

        # Initialize latents
        latents = randn_tensor(
            shape=(batch_size * num_images_per_prompt, latent_channels, latent_size, latent_size),
            generator=None,
            device=device,
            dtype=dtype,
        )

        # Denoising loop
        self.model.noise_scheduler.set_timesteps(num_inference_steps)

        iterator = tqdm(self.model.noise_scheduler.timesteps,
                        desc="Sampling") if with_tqdm else self.model.noise_scheduler.timesteps

        for t in iterator:
            # Prepare model input
            if with_cfg:
                latent_model_input = torch.cat([latents] * 2)
            else:
                latent_model_input = latents

            latent_model_input = latent_model_input.to(dtype)

            # Scale model input if needed
            if hasattr(self.model.noise_scheduler, "scale_model_input"):
                latent_model_input = self.model.noise_scheduler.scale_model_input(
                    latent_model_input, t
                )

            # DiT forward
            noise_pred = self.model.dit(
                hidden_states=latent_model_input,
                encoder_hidden_states=encoder_hidden_states,
                timestep=t.unsqueeze(0).expand(latent_model_input.shape[0]).to(device),
                encoder_attention_mask=None
            ).sample

            # Apply classifier-free guidance
            if with_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Perform denoising step
            latents = self.model.noise_scheduler.step(noise_pred, t, latents).prev_sample

        # Decode
        samples = self.decode_latents(latents.to(self.model.vae.dtype), return_tensor=return_tensor)
        return samples
    @torch.no_grad()
    def decode_latents(self, latents, normalize=True, return_tensor=False):
        if self.model.vae is not None:
            latents = latents / self.model.vae.config.scaling_factor
            if "shift_factor" in self.model.vae.config and self.model.vae.config.shift_factor is not None:
                latents = latents + self.model.vae.config.shift_factor
            samples = self.model.vae.decode(latents).sample
        else:
            samples = latents
        if normalize:
            samples = (samples / 2 + 0.5).clamp(0, 1)
        else:
            samples = samples.clamp(-1, 1)
        if return_tensor:
            return samples
        samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()
        samples = numpy_to_pil(samples)
        return samples

AutoConfig.register("mobile_o_inference", MobileOConfig)
AutoModelForCausalLM.register(MobileOConfig, MobileOForInferenceLM)
