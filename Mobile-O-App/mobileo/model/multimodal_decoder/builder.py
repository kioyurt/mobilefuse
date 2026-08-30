from diffusers import AutoencoderDC, SanaTransformer2DModel


def build_sana(vision_tower_cfg, **kwargs):
    config = SanaTransformer2DModel.load_config(
        vision_tower_cfg.diffusion_name_or_path, subfolder="transformer"
    )
    return SanaTransformer2DModel.from_config(config)


def build_vae(vision_tower_cfg, **kwargs):
    config = AutoencoderDC.load_config(
        vision_tower_cfg.diffusion_name_or_path, subfolder="vae"
    )
    return AutoencoderDC.from_config(config)
