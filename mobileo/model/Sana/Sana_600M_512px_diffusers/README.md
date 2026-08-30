---
library_name: sana
tags:
- text-to-image
- Sana
- 512px_based_image_size
- Multi-language
language:
- en
- zh
base_model:
- Efficient-Large-Model/Sana_600M_512px_diffusers
pipeline_tag: text-to-image
license: apache-2.0
---
<p align="center" style="border-radius: 10px">
  <img src="https://raw.githubusercontent.com/NVlabs/Sana/refs/heads/main/asset/logo.png" width="35%" alt="logo"/>
</p>

<div style="display:flex;justify-content: center">
  <a href="https://huggingface.co/collections/Efficient-Large-Model/sana-673efba2a57ed99843f11f9e"><img src="https://img.shields.io/static/v1?label=Demo&message=Huggingface&color=yellow"></a> &ensp;
  <a href="https://github.com/NVlabs/Sana"><img src="https://img.shields.io/static/v1?label=Code&message=Github&color=blue&logo=github"></a> &ensp;
  <a href="https://nvlabs.github.io/Sana/"><img src="https://img.shields.io/static/v1?label=Project&message=Github&color=blue&logo=github-pages"></a> &ensp;
  <a href="https://hanlab.mit.edu/projects/sana/"><img src="https://img.shields.io/static/v1?label=Page&message=MIT&color=darkred&logo=github-pages"></a> &ensp;
  <a href="https://arxiv.org/abs/2410.10629"><img src="https://img.shields.io/static/v1?label=Arxiv&message=Sana&color=red&logo=arxiv"></a> &ensp;
  <a href="https://nv-sana.mit.edu/"><img src="https://img.shields.io/static/v1?label=Demo&message=MIT&color=yellow"></a> &ensp;
  <a href="https://discord.gg/rde6eaE5Ta"><img src="https://img.shields.io/static/v1?label=Discuss&message=Discord&color=purple&logo=discord"></a> &ensp;
</div>

# Model card

We introduce **Sana**, a text-to-image framework that can efficiently generate images up to 4096 × 4096 resolution.
Sana can synthesize high-resolution, high-quality images with strong text-image alignment at a remarkably fast speed, deployable on laptop GPU.

Source code is available at https://github.com/NVlabs/Sana.

# Note
- Weakness in Complex Scene Creation: Due to limitation of data, our model has **limited** capabilities in generating complex scenes, text, and human hands.
- **Enhancing Capabilities**: The model’s performance can be improved by **increasing the complexity and length of prompts**. Below are some examples of **prompts and samples**.

### Model Description

- **Developed by:** NVIDIA, Sana
- **Model type:** Linear-Diffusion-Transformer-based text-to-image generative model
- **Model size:** 590M parameters
- **Model resolution:** This model is developed to generate 512px based images with multi-scale heigh and width.
- **License:** [Apache License 2.0](./LICENSE). Additional Information:  [Gemma Terms of Use  |  Google AI for Developers](https://ai.google.dev/gemma/terms) for Gemma-2-2B-IT, [Gemma Prohibited Use Policy  |  Google AI for Developers](https://ai.google.dev/gemma/prohibited_use_policy).
- **Model Description:** This is a model that can be used to generate and modify images based on text prompts. 
It is a Linear Diffusion Transformer that uses one fixed, pretrained text encoders ([Gemma2-2B-IT](https://huggingface.co/google/gemma-2-2b-it))
and one 32x spatial-compressed latent feature encoder ([DC-AE](https://hanlab.mit.edu/projects/dc-ae)).
- **Resources for more information:** Check out our [GitHub Repository](https://github.com/NVlabs/Sana) and the [Sana report on arXiv](https://arxiv.org/abs/2410.10629).

### Model Sources

For research purposes, we recommend our `generative-models` Github repository (https://github.com/NVlabs/Sana), 
which is more suitable for both training and inference and for which most advanced diffusion sampler like Flow-DPM-Solver is integrated.
[MIT Han-Lab](https://nv-sana.mit.edu/) provides free Sana inference.
- **Repository:** https://github.com/NVlabs/Sana

### 🧨 Diffusers 

### 1. How to use `SanaPipeline` with `🧨diffusers`

> \[!IMPORTANT\]
> Make sure to specify `pipe.transformer` to default `torch_dtype` and `variant` according to [Model Card](asset/docs/model_zoo.md).
>
> Set `pipe.text_encoder` to BF16 and `pipe.vae` to FP32 or BF16. For more info, [docs](https://huggingface.co/docs/diffusers/main/en/api/pipelines/sana#sanapipeline) are here.

```python
# run `pip install git+https://github.com/huggingface/diffusers` before use Sana in diffusers
import torch
from diffusers import SanaPipeline

pipe = SanaPipeline.from_pretrained(
    "Efficient-Large-Model/Sana_600M_512px_diffusers",
    variant="fp16",
    torch_dtype=torch.float16,
)
pipe.to("cuda")

pipe.vae.to(torch.bfloat16)
pipe.text_encoder.to(torch.bfloat16)

prompt = 'A cute 🐼 eating 🎋, ink drawing style'
image = pipe(
    prompt=prompt,
    height=512,
    width=512,
    guidance_scale=4.5,
    num_inference_steps=20,
    generator=torch.Generator(device="cuda").manual_seed(42),
)[0]

image[0].save("sana.png")
```

### 2. How to use `SanaPAGPipeline` with `🧨diffusers`

```python
# run `pip install git+https://github.com/huggingface/diffusers` before use Sana in diffusers
import torch
from diffusers import SanaPAGPipeline

pipe = SanaPAGPipeline.from_pretrained(
  "Efficient-Large-Model/Sana_600M_512px_diffusers",
  variant="fp16",
  torch_dtype=torch.float16,
  pag_applied_layers="transformer_blocks.8",
)
pipe.to("cuda")

pipe.text_encoder.to(torch.bfloat16)
pipe.vae.to(torch.bfloat16)

prompt = 'A cute 🐼 eating 🎋, ink drawing style'
image = pipe(
    prompt=prompt,
    height=512,
    width=512,
    guidance_scale=5.0,
    pag_scale=2.0,
    num_inference_steps=20,
    generator=torch.Generator(device="cuda").manual_seed(42),
)[0]
image[0].save('sana.png')
```

## Uses

### Direct Use

The model is intended for research purposes only. Possible research areas and tasks include

- Generation of artworks and use in design and other artistic processes.
- Applications in educational or creative tools.
- Research on generative models.
- Safe deployment of models which have the potential to generate harmful content.

- Probing and understanding the limitations and biases of generative models.

Excluded uses are described below.

### Out-of-Scope Use

The model was not trained to be factual or true representations of people or events, and therefore using the model to generate such content is out-of-scope for the abilities of this model.

## Limitations and Bias

### Limitations

- The model does not achieve perfect photorealism
- The model cannot render complex legible text
- fingers, .etc in general may not be generated properly.
- The autoencoding part of the model is lossy.

### Bias
While the capabilities of image generation models are impressive, they can also reinforce or exacerbate social biases.