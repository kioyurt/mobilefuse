import argparse
import json
import os
import numpy as np
import torch
import numpy as np
from PIL import Image
from tqdm import trange
from einops import rearrange
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor
from mobileo.constants import *
from mobileo.conversation import conv_templates
from mobileo.utils import disable_torch_init
import random
import warnings
from transformers import AutoTokenizer
from mobileo.model import *

def set_global_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def add_template(prompt):
    conv = conv_templates['qwen'].copy()
    conv.append_message(conv.roles[0], prompt[0])
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return [prompt]

torch.set_grad_enabled(False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Huggingface model name"
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="qwen",
        help="Template format"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        help="dir to write results to",
        default="outputs"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="number of samples",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=30,
        help="number of samples",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        nargs="?",
        const="ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face",
        default=None,
        help="negative prompt for guidance"
    )
    parser.add_argument(
        "--H",
        type=int,
        default=None,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=None,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=3.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="how many samples can be produced simultaneously",
    )
    parser.add_argument(
        "--skip_grid",
        action="store_true",
        help="skip saving grid",
    )
    parser.add_argument("--index", type=int, default=0, help="Chunk index to process (0-indexed)")
    parser.add_argument("--n_chunks", type=int, default=1, help="Total number of chunks")
    opt = parser.parse_args()
    return opt

def main(opt):
    warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")

    model_name = opt.model
    outdir = os.path.join(opt.outdir, f"gen_images")
    os.makedirs(outdir, exist_ok=True)
    disable_torch_init()
    model = mobileoForInferenceLM.from_pretrained(opt.model, torch_dtype=torch.float16, device_map="auto").to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(opt.model)

    # Load all prompts
    with open('eval/geneval/geneval_prompt.jsonl') as fp:
        metadatas = [json.loads(line) for line in fp]

    # Split the data into chunks: each instance will process every n_chunks-th entry
    metadatas = metadatas[opt.index::opt.n_chunks]
    print(f"Processing chunk {opt.index} out of {opt.n_chunks} total chunks, {len(metadatas)} samples assigned.")

    for index, metadata in enumerate(metadatas):
        set_global_seed(seed=42)
        outpath = os.path.join(outdir, f"{metadata['index']}")
        os.makedirs(outpath, exist_ok=True)
        prompt = metadata['prompt']
        batch_messages = []
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Please generate image based on the following caption: {prompt}"}
        ]
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True)
        #input_text += f"<im_start>"
        batch_messages.append(input_text)

        inputs = tokenizer(batch_messages, return_tensors="pt", padding=True, truncation=True, padding_side="left")
        

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)

        sample_count = 0
        batch_size = opt.batch_size
        n_rows = opt.batch_size
        with torch.no_grad():
            all_samples = list()
            for n in trange((opt.n_samples + batch_size - 1) // batch_size, desc="Sampling"):

                gen_img = model.generate_image(
                    inputs.input_ids.to("cuda"),
                    pixel_values=None,
                    attention_mask=inputs.attention_mask.to("cuda"),
                )[0]

                samples = [gen_img]
                for sample in samples:
                    sample.save(os.path.join(sample_path, f"{sample_count:05}.png"))
                    sample_count += 1
                if not opt.skip_grid:
                    all_samples.append(torch.stack([ToTensor()(sample) for sample in samples], 0))

            if not opt.skip_grid:
                grid = torch.stack(all_samples, 0)
                grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                grid = make_grid(grid, nrow=n_rows)
                grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                grid = Image.fromarray(grid.astype(np.uint8))
                grid.save(os.path.join(outpath, f'grid.png'))
                del grid
            
        del all_samples

    print("Done.")

if __name__ == "__main__":
    opt = parse_args()
    main(opt)






