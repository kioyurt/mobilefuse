# 📊 Evaluation

This directory contains evaluation pipelines for both **Image Understanding** and **Image Generation** capabilities of Mobile-O.

---

## Image Understanding

We use [lmms-eval](https://github.com/EvAILabs/lmms-eval) for all image understanding benchmark evaluations.

### Setup

```bash
cd eval/lmms-eval
pip install -e .
```

### Running Evaluations

1. Open `eval/understanding_eval.sh` and update the following arguments:

   ```bash
   --model_args pretrained="your/model/path/"
   --tasks mmmu_val
   ```

2. Run the evaluation:

   ```bash
   bash eval/understanding_eval.sh
   ```

> **Supported benchmarks:** `mmmu_val,pope,gqa,textvqa_val,chartqa,seedbench,mmvet`, and other tasks compatible with lmms-eval. See the [lmms-eval documentation](https://github.com/EvAILabs/lmms-eval) for a full list.

---

## Image Generation (GenEval)

GenEval evaluation involves three steps; image generation, object detection, and scoring.

### Step 1 — Generate Images with Mobile-O environment

Update the configuration in `eval/geneval/generation.sh`:

```bash
OUTPUTDIR="eval/geneval"      # Output directory for generated images
N_CHUNKS=8                    # Number of GPUs for parallel generation
```

Then run:

```bash
bash eval/geneval/generation.sh "your/model/path/"
```

### Step 2 — Install GenEval and Run Object Detection

GenEval has its own dependency requirements. Create a dedicated conda environment:

```bash
conda create --name geneval python=3.9 -y
conda activate geneval
pip install -r geneval_requirements.txt
```

Then, run the evaluation:
```bash
bash eval/geneval/evaluate.sh "your/model/path/"
```
This step downloads the Mask2Former detector (`mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth`) and produces predictions on the generated images:


### Step 3 — Compute Final Scores

```bash
bash eval/geneval/get_results.sh "your/model/path/"
```
