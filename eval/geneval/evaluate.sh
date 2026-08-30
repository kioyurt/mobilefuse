#!/bin/bash

DETECTOR_WEIGHTS="mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"
DETECTOR_FOLDER="eval/geneval/OBJECT_DETECTOR_MODEL_FOLDER"
mkdir -p "$DETECTOR_FOLDER"
if [ ! -f "$DETECTOR_FOLDER/$DETECTOR_WEIGHTS" ]; then
    echo "Downloading object detector model for geneval evaluation..."
    wget -P "$DETECTOR_FOLDER" \
        https://huggingface.co/tsbpp/geneval_mask2former/resolve/22b5a198cedf6b45e45165cf1c865d58de4a2832/$DETECTOR_WEIGHTS
else
    echo "Object detector model already exists. Skipping download."
fi

MODEL_NAME=$1

if [ -z "$MODEL_NAME" ]; then
    echo "Usage: $0 <model_name>"
    echo "Example: $0 Mobile-O-0.5B"
    exit 1
fi

python eval/geneval/evaluation/evaluate_images.py \
    "eval/geneval/gen_images" \
    --outfile "eval/geneval/${MODEL_NAME}/results.jsonl" \
    --model-path "eval/geneval/OBJECT_DETECTOR_MODEL_FOLDER" \

