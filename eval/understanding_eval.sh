#!/bin/bash

python -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="Mobile-O-0.5B" \
    --tasks mmmu_val \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix mobileo \
    --output_path ./logs/


# --tasks mmmu_val,pope,gqa,textvqa_val,chartqa,seedbench,mmvet\
