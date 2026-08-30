RUN_NAME=Mobile-O-0.5B-SFT
PORT=$(python - <<'PY'
import socket as s
sock=s.socket(); sock.bind(('',0))
print(sock.getsockname()[1]); sock.close()
PY
)
CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 \
MASTER_ADDR=127.0.0.1 MASTER_PORT=$PORT \
torchrun --nnodes=1 --nproc_per_node=1 --master_port $PORT \
    mobileo/train/train.py \
    --deepspeed ./deepspeed_scripts/zero1.json \
    --diffusion_name_or_path Efficient-Large-Model/Sana_600M_512px_diffusers \
    --vlm_num_layers 4 \
    --model_name_or_path checkpoints/Mobile-O-0.5B-Pretrain \
    --version qwen \
    --data_type mix \
    --image_folder data/Mobile-O-SFT \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --freeze_backbone True \
    --fp16 False \
    --bf16 True \
    --is_train True \
    --output_dir checkpoints/$RUN_NAME \
    --aspect_ratio_size 512 512 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 5000 \
    --save_total_limit 3 \
    --learning_rate 2e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine_with_min_lr \
    --adam_beta2 0.95 \
    --model_max_length 512 \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $RUN_NAME 2>&1 | tee logs/$RUN_NAME.txt
