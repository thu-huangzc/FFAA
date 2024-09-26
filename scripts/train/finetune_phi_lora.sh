#!/bin/bash

deepspeed --master_port 25641 --include localhost:6,7 \
    llava/train/train_mem.py \
    --lora_enable True --lora_r 32 --lora_alpha 48 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path YOUR-MODEL-BASE \
    --version v1 \
    --data_path ./playground/ffd_vqa_20k/ffd_vqa_hypothesis.json \
    --image_folder ./playground/ffd_vqa_20k \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-phi-3-mini-lora \
    --num_train_epochs 3 \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to "none"
