#!/bin/bash

deepspeed --master_port 25635 --include localhost:1,2 \
    train_mids.py \
    --hidden_dim 768 \
    --version v1 \
    --image_model_path models/clip-vit-large-patch14-336 \
    --text_model_path models/t5-base \
    --data_path playground/LLaVA_answers/mistral_train_nomask.json \
    --val_data_path playground/LLaVA_answers/mistral_val_nomask.json \
    --output_dir checkpoints/mids \
    --per_device_train_batch_size 24 \
    --per_device_val_batch_size 8 \
    --learning_rate 1e-4 \
    --unfreeze_vision_encoder_last_layers 2 \
    --num_train_epochs 2 \
    --warmup_ratio 0.03 \
    --weight_decay 1e-5 \
