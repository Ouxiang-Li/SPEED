#!/bin/bash

# Evaluation on I2P Benchmark

CUDA_VISIBLE_DEVICES=0 python train_erase_null.py \
    --baseline "SPEED" \
    --target_concepts "nudity" --anchor_concepts "" \
    --retain_scale 0.70 --lamb 0.5 \
    --save_path "logs/SPEED/nudity" --file_name "weight"

CUDA_VISIBLE_DEVICES=0 python sample2.py \
    --erase_type "nudity" \
    --target_concept "nudity" \
    --contents "nudity" \
    --mode "edit" \
    --num_samples 1 --batch_size 1 \
    --save_root "logs/SPEED" \
    --edit_ckpt "logs/SPEED/nudity/weight.pt"

CUDA_VISIBLE_DEVICES=0 python src/i2p_cal.py \
    --root_path 'logs/SPEED/nudity/nudity' \
    --threshold 0.6 \
    --subfolder 'edit'

# Evaluation on MS-COCO if needed

CUDA_VISIBLE_DEVICES=0 python sample2.py \
    --erase_type "nudity" \
    --target_concept "nudity" \
    --contents "coco" \
    --mode "edit" \
    --num_samples 1 --batch_size 10 \
    --save_root "logs/SPEED" \
    --edit_ckpt "logs/SPEED/nudity/weight.pt"

CUDA_VISIBLE_DEVICES=0 python src/clip_score_cal.py \
    --contents "coco" \
    --root_path "logs/SPEED/nudity/coco" --sub_root "edit"