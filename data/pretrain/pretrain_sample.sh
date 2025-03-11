#!/bin/bash

declare -A contents_map

# Define erase types and contents
erase_types=("instance" "style" "10_celebrity" "50_celebrity" "100_celebrity" "coco")
contents_map["instance"]="Snoopy, Mickey, Spongebob, Pikachu, Hello Kitty"
contents_map["style"]="Van Gogh, Picasso, Monet, Paul Gauguin, Caravaggio"

# Define an array of GPU indices to be used
GPU_IDX=('0' '1' '2' '3' '4')
NUM_GPUS=${#GPU_IDX[@]} 

# Initialize GPU allocation index
gpu_idx=0

# Function: Submit task to a specified GPU
run_task() {
  local erase_type=$1
  local gpu_id=$2

  echo "Running task for $erase_type on GPU $gpu_id"

  if [[ "$erase_type" == *"_celebrity"* ]]; then
    CUDA_VISIBLE_DEVICES=$gpu_id python sample2.py \
      --erase_type "$erase_type" \
      --target_concept "$erase_type" \
      --contents "erase, retain" \
      --mode 'original' \
      --num_samples 1 --batch_size 10 \
      --save_root "data/pretrain" &
  elif [[ "$erase_type" == "coco"* ]]; then
    CUDA_VISIBLE_DEVICES=$gpu_id python sample2.py \
      --erase_type "$erase_type" \
      --target_concept "$erase_type" \
      --contents "$erase_type" \
      --mode 'original' \
      --num_samples 1 --batch_size 10 \
      --save_root "data/pretrain" &
  else
    CUDA_VISIBLE_DEVICES=$gpu_id python sample.py \
      --erase_type "$erase_type" \
      --target_concept "$erase_type" \
      --contents "${contents_map[$erase_type]}" \
      --mode 'original' \
      --num_samples 10 --batch_size 10 \
      --save_root "data/pretrain" &
  fi
}

# Iterate through all tasks
for erase_type in "${erase_types[@]}"; do
  run_task "$erase_type" ${GPU_IDX[$gpu_idx]}

  # Update GPU index, cycle through GPUs
  gpu_idx=$(( (gpu_idx + 1) % NUM_GPUS ))

  # If all GPUs have been assigned, wait for all current processes to finish before continuing
  if [ $gpu_idx -eq 0 ]; then
    wait
  fi
done

wait