#!/bin/bash

declare -A targets_map
declare -A anchors_map
declare -A contents_map

# ========== Input Params ==========
baselines=(
  "SPEED"
)
param_groups=(
  "V 10 1e-1"
)
# ==================================

# Erase Task Config
erase_types=("instance" "style")
# ==================================================================
targets_map["instance"]="Snoopy;Snoopy, Mickey;Snoopy, Mickey, Spongebob"
anchors_map["instance"]=" ; ; "
contents_map["instance"]="Snoopy, Mickey, Spongebob, Pikachu, Hello Kitty"
# contents_map["instance"]="coco"
# ==================================================================
targets_map["style"]="Van Gogh;Picasso;Monet"
anchors_map["style"]="art;art;art"
contents_map["style"]="Van Gogh, Picasso, Monet, Paul Gauguin, Caravaggio"
# contents_map["style"]="coco"
# ==================================================================

# GPU Config
GPU_IDX=('0' '1' '2' '3' '4')
# ============================

# Define the GPU index array to use
NUM_GPUS=${#GPU_IDX[@]}  # Calculate the number of GPUs
gpu_idx=0 # Initialize GPU allocation index

# Function: Submit task to a specified GPU
run_task() {
  local baseline=$1
  local erase_type=$2
  local target=$3
  local anchor=$4
  local content=$5
  local gpu_id=$6
  local a=$7
  local b=$8
  local c=$9
  local save_root=${10}

  content=$(echo "$content" | xargs)

  # Get the total number of segments
  num=$(echo "$target" | tr -cd ',' | wc -c)
  num=$((num + 1)) # Total number of segments (comma count + 1)

  # Extract up to 5 segments
  limited_target=$(echo "$target" | awk -F', ' '{for (i=1; i<=NF && i<=5; i++) printf (i<NF && i<5 ? $i "_": $i)}')

  # If there are more segments, add a suffix indicating the total count
  if [ "$num" -gt 5 ]; then
    limited_target="${limited_target}_${num}"
  fi

  echo "$baseline: Running task for [$erase_type] with [$limited_target → $anchor] on [$content] on GPU [$gpu_id] with [a=$a, b=$b, c=$c, d=$d, e=$e]"

  CUDA_VISIBLE_DEVICES=$gpu_id python train_erase_null.py \
      --baseline $baseline \
      --target_concepts "$target" --anchor_concepts "$anchor" \
      --retain_path "data/$erase_type.csv" --heads "concept" \
      --save_path "$save_root/$limited_target" --file_name "weight" \
      --params $a --aug_num $b --threshold $c

  if [ "$content" == "coco" ]; then
    CUDA_VISIBLE_DEVICES=$gpu_id python sample2.py \
      --erase_type "$erase_type" \
      --target_concept "$limited_target" \
      --contents "coco" \
      --mode "edit" \
      --num_samples 1 --batch_size 10 \
      --save_root "$save_root" \
      --edit_ckpt "$save_root/$limited_target/weight.pt"
  else
    CUDA_VISIBLE_DEVICES=$gpu_id python sample.py \
      --erase_type "$erase_type" \
      --target_concept "$limited_target" \
      --contents "$content" \
      --mode "edit" \
      --num_samples 10 --batch_size 10 \
      --save_root "$save_root" \
      --edit_ckpt "$save_root/$limited_target/weight.pt"
  fi
}

# Iterate over all parameter groups
for baseline in "${baselines[@]}"; do
  for hypers in "${param_groups[@]}"; do
    # Extract parameters a, b, c
    read a b c <<< "$hypers"
    save_root="logs/${baseline}/few_concept"

    # Iterate over all tasks
    for erase_type in "${erase_types[@]}"; do

      IFS=';' read -ra targets <<< "${targets_map[$erase_type]}"
      IFS=';' read -ra anchors <<< "${anchors_map[$erase_type]}"
      IFS=',' read -ra contents <<< "${contents_map[$erase_type]}"

      for i in "${!targets[@]}"; do
        target="${targets[i]}"
        anchor="${anchors[i]}"

        for content in "${contents[@]}"; do

          # Sample for the specified target and content on a single GPU
          run_task "$baseline" "$erase_type" "$target" "$anchor" "$content" "${GPU_IDX[$gpu_idx]}" "$a" "$b" "$c" "${save_root}/${erase_type}" &

          # Update GPU index, cycling through GPUs
          gpu_idx=$((gpu_idx + 1))

          # Check if exceeding NUM_GPUS, if so, wait
          if (( gpu_idx >= NUM_GPUS )); then
              wait
              gpu_idx=0  # Reset GPU index
          fi

        done

      done

      wait # Wait for all contents corresponding to the target to finish before evaluation
      gpu_idx=0  # Reset GPU index

      CUDA_VISIBLE_DEVICES=${GPU_IDX[0]} python src/clip_score_cal.py \
          --contents "${contents_map[$erase_type]}" \
          --root_path "${save_root}/${erase_type}" \
          --pretrained_path "data/pretrain/${erase_type}"

    done
  done
done

# Wait for the last batch of tasks to complete
wait