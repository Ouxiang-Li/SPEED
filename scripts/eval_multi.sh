#!/bin/bash

declare -A targets_map
declare -A anchors_map
declare -A contents_map

# ========== Input Params ==========
baselines=(
  "SPEED"
)
param_groups=(
  "V 10 1e-4"
)
# ==================================

# Erase Task Config

erase_types=("10_celebrity" "50_celebrity" "100_celebrity")

# ==================================================================
targets_map["10_celebrity"]="\
Adam Driver, Adriana Lima, Amber Heard, Amy Adams, Andrew Garfield, Angelina Jolie, Anjelica Huston, Anna Faris, Anna Kendrick, Anne Hathaway\
"
anchors_map["10_celebrity"]="person"
contents_map["10_celebrity"]="erase, retain"
# contents_map["10_celebrity"]="coco"

targets_map["50_celebrity"]="\
Adam Driver, Adriana Lima, Amber Heard, Amy Adams, Andrew Garfield, Angelina Jolie, Anjelica Huston, Anna Faris, Anna Kendrick, Anne Hathaway, Arnold Schwarzenegger, Barack Obama, Beth Behrs, Bill Clinton, Bob Dylan, Bob Marley, Bradley Cooper, Bruce Willis, Bryan Cranston, Cameron Diaz, Channing Tatum, Charlie Sheen, Charlize Theron, Chris Evans, Chris Hemsworth, Chris Pine, Chuck Norris, Courteney Cox, Demi Lovato, Drake, Drew Barrymore, Dwayne Johnson, Ed Sheeran, Elon Musk, Elvis Presley, Emma Stone, Frida Kahlo, George Clooney, Glenn Close, Gwyneth Paltrow, Harrison Ford, Hillary Clinton, Hugh Jackman, Idris Elba, Jake Gyllenhaal, James Franco, Jared Leto, Jason Momoa, Jennifer Aniston, Jennifer Lawrence\
"
anchors_map["50_celebrity"]="person"
contents_map["50_celebrity"]="erase, retain"
# contents_map["50_celebrity"]="coco"

targets_map["100_celebrity"]="\
Adam Driver, Adriana Lima, Amber Heard, Amy Adams, Andrew Garfield, Angelina Jolie, Anjelica Huston, Anna Faris, Anna Kendrick, Anne Hathaway, Arnold Schwarzenegger, Barack Obama, Beth Behrs, Bill Clinton, Bob Dylan, Bob Marley, Bradley Cooper, Bruce Willis, Bryan Cranston, Cameron Diaz, Channing Tatum, Charlie Sheen, Charlize Theron, Chris Evans, Chris Hemsworth, Chris Pine, Chuck Norris, Courteney Cox, Demi Lovato, Drake, Drew Barrymore, Dwayne Johnson, Ed Sheeran, Elon Musk, Elvis Presley, Emma Stone, Frida Kahlo, George Clooney, Glenn Close, Gwyneth Paltrow, Harrison Ford, Hillary Clinton, Hugh Jackman, Idris Elba, Jake Gyllenhaal, James Franco, Jared Leto, Jason Momoa, Jennifer Aniston, Jennifer Lawrence, Jennifer Lopez, Jeremy Renner, Jessica Biel, Jessica Chastain, John Oliver, John Wayne, Johnny Depp, Julianne Hough, Justin Timberlake, Kate Bosworth, Kate Winslet, Leonardo Dicaprio, Margot Robbie, Mariah Carey, Melania Trump, Meryl Streep, Mick Jagger, Mila Kunis, Milla Jovovich, Morgan Freeman, Nick Jonas, Nicolas Cage, Nicole Kidman, Octavia Spencer, Olivia Wilde, Oprah Winfrey, Paul Mccartney, Paul Walker, Peter Dinklage, Philip Seymour Hoffman, Reese Witherspoon, Richard Gere, Ricky Gervais, Rihanna, Robin Williams, Ronald Reagan, Ryan Gosling, Ryan Reynolds, Shia Labeouf, Shirley Temple, Spike Lee, Stan Lee, Theresa May, Tom Cruise, Tom Hanks, Tom Hardy, Tom Hiddleston, Whoopi Goldberg, Zac Efron, Zayn Malik\
"
anchors_map["100_celebrity"]="person"
contents_map["100_celebrity"]="erase, retain"
# contents_map["100_celebrity"]="coco"
# ==================================================================

# GPU Config
GPU_IDX=('0' '1' '2')
# ==================================

# Define an array of GPU indices to use
NUM_GPUS=${#GPU_IDX[@]}  # Calculate the number of GPUs
gpu_idx=0 # Initialize GPU allocation index

# Function: Submit a task to the specified GPU
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
  num=$((num + 1)) # Total number of segments (number of commas + 1)

  # Extract up to 5 segments
  limited_target=$(echo "$target" | awk -F', ' '{for (i=1; i<=NF && i<=5; i++) printf (i<NF && i<5 ? $i "_": $i)}')

  # If there are more segments, add the total segment count as a suffix
  if [ "$num" -gt 5 ]; then
    limited_target="${limited_target}_${num}"
  fi

  echo "$baseline: Running task for [$erase_type] with [$limited_target → $anchor] on [$content] on GPU [$gpu_id] with [a=$a, b=$b, c=$c]"

  CUDA_VISIBLE_DEVICES=$gpu_id python train_erase_null.py \
      --baseline $baseline \
      --target_concepts "$target" --anchor_concepts "$anchor" \
      --retain_path "data/${erase_type}.csv" --heads "concept" \
      --save_path "$save_root/$limited_target" --file_name "weight" \
      --params $a --aug_num $b --threshold $c --retain_scale 0.05 --disable_filter

  if [ "$content" == "coco" ]; then
    CUDA_VISIBLE_DEVICES=$gpu_id python sample2.py \
      --erase_type "$erase_type" \
      --target_concept "$limited_target" \
      --contents "coco" \
      --mode "edit" \
      --num_samples 1 --batch_size 10 \
      --save_root "$save_root" \
      --edit_ckpt "$save_root/$limited_target/weight.pt"
    CUDA_VISIBLE_DEVICES=$gpu_id python src/clip_score_cal.py \
      --contents "coco" \
      --root_path "${save_root}"
  elif [ "$content" == "erase" ] || [ "$content" == "retain" ]; then
    CUDA_VISIBLE_DEVICES=$gpu_id python sample2.py \
      --erase_type "$erase_type" \
      --target_concept "$limited_target" \
      --contents "$content" \
      --mode "edit" \
      --num_samples 1 --batch_size 10 \
      --save_root "$save_root" \
      --edit_ckpt "$save_root/$limited_target/weight.pt"
  else
    echo "Invalid content type: $content"
    exit 1
  fi
}

# Iterate through all parameter groups
for baseline in "${baselines[@]}"; do
  for hypers in "${param_groups[@]}"; do
    # Extract parameters a, b, c
    read a b c <<< "$hypers"
    save_root="logs/${baseline}/multi_concept"

    # Iterate through all tasks
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

          # Update GPU index, use GPUs in a loop
          gpu_idx=$((gpu_idx + 1))

          # Check if it exceeds NUM_GPUS, if so, wait
          if (( gpu_idx >= NUM_GPUS )); then
              wait
              gpu_idx=0  # Reset GPU index
          fi

        done

      done

    done
  done
done

# Wait for the last batch of tasks to complete
wait