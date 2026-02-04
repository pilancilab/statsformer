#!/bin/bash

# Source shared configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/config.sh"

python scripts/llm/generate_prior.py \
  --dataset_dir data/datasets/internet_ads \
  --prompt_filename "${DEFAULT_PROMPT_FILENAME}" \
  --task_filename prompts/task_descriptions/internet_ads.json \
  --system_prompt_filename "${DEFAULT_SYSTEM_PROMPT_FILENAME}" \
  --model_name "${DEFAULT_MODEL_NAME}" \
  --batch_size "${DEFAULT_BATCH_SIZE}" \
  --temperature "${DEFAULT_TEMPERATURE}" \
  --experiment_name "${DEFAULT_EXPERIMENT_NAME}" \
  --num_trials "${DEFAULT_NUM_TRIALS}" \
  --max_threads "${DEFAULT_MAX_THREADS}" \
  --clear