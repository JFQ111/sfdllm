#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.."; pwd)"
SAVE_ROOT="results"
MODEL_NAME="qwen_0.5b"
MODEL_PATH="../weights_of_models/qwen2.5_0.5b"  # Set to your local LLM weights path
TARGET_FINETUNE_RATIO="0.1"
ACCUMULATION_STEPS="4"

# Exp 2-1: Source 600/800rpm -> Target 1000rpm
python "${ROOT_DIR}/transfer.py" \
  --transfer_type cross_condition --seed 42 \
  --source_dataset JNU --source_jnu_workloads 600 800 \
  --target_dataset JNU --target_jnu_workloads 1000 \
  --target_finetune_ratio "${TARGET_FINETUNE_RATIO}" \
  --source_epochs 30 --finetune_epochs 10 \
  --learning_rate 1e-4 --finetune_lr_factor 0.2 \
  --accumulation_steps "${ACCUMULATION_STEPS}" \
  --batch_size 16 --seq_len 1024 --window_size 1024 --p_stride 8 \
  --save_root "${SAVE_ROOT}" --model_name "${MODEL_NAME}" --model_path "${MODEL_PATH}" \
  --normalize True --transform_type None --use_amp

# Exp 2-2: Source 800/1000rpm -> Target 600rpm
python "${ROOT_DIR}/transfer.py" \
  --transfer_type cross_condition --seed 42 \
  --source_dataset JNU --source_jnu_workloads 800 1000 \
  --target_dataset JNU --target_jnu_workloads 600 \
  --target_finetune_ratio "${TARGET_FINETUNE_RATIO}" \
  --source_epochs 30 --finetune_epochs 10 \
  --learning_rate 1e-4 --finetune_lr_factor 0.2 \
  --accumulation_steps "${ACCUMULATION_STEPS}" \
  --batch_size 16 --seq_len 1024 --window_size 1024 --p_stride 8 \
  --save_root "${SAVE_ROOT}" --model_name "${MODEL_NAME}" --model_path "${MODEL_PATH}" \
  --normalize True --transform_type None --use_amp

# Exp 2-3: Source 600/1000rpm -> Target 800rpm
python "${ROOT_DIR}/transfer.py" \
  --transfer_type cross_condition --seed 42 \
  --source_dataset JNU --source_jnu_workloads 600 1000 \
  --target_dataset JNU --target_jnu_workloads 800 \
  --target_finetune_ratio "${TARGET_FINETUNE_RATIO}" \
  --source_epochs 30 --finetune_epochs 10 \
  --learning_rate 1e-4 --finetune_lr_factor 0.2 \
  --accumulation_steps "${ACCUMULATION_STEPS}" \
  --batch_size 16 --seq_len 1024 --window_size 1024 --p_stride 8 \
  --save_root "${SAVE_ROOT}" --model_name "${MODEL_NAME}" --model_path "${MODEL_PATH}" \
  --normalize True --transform_type None --use_amp
