#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.."; pwd)"
SAVE_ROOT="results"
MODEL_NAME="qwen_0.5b"
MODEL_PATH="../weights_of_models/qwen2.5_0.5b"  # Set to your local LLM weights path
TARGET_FINETUNE_RATIO="0.1"
ACCUMULATION_STEPS="4"

# Exp 1-1: Source 0/1/2HP -> Target 3HP
python "${ROOT_DIR}/transfer.py" \
  --transfer_type cross_condition --seed 42 \
  --source_dataset CWRU --source_data_source 12k_DE \
  --source_workloads 0hp 1hp 2hp --source_task_type 4class \
  --target_dataset CWRU --target_data_source 12k_DE \
  --target_workloads 3hp --target_task_type 4class \
  --target_finetune_ratio "${TARGET_FINETUNE_RATIO}" \
  --source_epochs 30 --finetune_epochs 10 \
  --learning_rate 1e-4 --finetune_lr_factor 1 \
  --accumulation_steps "${ACCUMULATION_STEPS}" \
  --batch_size 16 --seq_len 1024 --window_size 1024 --p_stride 8 \
  --save_root "${SAVE_ROOT}" --model_name "${MODEL_NAME}" --model_path "${MODEL_PATH}" \
  --normalize True --transform_type None --use_amp

# Exp 1-2: Source 1/2/3HP -> Target 0HP
python "${ROOT_DIR}/transfer.py" \
  --transfer_type cross_condition --seed 42 \
  --source_dataset CWRU --source_data_source 12k_DE \
  --source_workloads 1hp 2hp 3hp --source_task_type 4class \
  --target_dataset CWRU --target_data_source 12k_DE \
  --target_workloads 0hp --target_task_type 4class \
  --target_finetune_ratio "${TARGET_FINETUNE_RATIO}" \
  --source_epochs 30 --finetune_epochs 10 \
  --learning_rate 1e-4 --finetune_lr_factor 1 \
  --accumulation_steps "${ACCUMULATION_STEPS}" \
  --batch_size 16 --seq_len 1024 --window_size 1024 --p_stride 8 \
  --save_root "${SAVE_ROOT}" --model_name "${MODEL_NAME}" --model_path "${MODEL_PATH}" \
  --normalize True --transform_type None --use_amp

# Exp 1-3: Source 0/3HP -> Target 1HP
python "${ROOT_DIR}/transfer.py" \
  --transfer_type cross_condition --seed 42 \
  --source_dataset CWRU --source_data_source 12k_DE \
  --source_workloads 0hp 3hp --source_task_type 4class \
  --target_dataset CWRU --target_data_source 12k_DE \
  --target_workloads 1hp --target_task_type 4class \
  --target_finetune_ratio "${TARGET_FINETUNE_RATIO}" \
  --source_epochs 30 --finetune_epochs 10 \
  --learning_rate 1e-4 --finetune_lr_factor 1 \
  --accumulation_steps "${ACCUMULATION_STEPS}" \
  --batch_size 16 --seq_len 1024 --window_size 1024 --p_stride 8 \
  --save_root "${SAVE_ROOT}" --model_name "${MODEL_NAME}" --model_path "${MODEL_PATH}" \
  --normalize True --transform_type None --use_amp
