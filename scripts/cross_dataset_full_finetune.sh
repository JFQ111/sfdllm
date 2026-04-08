#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.."; pwd)"
SAVE_ROOT="results"
MODEL_NAME="qwen_0.5b"
MODEL_PATH="../weights_of_models/qwen2.5_0.5b"  # Set to your local LLM weights path
TARGET_FINETUNE_RATIO="1.0"
SOURCE_PU_REDUCTION_RATIO="0.1"
TARGET_PU_REDUCTION_RATIO="1.0"
ACCUMULATION_STEPS="16"

# Exp 4-1: Source CWRU+MFPT+JNU -> Target PU
python "${ROOT_DIR}/transfer.py" \
  --transfer_type cross_dataset --seed 42 \
  --source_dataset CWRU MFPT JNU \
  --target_dataset PU \
  --source_workloads 0hp 1hp 2hp 3hp --source_data_source 12k_DE --source_task_type 4class \
  --source_pu_workloads N15_M07_F10 N09_M07_F10 --source_pu_task_type 3class_artificial --source_pu_signal_type vibration \
  --source_jnu_workloads 600 800 1000 \
  --target_pu_workloads N15_M07_F10 N09_M07_F10 --target_pu_task_type 3class_artificial --target_pu_signal_type vibration \
  --target_finetune_ratio "${TARGET_FINETUNE_RATIO}" \
  --source_epochs 30 --finetune_epochs 10 \
  --learning_rate 1e-4 --finetune_lr_factor 0.2 \
  --accumulation_steps "${ACCUMULATION_STEPS}" \
  --batch_size 4 --seq_len 1024 --window_size 1024 --p_stride 8 \
  --save_root "${SAVE_ROOT}" --model_name "${MODEL_NAME}" --model_path "${MODEL_PATH}" \
  --normalize True --transform_type None \
  --source_pu_reduction_ratio "${SOURCE_PU_REDUCTION_RATIO}" \
  --target_pu_reduction_ratio "${TARGET_PU_REDUCTION_RATIO}" --use_amp

# Exp 4-2: Source CWRU+MFPT+PU -> Target JNU
python "${ROOT_DIR}/transfer.py" \
  --transfer_type cross_dataset --seed 42 \
  --source_dataset CWRU MFPT PU \
  --target_dataset JNU \
  --source_workloads 0hp 1hp 2hp 3hp --source_data_source 12k_DE --source_task_type 4class \
  --source_pu_workloads N15_M07_F10 N09_M07_F10 --source_pu_task_type 3class_artificial --source_pu_signal_type vibration \
  --target_jnu_workloads 600 800 1000 \
  --target_finetune_ratio "${TARGET_FINETUNE_RATIO}" \
  --source_epochs 30 --finetune_epochs 10 \
  --learning_rate 1e-4 --finetune_lr_factor 0.2 \
  --accumulation_steps "${ACCUMULATION_STEPS}" \
  --batch_size 4 --seq_len 1024 --window_size 1024 --p_stride 8 \
  --save_root "${SAVE_ROOT}" --model_name "${MODEL_NAME}" --model_path "${MODEL_PATH}" \
  --normalize True --transform_type None \
  --source_pu_reduction_ratio "${SOURCE_PU_REDUCTION_RATIO}" \
  --target_pu_reduction_ratio "${TARGET_PU_REDUCTION_RATIO}" --use_amp

# Exp 4-3: Source CWRU+JNU+PU -> Target MFPT
python "${ROOT_DIR}/transfer.py" \
  --transfer_type cross_dataset --seed 42 \
  --source_dataset CWRU JNU PU \
  --target_dataset MFPT \
  --source_workloads 0hp 1hp 2hp 3hp --source_data_source 12k_DE --source_task_type 4class \
  --source_pu_workloads N15_M07_F10 N09_M07_F10 --source_pu_task_type 3class_artificial --source_pu_signal_type vibration \
  --source_jnu_workloads 600 800 1000 \
  --target_finetune_ratio "${TARGET_FINETUNE_RATIO}" \
  --source_epochs 30 --finetune_epochs 10 \
  --learning_rate 1e-4 --finetune_lr_factor 0.2 \
  --accumulation_steps "${ACCUMULATION_STEPS}" \
  --batch_size 4 --seq_len 1024 --window_size 1024 --p_stride 8 \
  --save_root "${SAVE_ROOT}" --model_name "${MODEL_NAME}" --model_path "${MODEL_PATH}" \
  --normalize True --transform_type None \
  --source_pu_reduction_ratio "${SOURCE_PU_REDUCTION_RATIO}" \
  --target_pu_reduction_ratio "${TARGET_PU_REDUCTION_RATIO}" --use_amp

# Exp 4-4: Source MFPT+JNU+PU -> Target CWRU
python "${ROOT_DIR}/transfer.py" \
  --transfer_type cross_dataset --seed 42 \
  --source_dataset MFPT JNU PU \
  --target_dataset CWRU \
  --source_pu_workloads N15_M07_F10 N09_M07_F10 --source_pu_task_type 3class_artificial --source_pu_signal_type vibration \
  --source_jnu_workloads 600 800 1000 \
  --target_workloads 0hp 1hp 2hp 3hp --target_data_source 12k_DE --target_task_type 4class \
  --target_finetune_ratio "${TARGET_FINETUNE_RATIO}" \
  --source_epochs 30 --finetune_epochs 10 \
  --learning_rate 1e-4 --finetune_lr_factor 0.2 \
  --accumulation_steps "${ACCUMULATION_STEPS}" \
  --batch_size 4 --seq_len 1024 --window_size 1024 --p_stride 8 \
  --save_root "${SAVE_ROOT}" --model_name "${MODEL_NAME}" --model_path "${MODEL_PATH}" \
  --normalize True --transform_type None \
  --source_pu_reduction_ratio "${SOURCE_PU_REDUCTION_RATIO}" \
  --target_pu_reduction_ratio "${TARGET_PU_REDUCTION_RATIO}" --use_amp
