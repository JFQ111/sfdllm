# SFDLLM: Semantic-Anchored Fault Diagnosis via Large Language Models

This repository contains a pre-release implementation of SFDLLM for bearing fault diagnosis with large language models.

The associated paper is still under review. Some code structure, experiment settings, and documentation may be adjusted before a formal release.

---

## Overview

SFDLLM is a fault diagnosis framework that aligns vibration signals with textual fault semantics in the representation space of a frozen large language model. The current release includes the core model, training scripts, transfer scripts, and the data pipeline needed to reproduce the main experiments in this repository.

### Supported Backbone LLMs

| LLM | Type |
|-----|------|
| Qwen2.5-0.5B | Causal LM |
| GPT-2 | Causal LM |
| LLaMA-3 | Causal LM |
| BERT | Encoder |

---

## Environment Setup

```bash
conda create -n sfdllm python=3.10
conda activate sfdllm
pip install -r requirements.txt
```

Download a supported LLM checkpoint and place it under `../weights_of_models/`, or pass an absolute path through `--model_path`.

---

## Datasets and Data Loading

The dataset organization, preprocessing logic, and data-loading utilities used in this project are also maintained in:

- [fd_datasets](https://github.com/JFQ111/fd_datasets)

If you need dataset download instructions, directory structure details, or the standalone data pipeline, please refer to that repository first.

This pre-release repository expects the raw datasets to be available under `./datasets/` when running experiments locally. In addition to the main SFDLLM pipeline, this release also includes comparison baselines for `FeatureBasedLLM`, `DataBasedLLM`, `FEDformer`, and semi-supervised `DANN`. To keep the repository easier to navigate, all model implementations are organized under `model/`, while the repository root keeps only the runnable entry scripts.

The main datasets used here include:

- CWRU
- MFPT
- JNU
- PU

After preparing the data, place the corresponding dataset folders under `./datasets/`.

---

## Quickstart

### Single-domain training

```bash
python train.py \
  --dataset MFPT \
  --model_path ../weights_of_models/qwen2.5_0.5b \
  --model_name qwen_0.5b \
  --epochs 30 \
  --batch_size 16 \
  --normalize True \
  --transform_type None \
  --use_amp
```

### Cross-condition transfer

```bash
bash scripts/cross_condition_cwru.sh
```

### Cross-dataset transfer

```bash
bash scripts/cross_dataset_fewshot.sh
```

### Comparison baselines (single-domain)

```bash
python baseline_train.py \
  --dataset MFPT \
  --models feature data fedformer \
  --model_path ../weights_of_models/qwen2.5_0.5b
```

### Comparison baselines (transfer)

```bash
python baseline_transfer.py \
  --transfer_type cross_dataset \
  --source_dataset CWRU MFPT JNU \
  --target_dataset PU \
  --target_finetune_ratio 0.1 \
  --models feature data fedformer \
  --model_path ../weights_of_models/qwen2.5_0.5b
```

### Semi-supervised DANN transfer

```bash
python dann_transfer.py \
  --transfer_type cross_dataset \
  --source_dataset CWRU MFPT JNU \
  --target_dataset PU \
  --target_finetune_ratio 0.1
```

### Clear dataset cache

```python
from dataprovider.data_factory import clear_cache
clear_cache()
```

---

## Project Structure

```text
SFDLLM-release/
├── model/
│   ├── __init__.py
│   ├── sfdllm.py
│   └── baselines/
│       ├── feature_based_llm.py
│       ├── data_based_llm.py
│       ├── fedformer.py
│       └── dann.py
├── dataprovider/
│   ├── data_factory.py
│   ├── cwru_dataset.py
│   ├── pu_dataset.py
│   ├── jnu_dataset.py
│   └── mfpt_dataset.py
├── transforms/
│   └── signal_transforms.py
├── train.py
├── transfer.py
├── baseline_train.py
├── baseline_transfer.py
├── dann_transfer.py
├── scripts/
└── datasets/
```

Note: the dataset-related code in this repository is provided for running SFDLLM directly, while the standalone dataset and loader repository is [fd_datasets](https://github.com/JFQ111/fd_datasets).

---

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--seq_len` / `--window_size` | 1024 | Signal window length |
| `--patch_len` | 16 | Patch size for tokenization |
| `--p_stride` | 8 | Patch stride |
| `--num_mapping` | 1000 | Size of compressed semantic prototype set |
| `--lambda_rec` | 0.1 | Reconstruction loss weight |
| `--temperature` | 0.07 | Contrastive loss temperature |
| `--llm_layers` | 6 | Number of LLM layers used for LLaMA |
| `--redimension` | 64 | Intermediate dimension for channel reduction |

---

## Citation

The related paper has not been formally accepted yet. Citation information will be added or updated after publication.

If you need to reference this codebase before then, please cite the repository link directly.

---

## License

This project is released for academic research purposes.
