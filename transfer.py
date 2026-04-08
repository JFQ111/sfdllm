import argparse
import copy
import json
import os
import random
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
import tqdm

from dataprovider import create_dataloaders
from model import SemanticFaultAligner


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def squeeze_batch(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1: return x.unsqueeze(0)
    if x.dim() == 2: return x
    if x.dim() == 3 and x.size(1) == 1: return x[:, 0, :]
    return x.view(x.size(0), -1)


def reduce_dataloader(loader: DataLoader, ratio: float,
                      shuffle: Optional[bool] = None) -> DataLoader:
    if ratio >= 1.0:
        return loader
    n = max(1, int(len(loader.dataset) * ratio))
    idx = torch.randperm(len(loader.dataset))[:n]
    sh = True if shuffle is None else shuffle
    return DataLoader(
        Subset(loader.dataset, idx),
        batch_size=loader.batch_size, shuffle=sh,
        num_workers=loader.num_workers, pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
    )


def extract_class_names(dataset) -> Optional[List[str]]:
    candidates = []
    if hasattr(dataset, "get_class_names"): candidates.append(dataset.get_class_names())
    if hasattr(dataset, "class_names"):     candidates.append(dataset.class_names)
    if hasattr(dataset, "dataset"):
        n = extract_class_names(dataset.dataset)
        if n: candidates.append(n)
    if hasattr(dataset, "datasets"):
        for sub in dataset.datasets:
            n = extract_class_names(sub)
            if n: candidates.append(n)
    return max(candidates, key=len) if candidates else None


def infer_num_classes(dataset) -> Optional[int]:
    if hasattr(dataset, "num_classes"): return dataset.num_classes
    if hasattr(dataset, "dataset"):     return infer_num_classes(dataset.dataset)
    if hasattr(dataset, "datasets"):
        nums = [infer_num_classes(d) for d in dataset.datasets]
        nums = [n for n in nums if n is not None]
        return max(nums) if nums else None
    return None


def build_text_descriptions(class_names: List[str]) -> List[str]:
    descs = []
    for name in class_names:
        lo = name.lower()
        if "normal" in lo:
            detail = "Normal state: smooth vibration with no impulsive components."
        elif "inner" in lo:
            detail = "Inner race fault: periodic impulses at BPFI frequency."
        elif "outer" in lo:
            detail = "Outer race fault: repetitive transient shocks at BPFO frequency in the load zone."
        elif "ball" in lo:
            detail = "Rolling element fault: irregular impacts at BSF frequency."
        else:
            detail = f"{name} fault: identify by vibration pattern."
        descs.append(f"Fault diagnosis result: {detail}")
    return descs


def build_text_library(loader: DataLoader) -> Tuple[List[str], List[str]]:
    raw_names = extract_class_names(loader.dataset) or []
    num_classes = infer_num_classes(loader.dataset)

    if num_classes is not None and num_classes <= 4:
        class_names = ["normal", "inner", "outer", "ball"][:num_classes]
    else:
        class_names = raw_names
        if num_classes and len(class_names) < num_classes:
            class_names = class_names + [f"class_{i}" for i in range(len(class_names), num_classes)]
        if not class_names:
            class_names = [f"class_{i}" for i in range(num_classes or 4)]

    if set(class_names) == {"normal", "inner", "outer", "ball"}:
        class_names = ["normal", "inner", "outer", "ball"]
    return class_names, build_text_descriptions(class_names)


def clone_args(args, prefix: str) -> argparse.Namespace:
    da = copy.deepcopy(args)
    for attr in ["dataset", "dataset_weights", "data_source", "workloads", "task_type",
                 "pu_workloads", "pu_task_type", "pu_signal_type", "jnu_workloads"]:
        val = getattr(args, f"{prefix}_{attr}", None)
        if val is not None:
            setattr(da, attr, val)
    return da


def prepare_loaders(args, prefix: str, ratio: float) -> Tuple[DataLoader, DataLoader, DataLoader]:
    da = clone_args(args, prefix)
    train_l, val_l, test_l = create_dataloaders(da)
    if ratio < 1.0:
        train_l = reduce_dataloader(train_l, ratio, shuffle=True)
        val_l   = reduce_dataloader(val_l,   ratio, shuffle=False)
        test_l  = reduce_dataloader(test_l,  ratio, shuffle=False)
    return train_l, val_l, test_l


def train_one_epoch(model, loader, optimizer, scaler, device, text_desc,
                    accum_steps, use_amp) -> dict:
    model.train()
    total_loss = total_rec = correct = n = 0
    loader_len = len(loader)
    bar = tqdm.tqdm(loader, desc="Train", leave=False)
    for step, batch in enumerate(bar):
        x = squeeze_batch(batch["data"].to(device))
        y = batch["label"].to(device).long().view(-1)
        with torch.cuda.amp.autocast(enabled=use_amp):
            loss, logits, rec_loss = model(x, text_desc, y)
            loss_norm = loss / accum_steps
        if use_amp:
            scaler.scale(loss_norm).backward()
        else:
            loss_norm.backward()
        if (step + 1) % accum_steps == 0 or (step + 1) == loader_len:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            model.zero_grad()
        total_loss += loss.item()
        total_rec  += rec_loss.item()
        correct += (logits.argmax(1) == y).sum().item()
        n += y.size(0)
        bar.set_postfix(loss=f"{total_loss/(step+1):.4f}", acc=f"{correct/max(1,n):.4f}")
    nb = max(1, loader_len)
    return {"loss": total_loss/nb, "reconstruction": total_rec/nb, "accuracy": correct/max(1,n)}


def evaluate(model, loader, device, text_desc) -> dict:
    model.eval()
    total_loss = correct = n = 0
    with torch.no_grad():
        for batch in loader:
            x = squeeze_batch(batch["data"].to(device))
            y = batch["label"].to(device).long().view(-1)
            loss, logits, _ = model(x, text_desc, y)
            total_loss += loss.item()
            correct += (logits.argmax(1) == y).sum().item()
            n += y.size(0)
    nb = max(1, len(loader))
    return {"loss": total_loss/nb, "accuracy": correct/max(1,n)}


def parse_args():
    p = argparse.ArgumentParser(description="SFDLLM cross-domain transfer")
    p.add_argument("--transfer_type", type=str, default="cross_condition",
                   choices=["cross_condition", "cross_dataset"])
    p.add_argument("--target_finetune_ratio", type=float, default=0.1)
    p.add_argument("--source_epochs", type=int, default=30)
    p.add_argument("--finetune_epochs", type=int, default=10)
    p.add_argument("--finetune_lr_factor", type=float, default=0.2)
    p.add_argument("--accumulation_steps", type=int, default=4)
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--source_reduction_ratio", type=float, default=1.0)
    p.add_argument("--target_reduction_ratio", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset_weights", type=str, default=None)
    p.add_argument("--source_dataset_weights", type=str, default=None)
    p.add_argument("--target_dataset_weights", type=str, default=None)

    for prefix in ("source", "target"):
        p.add_argument(f"--{prefix}_dataset", type=str, nargs="+", default=["CWRU"],
                       choices=["CWRU", "PU", "JNU", "MFPT"])
        p.add_argument(f"--{prefix}_data_source", type=str, default="12k_DE",
                       choices=["12k_DE", "48k_DE", "both"])
        p.add_argument(f"--{prefix}_workloads", type=str, nargs="+",
                       default=["0hp", "1hp", "2hp"] if prefix == "source" else ["3hp"])
        p.add_argument(f"--{prefix}_task_type", type=str, default="4class",
                       choices=["4class", "10class"])
        p.add_argument(f"--{prefix}_pu_workloads", type=str, nargs="+",
                       default=["N15_M07_F10", "N09_M07_F10"])
        p.add_argument(f"--{prefix}_pu_task_type", type=str, default="3class_artificial")
        p.add_argument(f"--{prefix}_pu_signal_type", type=str, default="vibration",
                       choices=["vibration", "current", "both"])
        p.add_argument(f"--{prefix}_jnu_workloads", type=str, nargs="+", default=["600"])

    p.add_argument("--root_path", type=str, default="./datasets")
    p.add_argument("--sampling_rate", type=int, default=50000)
    p.add_argument("--window_size", type=int, default=1024)
    p.add_argument("--stride", type=int, default=512)
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--normalize", default=True)
    p.add_argument("--transform_type", type=str, default="None",
                   choices=["None", "cwt", "stft", "gaf", "rp", "scalogram"])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", type=bool, default=True)
    p.add_argument("--shuffle", type=bool, default=True)
    p.add_argument("--drop_last", type=bool, default=True)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--patch_len", type=int, default=16)
    p.add_argument("--p_stride", type=int, default=8)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--d_keys", type=int, default=None)
    p.add_argument("--num_mapping", type=int, default=1000)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--redimension", type=int, default=64)
    p.add_argument("--llm_layers", type=int, default=6)
    p.add_argument("--model_name", type=str, default="qwen_0.5b")
    p.add_argument("--model_path", type=str, default="../weights_of_models/qwen2.5_0.5b")
    p.add_argument("--lambda_rec", type=float, default=0.1)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--source_lr", type=float, default=None)
    p.add_argument("--finetune_lr", type=float, default=None)
    p.add_argument("--lr_step_size", type=int, default=10)
    p.add_argument("--lr_gamma", type=float, default=0.5)
    p.add_argument("--desc", type=str, default="Bearing Fault Data")
    p.add_argument("--save_root", type=str, default="results")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    set_seed(args.seed)
    if args.window_size != args.seq_len:
        args.seq_len = args.window_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    src_loaders = prepare_loaders(args, "source", args.source_reduction_ratio)
    tgt_loaders = prepare_loaders(args, "target", args.target_reduction_ratio)
    src_train, src_val, src_test = src_loaders
    tgt_train, tgt_val, tgt_test = tgt_loaders

    tgt_finetune = None
    if args.target_finetune_ratio > 0:
        tgt_finetune = reduce_dataloader(tgt_train, args.target_finetune_ratio, shuffle=True)

    src_class_names, src_text = build_text_library(src_train)
    tgt_class_names, tgt_text = build_text_library(tgt_train)

    model = SemanticFaultAligner(args).to(device)
    model.lambda_rec = args.lambda_rec
    if hasattr(model, "temperature"):
        model.temperature.data.fill_(args.temperature)
        model.temperature.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    src_lr = args.source_lr or args.learning_rate
    ft_lr  = args.finetune_lr or args.learning_rate * args.finetune_lr_factor

    optimizer = optim.AdamW(trainable, lr=src_lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    amp_scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    nowtime = time.strftime("%Y%m%d_%H%M%S")
    src_tag = "_".join(args.source_dataset)
    tgt_tag = "_".join(args.target_dataset)
    exp_dir = os.path.join(args.save_root, f"sfdllm_transfer_{args.transfer_type}_{src_tag}_to_{tgt_tag}_{nowtime}")
    os.makedirs(exp_dir, exist_ok=True)
    log_file     = os.path.join(exp_dir, "transfer_log.txt")
    metrics_file = os.path.join(exp_dir, "metrics_history.json")

    with open(os.path.join(exp_dir, "run_config.json"), "w") as f:
        json.dump({"args": vars(args), "source_classes": src_class_names,
                   "target_classes": tgt_class_names}, f, indent=2)

    history = []
    best_src_acc = 0.0
    best_src_path = os.path.join(exp_dir, "best_source_model.pth")

    print("\n===== Phase 1: Source Domain Training =====")
    for epoch in range(args.source_epochs):
        tm = train_one_epoch(model, src_train, optimizer, amp_scaler, device,
                             src_text, args.accumulation_steps, args.use_amp)
        vm = evaluate(model, src_val, device, src_text)
        scheduler.step()
        if vm["accuracy"] > best_src_acc:
            best_src_acc = vm["accuracy"]
            torch.save(model.state_dict(), best_src_path)
        history.append({"phase": "source", "epoch": epoch+1, "train": tm, "val": vm})
        with open(metrics_file, "w") as f: json.dump(history, f, indent=2)
        with open(log_file, "a") as f:
            f.write(f"[Source][Epoch {epoch+1}] train_acc={tm['accuracy']:.4f}, val_acc={vm['accuracy']:.4f}\n")
        print(f"Epoch {epoch+1}/{args.source_epochs} | Train={tm['accuracy']:.4f} Val={vm['accuracy']:.4f}")

    if os.path.exists(best_src_path):
        model.load_state_dict(torch.load(best_src_path, map_location="cpu"))
    src_test_metrics = evaluate(model, src_test, device, src_text)
    print(f"Source test accuracy: {src_test_metrics['accuracy']:.4f}")

    print("\n===== Phase 2: Zero-Shot Target Evaluation =====")
    tgt_before = evaluate(model, tgt_test, device, tgt_text)
    print(f"Target accuracy (before finetune): {tgt_before['accuracy']:.4f}")

    best_ft_acc = None
    if tgt_finetune is not None:
        print("\n===== Phase 3: Target Domain Fine-Tuning =====")
        ft_optimizer = optim.AdamW(trainable, lr=ft_lr, weight_decay=0.01)
        ft_scheduler = optim.lr_scheduler.StepLR(ft_optimizer, step_size=max(1, args.lr_step_size//2), gamma=args.lr_gamma)
        ft_scaler    = torch.cuda.amp.GradScaler(enabled=args.use_amp)
        best_ft_path = os.path.join(exp_dir, "best_finetune_model.pth")
        best_ft_acc = 0.0
        for epoch in range(args.finetune_epochs):
            tm = train_one_epoch(model, tgt_finetune, ft_optimizer, ft_scaler, device,
                                 tgt_text, args.accumulation_steps, args.use_amp)
            vm = evaluate(model, tgt_val, device, tgt_text)
            ft_scheduler.step()
            if vm["accuracy"] > best_ft_acc:
                best_ft_acc = vm["accuracy"]
                torch.save(model.state_dict(), best_ft_path)
            history.append({"phase": "finetune", "epoch": epoch+1, "train": tm, "val": vm})
            with open(metrics_file, "w") as f: json.dump(history, f, indent=2)
            with open(log_file, "a") as f:
                f.write(f"[Finetune][Epoch {epoch+1}] train_acc={tm['accuracy']:.4f}, val_acc={vm['accuracy']:.4f}\n")
            print(f"[Finetune] Epoch {epoch+1}/{args.finetune_epochs} | Train={tm['accuracy']:.4f} Val={vm['accuracy']:.4f}")
        if os.path.exists(best_ft_path):
            model.load_state_dict(torch.load(best_ft_path, map_location="cpu"))

    print("\n===== Phase 4: Final Target Evaluation =====")
    tgt_after = evaluate(model, tgt_test, device, tgt_text)
    print(f"Target accuracy (after finetune): {tgt_after['accuracy']:.4f}")

    summary = {
        "source_test": src_test_metrics, "target_before": tgt_before, "target_after": tgt_after,
        "best_source_val_acc": best_src_acc, "best_finetune_val_acc": best_ft_acc,
        "end_time": time.strftime("%Y%m%d_%H%M%S"),
    }
    with open(os.path.join(exp_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(log_file, "a") as f:
        f.write(f"\n=== Final Results ===\n"
                f"Source test: {src_test_metrics['accuracy']:.4f}\n"
                f"Target before: {tgt_before['accuracy']:.4f}\n"
                f"Target after: {tgt_after['accuracy']:.4f}\n")
    print(f"Experiment saved to: {exp_dir}")


if __name__ == "__main__":
    main()
