import argparse
import json
import os
import random
import time
from typing import List, Optional

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


def reduce_dataloader(loader: DataLoader, ratio: float) -> DataLoader:
    if ratio >= 1.0:
        return loader
    n = max(1, int(len(loader.dataset) * ratio))
    idx = torch.randperm(len(loader.dataset))[:n]
    return DataLoader(
        Subset(loader.dataset, idx),
        batch_size=loader.batch_size, shuffle=True,
        num_workers=loader.num_workers, pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
    )


def squeeze_batch(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1: return x.unsqueeze(0)
    if x.dim() == 2: return x
    if x.dim() == 3 and x.size(1) == 1: return x[:, 0, :]
    return x.view(x.size(0), -1)


def extract_class_names(dataset) -> Optional[List[str]]:
    if hasattr(dataset, "get_class_names"): return dataset.get_class_names()
    if hasattr(dataset, "dataset"): return extract_class_names(dataset.dataset)
    if hasattr(dataset, "datasets"):
        for ds in dataset.datasets:
            n = extract_class_names(ds)
            if n: return n
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


def evaluate(model: SemanticFaultAligner, loader: DataLoader, device: torch.device,
             text_desc: List[str]) -> dict:
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
    return {"loss": total_loss / nb, "accuracy": correct / max(1, n)}


def parse_args():
    p = argparse.ArgumentParser(description="SFDLLM single-domain training")
    p.add_argument("--dataset", type=str, nargs="+", default=["MFPT"],
                   choices=["CWRU", "PU", "JNU", "MFPT"])
    p.add_argument("--dataset_weights", type=str, default=None)
    p.add_argument("--root_path", type=str, default="./datasets")
    p.add_argument("--data_source", type=str, default="12k_DE",
                   choices=["12k_DE", "48k_DE", "both"])
    p.add_argument("--workloads", type=str, nargs="+", default=["0hp", "1hp", "2hp", "3hp"])
    p.add_argument("--task_type", type=str, default="4class", choices=["4class", "10class"])
    p.add_argument("--pu_workloads", type=str, nargs="+", default=["N15_M07_F10"])
    p.add_argument("--pu_task_type", type=str, default="3class_artificial")
    p.add_argument("--pu_signal_type", type=str, default="vibration",
                   choices=["vibration", "current", "both"])
    p.add_argument("--jnu_workloads", type=str, nargs="+", default=["600"])
    p.add_argument("--sampling_rate", type=int, default=50000)
    p.add_argument("--window_size", type=int, default=1024)
    p.add_argument("--stride", type=int, default=512)
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--normalize", default=True)
    p.add_argument("--transform_type", type=str, default="None",
                   choices=["None", "cwt", "stft", "gaf", "rp", "scalogram"])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--accumulation_steps", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", type=bool, default=True)
    p.add_argument("--shuffle", type=bool, default=True)
    p.add_argument("--drop_last", type=bool, default=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--lr_step_size", type=int, default=10)
    p.add_argument("--lr_gamma", type=float, default=0.1)
    p.add_argument("--reduction_ratio", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
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
    train_loader, val_loader, test_loader = create_dataloaders(args)

    if args.reduction_ratio < 1.0:
        train_loader = reduce_dataloader(train_loader, args.reduction_ratio)
        val_loader   = reduce_dataloader(val_loader, args.reduction_ratio)

    class_names = extract_class_names(train_loader.dataset) or [f"class_{i}" for i in range(4)]
    if set(args.dataset) == {"CWRU"} and args.task_type == "4class":
        ordered = ["normal", "inner", "outer", "ball"]
        if set(class_names) == set(ordered):
            class_names = ordered
    text_desc = build_text_descriptions(class_names)

    model = SemanticFaultAligner(args).to(device)
    model.lambda_rec = args.lambda_rec
    if hasattr(model, "temperature"):
        model.temperature.data.fill_(args.temperature)
        model.temperature.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    nowtime = time.strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_root, f"sfdllm_{'_'.join(args.dataset)}_{nowtime}")
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, "train_log.txt")
    metrics_file = os.path.join(save_dir, "metrics_history.json")
    best_path = os.path.join(save_dir, "best_model.pth")

    with open(os.path.join(save_dir, "run_config.json"), "w") as f:
        json.dump({"args": vars(args), "class_names": class_names, "text_desc": text_desc}, f, indent=2)

    best_acc = 0.0
    records = []
    model.zero_grad()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = epoch_rec = correct = total = 0
        loader_len = len(train_loader)
        bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)

        for i, batch in enumerate(bar):
            x = squeeze_batch(batch["data"].to(device))
            y = batch["label"].to(device).long().view(-1)

            with torch.cuda.amp.autocast(enabled=args.use_amp):
                loss, logits, rec_loss = model(x, text_desc, y)
                loss_norm = loss / args.accumulation_steps

            if args.use_amp:
                scaler.scale(loss_norm).backward()
            else:
                loss_norm.backward()

            if (i + 1) % args.accumulation_steps == 0 or (i + 1) == loader_len:
                if args.use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                model.zero_grad()

            epoch_loss += loss.item()
            epoch_rec  += rec_loss.item()
            correct += (logits.argmax(1) == y).sum().item()
            total   += y.size(0)
            bar.set_postfix(loss=f"{epoch_loss/(i+1):.4f}", acc=f"{correct/max(1,total):.4f}")

        nb = max(1, loader_len)
        train_metrics = {
            "loss": epoch_loss / nb, "reconstruction": epoch_rec / nb,
            "accuracy": correct / max(1, total),
        }
        val_metrics = evaluate(model, val_loader, device, text_desc)
        scheduler.step()

        records.append({"epoch": epoch+1, "train": train_metrics, "val": val_metrics,
                        "lr": optimizer.param_groups[0]["lr"]})
        with open(metrics_file, "w") as f:
            json.dump(records, f, indent=2)
        with open(log_file, "a") as f:
            f.write(f"Epoch {epoch+1}: train_acc={train_metrics['accuracy']:.4f}, "
                    f"val_acc={val_metrics['accuracy']:.4f}\n")

        print(f"[Epoch {epoch+1}/{args.epochs}] Loss={train_metrics['loss']:.4f} "
              f"TrainAcc={train_metrics['accuracy']:.4f} ValAcc={val_metrics['accuracy']:.4f}")

        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            torch.save(model.state_dict(), best_path)

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location="cpu"))
    test_metrics = evaluate(model, test_loader, device, text_desc)
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")

    summary = {
        "best_val_acc": best_acc, "test_metrics": test_metrics,
        "save_dir": save_dir,
        "end_time": time.strftime("%Y%m%d_%H%M%S"),
    }
    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
