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
from torch.utils.data import ConcatDataset, DataLoader, Subset, WeightedRandomSampler
import tqdm

from dataprovider import create_dataloaders
from model import DANNModel

# ============== 通用工具函数 ==============
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def reduce_dataloader(original_loader: DataLoader, reduction_ratio: float, shuffle_override: Optional[bool] = None) -> DataLoader:
    if reduction_ratio >= 1.0:
        return original_loader
    original_dataset = original_loader.dataset
    total_size = len(original_dataset)
    subset_size = max(1, int(total_size * reduction_ratio))
    indices = torch.randperm(total_size)[:subset_size]
    subset_dataset = Subset(original_dataset, indices)
    shuffle = True if shuffle_override is None else shuffle_override
    return DataLoader(
        subset_dataset,
        batch_size=original_loader.batch_size,
        shuffle=shuffle,
        num_workers=original_loader.num_workers,
        pin_memory=original_loader.pin_memory,
        drop_last=original_loader.drop_last,
        sampler=None,
    )

def squeeze_signal_batch(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        return x.unsqueeze(0)
    if x.dim() == 2:
        return x
    if x.dim() == 3 and x.size(1) == 1:
        return x[:, 0, :]
    return x.view(x.size(0), -1)

# ============== 数据加载与配置处理 ==============
def clone_domain_args(args, prefix: str) -> argparse.Namespace:
    domain_args = copy.deepcopy(args)
    domain_args.dataset = getattr(args, f"{prefix}_dataset")
    domain_args.dataset_weights = getattr(args, f"{prefix}_dataset_weights") or args.dataset_weights
    domain_args.data_source = getattr(args, f"{prefix}_data_source")
    domain_args.workloads = getattr(args, f"{prefix}_workloads")
    domain_args.task_type = getattr(args, f"{prefix}_task_type")
    domain_args.pu_workloads = getattr(args, f"{prefix}_pu_workloads")
    domain_args.pu_task_type = getattr(args, f"{prefix}_pu_task_type")
    domain_args.pu_signal_type = getattr(args, f"{prefix}_pu_signal_type")
    domain_args.jnu_workloads = getattr(args, f"{prefix}_jnu_workloads")
    return domain_args

def prepare_domain_loaders(args, prefix: str, reduction_ratio: float) -> Tuple[DataLoader, DataLoader, DataLoader]:
    domain_args = clone_domain_args(args, prefix)
    train_loader, val_loader, test_loader = create_dataloaders(domain_args)
    if reduction_ratio < 1.0:
        print(f"[{prefix}] 启用全局数据裁剪，比例: {reduction_ratio:.2f}")
        train_loader = reduce_dataloader(train_loader, reduction_ratio, shuffle_override=True)
        val_loader = reduce_dataloader(val_loader, reduction_ratio, shuffle_override=False)
        test_loader = reduce_dataloader(test_loader, reduction_ratio, shuffle_override=False)
    return train_loader, val_loader, test_loader

# ============== 训练与评估 ==============
def train_dann_epoch(model: DANNModel,
                     source_loader: DataLoader,
                     target_loader: DataLoader,
                     optimizer,
                     scaler,
                     device: torch.device,
                     epoch: int,
                     total_epochs: int,
                     dann_mode: str,
                     use_amp: bool):
    model.train()
    
    len_source = len(source_loader)
    len_target = len(target_loader) if target_loader is not None else 0
    num_batches = max(len_source, len_target) if len_target > 0 else len_source
    
    source_iter = iter(source_loader)
    target_iter = iter(target_loader) if target_loader is not None else None
    
    epoch_cls_loss = 0.0
    epoch_domain_loss = 0.0
    correct_s = 0
    total_s = 0
    
    progress_bar = tqdm.tqdm(range(num_batches), desc=f"Train Epoch {epoch}", leave=False)
    
    for step in progress_bar:
        # GRL alpha scaling 0 to 1
        p = float(step + epoch * num_batches) / (total_epochs * num_batches)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        
        try:
            batch_s = next(source_iter)
        except StopIteration:
            source_iter = iter(source_loader)
            batch_s = next(source_iter)
            
        x_s = squeeze_signal_batch(batch_s["data"]).to(device)
        y_s = batch_s["label"].to(device).long().view(-1)
        
        if target_iter is not None:
            try:
                batch_t = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                batch_t = next(target_iter)
                
            x_t = squeeze_signal_batch(batch_t["data"]).to(device)
            y_t = batch_t["label"].to(device).long().view(-1)
        else:
            x_t = None
            y_t = None
            
        with torch.cuda.amp.autocast(enabled=use_amp):
            class_out_s, domain_out_s = model(x_s, alpha)
            loss_cls_s = F.cross_entropy(class_out_s, y_s)
            
            domain_label_s = torch.zeros(x_s.size(0), dtype=torch.long).to(device)
            loss_domain_s = F.cross_entropy(domain_out_s, domain_label_s)
            
            if x_t is not None:
                class_out_t, domain_out_t = model(x_t, alpha)
                domain_label_t = torch.ones(x_t.size(0), dtype=torch.long).to(device)
                loss_domain_t = F.cross_entropy(domain_out_t, domain_label_t)
                
                if dann_mode == "semi":
                    loss_cls_t = F.cross_entropy(class_out_t, y_t)
                    loss_cls = (loss_cls_s + loss_cls_t) / 2.0
                else:
                    loss_cls = loss_cls_s
                    
                loss_domain = (loss_domain_s + loss_domain_t) / 2.0
            else:
                loss_cls = loss_cls_s
                loss_domain = loss_domain_s
                
            total_loss = loss_cls + loss_domain
            
        if use_amp:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        optimizer.zero_grad()
        
        preds_s = torch.argmax(class_out_s, dim=1)
        correct_s += (preds_s == y_s).sum().item()
        total_s += y_s.size(0)
        
        epoch_cls_loss += loss_cls.item()
        epoch_domain_loss += loss_domain.item()
        
        progress_bar.set_postfix({
            "cls_loss": f"{epoch_cls_loss / (step + 1):.4f}",
            "dom_loss": f"{epoch_domain_loss / (step + 1):.4f}",
            "acc_s": f"{correct_s / max(1, total_s):.4f}"
        })
        
    return {
        "loss_cls": epoch_cls_loss / num_batches,
        "loss_domain": epoch_domain_loss / num_batches,
        "accuracy": correct_s / max(1, total_s)
    }

def evaluate_dann(model: DANNModel, data_loader: DataLoader, device: torch.device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in data_loader:
            x = squeeze_signal_batch(batch["data"]).to(device)
            y = batch["label"].to(device).long().view(-1)
            
            class_out, _ = model(x, alpha=0.0)
            loss_cls = F.cross_entropy(class_out, y)
            preds = torch.argmax(class_out, dim=1)
            
            total_loss += loss_cls.item()
            correct += (preds == y).sum().item()
            total_samples += y.size(0)
            
    num_batches = max(1, len(data_loader))
    return {
        "loss": total_loss / num_batches,
        "accuracy": correct / max(1, total_samples)
    }

# ============== 参数解析 ==============
def parse_args():
    parser = argparse.ArgumentParser(description="半监督 DANN 跨工况/跨数据集迁移脚本")
    
    parser.add_argument('--transfer_type', type=str, default='cross_condition',
                        choices=['cross_condition', 'cross_dataset'])
    parser.add_argument('--dann_mode', type=str, default='semi',
                        choices=['semi'])
    parser.add_argument('--target_finetune_ratio', type=float, default=0.1,
                        help='目标域训练集用于DANN域损失(及半监督分类)的采样比例')
    parser.add_argument('--source_epochs', type=int, default=30)
    parser.add_argument('--use_amp', action='store_true', help='是否启用混合精度')
    parser.add_argument('--source_reduction_ratio', type=float, default=1.0)
    parser.add_argument('--target_reduction_ratio', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--dataset_weights', type=str, default=None)
    parser.add_argument('--source_dataset_weights', type=str, default=None)
    parser.add_argument('--target_dataset_weights', type=str, default=None)

    # 源域参数
    parser.add_argument('--source_dataset', type=str, nargs='+', default=['CWRU'],
                        choices=['CWRU', 'PU', 'JNU', 'MFPT'])
    parser.add_argument('--source_data_source', type=str, default='12k_DE')
    parser.add_argument('--source_workloads', type=str, nargs='+', default=['0hp', '1hp', '2hp'])
    parser.add_argument('--source_task_type', type=str, default='4class')
    parser.add_argument('--source_pu_workloads', type=str, nargs='+', default=['N15_M07_F10', 'N09_M07_F10'])
    parser.add_argument('--source_pu_task_type', type=str, default='3class_artificial')
    parser.add_argument('--source_pu_signal_type', type=str, default='vibration')
    parser.add_argument('--source_jnu_workloads', type=str, nargs='+', default=['600'])

    # 目标域参数
    parser.add_argument('--target_dataset', type=str, nargs='+', default=['PU'],
                        choices=['CWRU', 'PU', 'JNU', 'MFPT'])
    parser.add_argument('--target_data_source', type=str, default='12k_DE')
    parser.add_argument('--target_workloads', type=str, nargs='+', default=['0hp', '1hp'])
    parser.add_argument('--target_task_type', type=str, default='4class')
    parser.add_argument('--target_pu_workloads', type=str, nargs='+', default=['N09_M07_F10'])
    parser.add_argument('--target_pu_task_type', type=str, default='3class_artificial')
    parser.add_argument('--target_pu_signal_type', type=str, default='vibration')
    parser.add_argument('--target_jnu_workloads', type=str, nargs='+', default=['1000'])

    # 数据集基础参数
    parser.add_argument('--root_path', type=str, default='./datasets')
    parser.add_argument('--sampling_rate', type=int, default=50000)
    parser.add_argument('--window_size', type=int, default=1024)
    parser.add_argument('--stride', type=int, default=512)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--normalize', default=True)
    parser.add_argument('--transform_type', type=str, default='None')

    # dataloader设置
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--drop_last', type=bool, default=True)

    # DANN训练参数
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--lr_step_size', type=int, default=10)
    parser.add_argument('--lr_gamma', type=float, default=0.5)
    parser.add_argument('--save_root', type=str, default='results')

    return parser.parse_args()

def main():
    args = parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DANN Mode: {args.dann_mode}")
    print(f"Target Finetune Ratio: {args.target_finetune_ratio}")

    source_loaders = prepare_domain_loaders(args, "source", args.source_reduction_ratio)
    target_loaders = prepare_domain_loaders(args, "target", args.target_reduction_ratio)
    source_train_loader, source_val_loader, source_test_loader = source_loaders
    target_train_loader, target_val_loader, target_test_loader = target_loaders

    target_domain_loader = None
    if args.target_finetune_ratio > 0:
        target_domain_loader = reduce_dataloader(
            target_train_loader, args.target_finetune_ratio, shuffle_override=True
        )

    # Model inference num classes
    dataset_example = source_train_loader.dataset
    if hasattr(dataset_example, "num_classes"):
        num_classes = dataset_example.num_classes
    elif hasattr(dataset_example, "datasets") and hasattr(dataset_example.datasets[0], "num_classes"):
        num_classes = dataset_example.datasets[0].num_classes
    else:
        num_classes = 4

    model = DANNModel(num_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    nowtime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    source_tag = "_".join(args.source_dataset)
    target_tag = "_".join(args.target_dataset)
    mode_tag = f"dann_{args.dann_mode}"
    exp_dir = os.path.join(args.save_root, f"{mode_tag}_transfer_{args.transfer_type}_{source_tag}_to_{target_tag}_{nowtime}")
    os.makedirs(exp_dir, exist_ok=True)
    
    log_file = os.path.join(exp_dir, "transfer_log.txt")
    summary_file = os.path.join(exp_dir, "summary.json")

    best_source_acc = 0.0
    best_target_val_acc = 0.0
    best_model_path = os.path.join(exp_dir, "best_dann_model.pth")

    for epoch in range(args.source_epochs):
        train_metrics = train_dann_epoch(
            model, source_train_loader, target_domain_loader, optimizer, scaler,
            device, epoch, args.source_epochs, args.dann_mode, args.use_amp
        )
        
        val_metrics_s = evaluate_dann(model, source_val_loader, device)
        val_metrics_t = evaluate_dann(model, target_val_loader, device)
        scheduler.step()

        if val_metrics_t["accuracy"] > best_target_val_acc:
            best_target_val_acc = val_metrics_t["accuracy"]
            torch.save(model.state_dict(), best_model_path)

        with open(log_file, "a") as f:
            f.write(f"[Epoch {epoch + 1}] "
                    f"train_cls_loss={train_metrics['loss_cls']:.4f}, "
                    f"train_dom_loss={train_metrics['loss_domain']:.4f}, "
                    f"val_acc_s={val_metrics_s['accuracy']:.4f}, "
                    f"val_acc_t={val_metrics_t['accuracy']:.4f}\n")
        print(f"Epoch {epoch + 1}/{args.source_epochs} | "
              f"Val Acc S: {val_metrics_s['accuracy']:.4f} | "
              f"Val Acc T: {val_metrics_t['accuracy']:.4f}")

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        
    source_test_metrics = evaluate_dann(model, source_test_loader, device)
    target_test_metrics = evaluate_dann(model, target_test_loader, device)
    
    print(f"源域测试准确率: {source_test_metrics['accuracy']:.4f}")
    print(f"目标域测试准确率: {target_test_metrics['accuracy']:.4f}")

    summary = {
        "dann_mode": args.dann_mode,
        "target_finetune_ratio": args.target_finetune_ratio,
        "exp_dir": exp_dir,
        "source_test": source_test_metrics,
        "target_test": target_test_metrics,
        "best_target_val_acc": best_target_val_acc,
        "end_time": time.strftime("%Y%m%d_%H%M%S", time.localtime()),
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
