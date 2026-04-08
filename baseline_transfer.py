import argparse
import copy
import json
import os
import random
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from dataprovider import create_dataloaders
from model import DataBasedLLM, FEDformer, FeatureBasedLLM, extract_class_names, extract_sampling_rate


def set_seed(seed: int) -> None:
    """固定随机种子，保证可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def squeeze_signal_batch(x: torch.Tensor) -> torch.Tensor:
    """将输入压缩为[B, L]，适配一维信号模型。"""
    if x.dim() == 1:
        return x.unsqueeze(0)
    if x.dim() == 2:
        return x
    if x.dim() == 3 and x.size(1) == 1:
        return x[:, 0, :]
    return x.view(x.size(0), -1)


def unpack_batch(batch):
    if isinstance(batch, dict):
        return batch.get("data"), batch.get("label")
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise ValueError("Unsupported batch format")


def reduce_dataloader(original_loader: DataLoader, reduction_ratio: float, shuffle_override: Optional[bool] = None) -> DataLoader:
    """按比例裁剪DataLoader，便于快速调试或少量微调。"""
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


def infer_num_classes(dataset) -> Optional[int]:
    if hasattr(dataset, "num_classes"):
        return dataset.num_classes
    if hasattr(dataset, "dataset"):
        return infer_num_classes(dataset.dataset)
    if hasattr(dataset, "datasets"):
        nums = [infer_num_classes(ds) for ds in dataset.datasets]
        nums = [n for n in nums if n is not None]
        if nums:
            return max(nums)
    return None


def get_feature_llm_device(model):
    if hasattr(model, "model") and hasattr(model.model, "device"):
        return model.model.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sync_feature_model(model: FeatureBasedLLM, loader: DataLoader) -> None:
    class_names = extract_class_names(loader.dataset)
    if class_names:
        model.set_class_names(class_names)
    sampling_rate = extract_sampling_rate(loader.dataset)
    if sampling_rate:
        model.set_sampling_rate(sampling_rate)


def train_one_epoch_cls(model,
                        loader: DataLoader,
                        device: torch.device,
                        optimizer,
                        scaler=None,
                        accumulation_steps: int = 1,
                        max_batches: Optional[int] = None,
                        progress_desc: str = "Train"):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    steps = 0

    optimizer.zero_grad()
    total_batches = max_batches if max_batches is not None else len(loader)
    progress_bar = tqdm.tqdm(loader, total=total_batches, desc=progress_desc, leave=False)

    for step_idx, batch in enumerate(progress_bar):
        if step_idx >= total_batches:
            break
        data, labels = unpack_batch(batch)
        data = squeeze_signal_batch(data.to(device))
        labels = labels.to(device).long().view(-1)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(data)
                loss = F.cross_entropy(logits, labels)
                loss_scaled = loss / max(1, accumulation_steps)
            scaler.scale(loss_scaled).backward()
        else:
            logits = model(data)
            loss = F.cross_entropy(logits, labels)
            loss_scaled = loss / max(1, accumulation_steps)
            loss_scaled.backward()

        is_update = ((step_idx + 1) % max(1, accumulation_steps) == 0) or ((step_idx + 1) == total_batches)
        if is_update:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item()
        steps += 1

        progress_bar.set_postfix({
            "loss": f"{total_loss / max(1, steps):.4f}",
            "acc": f"{correct / max(1, total):.4f}",
        })

    avg_loss = total_loss / max(1, steps)
    acc = correct / max(1, total)
    return avg_loss, acc


def evaluate_cls(model, loader: DataLoader, device: torch.device, max_batches: Optional[int] = None):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    steps = 0

    with torch.no_grad():
        for step_idx, batch in enumerate(loader):
            if max_batches is not None and step_idx >= max_batches:
                break
            data, labels = unpack_batch(batch)
            data = squeeze_signal_batch(data.to(device))
            labels = labels.to(device).long().view(-1)

            logits = model(data)
            loss = F.cross_entropy(logits, labels)
            preds = torch.argmax(logits, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()
            steps += 1

    avg_loss = total_loss / max(1, steps)
    acc = correct / max(1, total)
    return avg_loss, acc


def train_one_epoch_feature(model: FeatureBasedLLM,
                            loader: DataLoader,
                            optimizer,
                            accumulation_steps: int = 1,
                            max_batches: Optional[int] = None,
                            progress_desc: str = "Train"):
    model.train()
    total_loss = 0.0
    steps = 0

    sync_feature_model(model, loader)
    device = get_feature_llm_device(model)
    optimizer.zero_grad()
    total_batches = max_batches if max_batches is not None else len(loader)
    progress_bar = tqdm.tqdm(loader, total=total_batches, desc=progress_desc, leave=False)

    for step_idx, batch in enumerate(progress_bar):
        if step_idx >= total_batches:
            break
        data, labels = unpack_batch(batch)
        data = squeeze_signal_batch(data.to(device))
        labels = labels.to(device).long().view(-1)

        loss = model(data, labels)
        loss_scaled = loss / max(1, accumulation_steps)
        loss_scaled.backward()

        is_update = ((step_idx + 1) % max(1, accumulation_steps) == 0) or ((step_idx + 1) == total_batches)
        if is_update:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        steps += 1

        progress_bar.set_postfix({
            "loss": f"{total_loss / max(1, steps):.4f}",
        })

    return total_loss / max(1, steps)


def evaluate_feature(model: FeatureBasedLLM, loader: DataLoader, max_batches: Optional[int] = None):
    model.eval()
    correct = 0
    total = 0

    sync_feature_model(model, loader)
    device = get_feature_llm_device(model)

    with torch.no_grad():
        for step_idx, batch in enumerate(loader):
            if max_batches is not None and step_idx >= max_batches:
                break
            data, labels = unpack_batch(batch)
            data = squeeze_signal_batch(data.to(device))
            labels = labels.to(device).long().view(-1)

            preds = model.generate_diagnosis(data)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / max(1, total)
    return acc


def clone_domain_args(args, prefix: str) -> argparse.Namespace:
    """复制一份args，并根据前缀(source/target)替换数据相关参数。"""
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
    """为指定域创建dataloader，并可选按比例裁剪。"""
    domain_args = clone_domain_args(args, prefix)
    train_loader, val_loader, test_loader = create_dataloaders(domain_args)

    if reduction_ratio < 1.0:
        print(f"[{prefix}] 启用全局数据裁剪，比例: {reduction_ratio:.2f}")
        train_loader = reduce_dataloader(train_loader, reduction_ratio, shuffle_override=True)
        val_loader = reduce_dataloader(val_loader, reduction_ratio, shuffle_override=False)
        test_loader = reduce_dataloader(test_loader, reduction_ratio, shuffle_override=False)

    return train_loader, val_loader, test_loader


def init_exp_dir(args, model_tag: str, source_tag: str, target_tag: str) -> Tuple[str, str]:
    nowtime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    exp_dir = os.path.join(
        args.save_root,
        f"baseline_transfer_{model_tag}_{args.transfer_type}_{source_tag}_to_{target_tag}_{nowtime}"
    )
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir, nowtime


def run_feature_transfer(args,
                         source_loaders: Tuple[DataLoader, DataLoader, DataLoader],
                         target_loaders: Tuple[DataLoader, DataLoader, DataLoader],
                         target_finetune_loader: Optional[DataLoader],
                         source_class_names: List[str],
                         target_class_names: List[str]):
    source_train_loader, source_val_loader, source_test_loader = source_loaders
    target_train_loader, target_val_loader, target_test_loader = target_loaders

    source_tag = "_".join(args.source_dataset)
    target_tag = "_".join(args.target_dataset)
    exp_dir, nowtime = init_exp_dir(args, "feature", source_tag, target_tag)
    log_file = os.path.join(exp_dir, "transfer_log.txt")
    metrics_file = os.path.join(exp_dir, "metrics_history.json")
    summary_file = os.path.join(exp_dir, "summary.json")
    config_file = os.path.join(exp_dir, "run_config.json")

    model = FeatureBasedLLM(args)
    model_device = get_feature_llm_device(model)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    source_lr = args.source_lr if args.source_lr is not None else args.learning_rate
    finetune_lr = args.finetune_lr if args.finetune_lr is not None else args.learning_rate * args.finetune_lr_factor
    print(f"[Feature] 可训练参数量: {sum(p.numel() for p in trainable_params)}")
    print(f"[Feature] 源域学习率: {source_lr}, 微调学习率: {finetune_lr}")

    run_config = {
        "args": vars(args),
        "device": str(model_device),
        "model": "FeatureBasedLLM",
        "model_tag": "feature",
        "start_time": nowtime,
        "exp_dir": exp_dir,
        "source_class_names": source_class_names,
        "target_class_names": target_class_names,
    }
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    metrics_history = []
    best_source_loss = None
    best_source_path = os.path.join(exp_dir, "best_source_model.pth")

    print("\n===== FeatureBasedLLM: 源域训练 =====")
    optimizer = optim.AdamW(trainable_params, lr=source_lr, weight_decay=0.01)
    for epoch in range(args.source_epochs):
        train_loss = train_one_epoch_feature(
            model,
            source_train_loader,
            optimizer,
            accumulation_steps=args.accumulation_steps,
            max_batches=args.max_batches,
            progress_desc="Feature Source",
        )
        if best_source_loss is None or train_loss < best_source_loss:
            best_source_loss = train_loss
            torch.save(model.state_dict(), best_source_path)

        metrics_history.append({
            "phase": "source",
            "epoch": epoch + 1,
            "train": {"loss": train_loss},
            "val": None,
            "lr": optimizer.param_groups[0]["lr"],
        })
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics_history, f, ensure_ascii=False, indent=2)
        with open(log_file, "a") as f:
            f.write(f"[Source][Epoch {epoch + 1}] train_loss={train_loss:.4f}\n")
        print(f"[Feature][Epoch {epoch + 1}/{args.source_epochs}] loss={train_loss:.4f}")

    if os.path.exists(best_source_path):
        model.load_state_dict(torch.load(best_source_path, map_location="cpu"))

    best_finetune_loss = None
    best_finetune_path = None
    if target_finetune_loader is not None:
        print("\n===== FeatureBasedLLM: 目标域微调 =====")
        finetune_optimizer = optim.AdamW(trainable_params, lr=finetune_lr, weight_decay=0.01)
        best_finetune_path = os.path.join(exp_dir, "best_finetune_model.pth")
        best_finetune_loss = None

        for epoch in range(args.finetune_epochs):
            train_loss = train_one_epoch_feature(
                model,
                target_finetune_loader,
                finetune_optimizer,
                accumulation_steps=args.accumulation_steps,
                max_batches=args.max_batches,
                progress_desc="Feature Finetune",
            )
            if best_finetune_loss is None or train_loss < best_finetune_loss:
                best_finetune_loss = train_loss
                torch.save(model.state_dict(), best_finetune_path)

            metrics_history.append({
                "phase": "finetune",
                "epoch": epoch + 1,
                "train": {"loss": train_loss},
                "val": None,
                "lr": finetune_optimizer.param_groups[0]["lr"],
            })
            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump(metrics_history, f, ensure_ascii=False, indent=2)
            with open(log_file, "a") as f:
                f.write(f"[Finetune][Epoch {epoch + 1}] train_loss={train_loss:.4f}\n")
            print(f"[Feature][Finetune {epoch + 1}/{args.finetune_epochs}] loss={train_loss:.4f}")

        if os.path.exists(best_finetune_path):
            model.load_state_dict(torch.load(best_finetune_path, map_location="cpu"))

    print("\n===== FeatureBasedLLM: 最终目标域测试 =====")
    target_after_acc = evaluate_feature(model, target_test_loader, max_batches=args.max_batches)
    print(f"[Feature] 目标域测试准确率（微调后）: {target_after_acc:.4f}")

    summary = {
        "exp_dir": exp_dir,
        "model": "FeatureBasedLLM",
        "model_tag": "feature",
        "source_test": None,
        "target_before": None,
        "target_after": {"acc": target_after_acc},
        "best_source_val_acc": None,
        "best_finetune_val_acc": None,
        "best_source_train_loss": best_source_loss,
        "best_finetune_train_loss": best_finetune_loss,
        "best_source_model": best_source_path,
        "best_finetune_model": best_finetune_path,
        "config_file": config_file,
        "log_file": log_file,
        "metrics_file": metrics_file,
        "end_time": time.strftime("%Y%m%d_%H%M%S", time.localtime()),
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(log_file, "a") as f:
        f.write("\n===== 最终结果 =====\n")
        f.write(f"目标域测试(后): acc={target_after_acc:.4f}\n")

    print(f"[Feature] 实验完成，结果保存在: {exp_dir}")
    del model
    torch.cuda.empty_cache()


def run_cls_transfer(model_tag: str,
                     model_ctor,
                     args,
                     device: torch.device,
                     source_loaders: Tuple[DataLoader, DataLoader, DataLoader],
                     target_loaders: Tuple[DataLoader, DataLoader, DataLoader],
                     target_finetune_loader: Optional[DataLoader],
                     source_class_names: List[str],
                     target_class_names: List[str]):
    source_train_loader, source_val_loader, source_test_loader = source_loaders
    target_train_loader, target_val_loader, target_test_loader = target_loaders

    source_tag = "_".join(args.source_dataset)
    target_tag = "_".join(args.target_dataset)
    exp_dir, nowtime = init_exp_dir(args, model_tag, source_tag, target_tag)
    log_file = os.path.join(exp_dir, "transfer_log.txt")
    metrics_file = os.path.join(exp_dir, "metrics_history.json")
    summary_file = os.path.join(exp_dir, "summary.json")
    config_file = os.path.join(exp_dir, "run_config.json")

    model = model_ctor(args).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    source_lr = args.source_lr if args.source_lr is not None else args.learning_rate
    finetune_lr = args.finetune_lr if args.finetune_lr is not None else args.learning_rate * args.finetune_lr_factor
    print(f"[{model_tag}] 可训练参数量: {sum(p.numel() for p in trainable_params)}")
    print(f"[{model_tag}] 源域学习率: {source_lr}, 微调学习率: {finetune_lr}")

    run_config = {
        "args": vars(args),
        "device": str(device),
        "model": model_tag,
        "model_tag": model_tag,
        "start_time": nowtime,
        "exp_dir": exp_dir,
        "source_class_names": source_class_names,
        "target_class_names": target_class_names,
    }
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    metrics_history = []
    best_source_train_acc = 0.0
    best_source_path = os.path.join(exp_dir, "best_source_model.pth")

    print(f"\n===== {model_tag}: 源域训练 =====")
    optimizer = optim.AdamW(trainable_params, lr=source_lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    for epoch in range(args.source_epochs):
        train_loss, train_acc = train_one_epoch_cls(
            model,
            source_train_loader,
            device,
            optimizer,
            scaler=scaler,
            accumulation_steps=args.accumulation_steps,
            max_batches=args.max_batches,
            progress_desc=f"{model_tag} Source",
        )
        if train_acc > best_source_train_acc:
            best_source_train_acc = train_acc
            torch.save(model.state_dict(), best_source_path)

        metrics_history.append({
            "phase": "source",
            "epoch": epoch + 1,
            "train": {"loss": train_loss, "acc": train_acc},
            "val": None,
            "lr": optimizer.param_groups[0]["lr"],
        })
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics_history, f, ensure_ascii=False, indent=2)
        with open(log_file, "a") as f:
            f.write(f"[Source][Epoch {epoch + 1}] "
                    f"train_acc={train_acc:.4f}\n")
        print(f"[{model_tag}][Epoch {epoch + 1}/{args.source_epochs}] "
              f"train_acc={train_acc:.4f}")

    if os.path.exists(best_source_path):
        model.load_state_dict(torch.load(best_source_path, map_location="cpu"))

    best_finetune_train_acc = None
    best_finetune_path = None
    if target_finetune_loader is not None:
        print(f"\n===== {model_tag}: 目标域微调 =====")
        finetune_optimizer = optim.AdamW(trainable_params, lr=finetune_lr, weight_decay=0.01)
        finetune_scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
        best_finetune_path = os.path.join(exp_dir, "best_finetune_model.pth")
        best_finetune_train_acc = 0.0

        for epoch in range(args.finetune_epochs):
            train_loss, train_acc = train_one_epoch_cls(
                model,
                target_finetune_loader,
                device,
                finetune_optimizer,
                scaler=finetune_scaler,
                accumulation_steps=args.accumulation_steps,
                max_batches=args.max_batches,
                progress_desc=f"{model_tag} Finetune",
            )
            if train_acc > best_finetune_train_acc:
                best_finetune_train_acc = train_acc
                torch.save(model.state_dict(), best_finetune_path)

            metrics_history.append({
                "phase": "finetune",
                "epoch": epoch + 1,
                "train": {"loss": train_loss, "acc": train_acc},
                "val": None,
                "lr": finetune_optimizer.param_groups[0]["lr"],
            })
            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump(metrics_history, f, ensure_ascii=False, indent=2)
            with open(log_file, "a") as f:
                f.write(f"[Finetune][Epoch {epoch + 1}] "
                        f"train_acc={train_acc:.4f}\n")
            print(f"[{model_tag}][Finetune {epoch + 1}/{args.finetune_epochs}] "
                  f"train_acc={train_acc:.4f}")

        if os.path.exists(best_finetune_path):
            model.load_state_dict(torch.load(best_finetune_path, map_location="cpu"))

    print(f"\n===== {model_tag}: 最终目标域测试 =====")
    target_after_loss, target_after_acc = evaluate_cls(model, target_test_loader, device, max_batches=args.max_batches)
    print(f"[{model_tag}] 目标域测试准确率（微调后）: {target_after_acc:.4f}")

    summary = {
        "exp_dir": exp_dir,
        "model": model_tag,
        "model_tag": model_tag,
        "source_test": None,
        "target_before": None,
        "target_after": {"loss": target_after_loss, "acc": target_after_acc},
        "best_source_val_acc": None,
        "best_finetune_val_acc": None,
        "best_source_train_acc": best_source_train_acc,
        "best_finetune_train_acc": best_finetune_train_acc,
        "best_source_model": best_source_path,
        "best_finetune_model": best_finetune_path,
        "config_file": config_file,
        "log_file": log_file,
        "metrics_file": metrics_file,
        "end_time": time.strftime("%Y%m%d_%H%M%S", time.localtime()),
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(log_file, "a") as f:
        f.write("\n===== 最终结果 =====\n")
        f.write(f"目标域测试(后): acc={target_after_acc:.4f}\n")

    print(f"[{model_tag}] 实验完成，结果保存在: {exp_dir}")
    del model
    torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description="对比模型跨工况/跨数据集迁移脚本")

    # 迁移与训练相关
    parser.add_argument('--transfer_type', type=str, default='cross_condition',
                        choices=['cross_condition', 'cross_dataset'])
    parser.add_argument('--target_finetune_ratio', type=float, default=0.1,
                        help='目标域训练集用于微调的比例，0表示不微调')
    parser.add_argument('--source_epochs', type=int, default=30)
    parser.add_argument('--finetune_epochs', type=int, default=10)
    parser.add_argument('--finetune_lr_factor', type=float, default=0.2)
    parser.add_argument('--accumulation_steps', type=int, default=4)
    parser.add_argument('--use_amp', action='store_true', help='是否启用混合精度')
    parser.add_argument('--source_reduction_ratio', type=float, default=1.0,
                        help='源域数据采样比例，用于快速调试')
    parser.add_argument('--target_reduction_ratio', type=float, default=1.0,
                        help='目标域数据采样比例，用于快速调试')
    parser.add_argument('--seed', type=int, default=42)

    # 统一/回退权重
    parser.add_argument('--dataset_weights', type=str, default=None,
                        help='通用的数据集权重，格式: "CWRU:1,PU:0.5"')
    parser.add_argument('--source_dataset_weights', type=str, default=None,
                        help='仅作用于源域的权重字符串，优先级高于dataset_weights')
    parser.add_argument('--target_dataset_weights', type=str, default=None,
                        help='仅作用于目标域的权重字符串，优先级高于dataset_weights')

    # 源域参数
    parser.add_argument('--source_dataset', type=str, nargs='+', default=['CWRU'],
                        choices=['CWRU', 'PU', 'JNU', 'MFPT'])
    parser.add_argument('--source_data_source', type=str, default='12k_DE',
                        choices=['12k_DE', '48k_DE', 'both'])
    parser.add_argument('--source_workloads', type=str, nargs='+', default=['0hp', '1hp', '2hp'])
    parser.add_argument('--source_task_type', type=str, default='4class',
                        choices=['4class', '10class'])
    parser.add_argument('--source_pu_workloads', type=str, nargs='+',
                        default=['N15_M07_F10', 'N09_M07_F10'])
    parser.add_argument('--source_pu_task_type', type=str, default='3class_artificial')
    parser.add_argument('--source_pu_signal_type', type=str, default='vibration',
                        choices=['vibration', 'current', 'both'])
    parser.add_argument('--source_jnu_workloads', type=str, nargs='+', default=['600'])

    # 目标域参数
    parser.add_argument('--target_dataset', type=str, nargs='+', default=['PU'],
                        choices=['CWRU', 'PU', 'JNU', 'MFPT'])
    parser.add_argument('--target_data_source', type=str, default='12k_DE',
                        choices=['12k_DE', '48k_DE', 'both'])
    parser.add_argument('--target_workloads', type=str, nargs='+', default=['0hp', '1hp'])
    parser.add_argument('--target_task_type', type=str, default='4class',
                        choices=['4class', '10class'])
    parser.add_argument('--target_pu_workloads', type=str, nargs='+',
                        default=['N09_M07_F10'])
    parser.add_argument('--target_pu_task_type', type=str, default='3class_artificial')
    parser.add_argument('--target_pu_signal_type', type=str, default='vibration',
                        choices=['vibration', 'current', 'both'])
    parser.add_argument('--target_jnu_workloads', type=str, nargs='+', default=['1000'])

    # 数据集基础参数
    parser.add_argument('--root_path', type=str, default='./datasets')
    parser.add_argument('--sampling_rate', type=int, default=50000)
    parser.add_argument('--window_size', type=int, default=1024)
    parser.add_argument('--stride', type=int, default=512)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--normalize', default=True, help='是否标准化')
    parser.add_argument('--transform_type', type=str, default='None',
                        choices=['None', 'cwt', 'stft', 'gaf', 'rp', 'scalogram'])

    # dataloader设置
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--drop_last', type=bool, default=True)

    # 模型参数
    parser.add_argument('--seq_len', type=int, default=1024)
    parser.add_argument('--patch_len', type=int, default=16)
    parser.add_argument('--p_stride', type=int, default=8)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--d_keys', type=int, default=None)
    parser.add_argument('--num_mapping', type=int, default=1000)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--redimension', type=int, default=64)
    parser.add_argument('--llm_layers', type=int, default=6)
    parser.add_argument('--model_name', type=str, default='qwen_0.5b')
    parser.add_argument('--model_path', type=str, default='../weights_of_models/qwen2.5_0.5b')

    # Baseline 模型额外参数
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--modes', type=int, default=32)
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--input_dim', type=int, default=1)
    parser.add_argument('--use_custom_pos', action='store_true', help='是否在DataBasedLLM中叠加自定义位置编码')
    parser.add_argument('--max_batches', type=int, default=None, help='每个epoch最多跑多少个batch')
    parser.add_argument('--models', type=str, nargs='+',
                        default=['feature', 'data', 'fedformer'],
                        help='可选: feature, data, fedformer')

    # 优化器
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--source_lr', type=float, default=None,
                        help='源域训练学习率，默认与learning_rate一致')
    parser.add_argument('--finetune_lr', type=float, default=None,
                        help='目标域微调学习率，默认使用learning_rate * finetune_lr_factor')
    parser.add_argument('--save_root', type=str, default='results')

    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    set_seed(args.seed)

    if args.window_size != args.seq_len:
        print(f"警告: window_size({args.window_size})与seq_len({args.seq_len})不一致，已同步为窗口长度。")
        args.seq_len = args.window_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"源域数据: {args.source_dataset} -> {args.source_workloads}，目标域数据: {args.target_dataset} -> {args.target_workloads}")

    source_loaders = prepare_domain_loaders(args, "source", args.source_reduction_ratio)
    target_loaders = prepare_domain_loaders(args, "target", args.target_reduction_ratio)
    source_train_loader, _, _ = source_loaders
    target_train_loader, _, _ = target_loaders

    target_finetune_loader = None
    if args.target_finetune_ratio > 0:
        target_finetune_loader = reduce_dataloader(
            target_loaders[0], args.target_finetune_ratio, shuffle_override=True
        )
        print(f"目标域微调数据比例: {args.target_finetune_ratio:.2f}, 批次数: {len(target_finetune_loader)}")

    source_class_names = extract_class_names(source_train_loader.dataset) or []
    target_class_names = extract_class_names(target_train_loader.dataset) or []

    candidate_nums: List[int] = []
    for num in (infer_num_classes(source_train_loader.dataset), infer_num_classes(target_train_loader.dataset)):
        if num is not None:
            candidate_nums.append(num)
    if source_class_names:
        candidate_nums.append(len(source_class_names))
    if target_class_names:
        candidate_nums.append(len(target_class_names))
    args.num_classes = max(candidate_nums) if candidate_nums else 4

    models = [m.lower() for m in args.models]
    if "feature" in models:
        run_feature_transfer(
            args,
            source_loaders,
            target_loaders,
            target_finetune_loader,
            source_class_names,
            target_class_names,
        )

    if "data" in models:
        run_cls_transfer(
            "data",
            DataBasedLLM,
            args,
            device,
            source_loaders,
            target_loaders,
            target_finetune_loader,
            source_class_names,
            target_class_names,
        )

    if "fedformer" in models:
        run_cls_transfer(
            "fedformer",
            FEDformer,
            args,
            device,
            source_loaders,
            target_loaders,
            target_finetune_loader,
            source_class_names,
            target_class_names,
        )


if __name__ == "__main__":
    main()
