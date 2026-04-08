import argparse
import json
import os
import random
import time
from typing import List, Optional

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


def reduce_dataloader(original_loader: DataLoader, reduction_ratio: float, shuffle_override: Optional[bool] = None) -> DataLoader:
    """按比例裁剪DataLoader，便于快速冒烟测试。"""
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


def unpack_batch(batch):
    if isinstance(batch, dict):
        return batch.get("data"), batch.get("label"), batch.get("cwt")
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return batch[0], batch[1], None
    raise ValueError("Unsupported batch format")


def train_one_epoch_cls(model, loader, device, optimizer, scaler=None, accumulation_steps: int = 1,
                        max_batches: Optional[int] = None, progress_desc: str = "Train", pass_cwt: bool = False):
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
        data, labels, cwt = unpack_batch(batch)
        data = squeeze_signal_batch(data.to(device))
        labels = labels.to(device).long().view(-1)
        if pass_cwt and cwt is not None:
            cwt = cwt.to(device)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(data, cwt=cwt) if pass_cwt else model(data)
                loss = F.cross_entropy(logits, labels)
                loss_scaled = loss / max(1, accumulation_steps)
            scaler.scale(loss_scaled).backward()
        else:
            logits = model(data, cwt=cwt) if pass_cwt else model(data)
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


def evaluate_cls(model, loader, device, max_batches: Optional[int] = None, pass_cwt: bool = False):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    steps = 0

    with torch.no_grad():
        for step_idx, batch in enumerate(loader):
            if max_batches is not None and step_idx >= max_batches:
                break
            data, labels, cwt = unpack_batch(batch)
            data = squeeze_signal_batch(data.to(device))
            labels = labels.to(device).long().view(-1)
            if pass_cwt and cwt is not None:
                cwt = cwt.to(device)

            logits = model(data, cwt=cwt) if pass_cwt else model(data)
            loss = F.cross_entropy(logits, labels)
            preds = torch.argmax(logits, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()
            steps += 1

    avg_loss = total_loss / max(1, steps)
    acc = correct / max(1, total)
    return avg_loss, acc


def get_feature_llm_device(model):
    if hasattr(model, "model") and hasattr(model.model, "device"):
        return model.model.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_save_dir(args, model_tag: str, dataset_tag: str, nowtime: str):
    save_dir = os.path.join(args.save_root, f"baseline_test_{model_tag}_{dataset_tag}_{nowtime}")
    os.makedirs(save_dir, exist_ok=True)
    return {
        "exp_dir": save_dir,
        "save_dir": save_dir,
        "log_file": os.path.join(save_dir, "train_log.txt"),
        "metrics_file": os.path.join(save_dir, "metrics_history.json"),
        "summary_file": os.path.join(save_dir, "summary.json"),
        "config_file": os.path.join(save_dir, "run_config.json"),
        "best_model": os.path.join(save_dir, "best_model.pth"),
    }


def train_one_epoch_feature(model, loader, optimizer, accumulation_steps: int = 1,
                            max_batches: Optional[int] = None, progress_desc: str = "Train"):
    model.train()
    total_loss = 0.0
    steps = 0

    device = get_feature_llm_device(model)
    optimizer.zero_grad()
    total_batches = max_batches if max_batches is not None else len(loader)
    progress_bar = tqdm.tqdm(loader, total=total_batches, desc=progress_desc, leave=False)
    for step_idx, batch in enumerate(progress_bar):
        if step_idx >= total_batches:
            break
        data, labels, _ = unpack_batch(batch)
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


def evaluate_feature(model, loader, max_batches: Optional[int] = None):
    model.eval()
    correct = 0
    total = 0
    device = get_feature_llm_device(model)

    with torch.no_grad():
        for step_idx, batch in enumerate(loader):
            if max_batches is not None and step_idx >= max_batches:
                break
            data, labels, _ = unpack_batch(batch)
            data = squeeze_signal_batch(data.to(device))
            labels = labels.to(device).long().view(-1)

            preds = model.generate_diagnosis(data)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / max(1, total)
    return acc


def parse_args():
    parser = argparse.ArgumentParser(description="对比模型单域训练脚本")
    # --- 基础数据参数 ---
    parser.add_argument('--dataset', type=str, nargs='+', default=['MFPT'], help='选择数据集')
    parser.add_argument('--dataset_weights', type=str, default=None)
    parser.add_argument('--root_path', type=str, default='./datasets')
    parser.add_argument('--data_source', type=str, default='12k_DE')
    parser.add_argument('--workloads', type=str, nargs='+', default=['0hp', '1hp', '2hp', '3hp'])
    parser.add_argument('--task_type', type=str, default='4class')
    parser.add_argument('--pu_workloads', type=str, nargs='+', default=['N15_M07_F10'])
    parser.add_argument('--pu_task_type', type=str, default='3class_artificial')
    parser.add_argument('--pu_signal_type', type=str, default='vibration')
    parser.add_argument('--jnu_workloads', type=str, nargs='+', default=['600'])
    parser.add_argument('--sampling_rate', type=int, default=50000)
    parser.add_argument('--window_size', type=int, default=1024)
    parser.add_argument('--stride', type=int, default=512)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--normalize', default=True, help='是否标准化')
    parser.add_argument('--transform_type', type=str, default='None')

    # --- 训练超参数 ---
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--accumulation_steps', type=int, default=4,
                        help='[新增] 梯度累积步数，模拟更大的Batch Size (16*4=64)')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--drop_last', type=bool, default=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--use_amp', action='store_true', help='是否启用混合精度')
    parser.add_argument('--lr_step_size', type=int, default=10)
    parser.add_argument('--lr_gamma', type=float, default=0.1)
    parser.add_argument('--reduction_ratio', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)

    # --- 模型参数 ---
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
    parser.add_argument('--lambda_rec', type=float, default=0.1, help='重建损失权重')
    parser.add_argument('--temperature', type=float, default=0.07, help='[建议] 固定为0.07')
    parser.add_argument('--desc', type=str, default='Bearing Fault Data')
    parser.add_argument('--save_root', type=str, default='results')

    # --- 对比模型额外参数 ---
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--modes', type=int, default=32)
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--input_dim', type=int, default=1)
    parser.add_argument('--use_custom_pos', action='store_true', help='是否在DataBasedLLM中叠加自定义位置编码')
    parser.add_argument('--max_batches', type=int, default=None, help='每个epoch最多跑多少个batch')
    parser.add_argument('--models', type=str, nargs='+',
                        default=['feature', 'data', 'fedformer'],
                        help='可选: feature, data, fedformer')

    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    train_loader, val_loader, test_loader = create_dataloaders(args)

    if args.reduction_ratio < 1.0:
        print(f"启用快速采样，比例: {args.reduction_ratio:.2f}")
        train_loader = reduce_dataloader(train_loader, args.reduction_ratio, shuffle_override=True)
        val_loader = reduce_dataloader(val_loader, args.reduction_ratio, shuffle_override=False)
        test_loader = reduce_dataloader(test_loader, args.reduction_ratio, shuffle_override=False)

    class_names = extract_class_names(train_loader.dataset)
    num_classes = infer_num_classes(train_loader.dataset)
    if num_classes is None:
        num_classes = len(class_names) if class_names else 4
    args.num_classes = num_classes
    args.class_names = class_names

    print(f"类别数: {args.num_classes}")
    if class_names:
        print(f"类别名: {class_names}")

    models = [m.lower() for m in args.models]
    dataset_tag = "_".join(args.dataset) if isinstance(args.dataset, list) else str(args.dataset)
    nowtime = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    if "feature" in models:
        print("\n===== FeatureBasedLLM 测试 =====")
        feature_model = FeatureBasedLLM(args)
        if class_names:
            feature_model.set_class_names(class_names)
        sampling_rate = extract_sampling_rate(train_loader.dataset)
        if sampling_rate:
            feature_model.set_sampling_rate(sampling_rate)
        feature_optimizer = optim.AdamW(feature_model.parameters(), lr=args.learning_rate)
        save_info = init_save_dir(args, "feature", dataset_tag, nowtime)
        run_config = {
            "args": vars(args),
            "device": str(get_feature_llm_device(feature_model)),
            "model": "feature",
            "model_tag": "feature",
            "start_time": nowtime,
            "exp_dir": save_info["exp_dir"],
            "class_names": class_names,
        }
        with open(save_info["config_file"], "w", encoding="utf-8") as f:
            json.dump(run_config, f, ensure_ascii=False, indent=2)

        best_val_acc = 0.0
        epoch_records = []

        for epoch in range(args.epochs):
            train_loss = train_one_epoch_feature(
                feature_model,
                train_loader,
                feature_optimizer,
                accumulation_steps=args.accumulation_steps,
                max_batches=args.max_batches,
                progress_desc="Feature Train",
            )
            val_acc = evaluate_feature(feature_model, val_loader, max_batches=args.max_batches)
            print(f"[Feature] Epoch {epoch + 1}: loss={train_loss:.4f}, val_acc={val_acc:.4f}")
            epoch_records.append({
                "phase": "train",
                "epoch": epoch + 1,
                "train": {"loss": train_loss},
                "val": {"acc": val_acc},
                "lr": feature_optimizer.param_groups[0]["lr"],
            })
            with open(save_info["metrics_file"], "w", encoding="utf-8") as f:
                json.dump(epoch_records, f, ensure_ascii=False, indent=2)
            with open(save_info["log_file"], "a") as f:
                f.write(
                    f"Epoch {epoch + 1}: train_loss={train_loss:.6f}, val_acc={val_acc:.6f}\n"
                )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(feature_model.state_dict(), save_info["best_model"])

        if os.path.exists(save_info["best_model"]):
            feature_model.load_state_dict(torch.load(save_info["best_model"], map_location="cpu"))
        test_acc = evaluate_feature(feature_model, test_loader, max_batches=args.max_batches)
        print(f"[Feature] Test acc={test_acc:.4f}")
        with open(save_info["log_file"], "a") as f:
            f.write(f"Test Final: acc={test_acc:.6f}\n")
        summary = {
            "exp_dir": save_info["exp_dir"],
            "save_dir": save_info["save_dir"],
            "best_model": save_info["best_model"],
            "log_file": save_info["log_file"],
            "metrics_file": save_info["metrics_file"],
            "config_file": save_info["config_file"],
            "model": "feature",
            "best_val_acc": best_val_acc,
            "test_metrics": {"acc": test_acc},
            "end_time": time.strftime("%Y%m%d_%H%M%S", time.localtime()),
        }
        with open(save_info["summary_file"], "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        del feature_model
        torch.cuda.empty_cache()

    if "data" in models:
        print("\n===== DataBasedLLM 测试 =====")
        data_model = DataBasedLLM(args).to(device)
        data_optimizer = optim.AdamW(
            [p for p in data_model.parameters() if p.requires_grad],
            lr=args.learning_rate
        )
        scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
        save_info = init_save_dir(args, "data", dataset_tag, nowtime)
        run_config = {
            "args": vars(args),
            "device": str(device),
            "model": "data",
            "model_tag": "data",
            "start_time": nowtime,
            "exp_dir": save_info["exp_dir"],
            "class_names": class_names,
        }
        with open(save_info["config_file"], "w", encoding="utf-8") as f:
            json.dump(run_config, f, ensure_ascii=False, indent=2)

        best_val_acc = 0.0
        epoch_records = []

        for epoch in range(args.epochs):
            train_loss, train_acc = train_one_epoch_cls(
                data_model,
                train_loader,
                device,
                data_optimizer,
                scaler=scaler,
                accumulation_steps=args.accumulation_steps,
                max_batches=args.max_batches,
                progress_desc="Data Train",
            )
            val_loss, val_acc = evaluate_cls(data_model, val_loader, device, max_batches=args.max_batches)
            print(
                f"[Data] Epoch {epoch + 1}: loss={train_loss:.4f}, "
                f"acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
            )
            epoch_records.append({
                "phase": "train",
                "epoch": epoch + 1,
                "train": {"loss": train_loss, "acc": train_acc},
                "val": {"loss": val_loss, "acc": val_acc},
                "lr": data_optimizer.param_groups[0]["lr"],
            })
            with open(save_info["metrics_file"], "w", encoding="utf-8") as f:
                json.dump(epoch_records, f, ensure_ascii=False, indent=2)
            with open(save_info["log_file"], "a") as f:
                f.write(
                    f"Epoch {epoch + 1}: train_loss={train_loss:.6f}, train_acc={train_acc:.6f}, "
                    f"val_loss={val_loss:.6f}, val_acc={val_acc:.6f}\n"
                )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(data_model.state_dict(), save_info["best_model"])

        if os.path.exists(save_info["best_model"]):
            data_model.load_state_dict(torch.load(save_info["best_model"], map_location="cpu"))
        test_loss, test_acc = evaluate_cls(data_model, test_loader, device, max_batches=args.max_batches)
        print(f"[Data] Test loss={test_loss:.4f}, acc={test_acc:.4f}")
        with open(save_info["log_file"], "a") as f:
            f.write(f"Test Final: loss={test_loss:.6f}, acc={test_acc:.6f}\n")
        summary = {
            "exp_dir": save_info["exp_dir"],
            "save_dir": save_info["save_dir"],
            "best_model": save_info["best_model"],
            "log_file": save_info["log_file"],
            "metrics_file": save_info["metrics_file"],
            "config_file": save_info["config_file"],
            "model": "data",
            "best_val_acc": best_val_acc,
            "test_metrics": {"loss": test_loss, "acc": test_acc},
            "end_time": time.strftime("%Y%m%d_%H%M%S", time.localtime()),
        }
        with open(save_info["summary_file"], "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        del data_model
        torch.cuda.empty_cache()

    if "fedformer" in models:
        print("\n===== FEDformer 测试 =====")
        fed_model = FEDformer(args).to(device)
        fed_optimizer = optim.AdamW(fed_model.parameters(), lr=args.learning_rate)
        scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
        save_info = init_save_dir(args, "fedformer", dataset_tag, nowtime)
        run_config = {
            "args": vars(args),
            "device": str(device),
            "model": "fedformer",
            "model_tag": "fedformer",
            "start_time": nowtime,
            "exp_dir": save_info["exp_dir"],
            "class_names": class_names,
        }
        with open(save_info["config_file"], "w", encoding="utf-8") as f:
            json.dump(run_config, f, ensure_ascii=False, indent=2)

        best_val_acc = 0.0
        epoch_records = []

        for epoch in range(args.epochs):
            train_loss, train_acc = train_one_epoch_cls(
                fed_model,
                train_loader,
                device,
                fed_optimizer,
                scaler=scaler,
                accumulation_steps=args.accumulation_steps,
                max_batches=args.max_batches,
                progress_desc="FEDformer Train",
            )
            val_loss, val_acc = evaluate_cls(fed_model, val_loader, device, max_batches=args.max_batches)
            print(
                f"[FEDformer] Epoch {epoch + 1}: loss={train_loss:.4f}, "
                f"acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
            )
            epoch_records.append({
                "phase": "train",
                "epoch": epoch + 1,
                "train": {"loss": train_loss, "acc": train_acc},
                "val": {"loss": val_loss, "acc": val_acc},
                "lr": fed_optimizer.param_groups[0]["lr"],
            })
            with open(save_info["metrics_file"], "w", encoding="utf-8") as f:
                json.dump(epoch_records, f, ensure_ascii=False, indent=2)
            with open(save_info["log_file"], "a") as f:
                f.write(
                    f"Epoch {epoch + 1}: train_loss={train_loss:.6f}, train_acc={train_acc:.6f}, "
                    f"val_loss={val_loss:.6f}, val_acc={val_acc:.6f}\n"
                )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(fed_model.state_dict(), save_info["best_model"])

        if os.path.exists(save_info["best_model"]):
            fed_model.load_state_dict(torch.load(save_info["best_model"], map_location="cpu"))
        test_loss, test_acc = evaluate_cls(fed_model, test_loader, device, max_batches=args.max_batches)
        print(f"[FEDformer] Test loss={test_loss:.4f}, acc={test_acc:.4f}")
        with open(save_info["log_file"], "a") as f:
            f.write(f"Test Final: loss={test_loss:.6f}, acc={test_acc:.6f}\n")
        summary = {
            "exp_dir": save_info["exp_dir"],
            "save_dir": save_info["save_dir"],
            "best_model": save_info["best_model"],
            "log_file": save_info["log_file"],
            "metrics_file": save_info["metrics_file"],
            "config_file": save_info["config_file"],
            "model": "fedformer",
            "best_val_acc": best_val_acc,
            "test_metrics": {"loss": test_loss, "acc": test_acc},
            "end_time": time.strftime("%Y%m%d_%H%M%S", time.localtime()),
        }
        with open(save_info["summary_file"], "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        del fed_model
        torch.cuda.empty_cache()



if __name__ == '__main__':
    main()
