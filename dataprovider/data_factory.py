import hashlib
import os
import pickle
import warnings
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from .cwru_dataset import CWRUBearingDataset
from .jnu_dataset import JNUBearingDataset
from .mfpt_dataset import MFPTBearingDataset
from .pu_dataset import PUBearingDataset

warnings.filterwarnings("ignore")

DATASET_CLASSES = {
    "CWRU": CWRUBearingDataset,
    "JNU":  JNUBearingDataset,
    "MFPT": MFPTBearingDataset,
    "PU":   PUBearingDataset,
}


class DomainLabeledDataset:
    def __init__(self, dataset, domain_name):
        self.dataset = dataset
        self.domain_name = domain_name

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if isinstance(sample, dict):
            sample["domain"] = self.domain_name
        else:
            sample = {"data": sample[0], "label": sample[1], "domain": self.domain_name}
        return sample

    def __len__(self):
        return len(self.dataset)


def _cache_key(args):
    params = {
        "dataset": sorted(args.dataset) if isinstance(args.dataset, list) else args.dataset,
        "dataset_weights": getattr(args, "dataset_weights", None),
        "root_path": args.root_path,
        "data_source": getattr(args, "data_source", None),
        "workloads": sorted(args.workloads) if hasattr(args, "workloads") else None,
        "task_type": getattr(args, "task_type", None),
        "pu_workloads": sorted(args.pu_workloads) if hasattr(args, "pu_workloads") else None,
        "pu_task_type": getattr(args, "pu_task_type", None),
        "pu_signal_type": getattr(args, "pu_signal_type", None),
        "jnu_workloads": sorted(args.jnu_workloads) if hasattr(args, "jnu_workloads") else None,
        "sampling_rate": args.sampling_rate,
        "window_size": args.window_size,
        "stride": args.stride,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "normalize": args.normalize,
        "transform_type": args.transform_type,
    }
    return hashlib.md5(str(sorted(params.items())).encode()).hexdigest()


def _create_datasets(args):
    datasets_to_use = args.dataset if isinstance(args.dataset, list) else [args.dataset]
    for ds in datasets_to_use:
        if ds not in DATASET_CLASSES:
            raise ValueError(f"Unknown dataset: {ds}. Available: {list(DATASET_CLASSES.keys())}")

    trains, vals, tests = [], [], []
    for ds_name in datasets_to_use:
        cls = DATASET_CLASSES[ds_name]
        train_ds = DomainLabeledDataset(cls(args, flag="train"), ds_name)
        val_ds   = DomainLabeledDataset(cls(args, flag="val"),   ds_name)
        test_ds  = DomainLabeledDataset(cls(args, flag="test"),  ds_name)
        trains.append(train_ds)
        vals.append(val_ds)
        tests.append(test_ds)

    if len(trains) == 1:
        return trains[0], vals[0], tests[0]
    return ConcatDataset(trains), ConcatDataset(vals), ConcatDataset(tests)


def create_dataloaders(args, cache_dir="./datasets/cache"):
    batch_size  = getattr(args, "batch_size", 32)
    num_workers = getattr(args, "num_workers", 4)
    pin_memory  = getattr(args, "pin_memory", True)
    drop_last   = getattr(args, "drop_last", False)
    shuffle     = getattr(args, "shuffle", True)

    cache_path = Path(cache_dir) / f"datasets_{_cache_key(args)}.pkl"
    train_ds = val_ds = test_ds = None

    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                train_ds, val_ds, test_ds = pickle.load(f)
        except Exception:
            train_ds = None

    if train_ds is None:
        train_ds, val_ds, test_ds = _create_datasets(args)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(cache_path, "wb") as f:
                pickle.dump((train_ds, val_ds, test_ds), f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass

    # Weighted sampler
    sampler = None
    datasets_to_use = args.dataset if isinstance(args.dataset, list) else [args.dataset]
    if hasattr(args, "dataset_weights") and args.dataset_weights not in (None, "None"):
        weights_dict = {}
        for pair in args.dataset_weights.split(","):
            name, w = pair.split(":")
            weights_dict[name.strip()] = float(w.strip())

        if isinstance(train_ds, ConcatDataset):
            sample_weights = []
            for sub in train_ds.datasets:
                dn = getattr(sub, "domain_name", datasets_to_use[0])
                sample_weights.extend([weights_dict.get(dn, 1.0)] * len(sub))
        else:
            dn = getattr(train_ds, "domain_name", datasets_to_use[0])
            sample_weights = [weights_dict.get(dn, 1.0)] * len(train_ds)

        sampler = WeightedRandomSampler(
            weights=torch.FloatTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(shuffle if sampler is None else False),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
    )
    return train_loader, val_loader, test_loader


def clear_cache(cache_dir="./datasets/cache"):
    cache_dir_path = Path(cache_dir)
    if not cache_dir_path.exists():
        return
    for f in cache_dir_path.glob("datasets_*.pkl"):
        f.unlink()
