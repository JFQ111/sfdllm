import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from scipy.signal import resample
from transforms.signal_transforms import apply_transform


class MFPTBearingDataset(Dataset):
    """MFPT Bearing Fault Dataset. 3-class: normal (0), inner race (1), outer race (2)."""

    TARGET_SAMPLING_RATE = 48828

    FOLDER_MAP = {
        "1 - Three Baseline Conditions": {
            "fault_type": "normal", "label": 0, "sampling_rate": 97656,
        },
        "2 - Three Outer Race Fault Conditions": {
            "fault_type": "outer_race", "label": 2, "sampling_rate": 97656,
        },
        "3 - Seven More Outer Race Fault Conditions": {
            "fault_type": "outer_race", "label": 2, "sampling_rate": 48828,
        },
        "4 - Seven Inner Race Fault Conditions": {
            "fault_type": "inner_race", "label": 1, "sampling_rate": 48828,
        },
    }

    def __init__(self, args, flag="train"):
        self.args = args
        self.args.sampling_rate = self.TARGET_SAMPLING_RATE
        self.flag = flag
        self.scaler = StandardScaler() if args.normalize else None
        self.class_names = ["normal", "inner_race", "outer_race"]
        self.num_classes = 3

        self._load_data()
        if self.scaler is not None:
            self.data = self.scaler.transform(self.data)
        if self.args.transform_type != "None":
            self.data = self._apply_transforms()

    def _apply_transforms(self):
        out = []
        for w in self.data:
            t = apply_transform(w, self.args.sampling_rate, self.args)
            if t.ndim == 2:
                t = np.expand_dims(t, 0)
            out.append(t)
        return np.array(out)

    def _extract_signal(self, mat_data, fname):
        try:
            if "bearing" in mat_data:
                bd = mat_data["bearing"]
                for i in [0, 1, 2]:
                    sig = bd[0][0][i]
                    if sig.shape[0] > 1024:
                        break
                sig = sig.reshape(-1) if sig.ndim == 1 else sig[:, 0]
                return sig
        except Exception:
            pass
        for k in mat_data:
            if not k.startswith("__"):
                d = mat_data[k]
                if isinstance(d, np.ndarray) and d.size > 10000:
                    return d.reshape(-1)
        return None

    def _load_all_data(self):
        all_data, all_labels = [], []
        root = os.path.join(self.args.root_path, "MFPT")
        if not os.path.exists(root):
            raise ValueError(f"MFPT root path '{root}' does not exist.")

        for folder, cfg in self.FOLDER_MAP.items():
            fpath = os.path.join(root, folder)
            if not os.path.exists(fpath):
                continue
            sr = cfg["sampling_rate"]
            for fname in sorted(f for f in os.listdir(fpath) if f.endswith(".mat")):
                try:
                    mat = loadmat(os.path.join(fpath, fname))
                    sig = self._extract_signal(mat, fname)
                    if sig is None:
                        continue
                    if sr != self.TARGET_SAMPLING_RATE:
                        new_len = int(len(sig) * self.TARGET_SAMPLING_RATE / sr)
                        sig = resample(sig, new_len)
                    windows = self._sliding_window(sig)
                    all_data.extend(windows)
                    all_labels.extend([cfg["label"]] * len(windows))
                except Exception:
                    continue

        if not all_data:
            raise ValueError("No MFPT data loaded.")
        return np.array(all_data, dtype=np.float32), np.array(all_labels, dtype=np.int64)

    def _load_data(self):
        all_data, all_labels = self._load_all_data()
        np.random.seed(42)
        idx = np.random.permutation(len(all_data))
        all_data, all_labels = all_data[idx], all_labels[idx]

        n = len(all_data)
        ts = int(n * self.args.train_ratio)
        vs = int(n * self.args.val_ratio)

        if self.scaler is not None:
            self.scaler.fit(all_data[:ts].reshape(-1, self.args.window_size))

        if self.flag == "train":
            self.data, self.labels = all_data[:ts], all_labels[:ts]
        elif self.flag == "val":
            self.data, self.labels = all_data[ts:ts+vs], all_labels[ts:ts+vs]
        else:
            self.data, self.labels = all_data[ts+vs:], all_labels[ts+vs:]

    def _sliding_window(self, sig):
        ws, st = self.args.window_size, self.args.stride
        return [sig[i:i+ws] for i in range(0, len(sig) - ws + 1, st)]

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.LongTensor([self.labels[idx]]).squeeze()

    def __len__(self):
        return len(self.data)

    def get_class_names(self):
        return self.class_names
