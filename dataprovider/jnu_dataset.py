import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from transforms.signal_transforms import apply_transform


class JNUBearingDataset(Dataset):
    """JNU (Jiangnan University) Bearing Fault Dataset. 4-class: normal, inner, outer, ball."""

    SAMPLING_RATE = 50000

    def __init__(self, args, flag="train"):
        self.args = args
        self.args.sampling_rate = self.SAMPLING_RATE
        self.flag = flag
        self.scaler = StandardScaler() if args.normalize else None

        self.fault_patterns = {
            "n":  {"class": 0, "name": "normal"},
            "ib": {"class": 1, "name": "inner"},
            "ob": {"class": 2, "name": "outer"},
            "tb": {"class": 3, "name": "ball"},
        }
        self.class_names = ["normal", "inner", "outer", "ball"]
        self.num_classes = 4

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

    def _parse_filename(self, filename):
        fl = filename.lower().replace(".csv", "")
        for ft in sorted(self.fault_patterns, key=len, reverse=True):
            if fl.startswith(ft):
                rem = fl[len(ft):]
                wl = "".join(c for c in rem if c.isdigit())[:4]
                return (ft, wl) if wl else (None, None)
        return None, None

    def _load_all_data(self):
        all_data, all_labels = [], []
        root = os.path.join(self.args.root_path, "JNU")
        if not os.path.exists(root):
            raise ValueError(f"JNU dataset path '{root}' does not exist.")

        all_files = sorted(f for f in os.listdir(root) if f.endswith(".csv"))
        for wl in self.args.jnu_workloads:
            for fname in all_files:
                ft, fwl = self._parse_filename(fname)
                if ft is None or fwl != wl:
                    continue
                try:
                    df = pd.read_csv(os.path.join(root, fname))
                    sig = df.iloc[:, 0].values.astype(np.float32)
                    windows = self._sliding_window(sig)
                    lbl = self.fault_patterns[ft]["class"]
                    all_data.extend(windows)
                    all_labels.extend([lbl] * len(windows))
                except Exception:
                    continue

        if not all_data:
            raise ValueError(f"No JNU data loaded for workloads {self.args.jnu_workloads}. "
                             f"Check root_path and CSV naming pattern (e.g. tb1000_2.csv).")
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
