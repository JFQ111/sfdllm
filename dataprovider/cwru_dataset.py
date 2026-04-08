import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from transforms.signal_transforms import apply_transform


class CWRUBearingDataset(Dataset):
    """CWRU Bearing Fault Dataset. Supports 4-class and 10-class tasks, multiple workloads, 12k/48k sampling rates."""

    def __init__(self, args, flag="train"):
        self.args = args
        self.args.sampling_rate = 12000 if args.data_source == "12k_DE" else 48000
        self.flag = flag
        self.scaler = StandardScaler() if args.normalize else None

        self.fault_patterns = {
            "normal": {"pattern": "normal", "class_10": 0, "class_4": 0},
            "IR007":  {"pattern": "IR007",  "class_10": 1, "class_4": 1},
            "B007":   {"pattern": "B007",   "class_10": 2, "class_4": 3},
            "OR007":  {"pattern": "OR007",  "class_10": 3, "class_4": 2},
            "IR014":  {"pattern": "IR014",  "class_10": 4, "class_4": 1},
            "B014":   {"pattern": "B014",   "class_10": 5, "class_4": 3},
            "OR014":  {"pattern": "OR014",  "class_10": 6, "class_4": 2},
            "IR021":  {"pattern": "IR021",  "class_10": 7, "class_4": 1},
            "B021":   {"pattern": "B021",   "class_10": 8, "class_4": 3},
            "OR021":  {"pattern": "OR021",  "class_10": 9, "class_4": 2},
        }

        if args.task_type == "4class":
            self.class_names = ["normal", "inner", "outer", "ball"]
            self.num_classes = 4
        else:
            self.class_names = [
                "normal", "inner_7", "ball_7", "outer_7",
                "inner_14", "ball_14", "outer_14",
                "inner_21", "ball_21", "outer_21",
            ]
            self.num_classes = 10

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

    def _identify_fault(self, filename):
        fl = filename.lower()
        for ft in sorted(self.fault_patterns, key=len, reverse=True):
            if self.fault_patterns[ft]["pattern"].lower() in fl:
                return ft
        return None

    def _find_de_column(self, mat_data):
        cols = [k for k in mat_data if "DE" in k.upper() and not k.startswith("__")
                and isinstance(mat_data[k], np.ndarray) and mat_data[k].size > 1000]
        if not cols:
            return None
        time_cols = [c for c in cols if "time" in c.lower()]
        return time_cols[0] if time_cols else cols[0]

    def _get_workload_files(self, ds_path, workload):
        wp = os.path.join(ds_path, workload)
        if not os.path.exists(wp):
            return []
        files_info = []
        for fname in sorted(f for f in os.listdir(wp) if f.endswith(".mat")):
            ft = self._identify_fault(fname)
            if ft is None:
                continue
            fpath = os.path.join(wp, fname)
            try:
                mat = loadmat(fpath)
                col = self._find_de_column(mat)
                if col:
                    files_info.append((fpath, col, ft))
            except Exception:
                continue
        return files_info

    def _load_all_data(self):
        all_data, all_labels = [], []
        cwru_root = os.path.join(self.args.root_path, "CWRU")
        sources = ["12k_DE", "48k_DE"] if self.args.data_source == "both" else [self.args.data_source]

        for src in sources:
            src_path = os.path.join(cwru_root, src)
            if not os.path.exists(src_path):
                continue
            for wl in self.args.workloads:
                for fpath, col, ft in self._get_workload_files(src_path, wl):
                    try:
                        sig = loadmat(fpath)[col].reshape(-1)
                        max_len = 480000 if self.args.sampling_rate == 48000 else (240000 if ft == "normal" else 119808)
                        sig = sig[:max_len] if len(sig) > max_len else sig
                        windows = self._sliding_window(sig)
                        lbl = self.fault_patterns[ft]["class_4" if self.args.task_type == "4class" else "class_10"]
                        all_data.extend(windows)
                        all_labels.extend([lbl] * len(windows))
                    except Exception:
                        continue

        if not all_data:
            raise ValueError(f"No CWRU data loaded. Check root_path and workloads.")
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
        return [sig[i:i+self.args.window_size]
                for i in range(0, len(sig) - self.args.window_size + 1, self.args.stride)]

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.LongTensor([self.labels[idx]]).squeeze()

    def __len__(self):
        return len(self.data)

    def get_class_names(self):
        return self.class_names
