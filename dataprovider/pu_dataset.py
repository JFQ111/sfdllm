import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from transforms.signal_transforms import apply_transform


class PUBearingDataset(Dataset):
    """PU (Paderborn University) Bearing Fault Dataset. Supports multiple classification tasks."""

    SAMPLING_RATE = 64000

    def __init__(self, args, flag="train"):
        self.args = args
        self.args.sampling_rate = self.SAMPLING_RATE
        self.flag = flag
        self.scaler = StandardScaler() if args.normalize else None
        self._setup_classes()
        self._load_data()
        if self.scaler is not None:
            self.data = self._standardize(self.data)
        if self.args.transform_type != "None":
            self.data = self._apply_transforms(self.data)

    def _setup_classes(self):
        tt = self.args.pu_task_type
        if tt == "3class_artificial":
            self.class_mapping = {
                "K001": 0,
                "KI01": 1, "KI03": 1, "KI05": 1, "KI07": 1, "KI08": 1,
                "KA01": 2, "KA03": 2, "KA05": 2, "KA07": 2, "KA09": 2,
            }
            self.class_names = ["normal", "inner_ring", "outer_ring"]
            self.num_classes = 3
            self.required_types = list(self.class_mapping.keys())
        elif tt == "5class_artificial":
            self.class_mapping = {
                "K001": 0,
                "KA05": 1, "KA07": 1,
                "KA03": 2, "KA06": 2, "KA08": 2, "KA09": 2,
                "KI01": 3, "KI03": 3, "KI05": 3,
                "KI07": 4, "KI08": 4,
            }
            self.class_names = ["normal", "outer_ring_1", "outer_ring_2", "inner_ring_1", "inner_ring_2"]
            self.num_classes = 5
            self.required_types = list(self.class_mapping.keys())
        elif tt == "9class_artificial":
            self.class_mapping = {
                "K001": 0,
                "KA01": 1, "KA03": 2, "KA07": 3, "KA05": 4, "KA09": 5,
                "KI01": 6, "KI03": 7, "KI08": 8,
            }
            self.class_names = ["normal", "KA01", "KA03", "KA07", "KA05", "KA09", "KI01", "KI03", "KI08"]
            self.num_classes = 9
            self.required_types = list(self.class_mapping.keys())
        else:
            raise ValueError(f"Unknown pu_task_type: {tt}")

    def _extract_workload(self, filename):
        parts = filename.split("_")
        return "_".join(parts[:3]) if len(parts) >= 3 else None

    def _extract_signal(self, mdata, fname):
        try:
            vib = mdata[0][0][2][0][6][2].reshape(-1)
            if self.args.pu_signal_type == "vibration":
                return vib
            c1 = mdata[0][0][2][0][1][2].reshape(-1)
            c2 = mdata[0][0][2][0][2][2].reshape(-1)
            cur = np.stack([c1, c2], axis=0) if len(c1) == len(c2) else \
                  np.stack([c1[:min(len(c1), len(c2))], c2[:min(len(c1), len(c2))]], axis=0)
            if self.args.pu_signal_type == "current":
                return cur
            ml = min(len(vib), cur.shape[1])
            return np.concatenate([vib[:ml].reshape(1, -1), cur[:, :ml]], axis=0)
        except Exception:
            return None

    def _load_data(self):
        all_data, all_labels = [], []
        root = os.path.join(self.args.root_path, "PU")

        for bt in self.required_types:
            bp = os.path.join(root, bt)
            if not os.path.exists(bp) or bt not in self.class_mapping:
                continue
            label = self.class_mapping[bt]
            for fname in sorted(f for f in os.listdir(bp) if f.endswith(".mat")):
                wl = self._extract_workload(fname)
                if wl not in self.args.pu_workloads:
                    continue
                fpath = os.path.join(bp, fname)
                try:
                    mat = loadmat(fpath)
                    key = os.path.splitext(fname)[0]
                    if key not in mat:
                        continue
                    sig = self._extract_signal(mat[key], fname)
                    if sig is None:
                        continue
                    windows = self._sliding_window(sig)
                    all_data.extend(windows)
                    all_labels.extend([label] * len(windows))
                except Exception:
                    continue

        if not all_data:
            raise ValueError("No PU data loaded. Check root_path, pu_workloads, and pu_task_type.")

        all_data = np.array(all_data, dtype=np.float32)
        all_labels = np.array(all_labels, dtype=np.int64)
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

    def _standardize(self, data):
        out = []
        for w in data:
            out.append(self.scaler.transform(w.reshape(1, -1)).reshape(w.shape))
        return np.array(out)

    def _apply_transforms(self, data):
        out = []
        for w in data:
            if w.ndim == 1:
                t = apply_transform(w, self.args.sampling_rate, self.args)
                if t.ndim == 2:
                    t = np.expand_dims(t, 0)
            else:
                ts = [apply_transform(w[i], self.args.sampling_rate, self.args) for i in range(w.shape[0])]
                t = np.stack([np.expand_dims(x, 0) if x.ndim == 2 else x for x in ts], axis=0)
            out.append(t)
        return np.array(out)

    def _sliding_window(self, sig):
        ws, st = self.args.window_size, self.args.stride
        if sig.ndim == 1:
            return [sig[i:i+ws] for i in range(0, len(sig) - ws + 1, st)]
        L = sig.shape[1]
        return [sig[:, i:i+ws] for i in range(0, L - ws + 1, st)]

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.LongTensor([self.labels[idx]]).squeeze()

    def __len__(self):
        return len(self.data)

    def get_class_names(self):
        return self.class_names
