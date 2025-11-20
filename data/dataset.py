import json
import numpy as np
import os
import random
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class Dataset100Style(Dataset):
    def __init__(self, config, styles, train=True, use_ids=None):
        self.motion_dir  = config["motion_dir"]
        self.mean        = np.load(config["mean_path"])
        self.std         = np.load(config["std_path"])
        self.unit_length = int(config["unit_length"])
        self.min_frames  = int(config["min_frames"])
        self.max_frames  = int(config["max_frames"]) if config["max_frames"] is not None else None
        self.train       = bool(train)
        self.use_ids     = set(use_ids) if use_ids is not None else None

        # Mixed-like toggles
        self.drop_first_frame = bool(config.get("drop_first_frame", False))

        # single/double rounding probability (Mixed used 2x "single", 1x "double")
        self.unit_double_prob = float(config.get("unit_double_prob", 0.0))

        # style map (stable indices by sorted style name)
        with open(config["style_json"], "r") as f:
            style_map_all = json.load(f)  # {style_name: [motion_ids]}
        styles_sorted = sorted(style_map_all.keys())
        self.style_to_style_idx = {s: i for i, s in enumerate(styles_sorted)}
        self.style_idx_to_style = {i: s for s, i in self.style_to_style_idx.items()}
        self.style_map = {self.style_to_style_idx[s]: style_map_all[s] for s in styles}

        # content map (used for filtering + default caption)
        with open(config["content_json"], "r") as f:
            content_map = json.load(f)  # {content_key: [motion_ids]}
        contents_sorted = sorted(content_map.keys())
        self.content_to_content_idx = {c: i for i, c in enumerate(contents_sorted)}
        self.content_idx_to_content = {i: c for c, i in self.content_to_content_idx.items()}

        # motion_id -> content_idx
        self.motion_to_content_idx = {}
        for content_key, mids in content_map.items():
            cidx = self.content_to_content_idx[content_key]
            for mid in mids:
                self.motion_to_content_idx[mid] = cidx

        # fixed captions by content code
        self.content_prompts = {
            "BR": "a person is running backward",
            "BW": "a person is walking backward",
            "FR": "a person is running forward",
            "FW": "a person is walking forward",
            "ID": "a person is standing still",
            "SR": "a person is running sideways",
            "SW": "a person is walking sideways",
        }

        # exclude TR1/2/3 contents
        exclude_keys = {"TR1", "TR2", "TR3"}
        self.exclude_content_idcs = {
            self.content_to_content_idx[k]
            for k in exclude_keys if k in self.content_to_content_idx
        }

        # index once (apply Mixed-like filters)
        kept = miss = short = outlier = filtered = 0
        self.items = []
        self.nfeats = None

        for style_idx, motion_ids in self.style_map.items():
            style_name = self.style_idx_to_style[style_idx]
            iterable = tqdm(motion_ids, desc=f"Index 100STYLE style={style_name}", leave=False)

            for motion_id in iterable:
                if self.use_ids is not None and motion_id not in self.use_ids:
                    filtered += 1
                    continue

                path = os.path.join(self.motion_dir, f"{motion_id}.npy")
                if not os.path.exists(path):
                    miss += 1
                    continue

                arr = np.load(path, mmap_mode="r")
                T = int(arr.shape[0])

                if self.drop_first_frame and T > 1:
                    arr = arr[1:]; T -= 1

                if T < self.min_frames:
                    short += 1
                    continue

                if np.max(np.abs((arr - self.mean) / self.std)) > 1e3:
                    outlier += 1
                    continue

                cidx = self.motion_to_content_idx.get(motion_id, None)
                if cidx is None or cidx in self.exclude_content_idcs:
                    filtered += 1
                    continue

                if self.nfeats is None:
                    self.nfeats = int(arr.shape[1])

                self.items.append({
                    "motion_id": motion_id,
                    "style_idx": style_idx,
                    "content_idx": cidx,
                    "length": T
                })
                kept += 1

                if kept % 200 == 0:
                    iterable.set_postfix(kept=kept, miss=miss, short=short, outlier=outlier, filt=filtered)

        self.items.sort(key=lambda x: x["length"])
        print(f"[100STYLE] kept={kept} miss={miss} short={short} outlier={outlier} filtered={filtered}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        meta   = self.items[idx]
        motion = np.load(os.path.join(self.motion_dir, f"{meta['motion_id']}.npy"), mmap_mode="r")
        T, D   = int(motion.shape[0]), int(motion.shape[1])

        if self.drop_first_frame and T > 1:
            motion = motion[1:]
            T -= 1

        # unit rounding with optional "double"
        U = self.unit_length
        if U <= 0:
            m_length = T
        else:
            k = max(1, T // U)
            m_length = k * U
            if self.train and U < 10 and self.unit_double_prob > 0.0 and k > 1:
                if random.random() < self.unit_double_prob:
                    m_length = (k - 1) * U
        if self.max_frames is not None:
            m_length = min(m_length, self.max_frames)
        m_length = max(1, min(m_length, T))

        # crop start
        if self.train:
            start_max = max(0, T - m_length)
            s = 0 if start_max == 0 else random.randint(0, start_max)
        else:
            s = max(0, (T - m_length) // 2)
        clip = motion[s:s + m_length]  # (m_length, D)

        # z-norm
        window = (clip - self.mean) / self.std
        window = torch.tensor(window, dtype=torch.float32)

        # right-pad
        if self.max_frames is not None and m_length < self.max_frames:
            pad = torch.zeros(self.max_frames - m_length, D, dtype=window.dtype)
            window = torch.cat([window, pad], dim=0)

        # caption from content code
        content_key = self.content_idx_to_content[meta["content_idx"]]
        caption = self.content_prompts.get(content_key, "a person is moving")

        # match Mixed ordering for 100STYLE side
        return caption, window, int(m_length), int(meta["style_idx"])


class DatasetHumanML3D(Dataset):
    def __init__(self, config, train=True):
        self.train       = bool(train)
        self.mean        = np.load(config["mean_path"])
        self.std         = np.load(config["std_path"])
        self.unit_length = int(config["unit_length"])
        self.min_frames  = int(config["min_frames"])
        self.max_frames  = int(config["max_frames"]) if config["max_frames"] is not None else None

        if train:
            split_base = config["split_base_train"]
        else:
            split_base = config["split_base_valid"]
        split_dir  = os.path.dirname(split_base)
        split_name = os.path.basename(split_base).split(".")[0]
        ids_file   = os.path.join(split_dir, f"{split_name}.txt")
        self.motion_dir = os.path.join(split_dir, "new_joint_vecs")
        self.text_dir   = os.path.join(split_dir, "texts")

        if not os.path.exists(ids_file):
            raise FileNotFoundError(f"Missing ID list: {ids_file}")

        # Load IDs
        with open(ids_file, "r", encoding="utf-8") as f:
            self.id_list = [ln.strip() for ln in f if ln.strip()]

        # Index
        miss = short = kept = long = 0
        self.items = []

        iterable = tqdm(self.id_list, desc=f"Index HumanML3D ({'train' if self.train else 'valid'})", leave=False)
        for motion_id in iterable:
            path = os.path.join(self.motion_dir, f"{motion_id}.npy")
            if not os.path.exists(path):
                miss += 1
                continue

            arr = np.load(path, mmap_mode="r")
            T = int(arr.shape[0])
            if T < self.min_frames:
                short += 1
                continue

            if T >= 200:
                long += 1
                continue

            self.items.append({"motion_id": motion_id, "length": T})
            kept += 1

            if kept % 500 == 0:
                iterable.set_postfix(kept=kept, miss=miss, short=short, long=long)

        self.items.sort(key=lambda x: x["length"])
        if not self.items:
            raise RuntimeError("No HumanML3D items found after filtering.")
        print(f"[HumanML3D] kept={kept} miss={miss} short={short} long={long}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        meta   = self.items[idx]
        motion = np.load(os.path.join(self.motion_dir, f"{meta['motion_id']}.npy"), mmap_mode="r")
        T, D   = int(motion.shape[0]), int(motion.shape[1])

        # Unit rounding
        U = self.unit_length
        if U <= 0:
            m_length = T
        else:
            k = max(1, T // U)
            m_length = k * U
        if self.max_frames is not None:
            m_length = min(m_length, self.max_frames)
        m_length = max(1, min(m_length, T))

        # Start index (random for train, center for eval)
        if self.train:
            start_max = max(0, T - m_length)
            s = 0 if start_max == 0 else random.randint(0, start_max)
        else:
            s = max(0, (T - m_length) // 2)

        clip = motion[s:s + m_length]

        # Z-normalize and tensor-ify
        window = (clip - self.mean) / self.std
        window = torch.tensor(window, dtype=torch.float32)

        # Right-pad to max_frames
        if self.max_frames is not None and m_length < self.max_frames:
            pad = torch.zeros(self.max_frames - m_length, D, dtype=window.dtype)
            window = torch.cat([window, pad], dim=0)

        # Caption: first non-empty line; strip tail after '#'
        cap_path = os.path.join(self.text_dir, f"{meta['motion_id']}.txt")
        caption = "a person is moving"
        if os.path.exists(cap_path):
            with open(cap_path, "r", encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    caption = ln.split("#")[0].strip() or caption
                    break

        # style_idx = -1 for HumanML3D
        return caption, window, int(m_length), -1