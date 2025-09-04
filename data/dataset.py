import json
import numpy as np
import os
import torch
from torch.utils.data import Dataset

class StyleDataset(Dataset):
    def __init__(self, motion_dir, mean, std, window_size, label_to_ids, style_to_label, ids_to_content):
        # Normalize motion data for VAE
        self.mean = mean
        self.std = std 
        self.window_size = window_size

        # Style-to-label mapping
        self.style_to_label = style_to_label
        self.label_to_style = {i: style for style, i in self.style_to_label.items()}

        # IDs to content
        self.ids_to_content = ids_to_content

        self.data = []       # List of (motion_id, motion_array)
        self.labels = []     # List of label indices
        self.lengths = []    # number of valid windows for each motion
        self.ids = []        #
        self.ids_to_idx = {} #

        for style_name, motion_ids in label_to_ids.items():
            label_idx = self.style_to_label[style_name]
            for motion_id in motion_ids:
                try:
                    motion_path = os.path.join(motion_dir, f"{motion_id}.npy")
                    motion = np.load(motion_path)
                    if motion.shape[0] < window_size:
                        continue
                    self.data.append((motion_id, motion))
                    self.labels.append(label_idx)
                    self.lengths.append(motion.shape[0] - window_size)
                    self.ids.append(motion_id)
                    self.ids_to_idx[motion_id] = len(self.data) - 1
                except Exception as e:
                    print(f"❌ Error loading {motion_id}: {e}")

        self.cumsum = np.cumsum([0] + self.lengths)

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, idx):
        if idx != 0:
            motion_idx = np.searchsorted(self.cumsum, idx) - 1
            offset = idx - self.cumsum[motion_idx] - 1
        else:
            motion_idx = 0
            offset = 0

        motion_id, motion = self.data[motion_idx]
        window = motion[offset:offset + self.window_size]
        window = (window - self.mean) / self.std
        label = self.labels[motion_idx]
        content = self.ids_to_content[motion_id]

        return torch.tensor(window, dtype=torch.float32), label, content, motion_id


class TextStyleDataset(Dataset):
    def __init__(self, config, styles, split=None):
        self.mean        = np.load(config["mean"])
        self.std         = np.load(config["std"])
        self.window_size = np.load(config["window_size"])

        # Style
        with open(config["style_json"], "r") as f:
            style_map = json.load(f)
        styles_sorted = sorted(style_map.keys())
        self.style_to_style_idx = {style: i for i, style in enumerate(styles_sorted)}
        self.style_idx_to_style = {i: style for style, i in self.style_to_style_idx.items()}
        self.style_map          = {self.style_to_style_idx[style]: style_map[style] for style in styles}
        
        # Metadata (?)
        self.items = []
        self.lengths = []
        for style_idx, motion_ids in style_map.items():
            for motion_id in motion_ids:
                motion_path = os.path.join(config["motion_dir"], f"{motion_id}.npy")
                motion      = np.load(motion_path, mmap_mode="r")
                if motion.shape[0] < self.window_size:
                    continue

                text_path   = os.path.join(config["text_dir"], f"{motion_id}.txt")
                captions = []
                with open(text_path, "r") as f:
                    for line in f:
                        parts = line.strip().split("#")
                        if parts and parts[0]:
                            captions.append(parts[0].strip())

                self.items.append({"motion_id": motion_id, "captions": captions, "style_idx": style_idx})
                self.lengths.append(motion.shape[0] - self.window_size)


    def __len__(self):
        return int(self.cumsum[-1])

    def __getitem__(self, idx):
        if idx != 0:
            motion_idx = int(np.searchsorted(self.cum))
            offset     = int(idx)
        else:
            motion_idx = 0
            offset     = 0

        meta = self.items[motion_idx]
        motion_id, captions, style_idx = meta["motion_id"], meta["captions"], meta["style_idx"]

        start, end = offset, offset + self.window_size
        motion = np.load(path, mmap_mode="r")
        window = motion[start:end]
        window = (window - self.mean) / self.std
        window = torch.tensor(window, dtype=torch.float32)
        caption = random.choice(captions) if self.split == "train" else captions[0]
        
        return window, caption, style_idx


#######################################################################

    def __len__(self):
        return self.

        # Storage mirroring your StyleDataset
        self.data      = []  # list of (motion_id, np.ndarray[T,D])
        self.labels    = []  # style label per motion
        self.lengths   = []  # number of valid windows per motion (T - W)
        self.ids       = []  # motion_ids in the same order as self.data
        self.ids_to_idx = {}

        # Per-motion captions (list of strings)
        self.id_to_captions = {}  # motion_id -> [caption, ...]

        for style_name, motion_ids in label_to_ids.items():
            label_idx = self.style_to_label[style_name]
            for motion_id in motion_ids:
                try:
                    motion_path = os.path.join(motion_dir, f"{motion_id}.npy")
                    motion = np.load(motion_path)  # [T, D]
                    T = motion.shape[0]
                    if T < min_motion_len or T >= max_motion_len or T < window_size:
                        continue

                    # load captions
                    cap_path = os.path.join(text_dir, f"{motion_id}.txt")
                    caps = []
                    if os.path.exists(cap_path):
                        with open(cap_path, "r") as f:
                            for line in f:
                                # file format: "caption#tok tok ...#f_tag#to_tag"
                                parts = line.strip().split("#")
                                if len(parts) >= 1:
                                    caps.append(parts[0].strip())
                    # fallback if no captions file
                    if len(caps) == 0:
                        caps = [""]  # allow CFG/unconditional

                    self.id_to_captions[motion_id] = caps

                    # register motion
                    self.data.append((motion_id, motion))
                    self.labels.append(label_idx)
                    self.lengths.append(T - window_size)  # sliding windows
                    self.ids.append(motion_id)
                    self.ids_to_idx[motion_id] = len(self.data) - 1

                except Exception as e:
                    print(f"❌ Error loading {motion_id}: {e}")

        self.cumsum = np.cumsum([0] + self.lengths)

    def __len__(self):
        return int(self.cumsum[-1])

    def __getitem__(self, idx):
        # map global index -> (motion_idx, start_offset)
        # if idx != 0:
        #     motion_idx = np.searchsorted(self.cumsum, idx) - 1
        #     offset = idx - self.cumsum[motion_idx] - 1
        # else:
        #     motion_idx = 0
        #     offset = 0

        motion_id, motion = self.data[motion_idx]
        label  = self.labels[motion_idx]
        T, D   = motion.shape

        # slice fixed window (like StyleDataset)
        start = offset
        end   = start + self.window_size
        window = motion[start:end]  # [W, D]

        # Z-normalize (same as StyleDataset)
        window = (window - self.mean) / self.std

        # length (for mask building in the trainer)
        m_len = self.window_size

        # pick a random caption for this motion
        caps = self.id_to_captions[motion_id]
        caption = random.choice(caps)

        content = self.ids_to_content.get(motion_id, None)

        return caption, torch.tensor(window, dtype=torch.float32), m_len, label, content, motion_id