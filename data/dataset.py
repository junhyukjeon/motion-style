import json
import numpy as np
import os
import torch
from torch.utils.data import Dataset

class StyleDataset(Dataset):
    def __init__(self, json_path, motion_dir, mean, std, window_size):
        # Normalize motion data for VAE
        self.mean = mean
        self.std = std

        # 
        self.window_size = window_size

        with open(json_path, 'r') as f:
            label_to_ids = json.load(f)

        self.data = []
        self.labels = []
        self.lengths = []

        # Style label from string to integers
        self.style_to_label = {style: i for i, style in enumerate(label_to_ids.keys())}
        self.label_to_style = {i: style for style, i in self.style_to_label.items()}

        for style_name, motion_ids in label_to_ids.items():
            label = self.style_to_label[style_name]
            for motion_id in motion_ids:
                try:
                    motion = np.load(os.path.join(motion_dir, f"{motion_id}.npy"))
                    if motion.shape[0] < window_size:
                        continue
                    self.lengths.append(motion.shape[0] - window_size)
                    self.data.append((motion_id, motion))
                    self.labels.append(label)
                except Exception as e:
                    print(f"Error loading motion {motion_id}: {e}")

        self.cumsum = np.cumsum([0] + self.lengths)

        # import pdb; pdb.set_trace()

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

        return torch.tensor(window, dtype=torch.float32), label