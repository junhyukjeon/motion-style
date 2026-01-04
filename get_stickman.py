# DISCLAIMER:
# This code was produced with heavy LLM assistance.
# It’s functional but not guaranteed to be clean or optimal.
# In fact, even this disclaimer was LLM-generated.

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from data.dataset import Dataset100Style, DatasetHumanML3D
from data.sampler import StyleSampler
from model.t2sm import Text2StylizedMotion
from utils.motion import recover_from_ric

from mld.data.humanml.utils.plot_script import plot_3d_motion
# from salad.utils.plot_script import plot_3d_motion

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def reset_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def get_unique_path(base_path):
    if not os.path.exists(base_path):
        return base_path

    base, ext = os.path.splitext(base_path)
    i = 1
    while True:
        new_path = f"{base}_{i}{ext}"
        if not os.path.exists(new_path):
            return new_path
        i += 1

def slug(s: str, maxlen: int = 60) -> str:
    # keep letters, digits, some punctuation; replace others with '_'
    s = ''.join(c if (c.isalnum() or c in " _-.,()[]{}") else '_' for c in s.strip())
    s = "_".join(s.split())  # spaces -> underscores
    return (s[:maxlen]).rstrip("_")

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file (YAML)')
    args = parser.parse_args()

    from pathlib import Path
    cfg_path = Path(args.config).resolve()

    with cfg_path.open('r') as f:
        config = yaml.safe_load(f)

    # run_name = the path inside "configs/" without the .yaml suffix
    # e.g., configs/loss/0.yaml  ->  run_name="loss/0"
    parts = cfg_path.parts
    if "configs" in parts:
        i = parts.index("configs")
        sub = Path(*parts[i+1:]).with_suffix("")   # loss/0 (Path)
        run_name = str(sub).replace("\\", "/")     # normalize on Windows just in case
    else:
        run_name = cfg_path.stem                   # fallback

    config["run_name"] = run_name

    # results/loss/0  and  checkpoints/loss/0
    config["result_dir"]     = os.path.join(config["result_dir"], "test")
    config["checkpoint_dir"] = os.path.join(config["checkpoint_dir"], run_name)
    return config

def load_model(config, device):
    model_cfg = config['model']
    model = Text2StylizedMotion(model_cfg).to(device)
    model.load_state_dict(torch.load(os.path.join(config["checkpoint_dir"], "best.ckpt"), map_location=device), strict=False)
    model.eval()
    return model

def _preprocess_motion(joints):
    # joints: (T, 22, 3)
    d = joints.copy().reshape(len(joints), -1, 3)
    mn, mx = d.min((0, 1)), d.max((0, 1))
    # lift to ground
    d[:, :, 1] -= mn[1]
    # root trajectory (x, z)
    traj = d[:, 0, (0, 2)]
    # center by root x/z
    d[..., 0] -= d[:, 0:1, 0]
    d[..., 2] -= d[:, 0:1, 2]
    return d, mn, mx, traj, d.shape[0]


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config()
    set_seed(config["random_seed"])

    # --- Style Split --- #
    # with open(config["dataset"]["style_json"]) as f:
    #     styles_to_ids = json.load(f)
    # styles_sorted = sorted(styles_to_ids.keys())
    # train_styles, valid_styles = train_test_split(styles_sorted, test_size=config['valid_size'], random_state=config["random_seed"])

    # --- Model --- #
    model = load_model(config, device)

    # --- Datasets (new) --- #
    style_cfg = config['dataset_style']  # your new block for 100STYLE
    with open(style_cfg["style_json"], "r", encoding="utf-8") as f:
        styles_to_ids = json.load(f)
    all_styles = sorted(styles_to_ids.keys())

    # 100STYLE dataset (eval mode so center-crop is used)
    ds_style = Dataset100Style(style_cfg, styles=all_styles, train=False)

    # Pick a target style index and subset to only those items
    target_style_idx = 81
    if target_style_idx not in ds_style.style_idx_to_style:
        raise ValueError(f"target_style_idx={target_style_idx} not in dataset.")

    indices = [i for i, it in enumerate(ds_style.items) if it["style_idx"] == target_style_idx]
    if len(indices) == 0:
        raise RuntimeError(f"No items found for style_idx={target_style_idx} ({ds_style.style_idx_to_style[target_style_idx]}).")

    # Build a DataLoader over that subset
    from torch.utils.data import Subset
    # B = config['sampler']['batch_size'] if 'sampler' in config else 16
    B = 16
    loader = DataLoader(Subset(ds_style, indices), batch_size=B, shuffle=True, num_workers=0)

    # --- Output dir --- #
    style_weight = config["model"].get("style_weight", None)
    style_guidance = config["model"].get("style_guidance", None)

    def fmt_tag(prefix: str, value) -> str:
        """
        Turn a numeric/bool config value into a filesystem-friendly tag.
        Examples:
            1.5   -> 'w1p5'
            0.0   -> 'g0p0'
            True  -> 'g1'
        """
        if isinstance(value, bool):
            return f"{prefix}{int(value)}"
        try:
            v = float(value)
            return f"{prefix}{str(v).replace('.', 'p')}"
        except (TypeError, ValueError):
            # fallback: just stringify
            return f"{prefix}{str(value)}"

    tag_parts = []
    if style_weight is not None:
        tag_parts.append(fmt_tag("w", style_weight))

    if style_guidance is not None:
        tag_parts.append(fmt_tag("g", style_guidance))

    if tag_parts:
        # e.g. "w1p5_g3p0"
        style_tag = "_".join(tag_parts)
        output_dir = os.path.join(config["result_dir"], style_tag)
    else:
        # fallback if neither is defined
        output_dir = os.path.join(config["result_dir"], "stylized")
    reset_dir(output_dir)

    # --- Mean & Std (tensors on device) --- #
    mean = torch.tensor(np.load(style_cfg["mean_path"]), dtype=torch.float32, device=device)
    std  = torch.tensor(np.load(style_cfg["std_path"]),  dtype=torch.float32, device=device)

    # --- One batch ---
    (cap1, win1, len1, sty1) = next(iter(loader)) # captions, (B,T,D), lengths, style_idx
    motions = win1.to(device)                     # normalized
    captions = list(cap1)                         # list[str]

    captions = ["a person walks forward and sits down"]*B

    # --- Generate stylized (uses your existing signature) ---
    stylized, captions_out = model.generate(motions, captions, len1, len1)

    # --- Denormalize stylized & reference (reference = input before swap, like before) ---
    stylized  = stylized * std + mean
    reference = motions * std + mean

    # --- Recover joints and render ---
    joints_stylized  = recover_from_ric(stylized, 22).detach().cpu().numpy()
    joints_reference = recover_from_ric(reference, 22).detach().cpu().numpy()

    kinematic_tree = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10],
                    [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21],
                    [9, 13, 16, 18, 20]]

    style_name = ds_style.style_idx_to_style[target_style_idx]
    lengths = len1.cpu().numpy().astype(int)

    num_samples = B
    num_repetitions = 1
    N = num_samples * num_repetitions

    T_max = int(lengths.max())
    J = 22

    all_motions = np.zeros((N, J, 3, T_max), dtype=np.float32)
    all_lengths = lengths.astype(np.int32)
    all_text = [str(captions_out[i]) for i in range(B)]

    for sample_i in range(B):
        L = int(all_lengths[sample_i])
        xyz = joints_stylized[sample_i][:L].astype(np.float32)  # (L,22,3)

        # fill as (J,3,T)
        all_motions[sample_i, :, :, :L] = np.transpose(xyz, (1, 2, 0))

        save_file = f"sample{sample_i:02d}_rep{0:02d}.mp4"
        animation_save_path = os.path.join(output_dir, save_file)

        plot_3d_motion(
            animation_save_path,
            kinematic_tree,
            xyz,                     # (L,22,3) for plotting
            title=all_text[sample_i],
            dataset="humanml",
            fps=20,
        )

    np.save(os.path.join(output_dir, "results.npy"), {
        "motion": all_motions,
        "text": all_text,
        "lengths": all_lengths,
        "num_samples": num_samples,
        "num_repetitions": num_repetitions,
    })