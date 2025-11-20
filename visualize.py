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
# from utils.skel import Skeleton

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
    config["result_dir"]     = os.path.join(config["result_dir"], run_name)
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

def plot_pair_3d_motion(
    save_path,
    kinematic_tree,
    joints_a,          # (T, 22, 3) stylized
    joints_b,          # (T, 22, 3) reference
    titles=("Stylized", "Reference"),
    figsize=(12, 6),
    fps=20,
    radius=4.0,
):
    # --- preprocess both sequences ---
    da, mna, mxa, trj_a, Ta = _preprocess_motion(joints_a)
    db, mnb, mxb, trj_b, Tb = _preprocess_motion(joints_b)
    T = min(Ta, Tb)  # sync to shortest

    colors = ['red','blue','black','red','blue',
              'darkblue','darkblue','darkblue','darkblue','darkblue',
              'darkred','darkred','darkred','darkred','darkred']

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=figsize)
    axL = fig.add_subplot(1, 2, 1, projection="3d")
    axR = fig.add_subplot(1, 2, 2, projection="3d")
    for ax, title in ((axL, titles[0]), (axR, titles[1])):
        ax.set(xlim=[-radius/2, radius/2], ylim=[0, radius], zlim=[0, radius], title=title)
        ax.grid(False); ax.set_axis_off(); ax.view_init(120, -90)

    # Ground planes
    planeL = Poly3DCollection([[[-radius/2, 0, 0], [-radius/2, 0, radius],
                                [ radius/2, 0, radius], [ radius/2, 0, 0]]],
                              facecolor=(0.5, 0.5, 0.5, 0.5))
    planeR = Poly3DCollection([[[-radius/2, 0, 0], [-radius/2, 0, radius],
                                [ radius/2, 0, radius], [ radius/2, 0, 0]]],
                              facecolor=(0.5, 0.5, 0.5, 0.5))
    axL.add_collection3d(planeL)
    axR.add_collection3d(planeR)

    # Past trajectory lines
    ltrajL, = axL.plot([], [], [], linewidth=1.0)
    ltrajR, = axR.plot([], [], [], linewidth=1.0)

    # Skeleton lines for each subplot
    linesL, linesR = [], []
    for i, chain in enumerate(kinematic_tree):
        lw = 4.0 if i < 5 else 2.0
        lineL, = axL.plot([], [], [], linewidth=lw, color=colors[i % len(colors)])
        lineR, = axR.plot([], [], [], linewidth=lw, color=colors[i % len(colors)])
        linesL.append(lineL)
        linesR.append(lineR)

    def update(t):
        # Left plane (A)
        px0, px1 = mna[0] - trj_a[t, 0], mxa[0] - trj_a[t, 0]
        pz0, pz1 = mna[2] - trj_a[t, 1], mxa[2] - trj_a[t, 1]
        planeL.set_verts([[[px0, 0, pz0], [px0, 0, pz1], [px1, 0, pz1], [px1, 0, pz0]]])

        # Right plane (B)
        px0, px1 = mnb[0] - trj_b[t, 0], mxb[0] - trj_b[t, 0]
        pz0, pz1 = mnb[2] - trj_b[t, 1], mxb[2] - trj_b[t, 1]
        planeR.set_verts([[[px0, 0, pz0], [px0, 0, pz1], [px1, 0, pz1], [px1, 0, pz0]]])

        # Past trajectories
        if t > 0:
            xa = trj_a[:t, 0] - trj_a[t, 0]
            za = trj_a[:t, 1] - trj_a[t, 1]
            ltrajL.set_data_3d(xa, np.zeros_like(xa), za)

            xb = trj_b[:t, 0] - trj_b[t, 0]
            zb = trj_b[:t, 1] - trj_b[t, 1]
            ltrajR.set_data_3d(xb, np.zeros_like(xb), zb)
        else:
            ltrajL.set_data_3d([], [], [])
            ltrajR.set_data_3d([], [], [])

        # Bones
        for line, chain in zip(linesL, kinematic_tree):
            line.set_data_3d(da[t, chain, 0], da[t, chain, 1], da[t, chain, 2])
        for line, chain in zip(linesR, kinematic_tree):
            line.set_data_3d(db[t, chain, 0], db[t, chain, 1], db[t, chain, 2])

        return (ltrajL, ltrajR, planeL, planeR, *linesL, *linesR)

    FuncAnimation(fig, update, frames=T, interval=1000 / fps, repeat=False).save(save_path, fps=fps)
    plt.close(fig)


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
    target_style_idx = 0
    if target_style_idx not in ds_style.style_idx_to_style:
        raise ValueError(f"target_style_idx={target_style_idx} not in dataset.")

    indices = [i for i, it in enumerate(ds_style.items) if it["style_idx"] == target_style_idx]
    if len(indices) == 0:
        raise RuntimeError(f"No items found for style_idx={target_style_idx} ({ds_style.style_idx_to_style[target_style_idx]}).")

    # Build a DataLoader over that subset
    from torch.utils.data import Subset
    B = config['sampler']['batch_size'] if 'sampler' in config else 16
    loader = DataLoader(Subset(ds_style, indices), batch_size=B, shuffle=True, num_workers=0)

    # --- Output dir --- #
    # Put videos under: result_dir/stylized/<style_weight>/
    style_weight = config["model"].get("style_weight", None)

    if style_weight is not None:
        # make it filesystem-friendly: e.g. 1.5 -> "w1p5"
        style_tag = f"w{style_weight}".replace(".", "p")
        output_dir = os.path.join(config["result_dir"], "stylized", style_tag)
    else:
        # fallback if style_weight isn't defined
        output_dir = os.path.join(config["result_dir"], "stylized")

    reset_dir(output_dir)

    # --- Mean & Std (tensors on device) --- #
    mean = torch.tensor(np.load(style_cfg["mean_path"]), dtype=torch.float32, device=device)
    std  = torch.tensor(np.load(style_cfg["std_path"]),  dtype=torch.float32, device=device)

    # --- One batch ---
    (cap1, win1, len1, sty1) = next(iter(loader)) # captions, (B,T,D), lengths, style_idx
    motions = win1.to(device)                     # normalized
    captions = list(cap1)                         # list[str]

    # Optional: pair/mismatch like your old code (swap every other item)
    # idx = torch.arange(motions.shape[0], device=motions.device)
    # motions_swapped = motions[idx ^ 1]

    captions = ["a person walks forward and sits down"]*32

    # --- Generate stylized (uses your existing signature) ---
    stylized, captions_out = model.generate(motions, captions, len1, len1)

    # --- Denormalize stylized & reference (reference = input before swap, like before) ---
    stylized  = stylized * std + mean
    reference = motions * std + mean

    # --- Recover joints and render ---
    joints_stylized  = recover_from_ric(stylized, 22).detach().cpu().numpy()    # (B, T, 22, 3)
    joints_reference = recover_from_ric(reference, 22).detach().cpu().numpy()

    kinematic_tree = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10],
                    [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21],
                    [9, 13, 16, 18, 20]]

    style_name = ds_style.style_idx_to_style[target_style_idx]
    for i in tqdm(range(stylized.size(0)), desc="Rendering videos", leave=False):
        cap = captions_out[i] if isinstance(captions_out, (list, tuple)) else str(captions_out[i])
        cap = slug(cap)
        save_path = os.path.join(output_dir, f"{cap}_{i}.mp4")

        plot_pair_3d_motion(
            save_path=save_path,
            kinematic_tree=kinematic_tree,
            joints_a=joints_stylized[i],
            joints_b=joints_reference[i],
            titles=(f"{style_name} — {cap}", "Reference"),
            figsize=(12, 6),
            fps=20, 
            radius=4.0,
        )