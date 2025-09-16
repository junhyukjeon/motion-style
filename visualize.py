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

from data.dataset import TextStyleDataset
from data.sampler import StyleSampler
from model.networks import NETWORK_REGISTRY
from model.t2sm import Text2StylizedMotion
from utils.motion import recover_from_ric
from utils.skel import Skeleton
from utils.train.early_stop import EarlyStopper
from utils.train.loss import LOSS_REGISTRY

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

    config_path = args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config_basename = os.path.basename(config_path)
    config["run_name"] = os.path.splitext(config_basename)[0]
    config["result_dir"] = os.path.join(config["result_dir"], config["run_name"])
    config["checkpoint_dir"] = os.path.join(config["checkpoint_dir"], config["run_name"])
    return config

def load_model(config, device):
    model_cfg = config['model']
    model = Text2StylizedMotion(model_cfg).to(device)
    model.load_state_dict(torch.load(os.path.join(config["checkpoint_dir"], "best.ckpt"), map_location=device))
    model.eval()
    return model

def plot_single_3d_motion(
    save_path,
    kinematic_tree,
    joints,          # (T, 22, 3)
    title="Motion",
    figsize=(6, 6),
    fps=120,
    radius=4.0,
):
    # Preprocess: lift to ground, center by root (x,z), keep traj for past path
    d = joints.copy().reshape(len(joints), -1, 3)
    mn, mx = d.min((0, 1)), d.max((0, 1))
    d[:, :, 1] -= mn[1]
    traj = d[:, 0, (0, 2)]
    d[..., 0] -= d[:, 0:1, 0]
    d[..., 2] -= d[:, 0:1, 2]
    T = d.shape[0]

    colors = ['red','blue','black','red','blue',
              'darkblue','darkblue','darkblue','darkblue','darkblue',
              'darkred','darkred','darkred','darkred','darkred']

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.set(xlim=[-radius/2, radius/2], ylim=[0, radius], zlim=[0, radius], title=title)
    ax.grid(False); ax.set_axis_off(); ax.view_init(120, -90)

    # Ground plane
    plane = Poly3DCollection([[[-radius/2, 0, 0], [-radius/2, 0, radius],
                               [ radius/2, 0, radius], [ radius/2, 0, 0]]],
                             facecolor=(0.5, 0.5, 0.5, 0.5))
    ax.add_collection3d(plane)

    # Past trajectory line
    ltraj, = ax.plot([], [], [], linewidth=1.0)

    # Skeleton lines per chain
    lines = []
    for i, chain in enumerate(kinematic_tree):
        lw = 4.0 if i < 5 else 2.0
        line, = ax.plot([], [], [], linewidth=lw, color=colors[i % len(colors)])
        lines.append(line)

    def update(t):
        # Update plane around current root position using global bounds
        px0, px1 = mn[0] - traj[t, 0], mx[0] - traj[t, 0]
        pz0, pz1 = mn[2] - traj[t, 1], mx[2] - traj[t, 1]
        plane.set_verts([[[px0, 0, pz0], [px0, 0, pz1], [px1, 0, pz1], [px1, 0, pz0]]])

        # Update past trajectory (centered)
        if t > 0:
            x = traj[:t, 0] - traj[t, 0]
            z = traj[:t, 1] - traj[t, 1]
            ltraj.set_data_3d(x, np.zeros_like(x), z)
        else:
            ltraj.set_data_3d([], [], [])

        # Update bones for this frame
        for line, chain in zip(lines, kinematic_tree):
            line.set_data_3d(d[t, chain, 0], d[t, chain, 1], d[t, chain, 2])

        return (ltraj, *lines, plane)

    FuncAnimation(fig, update, frames=T, interval=1000 / fps, repeat=False).save(save_path, fps=fps)
    plt.close(fig)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config()
    set_seed(config["random_seed"])

    # --- Style Split --- #
    with open(config["dataset"]["style_json"]) as f:
        styles_to_ids = json.load(f)
    styles_sorted = sorted(styles_to_ids.keys())
    # train_styles, valid_styles = train_test_split(styles_sorted, test_size=config['valid_size'], random_state=config["random_seed"])

    # --- Dataset & Loader --- #
    dataset_cfg = config['dataset']
    dataset = TextStyleDataset(dataset_cfg, styles_sorted)

    # --- Model --- #
    model_cfg = config['model']
    model = load_model(config, device)

    output_dir = os.path.join(config["result_dir"], "stylized")
    reset_dir(output_dir)

    # -- Mean & Standard Deviation --- #
    mean = np.load(dataset_cfg["mean_path"])
    mean = torch.tensor(mean, dtype=torch.float32, device=device)
    std = np.load(dataset_cfg["std_path"])
    std = torch.tensor(std, dtype=torch.float32, device=device)

    target_style_idx = 1
    target_style     = dataset.style_idx_to_style[target_style_idx]
    sampler_cfg = config['sampler']
    sampler = StyleSampler(sampler_cfg, dataset, target_style=target_style_idx)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)

    batch = next(iter(loader))
    motions, captions, style_idcs = batch
    motions = motions.to(device)

    stylized, captions = model.generate(motions, captions)
    stylized = stylized * std + mean
    joints   = recover_from_ric(stylized, 22).detach().cpu().numpy()
    kinematic_tree = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10],
                        [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21],
                        [9, 13, 16, 18, 20]]

    for i in tqdm(range(stylized.size(0)), desc="Rendering videos", leave=False):
        cap = captions[i] if isinstance(captions, (list, tuple)) else str(captions[i])
        cap = slug(cap)
        save_path = os.path.join(output_dir, f"{cap}_{i}.mp4")

        plot_single_3d_motion(
            save_path=save_path,
            kinematic_tree=kinematic_tree,
            joints=joints[i],
            title=f"{target_style} — {cap}",
            figsize=(6, 6),
            fps=20,
            radius=4.0,
        )