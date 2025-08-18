import argparse
import json
import numpy as np
import os
import random
import shutil
import torch
import yaml
from tqdm import tqdm

from data.dataset import StyleDataset
from model.networks import NETWORK_REGISTRY
from utils.motion import *
from utils.skel import Skeleton
# from utils.process import parse_humanml3d
# from utils.write import write_bvh

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


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    model = NETWORK_REGISTRY[model_cfg['type']](model_cfg).to(device)
    model.load_state_dict(torch.load(os.path.join(config["checkpoint_dir"], "best.ckpt"), map_location=device))
    model.eval()
    return model


import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_pair_3d_motion(
    save_path,
    kinematic_tree,
    joints_left,   # [T, 22, 3]  e.g. recon
    joints_right,  # [T, 22, 3]  e.g. swap
    title_left="Recon",
    title_right="Swap",
    figsize=(12, 6),
    fps=120,
    radius=4.0
):
    """
    Renders two sequences side-by-side in one video.
    Each sequence is normalized to its own root trajectory but shares the same
    camera settings per frame for fair visual comparison.
    """
    def prep(j):
        # j: [T, 22, 3] -> centered & trajectory extracted for ground plane
        data = j.copy()
        T = data.shape[0]

        # height normalize (subtract min Y over full seq)
        mins = data.min(axis=0).min(axis=0)
        maxs = data.max(axis=0).max(axis=0)

        height_offset = mins[1]
        data[:, :, 1] -= height_offset  # lift to ground

        # global trajectory from root (assume joint 0 is root)
        traj = data[:, 0, [0, 2]]  # [T, 2] (x, z)

        # center each frame on root in XZ (so subject stays near origin)
        data[..., 0] -= data[:, 0:1, 0]
        data[..., 2] -= data[:, 0:1, 2]

        return data, traj, mins, maxs

    data_L, traj_L, mins_L, maxs_L = prep(joints_left)
    data_R, traj_R, mins_R, maxs_R = prep(joints_right)

    # global bounds (use both clips to get shared axis limits)
    mins = np.minimum(mins_L, mins_R)
    maxs = np.maximum(maxs_L, maxs_R)

    colors = ['red','blue','black','red','blue',
              'darkblue','darkblue','darkblue','darkblue','darkblue',
              'darkred','darkred','darkred','darkred','darkred']

    fig = plt.figure(figsize=figsize)
    axL = fig.add_subplot(1, 2, 1, projection='3d')
    axR = fig.add_subplot(1, 2, 2, projection='3d')

    def init_ax(ax, title):
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([0, radius])
        ax.set_title(title, fontsize=12)
        ax.grid(False)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5

    def plot_xz_plane(ax, minx, maxx, minz, maxz):
        verts = [[minx, 0, minz],
                 [minx, 0, maxz],
                 [maxx, 0, maxz],
                 [maxx, 0, minz]]
        plane = Poly3DCollection([verts])
        plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(plane)

    init_ax(axL, title_left)
    init_ax(axR, title_right)
    fig.suptitle(f"{title_left} | {title_right}", fontsize=14)

    T = data_L.shape[0]
    assert data_R.shape[0] == T, "Left and Right sequences must have equal length"

    def draw_frame(ax, data, traj, idx):
        for line in ax.get_lines():
            line.remove()
        for coll in list(ax.collections):
            coll.remove()
        # ground plane aligned to current trajectory offset (XZ)
        plot_xz_plane(
            ax,
            mins[0] - traj[idx, 0], maxs[0] - traj[idx, 0],
            mins[2] - traj[idx, 1], maxs[2] - traj[idx, 1]
        )
        if idx > 1:
            ax.plot3D(traj[:idx, 0] - traj[idx, 0],
                      np.zeros_like(traj[:idx, 0]),
                      traj[:idx, 1] - traj[idx, 1],
                      linewidth=1.0, color='blue')

        for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
            lw = 4.0 if i < 5 else 2.0
            ax.plot3D(data[idx, chain, 0],
                      data[idx, chain, 1],
                      data[idx, chain, 2],
                      linewidth=lw, color=color)

    def update(idx):
        draw_frame(axL, data_L, traj_L, idx)
        draw_frame(axR, data_R, traj_R, idx)

    ani = FuncAnimation(fig, update, frames=T, interval=1000 / fps, repeat=False)
    ani.save(save_path, fps=fps)
    plt.close(fig)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config()
    set_seed(config["random_seed"])

    mean = np.load(config["mean_path"])
    std = np.load(config["std_path"])

    with open(config["style_json"]) as f:
        label_to_ids = yaml.safe_load(f)

    style_to_label = {style: i for i, style in enumerate(sorted(label_to_ids))}

    # --- Content Mapping ---
    with open(config["content_json"]) as f:
        content_to_ids = json.load(f)
    
    ids_to_content = {}
    for content_type, motion_ids in content_to_ids.items():
        for m_id in motion_ids:
            ids_to_content[m_id] = content_type

    dataset = StyleDataset(config["motion_dir"], mean, std, config["window_size"], label_to_ids, style_to_label, ids_to_content)
    model = load_model(config, device)

    output_dir = os.path.join(config["result_dir"], "style")
    reset_dir(output_dir)

    # === [1] Load reference style motion
    # style_idx = 1540217
    # style_idx = 815700
    style_idx = 0
    style_motion, _, _, style_motion_id = dataset[style_idx]
    style_motion = style_motion.unsqueeze(0).to(device)  # [1, T, J, D]

    # Find label name for style_motion_id
    for label, ids in label_to_ids.items():
        if style_motion_id in ids:
            style_label = label
            break
    else:
        raise ValueError(f"Could not find label for motion ID {style_motion_id}")

    with torch.no_grad():
        out = model.encode(style_motion)
        z_style_swap = out["z_style"].expand(32, -1)

    print(f"🎨 Using style from motion ID: {style_motion_id} (label: {style_label})")

    # === [2] Sample 32 motions to swap content
    sample_indices = random.sample(range(len(dataset)), 32)
    batch_size = 32
    batched_indices = [sample_indices[i:i+batch_size] for i in range(0, len(sample_indices), batch_size)]

    mean = torch.tensor(mean, dtype=torch.float32, device=device)
    std = torch.tensor(std, dtype=torch.float32, device=device)

    for batch_ids in batched_indices:
        motions, motion_ids = [], []
        for idx in batch_ids:
            motion, _, _, motion_id = dataset[idx]
            motions.append(motion)
            motion_ids.append(motion_id)

        motions = torch.stack(motions).to(device)  # [B, T, J, D]

        with torch.no_grad():
            out = model.encode(motions)
            z_style = out["z_style"]
            z_content = out["z_content"]

            z_recon = model.decode(z_style, z_content)
            z_swap  = model.decode(z_style_swap, z_content)
            motions_swap  = model.encoder.vae.decode(z_swap)   # [B, T, J, D]
            motions_recon = model.encoder.vae.decode(z_recon)

        motions_recon = motions_recon * std + mean
        motions_swap  = motions_swap  * std + mean

        joints_swap  = recover_from_ric(motions_swap, 22)
        joints_recon = recover_from_ric(motions_recon, 22)

        joints_recon_np = joints_recon.detach().cpu().numpy()
        joints_swap_np  = joints_swap.detach().cpu().numpy()

        kinematic_tree = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10],
                          [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21],
                          [9, 13, 16, 18, 20]]

        # >>> progress bar moved here <<<
        for i, motion_id in enumerate(tqdm(motion_ids, desc="Rendering videos", leave=False)):
            save_path = os.path.join(output_dir, f"{motion_id}_{style_label}_pair.mp4")

            plot_pair_3d_motion(
                save_path=save_path,
                kinematic_tree=kinematic_tree,
                joints_left=joints_recon_np[i],   # [T,22,3]
                joints_right=joints_swap_np[i],   # [T,22,3]
                title_left="Recon",
                title_right=f"Swap → {style_label}",
                figsize=(12, 6),
                fps=20,
                radius=4.0
            )