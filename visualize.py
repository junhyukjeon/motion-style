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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_triple_3d_motion(
    save_path,
    kinematic_tree,
    joints_gt,       # [T, 22, 3]
    joints_recon,    # [T, 22, 3]
    joints_swap,     # [T, 22, 3]
    title_gt="GT",
    title_recon="Recon",
    title_swap="Swap",
    figsize=(18, 6),
    fps=120,
    radius=4.0
):
    matplotlib.use('Agg')

    def prep_clip(joints):
        data = joints.copy().reshape(len(joints), -1, 3)
        mins = data.min(axis=0).min(axis=0)
        maxs = data.max(axis=0).max(axis=0)
        # lift to ground
        data[:, :, 1] -= mins[1]
        # root trajectory (x,z)
        traj = data[:, 0, [0, 2]]
        # center each frame on root XZ
        data[..., 0] -= data[:, 0:1, 0]
        data[..., 2] -= data[:, 0:1, 2]
        return {"data": data, "mins": mins, "maxs": maxs, "traj": traj, "T": data.shape[0]}

    G = prep_clip(joints_gt)
    R = prep_clip(joints_recon)
    S = prep_clip(joints_swap)
    T = G["T"]
    assert R["T"] == T and S["T"] == T, "All three sequences must have equal length"

    # same palette & widths as your original
    colors = ['red','blue','black','red','blue',
              'darkblue','darkblue','darkblue','darkblue','darkblue',
              'darkred','darkred','darkred','darkred','darkred']

    fig = plt.figure(figsize=figsize)
    axG = fig.add_subplot(1, 3, 1, projection='3d')
    axR = fig.add_subplot(1, 3, 2, projection='3d')
    axS = fig.add_subplot(1, 3, 3, projection='3d')
    axes  = [axG, axR, axS]
    clips = [G,   R,   S]
    titles = [title_gt, title_recon, title_swap]

    for ax, title in zip(axes, titles):
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([0, radius])
        ax.set_title(title, fontsize=12)
        ax.grid(False)
        ax.set_axis_off()  # same effect as plt.axis('off') but scoped to this ax
        ax.view_init(elev=120, azim=-90)
        try: ax.dist = 7.5
        except Exception: pass

    fig.suptitle(f"{title_gt} | {title_recon} | {title_swap}", fontsize=16)

    # ----- initialize artists once -----
    planes = []
    traj_lines = []
    bone_lines = []   # list per-axes, each contains per-chain line
    for ax in axes:
        # simple initial plane (will update verts each frame)
        verts0 = [[-radius/2, 0, 0],
                  [-radius/2, 0, radius],
                  [ radius/2, 0, radius],
                  [ radius/2, 0, 0]]
        plane = Poly3DCollection([verts0])
        plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(plane)
        planes.append(plane)

        # past trajectory line
        line_traj, = ax.plot([], [], [], linewidth=1.0, color='blue')
        traj_lines.append(line_traj)

        # skeleton chain lines
        lines = []
        for ci, chain in enumerate(kinematic_tree):
            lw = 4.0 if ci < 5 else 2.0
            line, = ax.plot([], [], [], linewidth=lw, color=colors[ci % len(colors)])
            lines.append(line)
        bone_lines.append(lines)

    # ----- per-frame update -----
    def update(idx):
        for ax, plane, ltraj, lines, clip in zip(axes, planes, traj_lines, bone_lines, clips):
            data, traj, mins, maxs = clip["data"], clip["traj"], clip["mins"], clip["maxs"]

            # update ground plane verts to follow global XZ bounds relative to current root
            px0, px1 = mins[0] - traj[idx, 0], maxs[0] - traj[idx, 0]
            pz0, pz1 = mins[2] - traj[idx, 1], maxs[2] - traj[idx, 1]
            plane.set_verts([[[px0, 0.0, pz0],
                               [px0, 0.0, pz1],
                               [px1, 0.0, pz1],
                               [px1, 0.0, pz0]]])

            # update past trajectory (XZ, centered)
            if idx > 1:
                x = traj[:idx, 0] - traj[idx, 0]
                y = np.zeros_like(x)
                z = traj[:idx, 1] - traj[idx, 1]
                ltraj.set_data_3d(x, y, z)
            else:
                ltraj.set_data_3d([], [], [])

            # update bones
            for line, chain in zip(lines, kinematic_tree):
                xs = data[idx, chain, 0]
                ys = data[idx, chain, 1]
                zs = data[idx, chain, 2]
                line.set_data_3d(xs, ys, zs)

        # return a flat tuple of artists for FuncAnimation (no blit)
        return tuple(traj_lines) + tuple(l for lines in bone_lines for l in lines) + tuple(planes)

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
        # z_style_swap = z_style_swap * 0

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
            z_style   = out["z_style"]
            z_content = out["z_content"]
            z_latent  = out["z_latent"]

            z_recon = model.decode(z_style, z_content)
            z_swap  = model.decode(z_style_swap, z_content)
            motions_gt = model.encoder.vae.decode(z_latent) 
            motions_swap  = model.encoder.vae.decode(z_swap)   # [B, T, J, D]
            motions_recon = model.encoder.vae.decode(z_recon)

        motions_gt = motions * std + mean
        motions_recon = motions_recon * std + mean
        motions_swap  = motions_swap  * std + mean

        joints_gt    = recover_from_ric(motions_gt, 22)
        joints_swap  = recover_from_ric(motions_swap, 22)
        joints_recon = recover_from_ric(motions_recon, 22)

        joints_gt       = joints_gt.detach().cpu().numpy()
        joints_recon_np = joints_recon.detach().cpu().numpy()
        joints_swap_np  = joints_swap.detach().cpu().numpy()

        kinematic_tree = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10],
                          [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21],
                          [9, 13, 16, 18, 20]]

        # >>> progress bar moved here <<<
        for i, motion_id in enumerate(tqdm(motion_ids, desc="Rendering videos", leave=False)):
            save_path = os.path.join(output_dir, f"{motion_id}_{style_label}_pair.mp4")

            plot_triple_3d_motion(
                save_path=os.path.join(output_dir, f"{motion_id}_{style_label}_triple.mp4"),
                kinematic_tree=kinematic_tree,
                joints_gt=joints_gt[i],
                joints_recon=joints_recon_np[i],
                joints_swap=joints_swap_np[i],
                title_gt="GT",
                title_recon="Recon",
                title_swap=f"Swap → {style_label}",
                figsize=(18,6),
                fps=20,
                radius=4.0
            )