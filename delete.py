#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# RIC (T, F) or (B, T, F) -> joints (T, 22, 3) or (B, T, 22, 3)
from utils.motion import recover_from_ric

# simple chains for drawing bones
KINEMATIC_TREE = [
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20],
]

def render_sequence(save_path, joints, fps=20, radius=4.0):
    """joints: (T, 22, 3) numpy"""
    d = joints.copy()
    T = d.shape[0]

    # lift & center by root x/z
    d[:, :, 1] -= d[:, :, 1].min()
    d[..., 0] -= d[:, 0:1, 0]
    d[..., 2] -= d[:, 0:1, 2]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set(xlim=[-radius/2, radius/2], ylim=[0, radius], zlim=[0, radius])
    ax.set_axis_off(); ax.view_init(120, -90)

    lines = []
    for _ in KINEMATIC_TREE:
        (ln,) = ax.plot([], [], [], lw=2)
        lines.append(ln)

    def update(t):
        for ln, chain in zip(lines, KINEMATIC_TREE):
            ln.set_data_3d(d[t, chain, 0], d[t, chain, 1], d[t, chain, 2])
        return lines

    FuncAnimation(fig, update, frames=T, interval=1000/fps, blit=False).save(save_path, fps=fps)
    plt.close(fig)

@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npy", required=True, help="Path to npy with shape (B, 196, 256)")
    p.add_argument("--outdir", default="viz", help="Output directory for mp4s")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    arr = np.load(args.npy)           # (B, 196, 256)
    assert arr.ndim == 3, f"Expected (B,T,F). Got {arr.shape}"
    B, T, F = arr.shape

    os.makedirs(args.outdir, exist_ok=True)

    # recover joints in batch on GPU, then loop to save
    x = torch.from_numpy(arr).float().to(args.device)          # (B,T,F)
    mean = np.load("./checkpoints/t2m/Comp_v6_KLD005/meta/mean.npy")
    mean = torch.tensor(mean, dtype=torch.float32, device='cuda:0')
    std = np.load("./checkpoints/t2m/Comp_v6_KLD005/meta/std.npy")
    std = torch.tensor(std, dtype=torch.float32, device='cuda:0')
    x = x * std + mean

    joints_bt = recover_from_ric(x, 22)                        # (B,T,22,3) torch
    joints_bt = joints_bt.detach().cpu().numpy()

    for b in range(B):
        save_path = os.path.join(args.outdir, f"sample_{b:03d}.mp4")
        render_sequence(save_path, joints_bt[b], fps=args.fps)

if __name__ == "__main__":
    main()
