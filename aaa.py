import os
import re
import random
import argparse
from collections import OrderedDict

import numpy as np
import torch

from utils.motion import recover_from_ric
from mld.data.humanml.utils.plot_script import plot_3d_motion


KINEMATIC_TREE = [
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20],
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def slug(s: str, maxlen: int = 80) -> str:
    s = ''.join(c if (c.isalnum() or c in "._-") else "_" for c in s.strip())
    s = re.sub(r"_+", "_", s)
    return s[:maxlen].rstrip("_")


def parse_style_name(bvh_name: str) -> str:
    """
    Examples:
        Aeroplane_BR_00.bvh      -> Aeroplane
        ArmsAboveHead_SW_02.bvh  -> ArmsAboveHead
    """
    stem = os.path.splitext(os.path.basename(bvh_name))[0]
    parts = stem.rsplit("_", 2)
    if len(parts) == 3:
        return parts[0]
    return stem


def parse_name_dict(txt_path: str, skip_m_prefix: bool = True):
    """
    Parses lines like:
        030001 Aeroplane_BR_00.bvh 0
        M030001 Aeroplane_BR_00.bvh 0
    Returns an ordered dict:
        style_idx -> list of dicts with keys: motion_id, bvh_name, style_idx, style_name
    """
    grouped = OrderedDict()

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 3:
                continue

            motion_id = parts[0]
            bvh_name = parts[1]
            style_idx = int(parts[-1])

            if skip_m_prefix and motion_id.startswith("M"):
                continue

            style_name = parse_style_name(bvh_name)

            grouped.setdefault(style_idx, []).append(
                {
                    "motion_id": motion_id,
                    "bvh_name": bvh_name,
                    "style_idx": style_idx,
                    "style_name": style_name,
                }
            )

    return grouped


def choose_one_per_style(grouped, mode="first", seed=42):
    """
    mode:
        - first: take the first motion listed for each style
        - random: choose a random motion from each style
    """
    rng = random.Random(seed)
    selected = []

    for style_idx, items in grouped.items():
        if not items:
            continue

        if mode == "random":
            picked = rng.choice(items)
        else:
            picked = items[0]

        selected.append(picked)

    return selected


def load_motion_npy(motion_dir: str, motion_id: str) -> np.ndarray:
    path = os.path.join(motion_dir, f"{motion_id}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Motion file not found: {path}")

    motion = np.load(path)

    if motion.ndim != 2:
        raise ValueError(f"Expected motion array shape (T, D), got {motion.shape} for {path}")

    return motion


def render_motion(
    motion: np.ndarray,
    save_path: str,
    fps: int = 20,
):
    """
    motion: (T, D) in HumanML/new_joint_vecs format
    """
    motion_tensor = torch.tensor(motion, dtype=torch.float32).unsqueeze(0)  # (1, T, D)
    joints = recover_from_ric(motion_tensor, 22)[0].cpu().numpy()  # (T, 22, 3)

    plot_3d_motion(
        save_path,
        KINEMATIC_TREE,
        joints.astype(np.float32),
        title="",
        dataset="humanml",
        fps=fps,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--txt_path",
        type=str,
        default="/mnt/data/100STYLE_name_dict.txt",
    )
    parser.add_argument(
        "--motion_dir",
        type=str,
        default="/source/junhyuk/motion-style/style-salad/dataset/100style/new_joint_vecs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./renders_one_per_style",
    )
    parser.add_argument(
        "--select_mode",
        type=str,
        choices=["first", "random"],
        default="first",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--skip_m_prefix",
        action="store_true",
        help="Skip entries like M030001 and only use 030001",
    )

    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    grouped = parse_name_dict(
        args.txt_path,
        skip_m_prefix=args.skip_m_prefix,
    )

    selected = choose_one_per_style(
        grouped,
        mode=args.select_mode,
        seed=args.seed,
    )

    print(f"Found {len(grouped)} styles")
    print(f"Selected {len(selected)} motions")

    for i, item in enumerate(selected):
        motion_id = item["motion_id"]
        style_idx = item["style_idx"]
        style_name = item["style_name"]

        try:
            motion = load_motion_npy(args.motion_dir, motion_id)
        except Exception as e:
            print(f"[{i+1}/{len(selected)}] SKIP {motion_id} ({style_name}): {e}")
            continue

        filename = f"{style_idx:03d}_{slug(style_name)}_{motion_id}.mp4"
        save_path = os.path.join(args.output_dir, filename)

        print(f"[{i+1}/{len(selected)}] Rendering style={style_idx} name={style_name} motion={motion_id}")
        try:
            render_motion(
                motion=motion,
                save_path=save_path,
                fps=args.fps,
            )
        except Exception as e:
            print(f"    FAILED: {e}")

    print("Done.")


if __name__ == "__main__":
    main()