import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import yaml

from data.dataset import Dataset100STYLE
from mld.data.humanml.utils.plot_script import plot_3d_motion
from mld.utils.joints import humanml3d_joints, mmm2smplh_correspondence
from model.t2sm import Text2StylizedMotion
from salad.utils.paramUtil import t2m_kinematic_chain
from utils.motion import recover_from_ric, recover_root_rot_pos


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def slug(s: str, maxlen: int = 80) -> str:
    s = "".join(c if (c.isalnum() or c in " _-.,()[]{}") else "_" for c in s.strip())
    s = "_".join(s.split())
    return s[:maxlen].rstrip("_") or "sample"


def load_config(config_path: str) -> Dict:
    cfg_path = Path(config_path).resolve()
    with cfg_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    parts = cfg_path.parts
    if "configs" in parts:
        idx = parts.index("configs")
        run_name = str(Path(*parts[idx + 1 :]).with_suffix("")).replace("\\", "/")
    else:
        run_name = cfg_path.stem

    config["run_name"] = run_name
    config["result_dir"] = os.path.join(config["result_dir"], os.path.basename(run_name))
    config["checkpoint_dir"] = os.path.join(config["checkpoint_dir"], run_name)
    return config


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_style_stats(style_cfg: Dict, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(np.load(style_cfg["mean_path"]), dtype=torch.float32, device=device)
    std = torch.tensor(np.load(style_cfg["std_path"]), dtype=torch.float32, device=device)
    return mean, std


def load_model(config: Dict, device: torch.device) -> Text2StylizedMotion:
    model = Text2StylizedMotion(config["model"]).to(device)
    ckpt_path = os.path.join(config["checkpoint_dir"], "latest.ckpt")
    model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
    style_cfg = config["dataset_style"]
    mean, std = load_style_stats(style_cfg, device)
    model.set_normalization_stats(mean, std)
    model.eval()
    return model


def build_style_dataset(config: Dict) -> Dataset100STYLE:
    style_cfg = dict(config["dataset_style"])
    style_cfg.setdefault("excluded_styles", [])
    return Dataset100STYLE(style_cfg)


def find_motion_meta(dataset: Dataset100STYLE, motion_id: str) -> Dict:
    motion_id = str(motion_id)
    for item in dataset.items:
        if item["motion_id"] == motion_id:
            return item
    raise ValueError(f"motion_id='{motion_id}' not found in Dataset100STYLE")


def get_full_motion(dataset: Dataset100STYLE, motion_id: str, device: torch.device) -> Tuple[torch.Tensor, int]:
    motion_id = str(motion_id)
    meta = find_motion_meta(dataset, motion_id)
    motion = dataset.motion_cache[motion_id].to(device)
    return motion, int(meta["length"])


def denormalize_motion(motion: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return motion * std.view(1, 1, -1) + mean.view(1, 1, -1)


def motion_to_joints(motion_real: torch.Tensor) -> torch.Tensor:
    return recover_from_ric(motion_real, 22)


def motion_to_root_xz(motion_real: torch.Tensor) -> torch.Tensor:
    _, root_pos = recover_root_rot_pos(motion_real)
    return root_pos[..., [0, 2]]


def repeat_batch(x: torch.Tensor, batch_size: int) -> torch.Tensor:
    return x.unsqueeze(0).repeat(batch_size, *([1] * x.ndim))


def save_motion_video(path: str, joints: np.ndarray, title: str = "", fps: int = 20):
    plot_3d_motion(
        save_path=path,
        kinematic_tree=t2m_kinematic_chain,
        joints=joints,
        title=title,
        dataset="humanml",
        fps=fps,
    )


def plot_root_trajectory(path: str, target_xz: np.ndarray, generated_xz: np.ndarray, title: str = ""):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(target_xz[:, 0], target_xz[:, 1], label="target", linewidth=2.0)
    ax.plot(generated_xz[:, 0], generated_xz[:, 1], label="generated", linewidth=2.0)
    ax.scatter(target_xz[0, 0], target_xz[0, 1], label="start", s=30)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("root x")
    ax.set_ylabel("root z")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_json(path: str, payload: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def build_joint_name_map() -> Dict[str, int]:
    name_to_idx: Dict[str, int] = {}
    for idx, joint in enumerate(humanml3d_joints):
        name_to_idx[joint.lower()] = idx
    for code, readable in mmm2smplh_correspondence.items():
        if code in humanml3d_joints:
            idx = humanml3d_joints.index(code)
            name_to_idx[readable.lower()] = idx
    name_to_idx["pelvis"] = humanml3d_joints.index("root")
    # Common aliases that are easier to guess than the dataset's canonical names.
    if "left_foot" in name_to_idx:
        name_to_idx["left_ankle"] = name_to_idx["left_foot"]
    if "right_foot" in name_to_idx:
        name_to_idx["right_ankle"] = name_to_idx["right_foot"]
    return name_to_idx


def resolve_joint_names(joint_names: Iterable[str]) -> Tuple[List[int], List[str]]:
    lookup = build_joint_name_map()
    indices: List[int] = []
    canonical: List[str] = []
    for name in joint_names:
        key = name.strip().lower()
        if key not in lookup:
            valid = ", ".join(sorted(lookup.keys()))
            raise ValueError(f"Unknown joint name '{name}'. Available names include: {valid}")
        indices.append(lookup[key])
        canonical.append(key)
    return indices, canonical
