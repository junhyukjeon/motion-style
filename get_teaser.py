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

from data.dataset import Dataset100Style
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

import re
def verb_after_person(caption: str, fallback: str = "unknown") -> str:
    if not caption:
        return fallback

    s = caption.strip().lower()

    # remove trailing punctuation
    s = re.sub(r"[^\w\s]", "", s)

    tokens = s.split()
    if not tokens:
        return fallback

    # walk backwards to find a valid word
    for tok in reversed(tokens):
        if tok.isalpha():
            return slug(tok, maxlen=24)

    return fallback

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file (YAML)')
    parser.add_argument('--ref_motion_id', type=str, default="030303",
                        help='Reference motion ID (default: 030303)')
    parser.add_argument('--caption', type=str, default=None,
                        help='Caption text for generation (default: dataset caption)')
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
        sub = Path(*parts[i+1:]).with_suffix("")
        run_name = str(sub).replace("\\", "/")
    else:
        run_name = cfg_path.stem

    config["run_name"] = run_name
    config["result_dir"]     = os.path.join(config["result_dir"], os.path.basename(run_name))
    config["checkpoint_dir"] = os.path.join(config["checkpoint_dir"], run_name)

    # attach CLI overrides
    config["_cli"] = {
        "ref_motion_id": args.ref_motion_id,
        "caption": args.caption,
    }

    return config

def load_model(config, device):
    model = Text2StylizedMotion(config["model"]).to(device)
    model.load_state_dict(torch.load(os.path.join(config["checkpoint_dir"], "latest.ckpt"), map_location=device), strict=False)
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

def find_index_by_motion_id(ds_style, motion_id: str) -> int:
    motion_id = str(motion_id)
    for i, it in enumerate(ds_style.items):
        if it["motion_id"] == motion_id:
            return i
    raise ValueError(f"motion_id='{motion_id}' not found in dataset.")

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

    B = 16

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

    REF_MOTION_ID = config["_cli"]["ref_motion_id"]
    style_name = REF_MOTION_ID

    style_tag = "_".join(tag_parts) if len(tag_parts) > 0 else "default"

    # --- Mean & Std (tensors on device) --- #
    mean = torch.tensor(np.load(style_cfg["mean_path"]), dtype=torch.float32, device=device)
    std  = torch.tensor(np.load(style_cfg["std_path"]),  dtype=torch.float32, device=device)

    # --- Pick reference motion by ID ---
    ref_i = find_index_by_motion_id(ds_style, REF_MOTION_ID)

    cap, win, L, sty = ds_style[ref_i]
    win = win[:L]
    motions  = win.unsqueeze(0).to(device)          # (1, T, D) normalized
    len1     = torch.tensor([L], dtype=torch.long)  # (1,)
    cli_caption = config["_cli"]["caption"]

    if cli_caption is not None:
        captions = [cli_caption]
    else:
        captions = [cap]

    output_dir_base = os.path.join(config["result_dir"], style_name, style_tag)

    # Use the *single* caption that defines this run (before repeating to B)
    caption_for_dir = captions[0] if isinstance(captions, list) and len(captions) > 0 else "unknown"
    caption_dir = slug(caption_for_dir, maxlen=80)

    output_dir = os.path.join(output_dir_base, caption_dir)
    reset_dir(output_dir)

    # Repeat reference to B samples
    motions  = motions.repeat(B, 1, 1)              # (B, T, D)
    len1     = len1.repeat(B)                       # (B,)
    captions = captions * B                         # (B,)

    len_out = torch.tensor(140, dtype=torch.long).repeat(B)

    # --- Generate stylized ---
    stylized, captions_out = model.generate(motions, captions, len_out, len1)

    # --- Denormalize stylized & reference ---
    stylized  = stylized * std + mean
    reference = motions * std + mean

    # --- Recover joints ---
    joints_stylized  = recover_from_ric(stylized, 22).detach().cpu().numpy()
    joints_reference = recover_from_ric(reference, 22).detach().cpu().numpy()

    # --- Kinematic tree (define once) ---
    kinematic_tree = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10],
                      [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21],
                      [9, 13, 16, 18, 20]]

    # --- Render reference ONCE (since it's identical for all batch entries) ---
    L_ref = int(len_out[0].item())
    xyz_ref = joints_reference[0][:L_ref].astype(np.float32)  # (L_ref, 22, 3)

    ref_path = os.path.join(output_dir, "sample00_rep00.mp4")
    plot_3d_motion(
        ref_path,
        kinematic_tree,
        xyz_ref,
        title="",
        dataset="humanml",
        fps=20,
    )

    # --- Prepare saving arrays ---
    lengths = len_out.cpu().numpy().astype(int)

    num_samples = B                  # total videos = B (sample00 is ref, sample01.. are generated)
    num_repetitions = 1
    N = num_samples * num_repetitions

    T_max = int(lengths.max())
    J = 22

    all_motions = np.zeros((N, J, 3, T_max), dtype=np.float32)
    all_lengths = lengths.astype(np.int32)

    # sample00 text can be anything; keep empty or mark as reference
    all_text = ["[REF]"] + [str(captions_out[i]) for i in range(1, B)]

    # --- Fill sample00 (reference) into results.npy ---
    all_motions[0, :, :, :L_ref] = np.transpose(xyz_ref, (1, 2, 0))
    all_lengths[0] = L_ref

    # --- Render generated from sample01..sample(B-1) and fill results.npy ---
    for sample_i in range(1, B):
        L = int(all_lengths[sample_i])
        xyz_gen = joints_stylized[sample_i][:L].astype(np.float32)  # (L, 22, 3)

        # fill as (J,3,T)
        all_motions[sample_i, :, :, :L] = np.transpose(xyz_gen, (1, 2, 0))

        gen_file = f"sample{sample_i:02d}_rep{0:02d}.mp4"
        gen_path = os.path.join(output_dir, gen_file)
        plot_3d_motion(
            gen_path,
            kinematic_tree,
            xyz_gen,
            title="",
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
