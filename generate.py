# DISCLAIMER:
# This code was produced with heavy LLM assistance.
# It’s functional but not guaranteed to be clean or optimal.
# In fact, even this disclaimer was LLM-generated.

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data.dataset import Dataset100Style, DatasetHumanML3D  # HumanML3D imported just in case
from model.t2sm import Text2StylizedMotion
from utils.motion import recover_from_ric  # not used inside here, but useful later if you want


# ================================================================
# Small utilities
# ================================================================

def set_seed(seed=42):
    """
    Fix random seeds for Python, NumPy, and PyTorch
    so that experiments are (mostly) reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config():
    """
    Read YAML config from --config,
    compute run_name from the path under configs/,
    and update:
        config["run_name"]
        config["result_dir"]
        config["checkpoint_dir"]

    Also parse:
        --style_idx  : which style to sample from (default 0)
        --out        : where to save the .npy file
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file (YAML)'
    )
    parser.add_argument(
        '--style_idx',
        type=int,
        default=0,
        help='Target style index to generate from (default: 0)'
    )
    parser.add_argument(
        '--out',
        type=str,
        default=None,
        help='Output .npy file path (default: <result_dir>/generated/style_<idx>.npy)'
    )

    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    with cfg_path.open('r') as f:
        config = yaml.safe_load(f)

    # Example:
    #   configs/loss/0.yaml -> run_name = "loss/0"
    parts = cfg_path.parts
    if "configs" in parts:
        i = parts.index("configs")
        sub = Path(*parts[i + 1:]).with_suffix("")  # e.g. loss/0
        run_name = str(sub).replace("\\", "/")
    else:
        run_name = cfg_path.stem

    config["run_name"] = run_name
    config["result_dir"] = os.path.join(config["result_dir"], run_name)
    config["checkpoint_dir"] = os.path.join(config["checkpoint_dir"], run_name)

    # CLI overrides we want to carry
    config["_cli_style_idx"] = args.style_idx
    config["_cli_out_path"] = args.out

    return config


def load_model(config, device):
    """
    Instantiate Text2StylizedMotion from config['model'],
    load its weights from `checkpoint_dir/best.ckpt`,
    and put it in eval mode.
    """
    model_cfg = config['model']
    model = Text2StylizedMotion(model_cfg).to(device)

    ckpt_path = os.path.join(config["checkpoint_dir"], "best.ckpt")
    print(f"[INFO] Loading checkpoint from: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)

    # strict=False to allow slight mismatches in keys
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config()
    set_seed(config["random_seed"])
    model = load_model(config, device)

    # Dataset
    style_cfg = config['dataset_style']

    # style_json maps style_name -> list of motion IDs
    with open(style_cfg["style_json"], "r", encoding="utf-8") as f:
        styles_to_ids = json.load(f)
    all_styles = sorted(styles_to_ids.keys())

    ds_style = Dataset100Style(style_cfg, styles=all_styles, train=False)

    target_style_idx = config.get("_cli_style_idx", 0)
    if target_style_idx not in ds_style.style_idx_to_style:
        raise ValueError(f"target_style_idx={target_style_idx} not in dataset.")

    style_name = ds_style.style_idx_to_style[target_style_idx]
    print(f"[INFO] Generating from style_idx={target_style_idx} ({style_name})")

    # Filter dataset indices for the chosen style
    indices = [
        i for i, it in enumerate(ds_style.items)
        if it["style_idx"] == target_style_idx
    ]
    if len(indices) == 0:
        raise RuntimeError(
            f"No items found for style_idx={target_style_idx} ({style_name})."
        )

    # Create loader over that subset.
    # We'll just grab the *first batch* and stop.
    B = config['sampler']['batch_size'] if 'sampler' in config else 16
    loader = DataLoader(
        torch.utils.data.Subset(ds_style, indices),
        batch_size=B,
        shuffle=False,
        num_workers=0
    )

    batch = next(iter(loader))
    cap1, win1, len1, sty1 = batch

    motions = win1.to(device)   # (B, T, D), normalized input motion
    captions = list(cap1)       # list[str]
    lengths = len1              # (B,)

    print(f"[INFO] Batch size: {motions.shape[0]}, T: {motions.shape[1]}, D: {motions.shape[2]}")

    # Optional: the swap trick you used in your visualize script.
    # If you don't want this behavior, set motions_swapped = motions.
    idx = torch.arange(motions.shape[0], device=motions.device)
    motions_swapped = motions[idx ^ 1]

    with torch.no_grad():
        stylized, captions_out = model.generate(
            motions_swapped,
            captions,
            lengths,
            lengths
        )

    # Make sure everything is on CPU for saving
    stylized_np = stylized.detach().cpu().numpy()  # (B, T, D)
    reference_np = motions.detach().cpu().numpy()  # (B, T, D)
    lengths_np = lengths.detach().cpu().numpy()    # (B,)
    style_idx_np = sty1.detach().cpu().numpy()     # (B,)

    # Normalize captions_out to list[str]
    if isinstance(captions_out, (list, tuple)):
        caps_out = [str(c) for c in captions_out]
    else:
        caps_out = [str(captions_out[i]) for i in range(stylized_np.shape[0])]

    # ------------------------------------------------------------
    # Decide output path
    # ------------------------------------------------------------
    out_path = config.get("_cli_out_path", None)
    if out_path is None:
        # Default: <result_dir>/generated/style_<idx>.npy
        out_dir = os.path.join(config["result_dir"], "generated")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"style_{target_style_idx}.npy")

    print(f"[INFO] Saving model output to: {out_path}")

    # ------------------------------------------------------------
    # Save everything into a single .npy file as a dict
    # ------------------------------------------------------------
    # This way you can later:
    #   data = np.load(out_path, allow_pickle=True).item()
    #   stylized = data["stylized"]
    #   reference = data["reference"]
    #   ...
    #
    # This keeps it simple while still giving you access to all fields.
    data_to_save = {
        "stylized": stylized_np,          # model output in normalized feature space
        "reference": reference_np,        # original input window in normalized feature space
        "lengths": lengths_np,            # valid lengths for each sequence
        "captions_in": np.array(captions, dtype=object),
        "captions_out": np.array(caps_out, dtype=object),
        "style_idx": style_idx_np,        # per-sample style index (should all be same here)
        "style_name": style_name,
    }

    # Save as a single .npy (pickled dict)
    np.save(out_path, data_to_save, allow_pickle=True)

    print("[DONE] Saved batch to .npy.")


if __name__ == "__main__":
    main()
