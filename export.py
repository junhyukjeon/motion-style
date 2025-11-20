import argparse
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from data.dataset import Dataset100Style, DatasetHumanML3D
from model.t2sm import Text2StylizedMotion
from utils.motion import recover_from_ric
from utils.skel import Skeleton


# -----------------------------
# Utility functions (seed, directories, config, model)
# -----------------------------

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


def reset_dir(path):
    """
    Remove a directory if it exists, then recreate it.
    Used here so that the output folder always starts empty.
    WARNING: this deletes anything already in `path`.
    """
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def slug(s: str, maxlen: int = 60) -> str:
    """
    Turn an arbitrary string (e.g., caption) into a filesystem-safe string:
      - keep alphanumerics and some punctuation,
      - replace others with underscore,
      - collapse spaces to underscore,
      - truncate to maxlen.
    Used for filenames so OS doesn’t freak out.
    """
    s = ''.join(c if (c.isalnum() or c in " _-.,()[]{}") else '_' for c in s.strip())
    s = "_".join(s.split())  # spaces -> underscores
    return (s[:maxlen]).rstrip("_")


def load_config():
    """
    Read YAML config from --config,
    compute run_name from the path under configs/,
    and update:
        config["run_name"]
        config["result_dir"]
        config["checkpoint_dir"]

    Also parse an extra CLI argument --style_idx for choosing which
    style index to export (e.g. Angry, Happy, etc.).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file (YAML)')
    parser.add_argument('--style_idx', type=int, default=0,
                        help='Target style index to export (default: 0)')
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    with cfg_path.open('r') as f:
        config = yaml.safe_load(f)

    # run_name = path inside "configs/" without .yaml suffix
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

    # also pass through style_idx
    config["_cli_style_idx"] = args.style_idx
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
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def main():
    # ------------------------------------------------------------
    # Setup: device, config, seed, model
    # ------------------------------------------------------------

    # Use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load YAML config and CLI overrides
    config = load_config()

    # Fix randomness
    set_seed(config["random_seed"])

    # Load model
    model = load_model(config, device)

    # ------------------------------------------------------------
    # Dataset: 100STYLE
    # ------------------------------------------------------------

    # Pull the style dataset config block (paths, JSON, etc.)
    style_cfg = config['dataset_style']

    # style_cfg["style_json"] should map style name -> list of motion IDs.
    # e.g., {"Angry_angry": ["030001", "030002", ...], ...}
    with open(style_cfg["style_json"], "r", encoding="utf-8") as f:
        styles_to_ids = json.load(f)

    import pdb; pdb.set_trace()

    # Sorted list of style names (so index order is consistent)
    all_styles = sorted(styles_to_ids.keys())

    # Create Dataset100Style in eval mode (train=False).
    # In your implementation, this likely does a center-crop or fixed sampling.
    ds_style = Dataset100Style(style_cfg, styles=all_styles, train=False)

    # Which style index do we want to export?
    # e.g., 0 might be "Neutral_neutral", 1 "Angry_angry", etc.
    target_style_idx = config.get("_cli_style_idx", 0)

    if target_style_idx not in ds_style.style_idx_to_style:
        raise ValueError(f"target_style_idx={target_style_idx} not in dataset.")

    # Human-readable style name (e.g. "Angry_angry")
    style_name = ds_style.style_idx_to_style[target_style_idx]
    print(f"[INFO] Exporting rotations for style_idx={target_style_idx} ({style_name})")

    # Filter dataset indices to only items that belong to this style index
    indices = [i for i, it in enumerate(ds_style.items) if it["style_idx"] == target_style_idx]

    if len(indices) == 0:
        raise RuntimeError(
            f"No items found for style_idx={target_style_idx} ({style_name})."
        )

    # DataLoader over the subset of that style only.
    # Each batch is: (captions, window (B,T,D), lengths, style_idx)
    B = config['sampler']['batch_size'] if 'sampler' in config else 16
    loader = DataLoader(Subset(ds_style, indices),
                        batch_size=B,
                        shuffle=False,
                        num_workers=0)

    # ------------------------------------------------------------
    # Output directory: result_dir/rotations/(style_tag)
    # ------------------------------------------------------------

    # If your model config has a style_weight, we fold it into the folder name
    # e.g., style_weight=1.5 -> "w1p5"
    style_weight = config["model"].get("style_weight", None)
    if style_weight is not None:
        style_tag = f"w{style_weight}".replace(".", "p")
        output_dir = os.path.join(config["result_dir"], "rotations", style_tag)
    else:
        output_dir = os.path.join(config["result_dir"], "rotations")

    # Clean + create the output directory
    reset_dir(output_dir)
    print(f"[INFO] Saving .npz rotations to: {output_dir}")

    # ------------------------------------------------------------
    # Denormalization stats (mean/std) for motion features
    # ------------------------------------------------------------

    # Those .npy files should store the mean and std used to normalize motions.
    # We reload them as tensors for easy broadcasting on GPU.

    # --- Normalization stats ---
    mean = torch.tensor(np.load(style_cfg["mean_path"]),
                        dtype=torch.float32, device=device)
    std = torch.tensor(np.load(style_cfg["std_path"]),
                       dtype=torch.float32, device=device)

    # --- Skeleton setup ---
    #
    # Expect something like:
    # config["skeleton"]["parents"] = [-1, 0, 1, ...]  (len = 22)
    # config["skeleton"]["lengths"] = [0.0, 1.0, 1.0, ...]  (len = 22)
    #
    # Fill these appropriately in your YAML.
    sk_cfg = config["skeleton"]
    parents = sk_cfg["parents"]
    lengths = sk_cfg["lengths"]

    # Instantiate Skeleton with all the policy knobs
    skeleton = Skeleton(
        parents=parents,
        lengths=lengths,
        face_indices=sk_cfg.get("face_indices", (1, 2, 11, 12)),
        up_axis=tuple(sk_cfg.get("up_axis", (0, 1, 0))),
        target_forward=tuple(sk_cfg.get("target_forward", (0, 0, 1))),
        smooth_forward_sigma=float(sk_cfg.get("smooth_forward_sigma", 0.0)),
        first_frame_root_identity=bool(sk_cfg.get("first_frame_root_identity", True)),
        zero_root_length=bool(sk_cfg.get("zero_root_length", True)),
    )

    # Global index counting how many clips we've exported so far
    global_clip_idx = 0

    # FPS to store in the .npz (probably 20 for T2M)
    fps = style_cfg.get("fps", 20)

    # ------------------------------------------------------------
    # Main loop: iterate over batches, run model, run IK, export .npz
    # ------------------------------------------------------------

    for batch in loader:
        cap1, win1, len1, sty1 = batch  # captions, (B,T,D), lengths, style_idx

        # Move motion window to device; still normalized
        motions = win1.to(device)      # normalized

        # Captions as Python list of str (easier to log + slugify)
        captions = list(cap1)          # list[str]

        # Sequence lengths as a torch tensor (stays on CPU; we only use it
        # to slice T later).
        lengths = len1                 # (B,)

        # Optional "swap every other index" trick
        # This came from your visualize script, presumably for mismatched pairings.
        # If you don't want that behavior, you can replace `motions_swapped` with `motions`.
        idx = torch.arange(motions.shape[0], device=motions.device)
        motions_swapped = motions[idx ^ 1]

        # -------------------------------
        # Stylized motion generation
        # -------------------------------
        # model.generate is assumed to have signature like:
        #   generate(motion_in, captions, len_in, len_out)
        # Returning (stylized_motion, returned_captions)
        # Generate stylized motion
        with torch.no_grad():
            stylized, captions_out = model.generate(
                motions_swapped,
                captions,
                lengths,
                lengths,
            )

        # -------------------------------
        # Denormalize stylized and reference motions
        # -------------------------------
        # shapewise: stylized/reference -> (B, T, D)
        stylized = stylized * std + mean
        reference = motions * std + mean

        # -------------------------------
        # Convert from RIC to joints
        # -------------------------------
        # recover_from_ric should output positions in R^3 for each joint.
        # shape: (B, T, 22, 3) if we pass 22 joints.
        joints_stylized = recover_from_ric(stylized, 22).detach().cpu().numpy()
        joints_reference = recover_from_ric(reference, 22).detach().cpu().numpy()

        # Normalize captions_out to Python list of strings
        if isinstance(captions_out, (list, tuple)):
            caps_out = [str(c) for c in captions_out]
        else:
            # if it's some other structure (like a tensor), we convert per item
            caps_out = [str(captions_out[i]) for i in range(stylized.size(0))]

        # B_cur is the number of sequences in this batch
        B_cur = stylized.size(0)

        # -------------------------------
        # Per-sequence export loop
        # -------------------------------
        for i in range(B_cur):
            # For each sequence i in the batch, we only keep the first length[i]
            # frames (since the rest are padding).
            T_i = int(lengths[i])  # actual time steps for this sequence

            # Pose sequence from stylized output: (T_i, J, 3)
            joints_seq = joints_stylized[i, :T_i]

            # Root trajectory (T_i, 3) – usually joint 0 is the root
            root_pos = joints_seq[:, 0, :]

            # ----------------------------------------------------
            # Run inverse kinematics to get local joint rotations
            # ----------------------------------------------------
            # skeleton.inverse_kinematics expects:
            #   joints: (T, J, 3)
            #   root_quat: (T,4) or None (if we let it estimate root from motion)
            #
            # It returns:
            #   quat_local: (T, J, 4) local rotations for each joint
            quat_local = skeleton.inverse_kinematics(
                joints=joints_seq,
                root_quat=None  # let Skeleton estimate forward direction
            )

            # ----------------------------------------------------
            # Optional: also export reference rotations
            # ----------------------------------------------------
            joints_ref_seq = joints_reference[i, :T_i]
            root_pos_ref = joints_ref_seq[:, 0, :]
            quat_local_ref = skeleton.inverse_kinematics(
                joints=joints_ref_seq,
                root_quat=None
            )

            # ----------------------------------------------------
            # Build safe filename (caption-based)
            # ----------------------------------------------------
            cap = caps_out[i]
            if len(cap) == 0:
                # fallback if caption is empty
                base_name = f"clip_{global_clip_idx}"
            else:
                base_name = cap

            # slug() ensures there are no forbidden characters for a filename
            cap_slug = slug(base_name)

            # 4-digit index suffix (e.g. 0000, 0001, ...)
            filename = f"{cap_slug}_{global_clip_idx:04d}.npz"

            # Final save path
            save_path = os.path.join(output_dir, filename)

            # ----------------------------------------------------
            # Save everything we might need in Blender as a .npz
            # ----------------------------------------------------
            # We'll store:
            #   rotations_stylized:   (T, J, 4) quaternions
            #   root_pos_stylized:    (T, 3)
            #   rotations_reference:  (T, J, 4)
            #   root_pos_reference:   (T, 3)
            #   parents:              (J,)
            #   style_idx:            scalar
            #   style_name:           string
            #   caption:              string
            #   fps:                  scalar
            np.savez_compressed(
                save_path,
                rotations_stylized=quat_local.astype(np.float32),
                root_pos_stylized=root_pos.astype(np.float32),
                rotations_reference=quat_local_ref.astype(np.float32),
                root_pos_reference=root_pos_ref.astype(np.float32),
                parents=np.asarray(skeleton.parents, dtype=np.int32),
                style_idx=int(target_style_idx),
                style_name=style_name,
                caption=caps_out[i],
                fps=int(fps),
            )

            print(f"[SAVED] {save_path}")
            global_clip_idx += 1  # increment global counter

    print("[DONE] Rotation export complete.")


# Standard Python entry point:
# this ensures that main() is only run if you call
#   python export_rotations.py ...
# and not when you import this file as a module.
if __name__ == "__main__":
    main()