# run_tsne_unseen.py
# Usage:
#   python run_tsne_unseen.py --config path/to/config.yaml
#   python run_tsne_unseen.py --config path/to/config.yaml --ckpt latest.ckpt --max_samples 3000

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# --- Your project imports ---
from data.dataset import Dataset100Style
from model.t2sm import Text2StylizedMotion

# If your plot_tsne lives in utils/plot.py, import it instead of redefining.
# from utils.plot import plot_tsne
# Otherwise, import your local function definition.
from utils.plot import plot_tsne, plot_tsne_2d  # <-- adjust if needed


def load_config_from_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file (YAML)")
    parser.add_argument("--ckpt", type=str, default="latest.ckpt", help="Checkpoint filename or full path")
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--title", type=str, default="unseen", help="Subfolder name under result_dir")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    with cfg_path.open("r") as f:
        config = yaml.safe_load(f)

    # mirror your run_name logic
    parts = cfg_path.parts
    if "configs" in parts:
        i = parts.index("configs")
        sub = Path(*parts[i + 1 :]).with_suffix("")
        run_name = str(sub).replace("\\", "/")
    else:
        run_name = cfg_path.stem

    config["run_name"] = run_name
    config["result_dir"] = os.path.join(config["result_dir"], run_name)
    config["checkpoint_dir"] = os.path.join(config["checkpoint_dir"], run_name)

    return config, args


def read_ids_if_exists(p: str):
    if p is None:
        return None
    p = os.path.expanduser(p)
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def load_trainable_only_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    """
    Your training code saves only trainable params:
      sd_trainable = {k: v for k, v in model.state_dict().items() if k in trainable}
    so we must load with strict=False.
    """
    state = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)

    # This is expected: non-trainable keys will be missing.
    print(f"[ckpt] Loaded: {ckpt_path}")
    if unexpected:
        print(f"[ckpt] Unexpected keys: {len(unexpected)}")
    print(f"[ckpt] Missing keys (expected w/ trainable-only ckpt): {len(missing)}")

    model.to(device)
    model.eval()
    return model


def build_label_to_name_dict(style_ds, styles_in_split):
    """
    plot_tsne expects label_to_name_dict: {label_int: "StyleName"}
    We try a few common dataset attributes; otherwise fall back.
    """
    # Common patterns:
    #   style_ds.style2idx : dict[str,int]
    #   style_ds.styles / style_ds.all_styles
    #   style_ds.idx2style : dict[int,str]
    if hasattr(style_ds, "idx2style"):
        idx2style = getattr(style_ds, "idx2style")
        if isinstance(idx2style, dict) and len(idx2style) > 0:
            return dict(idx2style)

    if hasattr(style_ds, "style2idx"):
        style2idx = getattr(style_ds, "style2idx")
        if isinstance(style2idx, dict) and len(style2idx) > 0:
            return {idx: name for name, idx in style2idx.items()}

    # Fallback: assume the label space corresponds to sorting of styles
    # (matches your earlier all_styles = sorted(styles_to_ids.keys()))
    styles_sorted = sorted(list(styles_in_split))
    return {i: s for i, s in enumerate(styles_sorted)}


def main():
    config, args = load_config_from_cli()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load all style names ---
    style_cfg = config["dataset_style"]
    with open(style_cfg["style_json"], "r") as f:
        styles_to_ids = json.load(f)
    all_styles = sorted(styles_to_ids.keys())

    # --- Reproduce the same split used in training ---
    # NOTE: your training code uses:
    #   train_styles, _ = train_test_split(all_styles, test_size=valid_size, random_state=random_seed)
    # That means the "unseen" set is whatever the second output would have been.
    train_styles, valid_styles = train_test_split(
        all_styles,
        test_size=config["valid_size"],
        random_state=config["random_seed"],
    )
    print(f"# Train styles: {len(train_styles)}")
    print(f"# Unseen/Valid styles: {len(valid_styles)}")

    # --- Optional: restrict to an ID list for unseen styles (if you have one) ---
    # If you *only* have a train id file, just omit this.
    ids_valid = None
    # Try config hooks first, else you can hardcode like you did for train.
    # Example optional config key: dataset_style: { valid_ids_txt: ".../valid_100STYLE_Full.txt" }
    def read_ids(p):
        with open(p, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    ids_train = read_ids("./dataset/100style/train_100STYLE_Full.txt")

    # import pdb; pdb.set_trace()

    # --- Dataset/Loader for unseen styles ---
    # You probably want train=False for evaluation sampling behavior (no augmentation / no random cropping, etc.)
    # import pdb; pdb.set_trace()
    # valid_styles = ['Sweep', 'OnHeels', 'Rushed', 'LimpRight', 'LimpLeft', 'LeanBack', 'DragRightLeg', 'Strutting', 'BentKnees', 'Aeroplane', 'CrowdAvoidance', 'GracefulArms', 'Skip', 'Heavyset', 'TwoFootJump', 'ArmsBehindBack', 'SpinClock', 'Star', 'BouncyLeft', 'HandsBetweenLegs', 'OnPhoneRight', 'Tiptoe', 'FairySteps', 'LeftHop', 'Rocket']
    style_unseen = Dataset100Style(style_cfg, styles=valid_styles, train=False, use_ids=ids_train)
    # IMPORTANT: your plot_tsne iterates "for batch in loader:" and calls model.style(batch),
    # so we need loader batches to be the raw dataset tuple (cap, win, len, sty), not a custom sampler dict.
    sampler_cfg = config.get("sampler", {})
    batch_size = sampler_cfg.get("batch_size", 32)

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    loader_unseen = DataLoader(
        style_unseen,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
    )

    # --- Build model & load checkpoint ---
    model_cfg = config["model"]
    model = Text2StylizedMotion(model_cfg)

    # Resolve ckpt path: allow either full path or just filename under checkpoint_dir
    ckpt_path = args.ckpt
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(config["checkpoint_dir"], ckpt_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found.\n"
            f"Tried: {args.ckpt}\n"
            f"and:   {ckpt_path}\n"
            f"checkpoint_dir: {config['checkpoint_dir']}"
        )

    model = load_trainable_only_checkpoint(model, ckpt_path, device)

    # --- Label mapping for legend/coloring ---
    label_to_name_dict = build_label_to_name_dict(style_unseen, valid_styles)

    # --- Run t-SNE and save to result_dir/<title>/ ---
    # plot_tsne signature:
    #   plot_tsne(model, loader, device, epoch=None, title="valid", result_dir="", label_to_name_dict=None, max_samples=3000, writer=None)
    # plot_tsne(
    #     model=model,
    #     loader=loader_unseen,
    #     device=device,
    #     epoch=0,  # set to 0 since this is a posthoc eval; you can parse from ckpt name if you want
    #     title=args.title,
    #     result_dir=config["result_dir"],
    #     label_to_name_dict=label_to_name_dict,
    #     max_samples=args.max_samples,
    #     writer=None,
    # )

    plot_tsne_2d(
        model=model,
        loader=loader_unseen,
        device=device,
        epoch=0,
        title=args.title + "_2d",
        result_dir=config["result_dir"],
        label_to_name_dict=label_to_name_dict,
        max_samples=args.max_samples,
        writer=None,
    )

    print(f"[done] Saved under: {os.path.join(config['result_dir'], args.title)}")


if __name__ == "__main__":
    main()
