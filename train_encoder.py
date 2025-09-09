# --- Imports ---
import argparse
import colorcet as cc
import contextlib
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import shutil
import time
import torch
import torch.nn.functional as F
import yaml
from collections import defaultdict
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from matplotlib.colors import to_rgb

from data.dataset import StyleDataset
from data.sampler import SAMPLER_REGISTRY
from model.networks import NETWORK_REGISTRY
from utils.train.early_stop import EarlyStopper
from utils.train.loss import LOSS_REGISTRY
from utils.train.loss_scaler import LossScaler
from utils.plot import plot_tsne


def compute_raw_losses(loss_fns, loss_cfg, model, out, labels):
    """
    Returns: dict[name] = (None, raw_loss_tensor)
    - Keeps 'style' and 'content' separate if your 'stylecon' fn returns two parts.
    - No weighting or normalization here.
    """
    losses = {}
    for name, fn in loss_fns.items():
        spec = loss_cfg[name]
        if name == "stylecon":
            style_loss, content_loss = fn(spec, model, out, labels)
            losses["style"]   = (None, style_loss)
            losses["content"] = (None, content_loss)
        else:
            val = fn(spec, model, out, labels)
            losses[name] = (None, val)
    return losses


def get_weight(loss_name, loss_cfg):
    if loss_name in loss_cfg and "weight" in loss_cfg[loss_name]:
        return float(loss_cfg[loss_name]["weight"])
    if loss_name in ("style", "content") and "stylecon" in loss_cfg:
        return float(loss_cfg["stylecon"].get("weight", 1.0))
    return float(loss_cfg.get("scaler", {}).get("default_weight", 1.0))


def calibrate(model, train_loader, device, loss_cfg, loss_fns, scaler, optimizer, K=500):
    """
    Runs K *training* steps (with updates) and simultaneously collects calibration stats
    from raw losses. After K steps, freezes denominators.
    """
    model.train()
    scaler.begin_calibration()

    pbar = tqdm(zip(range(K), train_loader), total=K, desc=f"[Burn-in Calib] K={K}")
    for _, (motions, labels, contents, _) in pbar:
        motions, labels = motions.to(device), labels.to(device)
        out = model.encode(motions)
        raw_losses = compute_raw_losses(loss_fns, loss_cfg, model, out, labels)
        scaler.accumulate_calibration(raw_losses)

        total_unscaled = None
        for name, (_, raw) in raw_losses.items():
            w = get_weight(name, loss_cfg)
            total_unscaled = raw*w if total_unscaled is None else total_unscaled + raw*w

        optimizer.zero_grad(set_to_none=True)
        total_unscaled.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
        optimizer.step()

    scaler.end_calibration(use="rms", tau=1e-3)
    print("📏 Burn-in calibration complete. Frozen denominators:", scaler.frozen_denom)


def train(model, loader, device, loss_cfg, loss_fns, scaler, optimizer, writer=None, epoch=None, clip_grad=5.0):
    model.train()
    losses_scaled_sum = defaultdict(float)
    losses_raw_sum    = defaultdict(float)

    pbar = tqdm(loader, desc=f"[Train] Epoch {epoch}")
    b_idx = -1
    for b_idx, (motions, labels, contents, _) in enumerate(pbar):
        motions, labels = motions.to(device), labels.to(device)

        # Forward
        out = model.encode(motions)

        # 1) raw losses (no weights here)
        raw_losses = compute_raw_losses(loss_fns, loss_cfg, model, out, labels)

        # 2) normalize + weight via LossScaler (updates EMAs internally)
        losses, total_loss = scaler.normalize_and_weight(raw_losses, train=True)

        # 3) safety
        if not torch.isfinite(total_loss):
            pbar.set_postfix(loss=float("nan"))
            optimizer.zero_grad(set_to_none=True)
            continue

        # 4) backward + step
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
        optimizer.step()

        # 5) tqdm
        unique_contents = set(contents)
        pbar.set_postfix(loss=total_loss.item(), contents=list(unique_contents))

        # 6) accumulate per-loss sums for epoch averages
        for name, (scaled, raw) in losses.items():
            losses_scaled_sum[name] += scaled.item()
            losses_raw_sum[name]    += raw.item()

        if MAX_TRAIN_BATCHES is not None and (b_idx + 1) >= MAX_TRAIN_BATCHES:
            break

    # epoch averages
    num_batches = max(b_idx + 1, 1)
    train_total_scaled = sum(losses_scaled_sum.values()) / max(num_batches, 1)
    train_total_raw = sum(losses_raw_sum.values()) / max(num_batches, 1)

    if writer is not None and epoch is not None:
        writer.add_scalar("Train/Raw/Total", train_total_raw, epoch)
        writer.add_scalar("Train/Scaled/Total", train_total_scaled, epoch)
        for name in losses_scaled_sum.keys():
            writer.add_scalar(f"Train/Raw/{name}",    losses_raw_sum[name]    / max(num_batches,1), epoch)
            writer.add_scalar(f"Train/Scaled/{name}", losses_scaled_sum[name] / max(num_batches,1), epoch)

    return train_total_scaled, train_total_raw


def validate(model, loader, device, loss_cfg, loss_fns, scaler, writer=None, epoch=None):
    model.eval()
    losses_scaled_sum = defaultdict(float)
    losses_raw_sum    = defaultdict(float)

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"[Valid] Epoch {epoch}")
        b_idx = -1
        for b_idx, (motions, labels, contents, _) in enumerate(pbar):
            motions, labels = motions.to(device), labels.to(device)
            out = model.encode(motions)

            raw_losses = compute_raw_losses(loss_fns, loss_cfg, model, out, labels) # Pure raw loss values. Check loss functions.
            losses, total_loss = scaler.normalize_and_weight(raw_losses, train=False) # Normalized and weighted losses, and sum.

            if not torch.isfinite(total_loss):
                pbar.set_postfix(loss=float("nan"))
                continue

            pbar.set_postfix(loss=total_loss.item())
            for name, (scaled, raw) in losses.items():
                losses_scaled_sum[name] += scaled.item()
                losses_raw_sum[name]    += raw.item()

            if MAX_TRAIN_BATCHES is not None and (b_idx + 1) >= MAX_TRAIN_BATCHES:
                break

    num_batches = max(b_idx + 1, 1)
    valid_total_scaled = sum(losses_scaled_sum.values()) / max(num_batches, 1)
    valid_total_raw = sum(losses_raw_sum.values()) / max(num_batches, 1)

    if writer is not None and epoch is not None:
        writer.add_scalar("Valid/Raw/Total", valid_total_raw, epoch)
        writer.add_scalar("Valid/Scaled/Total", valid_total_scaled, epoch)
        for name in losses_scaled_sum.keys():
            writer.add_scalar(f"Valid/Raw/{name}",    losses_raw_sum[name]    / max(num_batches,1), epoch)
            writer.add_scalar(f"Valid/Scaled/{name}", losses_scaled_sum[name] / max(num_batches,1), epoch)

    return valid_total_scaled, valid_total_raw


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


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config()

    # --- Result directories ---
    os.makedirs(config["result_dir"], exist_ok=True)
    os.makedirs(os.path.join(config["result_dir"], "valid"), exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join("./results/tensorboard", config["run_name"]))
    
    set_seed(config["random_seed"])

    # --- Dataset Stats ---
    MEAN = np.load(config["mean_path"])
    STD = np.load(config["std_path"])

    # --- Style Split ---
    with open(config["style_json"]) as f:
        full_label_to_ids = json.load(f)

    all_styles_sorted = sorted(full_label_to_ids.keys())
    global_style_to_label = {style: i for i, style in enumerate(all_styles_sorted)}
    global_label_to_style = {i: style for style, i in global_style_to_label.items()}

    non_neutral_styles = [s for s in all_styles_sorted if s != "Neutral"]
    train_styles, valid_styles = train_test_split(non_neutral_styles, test_size=config['valid_size'], random_state=config["random_seed"])
    train_styles.append("Neutral")
    valid_styles.append("Neutral")

    train_label_to_ids = {style: full_label_to_ids[style] for style in train_styles}
    valid_label_to_ids = {style: full_label_to_ids[style] for style in valid_styles}

    # --- Content Mapping ---
    with open(config["content_json"]) as f:
        content_to_ids = json.load(f)
    
    ids_to_content = {}
    for content_type, motion_ids in content_to_ids.items():
        for m_id in motion_ids:
            ids_to_content[m_id] = content_type

    # --- Dataset & Loader ---
    train_dataset = StyleDataset(config["motion_dir"], MEAN, STD, config["window_size"], train_label_to_ids, global_style_to_label, ids_to_content)
    valid_dataset = StyleDataset(config["motion_dir"], MEAN, STD, config["window_size"], valid_label_to_ids, global_style_to_label, ids_to_content)

    sampler_cfg = config['sampler']
    train_sampler = SAMPLER_REGISTRY[sampler_cfg['type']](sampler_cfg, train_dataset)
    valid_sampler = SAMPLER_REGISTRY[sampler_cfg['type']](sampler_cfg, valid_dataset)
    
    train_loader  = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_sampler=valid_sampler, num_workers=8)

    # --- Model, Optimizer, Loss ---
    model_cfg = config['model']
    model = NETWORK_REGISTRY[model_cfg['type']](model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    loss_cfg = config['loss'] 
    loss_fns = {name: LOSS_REGISTRY[name] for name in loss_cfg}

    # --- Early Stopper ---
    early_cfg = config['early']
    early = EarlyStopper(early_cfg)

    # --- Training ---
    ALPHA = 0.01
    NORM_EPS = 1e-8
    WARMUP_BATCHES = 100
    scaler = LossScaler(loss_cfg, alpha=ALPHA, eps=NORM_EPS, warmup=WARMUP_BATCHES)

    #####################
    MAX_TRAIN_BATCHES = 1000000

    best_val_loss = float('inf')
    model.encoder.vae.freeze()
    
    calibrate(model, train_loader, device, loss_cfg, loss_fns, scaler, optimizer, K=10000)

    for epoch in range(1, config['epochs'] + 1):
        # train_sampler.generate_batches()
        # valid_sampler.generate_batches()

        # --- Train ---
        train_scaled, train_raw = train(
            model=model,
            loader=train_loader,
            device=device,
            loss_cfg=loss_cfg,
            loss_fns=loss_fns,
            scaler=scaler,
            optimizer=optimizer,
            writer=writer,
            epoch=epoch,
            clip_grad=5.0,
        )

        # --- Validate ---
        valid_scaled, valid_task = validate(   # valid_task = raw-weighted total
            model=model,
            loader=valid_loader,
            device=device,
            loss_cfg=loss_cfg,
            loss_fns=loss_fns,
            scaler=scaler,
            writer=writer,
            epoch=epoch
        )

        print(
            f"Epoch {epoch} | "
            f"Train scaled: {train_scaled:.4f} | Train raw: {train_raw:.4f} | "
            f"Valid scaled: {valid_scaled:.4f} | Valid raw: {valid_task:.4f}"
        )
        model.train()  # ensure we’re back in train mode

        # --- t-SNE, checkpoints, best model ---
        plot_tsne(
            model, valid_loader, device, epoch, title="valid",
            result_dir=config["result_dir"],
            label_to_name_dict=valid_dataset.label_to_style,
            writer=writer
        )

        os.makedirs(config["checkpoint_dir"], exist_ok=True)
        torch.save(model.state_dict(), os.path.join(config["checkpoint_dir"], "latest.ckpt"))

        # Save best and early stop on the task metric
        if early.is_improvement(valid_task):
            print(f"✅ New best at epoch {epoch} (Val task: {valid_task:.4f})")
            torch.save(model.state_dict(), os.path.join(config["checkpoint_dir"], "best.ckpt"))

        if early.step(valid_task, epoch):
            print(f"⏹️ Early stopping at epoch {epoch} "
                f"(best task={early.best:.4f} at epoch {early.best_epoch})")
            break