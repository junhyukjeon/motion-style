# --- Imports ---
import argparse
import json
import numpy as np
import os
import random
import torch
import torch.nn.functional as F
import yaml
from collections import defaultdict
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.mix_dataset import MixedTextStyleDataset
from data.sampler import SAMPLER_REGISTRY
from model.t2sm_mix import Text2StylizedMotion
from utils.train.early_stopper import EarlyStopper
from utils.train.loss_mix import LOSS_REGISTRY
from utils.plot import plot_tsne_mix


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

    from pathlib import Path
    cfg_path = Path(args.config).resolve()

    with cfg_path.open('r') as f:
        config = yaml.safe_load(f)

    parts = cfg_path.parts
    if "configs" in parts:
        i = parts.index("configs")
        sub = Path(*parts[i+1:]).with_suffix("")
        run_name = str(sub).replace("\\", "/")
    else:
        run_name = cfg_path.stem

    config["run_name"] = run_name

    # results/loss/0  and  checkpoints/loss/0
    config["result_dir"]     = os.path.join(config["result_dir"], run_name)
    config["checkpoint_dir"] = os.path.join(config["checkpoint_dir"], run_name)
    return config


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config()
    set_seed(config["random_seed"])

    # --- Result directories --- #
    os.makedirs(config["result_dir"], exist_ok=True)
    os.makedirs(os.path.join(config["result_dir"], "valid"), exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join("./results/tensorboard", config["run_name"]))

    # --- Style Split --- #
    with open(config["dataset"]["style_json"]) as f:
        styles_to_ids = json.load(f)
    styles_sorted = sorted(styles_to_ids.keys())
    if config['valid_size'] == 0.0:
        train_styles, valid_styles = styles_sorted, []
    else:
        train_styles, valid_styles = train_test_split(styles_sorted, test_size=config['valid_size'], random_state=config["random_seed"])

    # --- Dataset & Loader --- #
    dataset_cfg   = config['dataset']
    train_dataset = MixedTextStyleDataset(dataset_cfg, train_styles)
    valid_dataset = MixedTextStyleDataset(dataset_cfg, valid_styles)

    # sampler_cfg = config['sampler']
    # train_sampler = SAMPLER_REGISTRY[sampler_cfg['type']](sampler_cfg, train_dataset)
    # valid_sampler = SAMPLER_REGISTRY[sampler_cfg['type']](sampler_cfg, valid_dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=dataset_cfg["batch_size"], num_workers=8, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=dataset_cfg["batch_size"], num_workers=8, shuffle=True, pin_memory=True)
    
    # --- Model --- #
    model_cfg = config['model']
    model = Text2StylizedMotion(model_cfg).to(device)
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=config['lr']
    )

    # --- Losses & Scaler --- #
    loss_cfg     = config['loss']
    loss_fns     = {name: LOSS_REGISTRY[name] for name in loss_cfg}

    # --- Early Stopper --- #
    early_cfg = config['early']
    early = EarlyStopper(early_cfg)

    # --- Calibration --- #
    scales  = {}
    sum_sq  = defaultdict(float)
    model.train()
    pbar = tqdm(zip(range(config['steps']), train_loader), total=min(config['steps'], len(train_loader)), desc="[Calibrating]")
    n = 0
    for _, batch in pbar:
        out = model(batch)
        losses = {}
        for name, fn in loss_fns.items():
            spec = loss_cfg[name]
            raw  = fn(spec, model, out)
            losses[name] = raw

            # =================== DEBUGGING ===================
            # if raw.isnan().any() or raw.isinf().any() or (raw > 1e5).any():
            #     breakpoint()
            # =================================================

        for name, val in losses.items():
            sum_sq[name] += float(val.detach()) ** 2

        total_loss = None
        for name, val in losses.items():
            scaled     = loss_cfg[name]['weight'] * val
            total_loss = scaled if total_loss is None else total_loss + scaled

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=float(total_loss.item()))
        n += 1

    for name, s2 in sum_sq.items():
        rms = (s2 / max(1, n)) ** 0.5
        scales[name] = max(rms, config['tau'])
    print("📏 Frozen RMS denominators:", {k: round(v, 6) for k, v in scales.items()})

    # --- Training --- #
    best_val_loss = float('inf')
    for epoch in range(1, config['epochs'] + 1):
        # Train
        model.train()
        losses_scaled_sum = defaultdict(float)
        losses_norm_sum   = defaultdict(float)
        losses_raw_sum    = defaultdict(float)

        pbar = tqdm(train_loader, desc=f"[Train] Epoch {epoch}")
        batch_idx = -1
        for batch_idx, batch in enumerate(pbar):
            out = model(batch)
            losses = {}
            total_loss = 0.0
            for name, fn in loss_fns.items():
                spec   = loss_cfg[name]
                raw    = fn(spec, model, out)
                norm   = raw / scales[name]
                scaled = norm * spec['weight']
                losses[name] = (scaled, norm, raw)
                total_loss = total_loss + scaled
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=float(total_loss.item()))
            for name, (scaled, norm, raw) in losses.items():
                losses_scaled_sum[name] += scaled.item()
                losses_norm_sum[name]   += norm.item()
                losses_raw_sum[name]    += raw.item()

        num_batches = max(batch_idx + 1, 1)
        train_total_scaled = sum(losses_scaled_sum.values()) / max(num_batches, 1)
        train_total_norm = sum(losses_norm_sum.values()) / max(num_batches, 1)
        train_total_raw = sum(losses_raw_sum.values()) / max(num_batches, 1)

        if writer is not None and epoch is not None:
            writer.add_scalar("Train/Raw/Total", train_total_raw, epoch)
            writer.add_scalar("Train/Norm/Total", train_total_norm, epoch)
            writer.add_scalar("Train/Scaled/Total", train_total_scaled, epoch)
            for name in losses_scaled_sum.keys():
                writer.add_scalar(f"Train/Raw/{name}",    losses_raw_sum[name]    / max(num_batches,1), epoch)
                writer.add_scalar(f"Train/Norm/{name}",   losses_norm_sum[name]   / max(num_batches,1), epoch)
                writer.add_scalar(f"Train/Scaled/{name}", losses_scaled_sum[name] / max(num_batches,1), epoch)

        # Validate
        model.eval()
        losses_scaled_sum = defaultdict(float)
        losses_norm_sum   = defaultdict(float)
        losses_raw_sum    = defaultdict(float)

        with torch.no_grad():
            pbar = tqdm(valid_loader, desc=f"[Valid] Epoch {epoch if epoch is not None else ''}".strip())
            batch_idx = -1
            for batch_idx, batch in enumerate(pbar):
                out = model(batch)
                losses = {}
                total_loss = 0.0
                for name, fn in loss_fns.items():
                    spec   = loss_cfg[name]
                    raw    = fn(spec, model, out)
                    norm   = raw / scales[name]
                    scaled = norm * spec['weight']
                    losses[name] = (scaled, norm, raw)
                    total_loss = total_loss + scaled
                pbar.set_postfix(loss=float(total_loss.item()))
                for name, (scaled, norm, raw) in losses.items():
                    losses_scaled_sum[name] += scaled.item()
                    losses_norm_sum[name]   += norm.item()
                    losses_raw_sum[name]    += raw.item()

            num_batches = max(batch_idx + 1, 1)
            val_total_scaled = sum(losses_scaled_sum.values()) / num_batches
            val_total_norm   = sum(losses_norm_sum.values()) / num_batches
            val_total_raw    = sum(losses_raw_sum.values())  / num_batches

            if writer is not None and epoch is not None:
                writer.add_scalar("Valid/Raw/Total",    val_total_raw,    epoch)
                writer.add_scalar("Valid/Norm/Total",   val_total_norm,   epoch)
                writer.add_scalar("Valid/Scaled/Total", val_total_scaled, epoch)
                for name in losses_scaled_sum.keys():
                    writer.add_scalar(f"Valid/Raw/{name}",    losses_raw_sum[name]    / num_batches, epoch)
                    writer.add_scalar(f"Valid/Norm/{name}",   losses_norm_sum[name]   / num_batches, epoch)
                    writer.add_scalar(f"Valid/Scaled/{name}", losses_scaled_sum[name] / num_batches, epoch)

            print(
                f"Epoch {epoch} | "
                f"Train scaled: {train_total_scaled:.4f} | Train norm: {train_total_norm:.4f} | Train raw: {train_total_raw:.4f} | "
                f"Valid scaled: {val_total_scaled:.4f}   | Valid norm: {val_total_norm:.4f}   | Valid raw: {val_total_raw:.4f} | "
            )

            plot_tsne_mix( 
                model, valid_loader, device, epoch, title="valid",
                result_dir=config["result_dir"],
                label_to_name_dict=valid_dataset.style_idx_to_style,
                writer=writer
            )

            trainable = {n for n, p in model.named_parameters() if p.requires_grad}
            sd = model.state_dict()
            sd_trainable = {k: v for k, v in sd.items() if k in trainable}

            os.makedirs(config["checkpoint_dir"], exist_ok=True)
            torch.save(sd_trainable, os.path.join(config["checkpoint_dir"], "latest.ckpt"))

            if epoch % 10 == 0:
                torch.save(sd_trainable, os.path.join(config["checkpoint_dir"], f"epoch{epoch:03d}.ckpt"))

            if early.is_improvement(val_total_scaled):
                print(f"✅ New best at epoch {epoch} (Val task: {val_total_scaled:.4f})")
                torch.save(sd_trainable, os.path.join(config["checkpoint_dir"], "best.ckpt"))

            if early.step(val_total_scaled, epoch):
                print(f"⏹️ Early stopping at epoch {epoch} "
                    f"(best task={early.best:.4f} at epoch {early.best_epoch})")
                break
