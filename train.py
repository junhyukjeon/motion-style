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
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataset import Dataset100Style, DatasetHumanML3D
from data.sampler import SAMPLER_REGISTRY
from model.t2sm import Text2StylizedMotion
from utils.train.early_stopper import EarlyStopper
from utils.train.loss import LOSS_REGISTRY
from utils.plot import plot_tsne


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
    style_cfg = config['dataset_style']
    with open(style_cfg["style_json"]) as f:
        styles_to_ids = json.load(f)
    all_styles = sorted(styles_to_ids.keys())

    #############################################################################################
    # train_styles, valid_styles = train_test_split(all_styles, test_size=config['valid_size'], random_state=config["random_seed"])
    # style_train = Dataset100Style(style_cfg, styles=train_styles, train=True)
    # style_valid = Dataset100Style(style_cfg, styles=valid_styles, train=False)

    # sampler_cfg = config['sampler']
    # train_sampler = SAMPLER_REGISTRY[sampler_cfg['type']](sampler_cfg, style_train)
    # valid_sampler = SAMPLER_REGISTRY[sampler_cfg['type']](sampler_cfg, style_valid)
    # train_loader_style  = DataLoader(style_train, batch_sampler=train_sampler)
    # valid_loader_style  = DataLoader(style_valid, batch_sampler=valid_sampler)

    #############################################################################################
    def read_ids(p):
        with open(p, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    ids_train = read_ids("./dataset/100style/train_random_100style.txt")
    ids_valid = read_ids("./dataset/100style/valid_random_100style.txt")

    # --- Dataset & Loader --- #
    style_train = Dataset100Style(style_cfg, styles=all_styles, train=True,  use_ids=ids_train)
    style_valid = Dataset100Style(style_cfg, styles=all_styles, train=False, use_ids=ids_valid)

    sampler_cfg = config['sampler']
    train_sampler = SAMPLER_REGISTRY[sampler_cfg['type']](sampler_cfg, style_train)
    valid_sampler = SAMPLER_REGISTRY[sampler_cfg['type']](sampler_cfg, style_valid)
    train_loader_style  = DataLoader(style_train, batch_sampler=train_sampler)
    valid_loader_style  = DataLoader(style_valid, batch_sampler=valid_sampler)
    # train_loader_style  = DataLoader(style_train, batch_size=sampler_cfg["batch_size"], shuffle=True, drop_last=True)
    # valid_loader_style  = DataLoader(style_valid, batch_size=sampler_cfg["batch_size"], shuffle=False, drop_last=True)
    #############################################################################################

    hml_cfg = config['dataset_hml']
    hml_train = DatasetHumanML3D(hml_cfg, train=True)
    hml_valid = DatasetHumanML3D(hml_cfg, train=False)
    train_loader_hml    = DataLoader(hml_train, batch_size=sampler_cfg["batch_size"], shuffle=True, drop_last=True)
    valid_loader_hml    = DataLoader(hml_valid, batch_size=sampler_cfg["batch_size"], shuffle=False, drop_last=True)

    # --- Model --- #
    model_cfg = config['model']
    model = Text2StylizedMotion(model_cfg).to(device)
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=config['lr']
    )
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    # --- Losses & Scaler --- #
    loss_cfg = config['loss']
    loss_fns = {name: LOSS_REGISTRY[name] for name in loss_cfg}
    normalize_flags = {
        name: loss_cfg[name].get("normalize", True)
        for name in loss_cfg
    }

    # --- Early Stopper --- #
    early_cfg = config['early']
    early = EarlyStopper(early_cfg)

    # --- Calibration --- #
    scales = {}
    sum_sq = defaultdict(float)
    model.train()
    pbar = tqdm(zip(train_loader_style, train_loader_hml), total=config['steps'])
    n = 0
    for i, (b_style, b_hml) in enumerate(pbar):
        if i >= config['steps']: break
        cap1, win1, len1, sty1 = b_style
        cap2, win2, len2, sty2 = b_hml
        batch = (cap1, win1, len1, sty1, cap2, win2, len2, sty2)
        # optimizer.zero_grad(set_to_none=True)
        with autocast(dtype=amp_dtype, enabled=torch.cuda.is_available()):
            out = model(batch)
            losses = {}
            losses_denoiser = {}
            for name, fn in loss_fns.items():
                spec = loss_cfg[name]
                raw  = fn(spec, model, out)
                if "cycle" in name:
                    losses_denoiser[name] = raw
                else:
                    losses[name] = raw
            total_loss = torch.zeros((), device=device)
            total_loss_denoiser = torch.zeros((), device=device)
            for name, val in losses.items():
                total_loss += loss_cfg[name]["weight"] * val
            for name, val in losses_denoiser.items():
                total_loss_denoiser += loss_cfg[name]["weight"] * val

        # total_loss.backward(retain_graph=True)

        # if total_loss_denoiser != 0.0:
        #     enc_req = [p.requires_grad for p in model.style_encoder.parameters()]
        #     for p in model.style_encoder.parameters():
        #         p.requires_grad_(False)
        #     try:
        #         total_loss_denoiser.backward()
        #     finally:
        #         for p, r in zip(model.style_encoder.parameters(), enc_req):
        #             p.requires_grad_(r)

        # optimizer.step()

        all_losses = {}
        all_losses.update(losses)
        all_losses.update(losses_denoiser)

        for name, val in all_losses.items():
            if normalize_flags.get(name, False):
                v = val.detach().float().item()
                sum_sq[name] += v * v

        n += 1
        pbar.set_postfix(loss=float(total_loss.item()))

    for name in loss_cfg.keys():
        if normalize_flags[name]:
            s2  = sum_sq[name]
            rms = (s2 / max(1, n)) ** 0.5
            scales[name] = max(rms, config['tau'])
        else:
            scales[name] = 1.0
    print("📏 Frozen RMS denominators:", {k: round(v, 6) for k, v in scales.items()})

    # --- Training --- #
    best_val_loss = float('inf')
    for epoch in range(1, config['epochs'] + 1):
        # Train
        model.train()
        losses_scaled_sum = defaultdict(float)
        losses_norm_sum   = defaultdict(float)
        losses_raw_sum    = defaultdict(float)
        pbar = tqdm(zip(train_loader_style, train_loader_hml),
                    total=len(train_loader_style),
                    desc=f"[Train] Epoch {epoch}")
        batch_idx = -1
        for batch_idx, (b_style, b_hml) in enumerate(pbar):
            cap1, win1, len1, sty1 = b_style
            cap2, win2, len2, sty2 = b_hml
            batch = (cap1, win1, len1, sty1, cap2, win2, len2, sty2)
            optimizer.zero_grad(set_to_none=True)
            with autocast(dtype=amp_dtype, enabled=torch.cuda.is_available()):
                out = model(batch)
                total_loss = torch.zeros((), device=device)         # non-cycle (updates BOTH)
                total_loss_cycle = torch.zeros((), device=device)   # cycle (denoiser-only)
                losses = {}
                for name, fn in loss_fns.items():
                    spec   = loss_cfg[name]
                    raw    = fn(spec, model, out)
                    norm   = raw / scales[name]
                    scaled = norm * spec["weight"]
                    losses[name] = (scaled, norm, raw)
                    if "cycle" in name:
                        total_loss_cycle = total_loss_cycle + scaled
                    else:
                        total_loss = total_loss + scaled

            # 1) Backprop non-cycle losses (updates BOTH denoiser + style_encoder)
            total_loss.backward(retain_graph=True)

            # 2) Backprop cycle losses (updates DENOISER ONLY)
            if len([k for k in losses.keys() if "cycle" in k]) > 0:
                enc_req = [p.requires_grad for p in model.style_encoder.parameters()]
                for p in model.style_encoder.parameters():
                    p.requires_grad_(False)
                try:
                    total_loss_cycle.backward()
                finally:
                    for p, r in zip(model.style_encoder.parameters(), enc_req):
                        p.requires_grad_(r)

            optimizer.step()

            pbar.set_postfix(loss=float((total_loss + total_loss_cycle).item()))

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

        with torch.no_grad(), autocast(dtype=amp_dtype, enabled=torch.cuda.is_available()):
            pbar = tqdm(zip(valid_loader_style, valid_loader_hml), total=len(valid_loader_style), desc=f"[Valid] Epoch {epoch if epoch is not None else ''}".strip())
            batch_idx = -1
            for batch_idx, (b_style, b_hml) in enumerate(pbar):
                cap1, win1, len1, sty1 = b_style
                cap2, win2, len2, sty2 = b_hml
                batch = (cap1, win1, len1, sty1, cap2, win2, len2, sty2)
                out = model(batch)
                total_loss = torch.zeros((), device=device)
                total_loss_cycle = torch.zeros((), device=device)
                losses = {}
                for name, fn in loss_fns.items():
                    spec   = loss_cfg[name]
                    raw    = fn(spec, model, out)
                    norm   = raw / scales[name]
                    scaled = norm * spec["weight"]
                    losses[name] = (scaled, norm, raw)
                    if "cycle" in name:
                        total_loss_cycle = total_loss_cycle + scaled
                    else:
                        total_loss = total_loss + scaled
                pbar.set_postfix(loss=float((total_loss + total_loss_cycle).item()))
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

            plot_tsne( 
                model, valid_loader_style, device, epoch, title="valid",
                result_dir=config["result_dir"],
                label_to_name_dict=style_valid.style_idx_to_style,
                writer=writer
            )

            trainable = {n for n, p in model.named_parameters() if p.requires_grad}
            sd = model.state_dict()
            sd_trainable = {k: v for k, v in sd.items() if k in trainable}

            os.makedirs(config["checkpoint_dir"], exist_ok=True)
            torch.save(sd_trainable, os.path.join(config["checkpoint_dir"], "latest.ckpt"))

            if early.is_improvement(val_total_scaled):
                print(f"✅ New best at epoch {epoch} (Val task: {val_total_scaled:.4f})")
                torch.save(sd_trainable, os.path.join(config["checkpoint_dir"], "best.ckpt"))

            if early.step(val_total_scaled, epoch):
                print(f"⏹️ Early stopping at epoch {epoch}"
                    f"(best task={early.best:.4f} at epoch {early.best_epoch})")
                break