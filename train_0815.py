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
from utils.loss import LOSS_REGISTRY
from utils.plot import plot_tsne

# import pdb; pdb.set_trace()

# def validate(model, loader, device, config, loss_cfg, loss_fns, writer=None, epoch=None):
#     model.eval()
#     losses_total = {}
#     num_batches = len(loader)

#     with torch.no_grad():
#         pbar = tqdm(loader, desc=f"[Valid] Epoch {epoch}")
#         for motions, labels, contents, _ in pbar:
#             motions, labels = motions.to(device), labels.to(device)
#             out = model.encode(motions)

#             # --- Loss calculation ---
#             losses = {}
#             for name, fn in loss_fns.items():
#                 spec = loss_cfg[name]
#                 if name == "stylecon":
#                     style_loss, content_loss = fn(spec, model, out, labels)
#                     losses["style"] = (spec["weight"] * style_loss, style_loss)
#                     losses["content"] = (spec["weight"] * content_loss, content_loss)
#                 else:
#                     val = fn(spec, model, out, labels)
#                     losses[name] = (spec["weight"] * val, val)

#             # --- Total loss ---
#             loss = sum(s for (s, _) in losses.values())
#             if not torch.isfinite(loss):
#                 print("❌ NaN or Inf detected in loss. Skipping batch.")
#                 continue

#             pbar.set_postfix(loss=loss.item())

#             for name, (scaled, raw) in losses.items():
#                 if name not in losses_total:
#                     losses_total[name] = (0.0, 0.0)
#                 prev_scaled, prev_raw = losses_total[name]
#                 losses_total[name] = (
#                     prev_scaled + scaled.item(),
#                     prev_raw + raw.item()
#                 )

#     # --- Logging ---
#     loss_total = sum(s for (s, _) in losses_total.values())

#     if writer is not None and epoch is not None:
#         writer.add_scalar("Valid/Total", loss_total / num_batches, epoch)
#         for name, (scaled_sum, raw_sum) in losses_total.items():
#             writer.add_scalar(f"Valid/Raw/{name}", raw_sum / num_batches, epoch)
#             writer.add_scalar(f"Valid/Scaled/{name}", scaled_sum / num_batches, epoch)

#     return loss_total / num_batches
def validate(model, loader, device, config, loss_cfg, loss_fns,
             running_mean,  # <-- pass the training EMA dict in
             alpha=0.01, eps=1e-8, use_norm=True,  # keep defaults aligned with train
             writer=None, epoch=None):

    model.eval()
    losses_total = {}
    num_batches = len(loader)

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"[Valid] Epoch {epoch}")
        for motions, labels, contents, _ in pbar:
            motions, labels = motions.to(device), labels.to(device)
            out = model.encode(motions)

            # --- Collect RAW losses only (no scaling here) ---
            losses = {}  # name -> (None, raw_tensor)
            for name, fn in loss_fns.items():
                spec = loss_cfg[name]
                if name == "stylecon":
                    style_loss, content_loss = fn(spec, model, out, labels)
                    losses["style"]   = (None, style_loss)
                    losses["content"] = (None, content_loss)
                else:
                    val = fn(spec, model, out, labels)
                    losses[name] = (None, val)

            # --- Apply the same normalization+weighting, but do NOT update EMAs in val ---
            losses, loss = normalize_and_weight(
                losses_dict=losses,
                loss_cfg=loss_cfg,
                running_mean=running_mean,
                use_norm=use_norm,       # usually True (matches train objective)
                update_stats=False,      # <---- freeze stats in validation
                alpha=alpha,
                eps=eps
            )

            if not torch.isfinite(loss):
                # Skip the batch but keep tqdm informative
                pbar.set_postfix(loss=float("nan"))
                continue

            pbar.set_postfix(loss=loss.item())

            # Accumulate totals for epoch averages
            for name, (scaled, raw) in losses.items():
                if name not in losses_total:
                    losses_total[name] = (0.0, 0.0)
                prev_scaled, prev_raw = losses_total[name]
                losses_total[name] = (
                    prev_scaled + scaled.item(),
                    prev_raw + raw.item()
                )

    # --- Logging ---
    loss_total = sum(s for (s, _) in losses_total.values())

    if writer is not None and epoch is not None:
        writer.add_scalar("Valid/Total", loss_total / num_batches, epoch)
        for name, (scaled_sum, raw_sum) in losses_total.items():
            writer.add_scalar(f"Valid/Raw/{name}",    raw_sum   / num_batches, epoch)
            writer.add_scalar(f"Valid/Scaled/{name}", scaled_sum / num_batches, epoch)

    return loss_total / num_batches


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


def get_weight(loss_name):
    if loss_name in loss_cfg:
        return loss_cfg[loss_name]["weight"]
    if loss_name in ("style", "content") and "stylecon" in loss_cfg:
        return loss_cfg["stylecon"]["weight"]
    return 1.0  # safe default


def normalize_and_weight(losses_dict, loss_cfg, running_mean, use_norm=True, update_stats=True, alpha=0.01, eps=1e-8):
    new_losses = {}
    final_loss = 0.0
    for name, (_, raw) in list(losses_dict.items()):
        if update_stats:
            prev = running_mean[name]
            rm = (1 - alpha) * prev + alpha * raw.detach().item()
            running_mean[name] = rm
        else:
            rm = running_mean[name]  # freeze stats in eval

        # 2) norm factor
        norm_factor = rm + eps

        # 3) weight after normalization
        base_w = get_weight(name)
        scaled = base_w * (raw / norm_factor) if use_norm else base_w * raw

        new_losses[name] = (scaled, raw)
        final_loss = final_loss + scaled

    return new_losses, final_loss

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
    train_styles, valid_styles = train_test_split(non_neutral_styles, test_size=19, random_state=config["random_seed"])
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

    # --- Training ---
    # I don't know what I'm doing
    best_val_loss = float('inf')
    running_mean = defaultdict(lambda: 1.0)  # EMA of *raw* loss values per loss name
    ALPHA = 0.01       # EMA smoothing; higher = faster adaptation
    NORM_EPS = 1e-8    # numerical safety
    WARMUP_BATCHES = 50  # optional: skip normalization at the very start
    global_step = 0

    model.train()
    model.encoder.vae.freeze()

    for epoch in range(1, config['epochs'] + 1):
        train_sampler.generate_batches()
        losses_total = {}
        
        pbar = tqdm(train_loader, desc=f"[Train] Epoch {epoch}")
        for motions, labels, contents, _ in pbar:
            motions, labels = motions.to(device), labels.to(device)
            out = model.encode(motions)

            # losses = {}
            # for name, fn in loss_fns.items():
            #     spec = loss_cfg[name]
            #     if name == "stylecon":
            #         style_loss, content_loss = fn(spec, model, out, labels)
            #         losses["style"] = (spec["weight"] * style_loss, style_loss)
            #         losses["content"] = (spec["weight"] * content_loss, content_loss)
            #     else:
            #         val = fn(spec, model, out, labels)
            #         losses[name] = (spec['weight'] * val, val)
            
            losses = {}  # will store: name -> (scaled_placeholder, raw_tensor)
            for name, fn in loss_fns.items():
                spec = loss_cfg[name]
                if name == "stylecon":
                    style_loss, content_loss = fn(spec, model, out, labels)
                    # store RAW parts separately so we can normalize/weight them independently
                    losses["style"]   = (None, style_loss)
                    losses["content"] = (None, content_loss)
                else:
                    val = fn(spec, model, out, labels)
                    losses[name] = (None, val)

            # --- Total loss ---
            # loss = sum(scaled for (scaled, _) in losses.values())
            # if not torch.isfinite(loss):
            #     import pdb; pdb.set_trace()
            #     continue

            use_norm = global_step >= WARMUP_BATCHES
            losses, loss = normalize_and_weight(
                losses_dict=losses,
                loss_cfg=loss_cfg,
                running_mean=running_mean,
                use_norm=use_norm,
                update_stats=True,          # training: update EMA
                alpha=ALPHA,
                eps=NORM_EPS
            )
            global_step += 1

            # --- Backpropagation ---
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            # --- Logging to tqdm ---
            unique_contents = set(contents)
            pbar.set_postfix(loss=loss.item(), contents=list(unique_contents))

            # --- Total loss ---
            for name, (scaled, raw) in losses.items():
                if name not in losses_total:
                    losses_total[name] = (0.0, 0.0)
                prev_scaled, prev_raw = losses_total[name]
                losses_total[name] = (
                    prev_scaled + scaled.item(),
                    prev_raw + raw.item()
                )

        # --- After each epoch ---
        num_batches = len(train_loader)
        loss_total = sum(scaled for (scaled, _) in losses_total.values())
        avg_train_loss = loss_total / num_batches
        writer.add_scalar("Train/Total", avg_train_loss, epoch)

        for name, (scaled_sum, raw_sum) in losses_total.items():
            writer.add_scalar(f"Train/Raw/{name}", raw_sum / num_batches, epoch)
            writer.add_scalar(f"Train/Scaled/{name}", scaled_sum / num_batches, epoch)

        # --- Validation ---
        valid_sampler.generate_batches()
        valid_loss = validate(
            model, valid_loader, device,
            config=config,
            loss_cfg=loss_cfg,
            loss_fns=loss_fns,
            writer=writer,
            epoch=epoch
        )

        print(f"Epoch {epoch} | Train: {avg_train_loss:.4f} | Valid: {valid_loss:.4f}")

        # Evaluation (t-SNE)
        plot_tsne(model, valid_loader, device, epoch, title="valid", result_dir=config["result_dir"], label_to_name_dict=valid_dataset.label_to_style, writer=writer)

        # Always save latest
        os.makedirs(config["checkpoint_dir"], exist_ok=True)
        torch.save(model.state_dict(), os.path.join(config["checkpoint_dir"], "latest.ckpt"))

        # Save best if validation improves
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            print(f"✅ New best model at epoch {epoch} (Val loss: {valid_loss:.4f})")
            torch.save(model.state_dict(), os.path.join(config["checkpoint_dir"], "best.ckpt"))
    writer.close()