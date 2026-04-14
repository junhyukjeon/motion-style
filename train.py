# --- Imports ---
import argparse
import json
import os
import random

# Cap host-side threading by default so concurrent training jobs do not
# oversubscribe the same CPU. These can still be overridden from the shell.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
os.environ.setdefault("TORCH_NUM_INTEROP_THREADS", "1")

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from collections import defaultdict
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.plot import plot_tsne

from data.dataset import DATASET_REGISTRY
from data.sampler import SAMPLER_REGISTRY
from model.t2sm import Text2StylizedMotion
from loss.losses import LOSS_REGISTRY


def set_seed(seed):
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
    config["result_dir"] = os.path.join(config["result_dir"], run_name)
    config["checkpoint_dir"] = os.path.join(config["checkpoint_dir"], run_name)
    return config


if __name__ == "__main__":
    device = torch.device("cuda")
    use_bf16 = (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8)

    amp_enabled = use_bf16
    amp_dtype = torch.bfloat16 if use_bf16 else None

    print(f"use_bf16: {use_bf16}")
    print(f"amp_enabled: {amp_enabled}")

    config = load_config()
    set_seed(config["random_seed"])

    # Match Torch's thread pools to the default caps above unless the user
    # overrides them via environment variables at launch time.
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    torch.set_num_interop_threads(int(os.environ["TORCH_NUM_INTEROP_THREADS"]))

    # Output directory
    os.makedirs(config["result_dir"], exist_ok=True)
    os.makedirs(os.path.join(config["result_dir"], "valid"), exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join("./results/tensorboard", config["run_name"]))

    # Dataset
    dataset_cfg = config['dataset']
    dataset = DATASET_REGISTRY[dataset_cfg['class']](dataset_cfg)

    # Dataloader (custom sampler)
    sampler_cfg = config['sampler']
    sampler = SAMPLER_REGISTRY[sampler_cfg['class']](sampler_cfg, dataset)
    dataloader = DataLoader(dataset, batch_sampler=sampler)

    # t-SNE
    label_to_name_dict = dict(dataset.idx_to_style)
    tsne_every = int(config.get("tsne_every", 10))
    tsne_max_samples = int(config.get("tsne_max_samples", 1000))

    # Model
    model_cfg = dict(config['model'])
    model_cfg.pop("class", None)
    model = Text2StylizedMotion(model_cfg).to(device)
    style_names = [dataset.idx_to_style[i] for i in range(len(dataset.idx_to_style))]
    model.set_style_text_prior(style_names)

    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=config['lr']
    )

    # Losses
    loss_cfg = config['loss']
    loss_fns = {name: LOSS_REGISTRY[name] for name in loss_cfg}
    normalize_flags = {
        name: loss_cfg[name].get("normalize", True)
        for name in loss_cfg
    }

    # Calibration
    scales = {}
    sum_sq = defaultdict(float)
    pbar = tqdm(dataloader, total=config['steps'])
    n = 0

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(pbar):
            if i >= config['steps']: break
            captions, motions, num_frames, style_idxs = batch
            motions, num_frames, style_idxs = motions.to(device), num_frames.to(device), style_idxs.to(device)

            with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
                out = model(captions, motions, num_frames, style_idxs)
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
                for name, val in losses.items():
                    total_loss += loss_cfg[name]["weight"] * val

            all_losses = {}
            all_losses.update(losses)

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

    # Training
    model.train()
    for epoch in range(1, config['epochs'] + 1):
        losses_scaled_sum = defaultdict(float)
        losses_norm_sum   = defaultdict(float)
        losses_raw_sum    = defaultdict(float)

        pbar = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"[Train] Epoch {epoch}"
        )

        batch_idx = -1
        for batch_idx, batch in enumerate(pbar):
            captions, motions, num_frames, style_idxs = batch
            motions = motions.to(device)
            num_frames = num_frames.to(device)
            style_idxs = style_idxs.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
                out = model(captions, motions, num_frames, style_idxs)
                losses = {}
                losses_denoiser = {}
                for name, fn in loss_fns.items():
                    spec = loss_cfg[name]
                    raw = fn(spec, model, out)
                    norm = raw / scales[name]
                    scaled = spec["weight"] * norm

                    if "cycle" in name:
                        losses_denoiser[name] = (scaled, norm, raw)
                    else:
                        losses[name] = (scaled, norm, raw)

                total_loss = torch.zeros((), device=device)

                for _, (scaled, _, _) in losses.items():
                    total_loss += scaled

            total_loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=float(total_loss.item()))

            all_losses = {}
            all_losses.update(losses)
            all_losses.update(losses_denoiser)

            for name, (scaled, norm, raw) in all_losses.items():
                losses_scaled_sum[name] += scaled.item()
                losses_norm_sum[name]   += norm.item()
                losses_raw_sum[name]    += raw.item()

        num_batches = max(batch_idx + 1, 1)
        train_total_scaled = sum(losses_scaled_sum.values()) / num_batches
        train_total_norm   = sum(losses_norm_sum.values()) / num_batches
        train_total_raw    = sum(losses_raw_sum.values()) / num_batches

        if writer is not None:
            writer.add_scalar("Train/Raw/Total", train_total_raw, epoch)
            writer.add_scalar("Train/Norm/Total", train_total_norm, epoch)
            writer.add_scalar("Train/Scaled/Total", train_total_scaled, epoch)

            for name in losses_scaled_sum.keys():
                writer.add_scalar(f"Train/Raw/{name}",    losses_raw_sum[name]    / num_batches, epoch)
                writer.add_scalar(f"Train/Norm/{name}",   losses_norm_sum[name]   / num_batches, epoch)
                writer.add_scalar(f"Train/Scaled/{name}", losses_scaled_sum[name] / num_batches, epoch)

        print(
            f"Epoch {epoch} | "
            f"Train scaled: {train_total_scaled:.4f} | "
            f"Train norm: {train_total_norm:.4f} | "
            f"Train raw: {train_total_raw:.4f}"
        )

        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        sd = model.state_dict()
        sd_trainable = {k: v for k, v in sd.items() if k in trainable}

        os.makedirs(config["checkpoint_dir"], exist_ok=True)
        torch.save(sd_trainable, os.path.join(config["checkpoint_dir"], "latest.ckpt"))

        save_every = config.get("save_every", 10)
        if save_every and (epoch % save_every == 0):
            torch.save(
                sd_trainable,
                os.path.join(config["checkpoint_dir"], f"epoch_{epoch:04d}.ckpt")
            )

        if tsne_every > 0 and epoch % tsne_every == 0:
            plot_tsne(
                model=model,
                loader=dataloader,
                device=device,
                epoch=epoch,
                title="train",
                result_dir=config["result_dir"],
                label_to_name_dict=label_to_name_dict,
                max_samples=tsne_max_samples,
                writer=writer,
            )
