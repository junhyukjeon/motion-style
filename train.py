# --- Imports ---
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import torch.nn.functional as F
import yaml
from datetime import datetime
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataset import StyleDataset
from data.sampler import StyleSampler
from model.networks import StyleEncoder
from salad.t2m import Text2Motion
from utils.losses import supervised_contrastive_loss

def evaluate(model, t2m, loader, device, epoch=None, label="valid", result_dir="", label_to_name_dict=None, max_samples=3000, writer=None):
    model.eval()
    all_embeddings, all_labels = [], []
    pbar = tqdm(total=max_samples, desc=f"[t-SNE] Extracting ({label})")

    with torch.no_grad():
        for motions, labels, _ in loader:
            motions = motions.to(device)
            labels = labels.to(device)
            z, _ = t2m.vae.encode(motions)     # (B, T, J, D)
            out_pooled = model(z).mean(dim=(1, 2))
            mask = torch.isfinite(out_pooled).all(dim=1)

            out_cpu = out_pooled[mask].cpu()[:max_samples - len(all_labels)]
            labels_cpu = labels[mask].cpu()[:len(out_cpu)]

            all_embeddings.append(out_cpu)
            all_labels.append(labels_cpu)
            pbar.update(len(out_cpu))

            if len(torch.cat(all_labels)) >= max_samples:
                break

    pbar.close()
    if not all_labels:
        print("❌ No valid samples collected for t-SNE!")
        return

    embeddings_np = torch.cat(all_embeddings).numpy()
    labels_np = torch.cat(all_labels).numpy()
    X = TSNE(n_components=2, perplexity=30, metric='cosine', random_state=42).fit_transform(embeddings_np)

    unique_labels = np.unique(labels_np)
    num_classes = len(unique_labels)
    cmap = plt.cm.get_cmap('nipy_spectral', num_classes)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels_np, cmap=cmap, alpha=0.7, s=10)
    cbar = plt.colorbar(scatter, ax=ax)

    if num_classes <= 20:
        cbar.set_ticks(unique_labels)
        if label_to_name_dict:
            label_names = [label_to_name_dict.get(idx, str(idx)) for idx in unique_labels]
            cbar.set_ticklabels(label_names)
    else:
        cbar.remove()

    # Save the plot
    os.makedirs(os.path.join(result_dir, label), exist_ok=True)
    save_path = os.path.join(result_dir, label, f"tsne_epoch{epoch:03d}.png")
    fig.savefig(save_path)

    # ✅ Correct: Pass figure explicitly
    if writer is not None:
        writer.add_figure(f"Plot/{label}", fig, global_step=epoch)

    plt.close(fig)
    print(f"[t-SNE] Saved → {save_path}")

def validate(model, t2m, loader, device, temperature):
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for motions, labels, _ in loader:
            motions = motions.to(device)
            labels = labels.to(device)

            z, _ = t2m.vae.encode(motions)
            if torch.isnan(z).any() or torch.isinf(z).any():
                print("Skipping validation batch due to NaN/Inf in z")
                continue

            out = model(z)
            out_pooled = out.mean(dim=(1, 2))
            loss = supervised_contrastive_loss(out_pooled, labels, temperature)

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / max(1, num_batches)
    return avg_loss

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

    os.makedirs(config["result_dir"], exist_ok=True)
    os.makedirs(os.path.join(config["result_dir"], "valid"), exist_ok=True)
    os.makedirs(os.path.join(config["result_dir"], "test"), exist_ok=True)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join("./result/tensorboard", config["run_name"]))

    set_seed(config["random_seed"])

    # --- Load dataset stats ---
    MEAN = np.load(config["mean_path"])
    STD = np.load(config["std_path"])

    # --- Load and split styles ---
    with open(config["style_json"]) as f:
        full_label_to_ids = json.load(f)

    all_styles = list(full_label_to_ids.keys())
    train_styles, test_styles = train_test_split(all_styles, test_size=1 - config["train_split"], random_state=config["random_seed"])

    train_label_to_ids, valid_label_to_ids = {}, {}
    for style in train_styles:
        motions = full_label_to_ids[style]
        if len(motions) < 2:
            train_label_to_ids[style] = motions
        else:
            train_ids, valid_ids = train_test_split(motions, test_size=config["val_split"], random_state=config["random_seed"])
            train_label_to_ids[style] = train_ids
            valid_label_to_ids[style] = valid_ids

    test_label_to_ids = {style: full_label_to_ids[style] for style in test_styles}
    if config.get("num_test_styles") is not None:
        rng = random.Random(config["random_seed"])
        selected_test_styles = rng.sample(list(test_label_to_ids.keys()), config["num_test_styles"])
        test_label_to_ids = {style: test_label_to_ids[style] for style in selected_test_styles}

    # --- Load datasets ---
    train_dataset = StyleDataset(
        motion_dir=config["motion_dir"],
        mean=MEAN,
        std=STD,
        window_size=config["window_size"],
        label_to_ids=train_label_to_ids
    )

    valid_dataset = StyleDataset(
        motion_dir=config["motion_dir"],
        mean=MEAN,
        std=STD,
        window_size=config["window_size"],
        label_to_ids=valid_label_to_ids
    )

    test_dataset = StyleDataset(
        motion_dir=config["motion_dir"],
        mean=MEAN,
        std=STD,
        window_size=config["window_size"],
        label_to_ids=test_label_to_ids
    )

    # --- Samplers & Loaders ---
    train_sampler = StyleSampler(train_dataset, config["batch_size"], config["samples_per_class"])
    train_loader  = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)

    # --- Model & Optimizer ---
    denoiser_name = config["denoiser_name"]
    dataset_name = config["dataset_name"]
    t2m = Text2Motion(denoiser_name, dataset_name)
    model = StyleEncoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # --- Training Loop ---
    t2m.vae.freeze()
    latest_model_path = f"{config['checkpoint_dir']}/style_encoder_latest.pt"
    checkpoint_dir = config["checkpoint_dir"]

    for epoch in range(1, config["epochs"] + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0

        for motions, labels, _ in tqdm(train_loader, desc=f"[Train] Epoch {epoch}"):
            motions = motions.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                z, _ = t2m.vae.encode(motions)
                if torch.isnan(z).any() or torch.isinf(z).any():
                    print("Skipping batch due to NaN/Inf in z") 
                    continue
                
            out = model(z) # (B, T, J, D)
            out_pooled = out.mean(dim=(1,2)) # (B, D)
            loss = supervised_contrastive_loss(out_pooled, labels, config["temperature"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            total_loss += loss_val

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}")

        # Save latest model always
        torch.save(model.state_dict(), latest_model_path)

        valid_loss = validate(model, t2m, valid_loader, device, config["temperature"])
        print(f"Epoch {epoch} - Valid Loss: {valid_loss:.4f}")

        writer.add_scalar("Loss/Train", avg_loss, epoch)
        writer.add_scalar("Loss/Valid", valid_loss, epoch)

        evaluate(model, t2m, valid_loader, device, epoch, label="valid", result_dir=config["result_dir"], label_to_name_dict=valid_dataset.label_to_style, writer=writer)
        evaluate(model, t2m, test_loader, device, epoch, label="test", result_dir=config["result_dir"], label_to_name_dict=test_dataset.label_to_style, writer=writer)

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            ckpt_path = f"{checkpoint_dir}/style_encoder_epoch{epoch:03d}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

    writer.close()