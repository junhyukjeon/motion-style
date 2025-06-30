# --- Imports ---
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import torch.nn.functional as F
from datetime import datetime
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import StyleEncoder
from salad_backup.t2m import Text2Motion
from data.style_dataset import StyleDataset

# --- Losses --- 
def supervised_contrastive_loss(z, labels, temperature=0.07):
    # Mean pool frames and joints
    z = z.mean(dim=(1,2))

    # Normalize
    z = F.normalize(z, dim=1)

    # labels: (B,)
    sim = torch.matmul(z, z.T) / temperature  # (B, B)
    N = sim.size(0)

    logits_mask = ~torch.eye(N, dtype=torch.bool, device=z.device)
    labels = labels.unsqueeze(0)  # (1, B)
    pos_mask = (labels == labels.T) & logits_mask  # (B, B)

    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8)
    loss = -mean_log_prob_pos.mean()
    return loss

def tsne_evaluate(model, t2m, loader, device, epoch=None, label="valid", result_dir="./result", label_to_name_dict=None):
    MAX_SAMPLES = 3000
    num_collected = 0

    model.eval()
    all_embeddings = []
    all_labels = []

    pbar = tqdm(total=MAX_SAMPLES, desc=f"[t-SNE] Extracting ({label})")

    with torch.no_grad():
        for motions, labels in loader:
            motions = motions.to(device)
            labels = labels.to(device)

            z, _ = t2m.vae.encode(motions)  # (B, T, J, D)

            if torch.isnan(z).any() or torch.isinf(z).any():
                print("❌ Detected NaN or Inf in z")
                print(f"  Stats: min={z.min().item():.4f}, max={z.max().item():.4f}, mean={z.mean().item():.4f}")
                print(f"  Count of invalid entries: {(~torch.isfinite(z)).sum().item()} / {z.numel()}")
                print(f"  Batch labels: {labels.tolist()}")
                continue

            out = model(z)
            out_pooled = out.mean(dim=(1, 2))  # (B, D)
            out_cpu = out_pooled.cpu()
            labels_cpu = labels.cpu()

            valid_mask = torch.isfinite(out_cpu).all(dim=1)
            if valid_mask.sum().item() < len(valid_mask):
                print(f"⚠️ Skipping {len(valid_mask) - valid_mask.sum().item()} invalid embeddings")
            out_cpu = out_cpu[valid_mask]
            labels_cpu = labels_cpu[valid_mask]

            # Trim to MAX_SAMPLES if needed
            if num_collected + len(out_cpu) > MAX_SAMPLES:
                needed = MAX_SAMPLES - num_collected
                out_cpu = out_cpu[:needed]
                labels_cpu = labels_cpu[:needed]

            all_embeddings.append(out_cpu)
            all_labels.append(labels_cpu)
            num_collected += len(out_cpu)
            pbar.update(len(out_cpu))

            if num_collected >= MAX_SAMPLES:
                break

    pbar.close()

    if num_collected == 0:
        print("❌ No valid samples collected for t-SNE!")
        return

    # Stack embeddings and labels
    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=30, metric='cosine', random_state=42)
    X_2d = tsne.fit_transform(all_embeddings)

    # Plot
    unique_labels = np.unique(all_labels)
    num_classes = len(unique_labels)
    cmap = plt.cm.get_cmap('nipy_spectral', num_classes)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=all_labels, cmap=cmap, alpha=0.7, s=10)

    cbar = plt.colorbar(scatter, ticks=unique_labels)
    cbar.set_label("Style Label Index")

    # Hide tick labels if too many classes
    if num_classes <= 20:
        cbar.set_ticks(unique_labels)
        if label_to_name_dict:
            label_names = [label_to_name_dict[idx] for idx in unique_labels]
            cbar.set_ticklabels(label_names)
        else:
            cbar.set_ticklabels([])

    title = f"t-SNE of Style Embeddings ({label}, Epoch {epoch})" if epoch else f"t-SNE of Style Embeddings ({label})"
    plt.title(title)
    plt.tight_layout()

    save_path = f"{result_dir}/{label}/tsne_epoch{epoch:03d}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"[t-SNE] Saved plot with {num_collected} samples → {save_path}")


if __name__ == "__main__":
    RUN_NAME = datetime.now().strftime("%Y%m%d_%H%M%S")  # e.g., '20250627_2012'
    RESULT_DIR = f"./result/{RUN_NAME}"
    CHECKPOINT_DIR = f"./checkpoints/style_encoder/{RUN_NAME}"

    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULT_DIR, "valid"), exist_ok=True)
    os.makedirs(os.path.join(RESULT_DIR, "test"), exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    set_seed(42)

    # --- Hyperparameters --- 
    BATCH_SIZE = 32
    SAMPLES_PER_CLASS = 4
    EMBED_DIM = 64
    HIDDEN_DIM = 128
    EPOCHS = 100
    LR = 1e-3
    TEMPERATURE = 0.07
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    TRAIN_JSON = './dataset/100style/100style_train_train.json'
    VAL_JSON   = './dataset/100style/100style_train_valid.json'
    TEST_JSON  = './dataset/100style/100style_test.json'
    MOTION_DIR = './dataset/100style/new_joint_vecs'
    MEAN = np.load('./checkpoints/t2m/Comp_v6_KLD005/meta/mean.npy')
    STD   = np.load('./checkpoints/t2m/Comp_v6_KLD005/meta/std.npy')
    WINDOW_SIZE = 64

    # --- Load datasets ---
    train_dataset = StyleDataset(TRAIN_JSON, MOTION_DIR, mean=MEAN, std=STD, window_size=WINDOW_SIZE)
    valid_dataset = StyleDataset(VAL_JSON, MOTION_DIR, mean=MEAN, std=STD, window_size=WINDOW_SIZE)
    test_dataset = StyleDataset(TEST_JSON, MOTION_DIR, mean=MEAN, std=STD, window_size=WINDOW_SIZE)

    # --- Samplers & Loaders ---
    train_sampler = StyleSampler(train_dataset, BATCH_SIZE, SAMPLES_PER_CLASS)
    train_loader  = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # --- Model & Optimizer ---
    denoiser_name = "t2m_denoiser_vpred_vaegelu"
    dataset_name = "t2m"
    t2m = Text2Motion(denoiser_name, dataset_name)
    model = StyleEncoder().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # --- Training Loop ---
    t2m.vae.freeze()
    latest_model_path = f"{CHECKPOINT_DIR}/style_encoder_latest.pt"
    checkpoint_dir = CHECKPOINT_DIR

    for epoch in range(1, EPOCHS + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0

        for motions, labels in tqdm(train_loader, desc=f"[Train] Epoch {epoch}"):
            motions = motions.to(DEVICE)
            labels = labels.to(DEVICE)

            with torch.no_grad():
                z, _ = t2m.vae.encode(motions)
                if torch.isnan(z).any() or torch.isinf(z).any():
                    print("Skipping batch due to NaN/Inf in z") 
                    continue
                
            out = model(z)
            loss = supervised_contrastive_loss(out, labels, TEMPERATURE)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            total_loss += loss_val

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}")

        # Save latest model always
        torch.save(model.state_dict(), latest_model_path)
        tsne_evaluate(model, t2m, valid_loader, DEVICE, epoch, label="valid", result_dir=RESULT_DIR, label_to_name_dict=valid_dataset.label_to_style)
        tsne_evaluate(model, t2m, test_loader, DEVICE, epoch, label="test", result_dir=RESULT_DIR, label_to_name_dict=test_dataset.label_to_style)

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            ckpt_path = f"{checkpoint_dir}/style_encoder_epoch{epoch:03d}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")