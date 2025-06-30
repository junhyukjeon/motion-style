import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE

from salad.t2m import Text2Motion
from style_dataset import StyleDataset

# --- Setup ---
BATCH_SIZE = 32
MAX_SAMPLES = 3000
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TEST_JSON = './dataset/100style/100style_test.json'
MOTION_DIR = './dataset/100style/new_joint_vecs'
MEAN = np.load('./checkpoints/t2m/Comp_v6_KLD005/meta/mean.npy')
STD  = np.load('./checkpoints/t2m/Comp_v6_KLD005/meta/std.npy')
WINDOW_SIZE = 64
RESULT_DIR = './result/vae_tsne'
os.makedirs(RESULT_DIR, exist_ok=True)

# --- Load dataset ---
test_dataset = StyleDataset(TEST_JSON, MOTION_DIR, mean=MEAN, std=STD, window_size=WINDOW_SIZE)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# --- Load VAE ---
denoiser_name = "t2m_denoiser_vpred_vaegelu"
dataset_name = "t2m"
t2m = Text2Motion(denoiser_name, dataset_name)
t2m.vae.freeze()
# t2m = t2m.to(DEVICE)

# --- Collect VAE Latents ---
all_embeddings = []
all_labels = []
num_collected = 0

with torch.no_grad():
    pbar = tqdm(total=MAX_SAMPLES, desc="Extracting VAE latents")
    for motions, labels in test_loader:
        motions = motions.to(DEVICE)
        z, _ = t2m.vae.encode(motions)  # (B, T, J, D)
        pooled = z.mean(dim=(1, 2))     # (B, D)

        if torch.isnan(pooled).any() or torch.isinf(pooled).any():
            print("⚠️ Skipping batch with invalid embeddings")
            continue

        pooled = pooled.cpu()
        labels = labels.cpu()

        if num_collected + len(pooled) > MAX_SAMPLES:
            needed = MAX_SAMPLES - num_collected
            pooled = pooled[:needed]
            labels = labels[:needed]

        all_embeddings.append(pooled)
        all_labels.append(labels)
        num_collected += len(pooled)
        pbar.update(len(pooled))
        if num_collected >= MAX_SAMPLES:
            break
    pbar.close()

# --- Run t-SNE ---
all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
all_labels = torch.cat(all_labels, dim=0).numpy()

tsne = TSNE(n_components=2, perplexity=30, metric='cosine', random_state=42)
X_2d = tsne.fit_transform(all_embeddings)

# --- Plot ---
unique_labels = np.unique(all_labels)
num_classes = len(unique_labels)
cmap = plt.cm.get_cmap('nipy_spectral', num_classes)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=all_labels, cmap=cmap, alpha=0.7, s=10)
cbar = plt.colorbar(scatter, ticks=unique_labels)
cbar.set_label("Style Label Index")

if num_classes <= 20:
    cbar.set_ticks(unique_labels)
    try:
        label_names = [test_dataset.label_to_style[idx] for idx in unique_labels]
        cbar.set_ticklabels(label_names)
    except:
        pass

plt.title("t-SNE of Raw VAE Latents (test set)")
plt.tight_layout()
save_path = os.path.join(RESULT_DIR, "tsne_vae_latents.png")
plt.savefig(save_path)
plt.close()
print(f"[t-SNE VAE] Saved plot to {save_path}")
