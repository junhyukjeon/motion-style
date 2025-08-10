import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from matplotlib.colors import to_rgb
from sklearn.manifold import TSNE
from tqdm import tqdm

def plot_tsne(model, loader, device, epoch=None, title="valid", result_dir="", label_to_name_dict=None, max_samples=3000, writer=None):
    model.eval()
    all_embeddings, all_labels = [], []

    save_dir = os.path.join(result_dir, title)
    os.makedirs(save_dir, exist_ok=True)

    pbar = tqdm(total=max_samples, desc=f"[t-SNE 3D] Extracting ({title})")
    with torch.no_grad():
        for motions, labels, _, _ in loader:
            motions = motions.to(device)
            labels = labels.to(device)

            out = model.encode(motions)
            z_style = out["z_style"].cpu()
            labels = labels.cpu()

            remaining = max_samples - sum(len(l) for l in all_labels)
            if remaining <= 0:
                break

            z_style = z_style[:remaining]
            labels = labels[:remaining]

            all_embeddings.append(z_style)
            all_labels.append(labels)
            pbar.update(len(z_style))

    pbar.close()

    if not all_labels:
        print("❌ No valid samples collected for t-SNE!")
        return

    embeddings_np = torch.cat(all_embeddings).numpy()
    labels_np = torch.cat(all_labels).numpy()

    print(f"[t-SNE] Running 3D t-SNE on {len(embeddings_np)} samples...")
    X = TSNE(n_components=3, perplexity=30, metric='cosine', random_state=42).fit_transform(embeddings_np)

    unique_labels = np.sort(np.unique(labels_np))
    num_classes = len(unique_labels)

    # === Set up colors ===
    style_colors = {}
    use_glasbey = num_classes > 20
    if use_glasbey:
        glasbey_colors = [to_rgb(c) for c in cc.glasbey]
    else:
        cmap = plt.get_cmap('tab20')

    for i, label in enumerate(unique_labels):
        name = label_to_name_dict.get(label, str(label)).lower() if label_to_name_dict else str(label).lower()
        if name == "neutral":
            style_colors[label] = "black"
        else:
            if use_glasbey:
                style_colors[label] = glasbey_colors[i % len(glasbey_colors)]
            else:
                style_colors[label] = cmap(i % cmap.N)

    # === Plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    for label in unique_labels:
        idx = labels_np == label
        color = style_colors[label]
        name = label_to_name_dict.get(label, str(label)) if label_to_name_dict else str(label)
        ax.scatter(X[idx, 0], X[idx, 1], X[idx, 2], label=name, s=10, alpha=0.8, color=color)

    ax.set_title(f"3D t-SNE of z_style ({title})")

    if num_classes <= 20:
        handles, legend_labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, legend_labels, markerscale=2, loc='upper right', bbox_to_anchor=(1.3, 1))

    save_path = os.path.join(save_dir, f"tsne3d_epoch{epoch:03d}.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"[t-SNE] Saved → {save_path}")

    if writer is not None:
        writer.add_figure(f"{title.capitalize()}/t-SNE-3D", fig, global_step=epoch)

    plt.close(fig)