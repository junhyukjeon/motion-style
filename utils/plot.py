
import os
import colorsys

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import to_rgb
from sklearn.manifold import TSNE
from tqdm import tqdm


def soften_rgb(rgb, sat_scale=0.85, light_shift=0.01):
    """
    Gentle softening for Glasbey:
      - sat_scale < 1 reduces neon slightly
      - light_shift > 0 lifts very dark colors a touch
    """
    r, g, b = rgb
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    s = np.clip(s * sat_scale, 0.0, 1.0)
    l = np.clip(l + light_shift, 0.0, 1.0)
    return colorsys.hls_to_rgb(h, l, s)


def style_3d_axes(ax):
    ax.set_facecolor("white")
    ax.grid(True)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["color"] = (0, 0, 0, 0.25)
        axis._axinfo["grid"]["linewidth"] = 0.5
        axis._axinfo["grid"]["linestyle"] = "-"

    # Transparent panes
    try:
        ax.xaxis.pane.set_facecolor((1, 1, 1, 0))
        ax.yaxis.pane.set_facecolor((1, 1, 1, 0))
        ax.zaxis.pane.set_facecolor((1, 1, 1, 0))

        ax.xaxis.pane.set_edgecolor((0, 0, 0, 0.25))
        ax.yaxis.pane.set_edgecolor((0, 0, 0, 0.25))
        ax.zaxis.pane.set_edgecolor((0, 0, 0, 0.25))
    except Exception:
        pass

    ax.tick_params(labelsize=8, width=0.8, length=3)


def make_style_colors_glasbey_first_k(unique_labels, label_to_name_dict=None,
                                     neutral_name="neutral",
                                     sat_scale=0.85, light_shift=0.01):
    """
    Matches your previous behavior: pick the first K colors from cc.glasbey,
    but optionally soften them.

    For 25 fixed styles, this will be consistent across runs as long as labels are consistent.
    """
    K = len(unique_labels)

    # First K from glasbey (old behavior)
    base = [to_rgb(c) for c in cc.glasbey[:K]]

    # Soften (optional)
    base = [soften_rgb(c, sat_scale=sat_scale, light_shift=light_shift) for c in base]

    style_colors = {}
    for i, label in enumerate(unique_labels):
        name = (label_to_name_dict.get(label, str(label)) if label_to_name_dict else str(label)).lower()
        if name == neutral_name:
            style_colors[label] = (0, 0, 0)
        else:
            style_colors[label] = base[i]

    return style_colors

def set_axes_equal(ax, X):
    """
    Make 3D axes have equal scale so the grid looks like a cube.
    """
    x_limits = (X[:, 0].min(), X[:, 0].max())
    y_limits = (X[:, 1].min(), X[:, 1].max())
    z_limits = (X[:, 2].min(), X[:, 2].max())

    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]

    max_range = max(x_range, y_range, z_range)

    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)

    ax.set_xlim(x_mid - max_range / 2, x_mid + max_range / 2)
    ax.set_ylim(y_mid - max_range / 2, y_mid + max_range / 2)
    ax.set_zlim(z_mid - max_range / 2, z_mid + max_range / 2)

def plot_tsne(model, loader, device, epoch=None, title="valid", result_dir="",
              label_to_name_dict=None, max_samples=3000, writer=None,
              sat_scale=0.85, light_shift=0.01):
    model.eval()
    all_embeddings, all_labels = [], []

    save_dir = os.path.join(result_dir, title)
    os.makedirs(save_dir, exist_ok=True)

    collected = 0
    pbar = tqdm(total=max_samples, desc=f"[t-SNE 3D] Extracting ({title})")
    with torch.no_grad():
        for batch in loader:
            style, style_idx = model.style(batch)

            remaining = max_samples - collected
            if remaining <= 0:
                break

            style = style[:remaining]
            style_idx = style_idx[:remaining]

            all_embeddings.append(style.detach().cpu())
            all_labels.append(style_idx.detach().cpu())

            collected += len(style)
            pbar.update(len(style))

    pbar.close()

    if not all_labels:
        print("❌ No valid samples collected for t-SNE!")
        return

    embeddings_np = torch.cat(all_embeddings).float().numpy()
    labels_np = torch.cat(all_labels).numpy()

    print(f"[t-SNE] Running 3D t-SNE on {len(embeddings_np)} samples...")
    X = TSNE(
        n_components=3,
        perplexity=30,
        metric="cosine",
        random_state=42
    ).fit_transform(embeddings_np)

    unique_labels = np.sort(np.unique(labels_np))
    num_classes = len(unique_labels)

    # --- Glasbey (first K) + softening ---
    style_colors = make_style_colors_glasbey_first_k(
        unique_labels,
        label_to_name_dict=label_to_name_dict,
        sat_scale=sat_scale,
        light_shift=light_shift,
    )

    # --- Plot ---
    fig = plt.figure(figsize=(6.0, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    style_3d_axes(ax)
    ax.view_init(elev=18, azim=35)

    for label in unique_labels:
        idx = labels_np == label
        name = label_to_name_dict.get(label, str(label)) if label_to_name_dict else str(label)

        ax.scatter(
            X[idx, 0], X[idx, 1], X[idx, 2],
            label=name,
            s=10,
            alpha=0.75,
            color=style_colors[label],
            edgecolors="none",
            rasterized=True
        )

    # ax.set_title(f"t-SNE (3D) of $z_{{style}}$ ({title})", fontsize=11, pad=10)
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass
    set_axes_equal(ax, X)
    

    # For 25 styles, legend usually too big; keep disabled by default.
    # If you want it anyway, change 15 -> 25 or remove the if.
    if num_classes <= 15:
        handles, legend_labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                handles, legend_labels,
                frameon=False,
                fontsize=8,
                markerscale=1.6,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
            )

    png_path = os.path.join(save_dir, f"tsne3d_epoch{epoch:03d}.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"[t-SNE] Saved → {png_path}")

    if writer is not None:
        writer.add_figure(f"{title.capitalize()}/t-SNE-3D", fig, global_step=epoch)

    plt.close(fig)


def plot_tsne_2d(
    model,
    loader,
    device,
    epoch=None,
    title="valid",
    result_dir="",
    label_to_name_dict=None,
    max_samples=3000,
    writer=None,
    sat_scale=0.85,
    light_shift=0.01,
    # plot tuning
    figsize=(6.0, 6.0),
    dpi=300,
    s=5,
    alpha=0.70,
    grid_alpha=0.25,
    grid_lw=0.5,
    limit_pad=0.02,
    show_ticks=False,
    show_legend=False,
):
    model.eval()
    all_embeddings, all_labels = [], []

    save_dir = os.path.join(result_dir, title)
    os.makedirs(save_dir, exist_ok=True)

    collected = 0
    pbar = tqdm(total=max_samples, desc=f"[t-SNE 2D] Extracting ({title})")
    with torch.no_grad():
        for batch in loader:
            style, style_idx = model.style(batch)

            remaining = max_samples - collected
            if remaining <= 0:
                break

            style = style[:remaining]
            style_idx = style_idx[:remaining]

            all_embeddings.append(style.detach().cpu())
            all_labels.append(style_idx.detach().cpu())

            collected += len(style)
            pbar.update(len(style))
    pbar.close()

    if not all_labels:
        print("❌ No valid samples collected for t-SNE!")
        return

    embeddings_np = torch.cat(all_embeddings).float().numpy()
    labels_np = torch.cat(all_labels).numpy()

    print(f"[t-SNE] Running 2D t-SNE on {len(embeddings_np)} samples...")
    X = TSNE(
        n_components=2,
        perplexity=30,
        metric="cosine",
        random_state=42
    ).fit_transform(embeddings_np)

    unique_labels = np.sort(np.unique(labels_np))
    num_classes = len(unique_labels)

    # Same palette logic as 3D
    style_colors = make_style_colors_glasbey_first_k(
        unique_labels,
        label_to_name_dict=label_to_name_dict,
        sat_scale=sat_scale,
        light_shift=light_shift,
    )

    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("white")

    # faint black grid (paper-friendly)
    ax.grid(True, linewidth=grid_lw, color=(0, 0, 0, grid_alpha))

    # tight limits (reduces whitespace)
    xmin, ymin = X.min(axis=0)
    xmax, ymax = X.max(axis=0)
    dx = max(xmax - xmin, 1e-9)
    dy = max(ymax - ymin, 1e-9)
    ax.set_xlim(xmin - limit_pad * dx, xmax + limit_pad * dx)
    ax.set_ylim(ymin - limit_pad * dy, ymax + limit_pad * dy)
    ax.margins(0)

    for label in unique_labels:
        idx = labels_np == label
        name = label_to_name_dict.get(label, str(label)) if label_to_name_dict else str(label)

        ax.scatter(
            X[idx, 0], X[idx, 1],
            label=name,
            s=s,
            alpha=alpha,
            color=style_colors[label],
            edgecolors="none",
            rasterized=True
        )

    # optional ticks (usually off for subpanels)
    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])

    # legend is usually too big for 25 styles
    if show_legend and num_classes <= 15:
        handles, legend_labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, legend_labels, frameon=False, fontsize=8)

    png_path = os.path.join(save_dir, f"tsne2d_epoch{epoch:03d}.png")
    plt.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.01)
    print(f"[t-SNE] Saved → {png_path}")

    if writer is not None:
        writer.add_figure(f"{title.capitalize()}/t-SNE-2D", fig, global_step=epoch)

    plt.close(fig)
