import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_tsne(embeddings, labels=None, save_path=None, title="t-SNE", perplexity=30):
    tsne = TSNE(n_components=2, perplexity=perplexity, metric='cosine', random_state=42)
    X_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    if labels is not None:
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='nipy_spectral', s=10, alpha=0.7)
        plt.colorbar(scatter)
    else:
        plt.scatter(X_2d[:, 0], X_2d[:, 1], s=10, alpha=0.7)

    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"t-SNE saved to {save_path}")
    else:
        plt.show()