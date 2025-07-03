import torch
import torch.nn.functional as F

def supervised_contrastive_loss(z, labels, temperature=0.07):
    """
    Computes the Supervised Contrastive Loss as introduced in:
    "Supervised Contrastive Learning" (Khosla et al., NeurIPS 2020)

    Args:
        z (Tensor): Embeddings of shape (B, D) where B is the batch size and D is the embedding dimension.
        labels (Tensor): Ground truth labels of shape (B,) as integers.
        temperature (float): Temperature scaling factor for the softmax.

    Returns:
        loss (Tensor): Scalar contrastive loss.
    """
    # Normalize embeddings to unit vectors
    z = F.normalize(z, dim=1)

    # Compute cosine similarity matrix between all pairs: (B, B)
    sim = torch.matmul(z, z.T) / temperature

    # Mask to exclude self-comparisons (diagonal)
    N = sim.size(0)
    logits_mask = ~torch.eye(N, dtype=torch.bool, device=z.device)

    # Create binary mask for positive pairs: (i, j) if labels[i] = labels[j] and i != j
    labels = labels.unsqueeze(0)  # (1, B)
    pos_mask = (labels == labels.T) & logits_mask  # (B, B)

    # Compute log-softmax denominator (only valid comparisons)
    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    # Average log-probabilities over positive pairs
    mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8)

    # Return mean loss
    loss = -mean_log_prob_pos.mean()
    return loss