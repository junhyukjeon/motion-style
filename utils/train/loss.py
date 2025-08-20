import torch
import torch.nn.functional as F


def loss_recon(config, model, out, labels):
    z_latent = out["z_latent"]
    z_style = out["z_style"]
    z_content = out["z_content"]
    mem = out["mem"]
    z_recon = model.decode(z_style, z_content, mem)

    return F.mse_loss(z_recon, z_latent)


def loss_supcon(config, model, out, labels):
    temperature = config['temperature']
    anchor = config['anchor']
    
    z = out['z_style']
    if anchor:
        z = z[4:]
        labels = labels[4:]

    # Normalize so magnitude of z is not penalized
    z = F.normalize(z, dim=1)

    # Cosine similarity between samples
    sim = torch.matmul(z, z.T) / temperature
    N = sim.size(0)
    logits_mask = ~torch.eye(N, dtype=torch.bool, device=z.device)

    # Positive pair mask
    labels = labels.unsqueeze(0)
    pos_mask = (labels == labels.T) & logits_mask

    # Log-softmax
    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
    mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8)

    return -mean_log_prob_pos.mean()


def loss_stylecon(config, model, out, labels):
    z_style = out['z_style']
    z_content = out['z_content']
    mem = out['mem']
    # gamma = out['gamma']
    # beta = out['beta']
    
    B = z_style.shape[0]
    perm = torch.randperm(B, device=z_style.device)
    
    # ---- Style consistency ----
    # swap style; detach content so style loss doesn't push content encoder
    z_fused_style = model.decode(z_style[perm], z_content.detach(), mem, detach_mem=True)
    new_style     = model.encoder.forward_from_latent(z_fused_style)
    style_loss    = F.mse_loss(new_style['z_style'], z_style[perm].detach())

    # ---- Content consistency ----
    # keep content; detach style so content loss doesn't push style encoder
    z_fused_content = model.decode(z_style.detach(), z_content, mem, detach_mem=True)
    new_content     = model.encoder.forward_from_latent(z_fused_content)
    content_loss    = F.mse_loss(new_content['z_content'], z_content.detach())

    return style_loss, content_loss


def loss_anchor(config, model, out, labels):
    z_style = out["z_style"]
    return F.mse_loss(z_style[:4], torch.zeros_like(z_style[:4]))


def loss_orthogonality(config, model, out, labels):
    z_style = out["z_style"]
    z_content = out["z_content"]
    z_s = F.normalize(z_style, dim=-1)
    z_c = F.normalize(z_content, dim=-1)
    return (z_s * z_c).sum(dim=-1).pow(2).mean()


LOSS_REGISTRY = {
    "recon": loss_recon,
    "supcon": loss_supcon,
    "stylecon": loss_stylecon,
    "anchor": loss_anchor,
    "orthogonality": loss_orthogonality
}
    