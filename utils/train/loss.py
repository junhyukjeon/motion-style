import torch
import torch.nn.functional as F


def loss_velocity(config, model, out):
    pred       = out["pred"]
    latent     = out["latent"]
    noise      = out["noise"]
    timesteps  = out["timesteps"]
    velocity   = model.scheduler.get_velocity(latent, noise, timesteps)
    return F.mse_loss(pred, velocity)


def loss_supcon(config, model, out):
    temperature = config['temperature']
    style       = out['style']
    style_idx   = out['style_idx']

    # Normalize so magnitude of z is not penalized
    style = F.normalize(style, dim=1)

    # Cosine similarity between samples
    sim = torch.matmul(style, style.T) / temperature
    N = sim.size(0)

    # Exclude self-comparison
    logits_mask = ~torch.eye(N, dtype=torch.bool, device=style.device)

    # Numerical stability for softmax
    sim_stable = sim - sim.max(dim=1, keepdim=True).values

    # Positive pair mask
    style_idx = style_idx.view(1, -1)
    pos_mask = (style_idx == style_idx.T) & logits_mask

    # Denominator over non-self entries
    exp_sim = torch.exp(sim_stable) * logits_mask
    denom = exp_sim.sum(dim=1, keepdim=True) + 1e-8

    # Log probability
    log_prob = sim_stable - torch.log(denom)
    mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8)
    return -mean_log_prob_pos.mean()


# def loss_stylecon(config, model, out, labels=None):
#     z_style   = out['z_style']
#     z_content = out['z_content']
#     B = z_style.shape[0]
#     perm = torch.randperm(B, device=z_style.device)

#     z_fused_s = model.decode(z_style[perm], z_content.detach())
#     z_fused_c = model.decode(z_style[perm].detach(), z_content)

#     new_s = model.encoder.forward_from_latent(z_fused_s)
#     new_c = model.encoder.forward_from_latent(z_fused_c)

#     style_loss    = F.mse_loss(new_s['z_style'], z_style[perm].detach())
#     content_loss  = F.mse_loss(new_c['z_content'], z_content.detach()) 
#     return style_loss, content_loss


# def loss_anchor(config, model, out, labels):
#     z = out["z_prime"]
#     if z.dim() == 4:
#         z = z.mean(dim=(1, 2)) 
#     return F.mse_loss(z[:8], torch.zeros_like(z[:8]))


# def loss_magnitude(config, model, out, labels):
#     z = out["z_style"]
#     z = z[8:]
#     norms = z.norm(dim=-1)
#     return (norms - 1.0).pow(2).mean()
    

LOSS_REGISTRY = {
    "velocity": loss_velocity,
    "supcon": loss_supcon,
    # "supcon": loss_supcon,
    # "stylecon": loss_stylecon,
    # "anchor": loss_anchor,
    # "magnitude": loss_magnitude,
    # "l1": torch.nn.L1Loss(),
    # "l2": torch.nn.MSELoss(),
    # "smooth_l1": torch.nn.SmoothL1Loss(),
}
    