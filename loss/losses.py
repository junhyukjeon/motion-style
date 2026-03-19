import torch
import torch.nn.functional as F


def loss_style(config, model, out):
    pred       = out["pred"]
    latent     = out["latent"]
    noise      = out["noise"]
    timesteps  = out["timesteps"]
    velocity   = model.scheduler.get_velocity(latent, noise, timesteps).detach()
    return F.mse_loss(pred, velocity)


def loss_supcon(config, model, out):
    temperature = config['temperature']
    style       = out['style']
    style_idx   = out['style_idx']
    len_mask  = out.get('len_mask', None)

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


def loss_soft_supcon(config, model, out):
    temperature = config["temperature"]
    style = out["style"]
    style_idx = out["style_idx"]

    style = F.normalize(style, dim=1)

    sim = torch.matmul(style, style.T) / temperature
    N = sim.size(0)

    logits_mask = ~torch.eye(N, dtype=torch.bool, device=style.device)
    sim_stable = sim - sim.max(dim=1, keepdim=True).values

    weight = model.style_affinity[style_idx][:, style_idx]
    weight = weight * logits_mask

    same = (style_idx.view(1, -1) == style_idx.view(-1, 1)) & logits_mask
    weight = torch.where(same, torch.ones_like(weight), weight)

    weight = weight / (weight.sum(dim=1, keepdim=True) + 1e-8)

    exp_sim = torch.exp(sim_stable) * logits_mask
    denom = exp_sim.sum(dim=1, keepdim=True) + 1e-8

    log_prob = sim_stable - torch.log(denom)
    mean_log_prob = (weight * log_prob).sum(dim=1)

    return -mean_log_prob.mean()


LOSS_REGISTRY = {
    "style"   : loss_style,
    "supcon"  : loss_supcon,
    "soft_supcon": loss_soft_supcon,
}
    
