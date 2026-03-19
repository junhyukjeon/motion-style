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

    style = F.normalize(out["style"], dim=1)
    style_idx = out["style_idx"]

    sim = torch.matmul(style, style.T) / temperature
    N = sim.size(0)

    logits_mask = ~torch.eye(N, dtype=torch.bool, device=style.device)

    pred_logits = sim.masked_fill(~logits_mask, float("-inf"))
    log_p = F.log_softmax(pred_logits, dim=1)

    target_logits = model.style_affinity[style_idx][:, style_idx] / temperature
    target_logits = target_logits.masked_fill(~logits_mask, float("-inf"))
    target = F.softmax(target_logits, dim=1)

    # zero out invalid entries after softmax/log_softmax to avoid 0 * -inf
    log_p = torch.where(logits_mask, log_p, torch.zeros_like(log_p))
    target = torch.where(logits_mask, target, torch.zeros_like(target))

    return -(target * log_p).sum(dim=1).mean()


LOSS_REGISTRY = {
    "style"   : loss_style,
    "supcon"  : loss_supcon,
    "soft_supcon": loss_soft_supcon,
}
    
