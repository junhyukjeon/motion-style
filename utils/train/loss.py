import torch
import torch.nn.functional as F

def _pool_style_tokens(style_tokens, len_mask):
    """
    style_tokens: [B, T, J, Ds]
    len_mask:     [B, T]  (True = valid)
    returns:      [B, Ds]
    """
    B, T, J, Ds = style_tokens.shape
    valid_tj = len_mask[:, :, None].expand(B, T, J)      # [B, T, J]
    w = valid_tj.float()[..., None]                      # [B, T, J, 1]
    num = (style_tokens * w).sum(dim=(1, 2))             # [B, Ds]
    den = w.sum(dim=(1, 2)).clamp_min(1e-5)              # [B, 1]
    return num / den

def loss_style(config, model, out):
    pred       = out["pred1"]
    latent     = out["latent1"]
    noise      = out["noise1"]
    timesteps  = out["timesteps"]
    velocity   = model.scheduler.get_velocity(latent, noise, timesteps).detach()
    return F.mse_loss(pred, velocity)

def loss_hml(config, model, out):
    pred       = out["pred2"]
    latent     = out["latent2"]
    noise      = out["noise2"]
    timesteps  = out["timesteps"]
    velocity   = model.scheduler.get_velocity(latent, noise, timesteps).detach()
    return F.mse_loss(pred, velocity)

def loss_cycle_style(config, model, out):
    """
    Cycle loss on x0_style (100STYLE branch):
      - anchor style: out["style"]       (pooled style from latent1 / 100STYLE)
      - stylized latent: out["pred_x0_style"]
      - re-encode x0_style and enforce style(x0_style) ≈ style_ref
    """
    # anchor / reference style from 100STYLE latent (do not backprop into this)
    style_ref = out["style"].detach()           # [B, Ds]

    # predicted clean latent for 100STYLE branch
    x0_style = out["pred_x0_style"]            # [B, T, J, D]
    B, T, J, D = x0_style.shape

    # derive a length mask from non-zero frames (x0_style already masked by len_mask1)
    with torch.no_grad():
        # [B, T]; True where any joint has non-zero magnitude
        len_mask = (x0_style.abs().sum(dim=(-1, -2)) > 0)

    # re-encode style tokens from x0_style
    # style_tokens_style: [B, T, J, Ds]
    style_tokens_style = model.style_encoder(x0_style, len_mask)

    # pooled style for x0_style
    style_pred = _pool_style_tokens(style_tokens_style, len_mask)  # [B, Ds]

    mode = config.get("style_cycle_mode", "cosine")
    if mode == "cosine":
        # 1 - cosine similarity (0 = perfect match)
        loss_per_sample = 1.0 - F.cosine_similarity(style_pred, style_ref, dim=-1)
        loss = loss_per_sample.mean()
    else:
        loss = F.mse_loss(style_pred, style_ref)

    w_cycle = config.get("lambda_cycle_x0_style", 1.0)
    return w_cycle * loss

def loss_cycle_hml(config, model, out):
    """
    Cycle loss on x0_hml (HumanML3D stylized branch):
      - anchor style: out["style"]       (pooled style from latent1 / 100STYLE)
      - stylized latent: out["pred_x0_hml"]
      - re-encode x0_hml and enforce style(x0_hml) ≈ style_ref
    """
    style_ref = out["style"].detach()          # [B, Ds]

    x0_hml = out["pred_x0_hml"]               # [B, T, J, D]
    B, T, J, D = x0_hml.shape

    with torch.no_grad():
        len_mask = (x0_hml.abs().sum(dim=(-1, -2)) > 0)  # [B, T]

    style_tokens_hml = model.style_encoder(x0_hml, len_mask)   # [B, T, J, Ds]
    style_pred = _pool_style_tokens(style_tokens_hml, len_mask)

    mode = config.get("style_cycle_mode", "cosine")
    if mode == "cosine":
        loss_per_sample = 1.0 - F.cosine_similarity(style_pred, style_ref, dim=-1)
        loss = loss_per_sample.mean()
    else:
        loss = F.mse_loss(style_pred, style_ref)

    w_cycle = config.get("lambda_cycle_x0_hml", 1.0)
    return w_cycle * loss


# def loss_content(config, model, out):
#     pred       = out["pred2"]
#     latent     = out["latent2"]
#     noise      = out["noise2"]
#     timesteps  = out["timesteps"]
#     velocity   = model.scheduler.get_velocity(latent, noise, timesteps).detach()
#     return F.mse_loss(pred, velocity)


# def loss_cycle(config, model, out):
#     pred_cycle1 = out["pred_cycle1"]
#     pred_cycle2 = out["pred_cycle2"]

#     latent1 = out["latent1"]
#     latent2 = out["latent2"]

#     noise1 = out["noise1"]
#     noise2 = out["noise2"]

#     timesteps = out["timesteps"]

#     velocity1 = model.scheduler.get_velocity(latent1, noise1, timesteps).detach()
#     velocity2 = model.scheduler.get_velocity(latent2, noise2, timesteps).detach()
#     return F.mse_loss(pred_cycle1 + pred_cycle2, velocity1 + velocity2)


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


LOSS_REGISTRY = {
    "style"   : loss_style,
    "hml"     : loss_hml,
    "cycle_style" : loss_cycle_style,
    "cycle_hml" : loss_cycle_hml,
    # "content" : loss_content,
    # "cycle"   : loss_cycle,
    "supcon"  : loss_supcon,
    # "stylecon": loss_stylecon,
    # "anchor": loss_anchor,
    # "magnitude": loss_magnitude,
    # "l1": torch.nn.L1Loss(),
    # "l2": torch.nn.MSELoss(),
    # "smooth_l1": torch.nn.SmoothL1Loss(),
}
    
