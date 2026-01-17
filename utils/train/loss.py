import torch
import torch.nn.functional as F

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
    style_ref = out["style"].detach()
    x0_style = out["pred_x0_style"]
    # enc_req = [p.requires_grad for p in model.style_encoder.parameters()]
    # for p in model.style_encoder.parameters():
    #     p.requires_grad_(False)
    style_tokens_style = model.style_encoder(x0_style, out["len_mask1"])
    style_pred = model.pool_style(style_tokens_style, out["len_mask1"])
    loss = F.mse_loss(style_pred, style_ref)
    # for p, r in zip(model.style_encoder.parameters(), enc_req):
    #     p.requires_grad_(r)
    return loss

def loss_cycle_hml(config, model, out):
    style_ref = out["style"].detach()
    x0_hml = out["pred_x0_hml"]
    # enc_req = [p.requires_grad for p in model.style_encoder.parameters()]
    # for p in model.style_encoder.parameters():
    #     p.requires_grad_(False)
    style_tokens_hml = model.style_encoder(x0_hml, out["len_mask2"])
    style_pred = model.pool_style(style_tokens_hml, out["len_mask2"])
    loss = F.mse_loss(style_pred, style_ref)
    # for p, r in zip(model.style_encoder.parameters(), enc_req):
    #     p.requires_grad_(r)
    return loss


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
    
