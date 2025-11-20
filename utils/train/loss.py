import torch
import torch.nn.functional as F


def loss_style(config, model, out):
    pred       = out["pred1"]
    latent     = out["latent1"]
    noise      = out["noise1"]
    timesteps  = out["timesteps"]
    velocity   = model.scheduler.get_velocity(latent, noise, timesteps).detach()
    return F.mse_loss(pred, velocity)


def loss_content(config, model, out):
    pred       = out["pred2"]
    latent     = out["latent2"]
    noise      = out["noise2"]
    timesteps  = out["timesteps"]
    velocity   = model.scheduler.get_velocity(latent, noise, timesteps).detach()
    return F.mse_loss(pred, velocity)


def loss_cycle(config, model, out):
    pred_cycle1 = out["pred_cycle1"]
    pred_cycle2 = out["pred_cycle2"]

    latent1 = out["latent1"]
    latent2 = out["latent2"]

    noise1 = out["noise1"]
    noise2 = out["noise2"]

    timesteps = out["timesteps"]

    velocity1 = model.scheduler.get_velocity(latent1, noise1, timesteps).detach()
    velocity2 = model.scheduler.get_velocity(latent2, noise2, timesteps).detach()
    return F.mse_loss(pred_cycle1 + pred_cycle2, velocity1 + velocity2)


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
    

# def loss_cycle(config, model, out, labels):
#     z_style = out["z_style"]     # [B, ...]
#     z_content = out["z_content"] # [B, ...]

#     # random permutation of styles
#     perm = torch.randperm(z_style.size(0), device=z_style.device)
#     z_fused = model.decode(z_style[perm].detach(), z_content)
#     out_fused = model.encoder.forward_from_latent(z_fused)

#     # cycle consistency
#     z_cycle = model.decode(z_style, out_fused["z_content"])

#     cycle_loss = F.mse_loss(z_cycle, out["z_latent"])

#     return cycle_loss


# def loss_cycle_v2(config, model, out, labels):
#     z_style = out["z_style"]     # [B, ...]
#     z_content = out["z_content"] # [B, ...]

#     # random permutation of styles
#     perm = torch.randperm(z_style.size(0), device=z_style.device)
#     z_fused = model.decode(z_style[perm].detach(), z_content)
#     out_fused = model.encoder.forward_from_latent(z_fused)

#     # cycle consistency
#     z_cycle = model.decode(z_style, out_fused["z_content"])
#     content_cycle_loss = F.mse_loss(z_cycle, out["z_latent"])

#     # root loss
#     pred_root = model.encoder.vae.decode(out["z_latent"])[..., :3]
#     pred_root_cycle = model.encoder.vae.decode(z_cycle)[..., :3]
#     root_cycle_loss = F.mse_loss(pred_root_cycle, pred_root)

#     return content_cycle_loss + root_cycle_loss


# def loss_cycle_v3(config, model, out, labels):
#     z_style = out["z_style"]     # [B, ...]
#     z_content = out["z_content"] # [B, ...]

#     # random permutation of styles
#     perm = torch.randperm(z_style.size(0), device=z_style.device)
#     z_fused = model.decode(z_style[perm], z_content)
#     out_fused = model.encoder.forward_from_latent(z_fused)

#     # cycle consistency
#     z_cycle = model.decode(z_style, out_fused["z_content"])

#     cycle_z_loss = F.mse_loss(z_cycle, out["z_latent"])
#     cycle_c_loss = F.mse_loss(out_fused["z_content"], z_content)
#     cycle_s_loss = F.mse_loss(out_fused["z_style"], z_style[perm])

#     return cycle_loss


LOSS_REGISTRY = {
    "style"   : loss_style,
    "content" : loss_content,
    "cycle"   : loss_cycle,
    "supcon"  : loss_supcon,
    # "supcon": loss_supcon,
    # "stylecon": loss_stylecon,
    # "anchor": loss_anchor,
    # "magnitude": loss_magnitude,
    # "l1": torch.nn.L1Loss(),
    # "l2": torch.nn.MSELoss(),
    # "smooth_l1": torch.nn.SmoothL1Loss(),
}
    
