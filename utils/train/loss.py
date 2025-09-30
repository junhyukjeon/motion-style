import torch
import torch.nn.functional as F


def loss_velocity(config, model, out):
    pred       = out["pred"]
    latent     = out["latent"]
    noise      = out["noise"]
    timesteps  = out["timesteps"]
    velocity   = model.scheduler.get_velocity(latent, noise, timesteps)
    return F.mse_loss(pred, velocity)


<<<<<<< HEAD
def loss_supcon(config, model, out):
=======
    return F.mse_loss(z_recon, z_latent)


def loss_recon_v2(config, model, out, labels):
    z_latent = out["z_latent"]
    z_style = out["z_style"]
    z_content = out["z_content"]

    z_recon = model.decode(z_style, z_content)

    return F.mse_loss(z_recon, z_latent)


def loss_motion_recon(config, model, out, labels):
    motion = out["motion"]
    z_style = out["z_style"]
    z_content = out["z_content"]

    pred_motion = model.decode(z_style, z_content)

    return F.mse_loss(pred_motion, motion)


def loss_supcon(config, model, out, labels):
>>>>>>> a259b41980b31d02bb0dee1bdc08f3f7b7844c9e
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

<<<<<<< HEAD
    # Log probability
    log_prob = sim_stable - torch.log(denom)
    mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8)
    return -mean_log_prob_pos.mean()
=======
def loss_supcon_v2(config, model, out, labels):
    temperature = config['temperature']
    anchor = config['anchor']
    
    z = out['z_style']

    if z.dim() == 4:                 # [B, T, J, D]
        z = z.mean(dim=(1, 2))

    # Euclidean distance between samples
    dist_sq = torch.cdist(z, z, p=2).pow(2)
    dist_sq = dist_sq / temperature
    sim = -dist_sq
    N = sim.size(0)
    logits_mask = ~torch.eye(N, dtype=torch.bool, device=z.device)

    # Positive pair mask
    labels = labels.unsqueeze(0)
    pos_mask = (labels == labels.T) & logits_mask
    pos_mask = pos_mask.float()

    # Log-softmax
    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8)

    return -mean_log_prob_pos.mean()


def loss_supcon_v3(config, model, out, labels):
    temperature = config['temperature']
    anchor = config['anchor']
    
    z_style = out['z_style']
    loss = 0.0

    for z in z_style:
        if z.dim() == 4:                 # [B, T, J, D]
            z = z.mean(dim=(1, 2)) 

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
        
        loss += -mean_log_prob_pos.mean()

    return loss



def loss_stylecon(config, model, out, labels=None):
    z_style   = out['z_style']     # [B, ...]
    z_content = out['z_content']   # [B, ...]
    B = z_style.shape[0]
    perm = torch.randperm(B, device=z_style.device)
>>>>>>> a259b41980b31d02bb0dee1bdc08f3f7b7844c9e


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

<<<<<<< HEAD

# def loss_anchor(config, model, out, labels):
#     z = out["z_prime"]
#     if z.dim() == 4:
#         z = z.mean(dim=(1, 2)) 
#     return F.mse_loss(z[:8], torch.zeros_like(z[:8]))
=======
def loss_stylecon_v2(config, model, out, labels=None):
    z_style   = out['z_style'].detach()     # [B, ...]
    z_content = out['z_content'].detach()   # [B, ...]
    B = z_style.shape[0]
    perm = torch.randperm(B, device=z_style.device)

    z_cycle = model.decode(z_style, z_content)
    z_fused = model.decode(z_style[perm], z_content)

    out_cycle = model.encoder.forward_from_latent(z_cycle)
    out_fused = model.encoder.forward_from_latent(z_fused)

    style_cycle_loss = F.mse_loss(out_cycle['z_style'], z_style)
    style_fused_loss = F.mse_loss(out_fused['z_style'], z_style[perm])

    content_cycle_loss = F.mse_loss(out_cycle['z_content'], z_content)
    content_fused_loss = F.mse_loss(out_fused['z_content'], z_content)

    style_loss    = style_cycle_loss + style_fused_loss
    content_loss  = content_cycle_loss + content_fused_loss

    return style_loss, content_loss


def loss_stylecon_v3(config, model, out, labels=None):
    z_A = out['z_latent']
    s_A = out['z_style']     # [B, ...]
    c_A = out['z_content']   # [B, ...]
    B = s_A.shape[0]
    perm = torch.randperm(B, device=s_A.device)

    # apply permutation
    z_B = z_A[perm]
    s_B = s_A[perm]
    c_B = c_A[perm]

    # cycle generation using style mixture
    z_AB = model.decode(s_B, c_A)
    out_AB = model.encoder.forward_from_latent(z_AB)
    s_AB = out_AB['z_style']
    c_AB = out_AB['z_content']

    # cycle loss
    z_ABA = model.decode(s_A, c_AB)
    z_ABB = model.decode(s_AB, c_B)

    content_loss = F.mse_loss(z_ABA, z_A)
    style_loss = F.mse_loss(z_ABB, z_B)

    return style_loss, content_loss


def loss_stylecon_v4(config, model, out, labels=None):
    m_A = out['motion']
    s_A = out['z_style']     # N x [B, ...]
    c_A = out['z_content']   # N x [B, ...]
    B = m_A.shape[0]
    
    perm = torch.randperm(B).tolist()

    # apply permutation
    m_B = m_A[perm]
    # s_B = [ s_A[i] for i in perm ]
    # c_B = [ c_A[i] for i in perm ]
    s_B = s_A.copy()
    c_B = c_A.copy()
    for i in range(len(s_A)):
        s_B[i] = s_B[i][perm]
        c_B[i] = c_B[i][perm]

    # cycle generation using style mixture
    m_AB = model.decode(s_B, c_A)
    out_AB = model.encode(m_AB)
    s_AB = out_AB['z_style']
    c_AB = out_AB['z_content']

    # cycle loss
    m_ABA = model.decode(s_A, c_AB)
    m_ABB = model.decode(s_AB, c_B)

    content_loss = F.mse_loss(m_ABA, m_A)
    style_loss = F.mse_loss(m_ABB, m_B)

    return style_loss, content_loss


def loss_anchor(config, model, out, labels):
    z_style = out["z_style"]
    return F.mse_loss(z_style[:4], torch.zeros_like(z_style[:4]))

def loss_anchor_v2(config, model, out, labels):
    z_style = out["z_style"]
    z_style_norm = torch.norm(z_style, dim=-1)
    
    neutral_idx = (labels == 51)

    neutral_loss = F.mse_loss(z_style_norm[neutral_idx], torch.zeros_like(z_style_norm[neutral_idx]))
    non_neutral_loss = F.mse_loss(z_style_norm[~neutral_idx], torch.ones_like(z_style_norm[~neutral_idx]))
    return neutral_loss + non_neutral_loss

>>>>>>> a259b41980b31d02bb0dee1bdc08f3f7b7844c9e


# def loss_magnitude(config, model, out, labels):
#     z = out["z_style"]
#     z = z[8:]
#     norms = z.norm(dim=-1)
#     return (norms - 1.0).pow(2).mean()
    

def loss_cycle(config, model, out, labels):
    z_style = out["z_style"]     # [B, ...]
    z_content = out["z_content"] # [B, ...]

    # random permutation of styles
    perm = torch.randperm(z_style.size(0), device=z_style.device)
    z_fused = model.decode(z_style[perm].detach(), z_content)
    out_fused = model.encoder.forward_from_latent(z_fused)

    # cycle consistency
    z_cycle = model.decode(z_style, out_fused["z_content"])

    cycle_loss = F.mse_loss(z_cycle, out["z_latent"])

    return cycle_loss


def loss_cycle_v2(config, model, out, labels):
    z_style = out["z_style"]     # [B, ...]
    z_content = out["z_content"] # [B, ...]

    # random permutation of styles
    perm = torch.randperm(z_style.size(0), device=z_style.device)
    z_fused = model.decode(z_style[perm].detach(), z_content)
    out_fused = model.encoder.forward_from_latent(z_fused)

    # cycle consistency
    z_cycle = model.decode(z_style, out_fused["z_content"])
    content_cycle_loss = F.mse_loss(z_cycle, out["z_latent"])

    # root loss
    pred_root = model.encoder.vae.decode(out["z_latent"])[..., :3]
    pred_root_cycle = model.encoder.vae.decode(z_cycle)[..., :3]
    root_cycle_loss = F.mse_loss(pred_root_cycle, pred_root)

    return content_cycle_loss + root_cycle_loss


def loss_cycle_v3(config, model, out, labels):
    z_style = out["z_style"]     # [B, ...]
    z_content = out["z_content"] # [B, ...]

    # random permutation of styles
    perm = torch.randperm(z_style.size(0), device=z_style.device)
    z_fused = model.decode(z_style[perm], z_content)
    out_fused = model.encoder.forward_from_latent(z_fused)

    # cycle consistency
    z_cycle = model.decode(z_style, out_fused["z_content"])

    cycle_z_loss = F.mse_loss(z_cycle, out["z_latent"])
    cycle_c_loss = F.mse_loss(out_fused["z_content"], z_content)
    cycle_s_loss = F.mse_loss(out_fused["z_style"], z_style[perm])

    return cycle_loss


LOSS_REGISTRY = {
<<<<<<< HEAD
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
    
=======
    "recon": loss_recon,
    "recon_v2": loss_recon_v2,
    "motion_recon": loss_motion_recon,

    "supcon": loss_supcon,
    "supcon_v2": loss_supcon_v2,
    "supcon_v3": loss_supcon_v3,

    "stylecon": loss_stylecon,
    "stylecon_v2": loss_stylecon_v2,
    "stylecon_v3": loss_stylecon_v3,
    "stylecon_v4": loss_stylecon_v4,

    "anchor": loss_anchor,
    "anchor_v2": loss_anchor_v2,

    "orthogonality": loss_orthogonality,

    "cycle": loss_cycle,
    "cycle_v2": loss_cycle_v2,
    "cycle_v3": loss_cycle_v3,
}
>>>>>>> a259b41980b31d02bb0dee1bdc08f3f7b7844c9e
