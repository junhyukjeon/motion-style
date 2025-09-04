# --- Imports ---
import torch
import torch.nn as nn

from diffusers import DDIMScheduler
from data.dataset import StyleDataset
from data.sampler import SAMPLER_REGISTRY
from model.denoiser import Denoiser
from model.networks import StyleContentNet
from salad.models.vae.model import VAE
from salad.utils.get_opt import get_opt
from utils.train.loss import LOSS_REGISTRY


def load_vae(vae_opt):
    print(f'Loading VAE Model {vae_opt.name}')
    model = VAE(vae_opt)
    ckpt = torch.load(pjoin(vae_opt.checkpoints_dir, vae_opt.dataset_name, vae_opt.name, 'model', 'net_best_fid.tar'),
                            map_location='cpu')
    model.load_state_dict(ckpt["vae"])
    model.freeze()
    return model


def load_denoiser(opt, vae_dim):
    print(f'Loading Denoiser Model {opt.name}')
    denoiser = Denoiser(opt, vae_dim)
    ckpt = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'model', 'net_best_fid.tar'),
                            map_location='cpu')
    missing_keys, unexpected_keys = denoiser.load_state_dict(ckpt["denoiser"], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    return denoiser


class Text2StylizedMotion(nn.Module):
    def __init__(self, config, opt, vae_dim):
        super(Text2StylizedMotion, self).__init__()
        self.device  = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.config  = config
        self.opt     = get_opt(f"checkpoints/{dataset_name}/{denoiser_name}/opt.txt", self.device)
        self.vae_opt = get_opt(f"checkpoints/{dataset_name}/{self.opt.vae_name}/opt.txt", self.device)

        # Components
        self.vae           = load_vae(self.vae_opt).to(self.device)
        self.style_encoder = StyleContentNet(config['style_encoder'], opt).to(device)
        self.denoiser      = Denoiser(config['denoiser'], opt, vae_dim).to(device)

        # Scheduler
        self.scheduler     = DDIMScheduler(
            num_train_timesteps=self.opt.num_train_timesteps,
            beta_start=self.opt.beta_start,
            beta_end=self.opt.beta_end,
            beta_schedule=self.opt.beta_schedule,
            prediction_type=self.opt.prediction_type,
            clip_sample=False,
        )

        self.tokenizer = self.denoiser.clip_model.tokenizer


    def forward_train(self, batch):
        # Setup input
        # Hm... I guess there's a corresponding text for each motion sample inside the batch?
        # For style embeddings I can use the motions in the same batch... maybe?
        text, motion, m_lens = batch

        # Random drop for text
        text = [
            "" if np.random.rand(1) < self.config['text_drop'] else t for t in text
        ]

        # Style embedding
        with torch.no_grad():
            style = self.style_encoder.encode(motion)
        
        # Random drop for style
        if np.random.rand() < self.config['style_drop']:
            style = torch.zeros_like(style)

        # To device
        motion   = motion.to(self.opt.device, dtype=torch.float32)
        m_lens   = m_lens.to(self.opt.device, dtype=torch.long)
        len_mask = lengths_to_mask(m_lens // 4)

        # Latent
        with torch.no_grad():
            latent, _ = self.vae.encode(motion)
            len_mask = F.pad(len_mask, (0, latent.shape[1] - len_mask.shape[1]), mode="constant", value=False)
            latent = latent * len_mask[..., None, None].float()

        # Sample diffusion timesteps
        timesteps = torch.randint(
            0,
            self.opt.num_train_timesteps,
            (latent.shape[0],),
            device=latent.device,
        ).long()

        # Add noise
        noise = torch.randn_like(latent)
        noise = noise * len_mask[..., None, None].float()
        noisy_latent = self.noise_scheduler.add_noise(latent, noise, timesteps)

        # Predict the noise
        pred, attn_list = self.denoiser.forward(noisy_latent, timesetps, text, len_mask=len_mask)
        pred = pred * len_mask[..., None, None].float()

        # Loss
        loss_dict = {}
        loss = 0
        if self.opt.prediction_type == "sample":
            loss_sample = self.recon_criterion(pred, latent)
            loss += loss_sample
            loss_dict["loss_sample"] = loss_sample

        elif self.opt.prediction_type == "epsilon":
            loss_eps = self.recon_criterion(pred, noise)
            loss += loss_eps
            loss_dict["loss_eps"] = loss_eps

        elif self.opt.prediction_type == "v_prediction":
            vel = self.noise_scheduler.get_velocity(latent, noise, timesteps)
            loss_vel = self.recon_criterion(pred, vel)
            loss += loss_vel
            loss_dict["loss_vel"] = loss_vel

        return {
            "pred": pred,
            "latent": latent,
            "noise": noise,
            "timesteps": timesteps,
            "len_mask": len_mask,
            "attn": attn_list,
            "text": text,
            "style": style,
        }

    
    def denoiser_step(model, batch, )
        model.train()
        model.style_encoder.eval()
        set_requires_grad(model)


    def style_step(model, batch, )