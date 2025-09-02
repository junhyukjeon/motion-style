# --- Imports ---
import torch
import torch.nn as nn

from model.denoiser import Denoiser
from model.networks import StyleEncoder
from salad.models.vae.model import VAE
from salad.utils.get_opt import get_opt

from data.dataset import StyleDataset
from data.sampler import SAMPLER_REGISTRY
from model.networks import StyleContentNet


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


    def optimize(self, batch_data):
        # Setup input
        text, motion, m_lens = batch_data

        # Random drop during training
        text = [
            "" if np.random.rand(1) < self.opt.cond_drop_prob else t for t in text
        ]

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
