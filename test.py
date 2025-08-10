import os
import torch
import numpy as np
from model.networks import StyleContentEncoder
from utils.process import parse_humanml3d
from utils.write import write_bvh
from data.dataset import StyleDataset
import yaml

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_model(config, device):
    model = StyleContentEncoder(config['style_content_encoder']).to(device)
    model.vae.eval()
    return model

def main():
    # === Load config
    with open("./configs/3.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(config["random_seed"])

    # === Load stats
    mean = np.load(config["mean_path"])
    std = np.load(config["std_path"])
    mean = torch.tensor(mean, dtype=torch.float32, device=device)
    std = torch.tensor(std, dtype=torch.float32, device=device)

    # === Load motion
    with open(config["style_json"]) as f:
        label_to_ids = yaml.safe_load(f)
    style_to_label = {style: i for i, style in enumerate(sorted(label_to_ids))}
    dataset = StyleDataset(config["motion_dir"], mean.cpu().numpy(), std.cpu().numpy(),
                           config["window_size"], label_to_ids, style_to_label)

    motion, _, motion_id = dataset[0]  # just one sample
    motion = motion.unsqueeze(0).to(device)  # [1, T, J, D]

    # === Load model
    model = load_model(config, device)

    # === Encode and decode
    with torch.no_grad():
        z_latent, _ = model.vae.encode(motion)
        recon = model.vae.decode(z_latent)  # [1, T, J, D]

    # === Denormalize
    recon = recon * std + mean

    # === Parse and export BVH
    parsed = parse_humanml3d(recon)
    root_pos = parsed["root_pos"][0].cpu().numpy()
    rot_mtx = parsed["rot_matrices"][0].cpu().numpy()

    output_dir = os.path.join(config["result_dir"], "vae_recon_test")
    os.makedirs(output_dir, exist_ok=True)
    bvh_path = os.path.join(output_dir, f"{motion_id}_vae_recon.bvh")
    write_bvh(bvh_path, root_pos, rot_mtx)

    print("✅ Finished writing BVH.")

if __name__ == "__main__":
    main()