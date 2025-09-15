import argparse
import json
import numpy as np
import os
import random
import shutil
import torch
import yaml
from tqdm import tqdm

from data.dataset import StyleDataset
from model.networks import NETWORK_REGISTRY
# from utils.process import parse_humanml3d
# from utils.write import write_bvh

def reset_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def get_unique_path(base_path):
    if not os.path.exists(base_path):
        return base_path

    base, ext = os.path.splitext(base_path)
    i = 1
    while True:
        new_path = f"{base}_{i}{ext}"
        if not os.path.exists(new_path):
            return new_path
        i += 1

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file (YAML)')
    args = parser.parse_args()

    config_path = args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config_basename = os.path.basename(config_path)
    config["run_name"] = os.path.splitext(config_basename)[0]
    config["result_dir"] = os.path.join(config["result_dir"], config["run_name"])
    config["checkpoint_dir"] = os.path.join(config["checkpoint_dir"], config["run_name"])
    return config


def load_model(config, device):
    model_cfg = config['model']
    model = NETWORK_REGISTRY[model_cfg['type']](model_cfg).to(device)
    model.load_state_dict(torch.load(os.path.join(config["checkpoint_dir"], "best.ckpt"), map_location=device))
    model.eval()
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config()
    set_seed(config["random_seed"])

    mean = np.load(config["mean_path"])
    std = np.load(config["std_path"])

    with open(config["style_json"]) as f:
        label_to_ids = yaml.safe_load(f)

    style_to_label = {style: i for i, style in enumerate(sorted(label_to_ids))}

    # --- Content Mapping ---
    with open(config["content_json"]) as f:
        content_to_ids = json.load(f)
    
    ids_to_content = {}
    for content_type, motion_ids in content_to_ids.items():
        for m_id in motion_ids:
            ids_to_content[m_id] = content_type

    dataset = StyleDataset(config["motion_dir"], mean, std, config["window_size"], label_to_ids, style_to_label, ids_to_content)
    model = load_model(config, device)

    output_dir = os.path.join(config["result_dir"], "style")
    reset_dir(output_dir)

    # === [1] Load reference style motion
    # style_idx = 1540217
    style_idx = 815700
    style_motion, _, _, style_motion_id = dataset[style_idx]
    style_motion = style_motion.unsqueeze(0).to(device)  # [1, T, J, D]

    # flapping_ids = label_to_ids.get("Flapping", [])

    # import pdb; pdb.set_trace()

    # Find label name for style_motion_id
    for label, ids in label_to_ids.items():
        if style_motion_id in ids:
            style_label = label
            break
    else:
        raise ValueError(f"Could not find label for motion ID {style_motion_id}")

    with torch.no_grad():
        out = model.encode(style_motion)
        z_style = out["z_style"]  # [1, D]
        gamma = out["gamma"]
        beta = out["beta"]

    print(f"🎨 Using style from motion ID: {style_motion_id} (label: {style_label})")

    # === [2] Sample 100 motions to swap content
    sample_indices = random.sample(range(len(dataset)), 100)
    batch_size = 32
    batched_indices = [sample_indices[i:i+batch_size] for i in range(0, len(sample_indices), batch_size)]

    mean = torch.tensor(mean, dtype=torch.float32, device=device)
    std = torch.tensor(std, dtype=torch.float32, device=device)

    for batch_ids in tqdm(batched_indices, desc="Swapping style onto content"):
        motions, motion_ids = [], []

        for idx in batch_ids:
            motion, _, _, motion_id = dataset[idx]
            motions.append(motion)
            motion_ids.append(motion_id)

        motions = torch.stack(motions).to(device)  # [B, T, J, D]

        with torch.no_grad():
            out = model.encode(motions)
            z_content = out["z_content"]  # [B, D]
            z_new = model.decode(gamma * z_content + beta)
            motions_recon = model.encoder.vae.decode(z_new)  # [B, T, J, D]

        motions_recon = motions_recon * std + mean

        for i, motion_id in enumerate(motion_ids):
            recon_npy = motions_recon[i].cpu().numpy()  # [T, 263]
            
            save_path = get_unique_path(os.path.join(output_dir, f"{motion_id}_{style_label}.npy"))
            np.save(save_path, recon_npy)