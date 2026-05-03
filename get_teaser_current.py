import argparse
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml

from data.dataset import Dataset100STYLE
from model.t2sm import Text2StylizedMotion
from utils.motion import recover_from_ric
from mld.data.humanml.utils.plot_script import plot_3d_motion


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reset_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def slug(s: str, maxlen: int = 80) -> str:
    s = "".join(c if (c.isalnum() or c in " _-.,()[]{}") else "_" for c in s.strip())
    s = "_".join(s.split())
    return (s[:maxlen]).rstrip("_")


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file (YAML)")
    parser.add_argument("--ref_motion_id", type=str, required=True, help="Reference motion ID")
    parser.add_argument("--caption", type=str, default=None, help="Caption text for generation")
    parser.add_argument("--num_samples", type=int, default=8, help="Total outputs including the reference slot")
    parser.add_argument("--output_length", type=int, default=68, help="Generated output length")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    with cfg_path.open("r") as f:
        config = yaml.safe_load(f)

    parts = cfg_path.parts
    if "configs" in parts:
        i = parts.index("configs")
        run_name = str(Path(*parts[i + 1:]).with_suffix("")).replace("\\", "/")
    else:
        run_name = cfg_path.stem

    config["run_name"] = run_name
    config["result_dir"] = os.path.join(config["result_dir"], os.path.basename(run_name))
    config["checkpoint_dir"] = os.path.join(config["checkpoint_dir"], run_name)
    config["_cli"] = {
        "ref_motion_id": str(args.ref_motion_id),
        "caption": args.caption,
        "num_samples": int(args.num_samples),
        "output_length": int(args.output_length),
    }
    return config


def load_model(config, device):
    model = Text2StylizedMotion(config["model"]).to(device)
    ckpt_path = os.path.join(config["checkpoint_dir"], "latest.ckpt")
    model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
    model.eval()
    return model


def find_index_by_motion_id(ds_style, motion_id: str) -> int:
    motion_id = str(motion_id)
    for i, item in enumerate(ds_style.items):
        if str(item["motion_id"]) == motion_id:
            return i
    raise ValueError(f"motion_id='{motion_id}' not found in dataset.")


def get_eval_window(ds_style, idx: int):
    meta = ds_style.items[idx]
    motion = ds_style.motion_cache[meta["motion_id"]]
    total_frames = int(motion.shape[0])
    feat_dim = int(motion.shape[1])

    unit = ds_style.unit_length
    if unit <= 0:
        motion_length = total_frames
    else:
        motion_length = max(1, total_frames // unit) * unit

    if ds_style.max_frames is not None:
        motion_length = min(motion_length, ds_style.max_frames)
    motion_length = max(1, min(motion_length, total_frames))

    start = max(0, (total_frames - motion_length) // 2)
    window = motion[start:start + motion_length]

    if ds_style.max_frames is not None and motion_length < ds_style.max_frames:
        pad = torch.zeros(ds_style.max_frames - motion_length, feat_dim, dtype=window.dtype)
        window = torch.cat([window, pad], dim=0)

    content_key = ds_style.idx_to_content[meta["content_idx"]]
    caption = ds_style.content_prompts[content_key]
    return caption, window, motion_length, int(meta["style_idx"])


def build_output_dir(config, caption):
    style_weight = config["model"].get("style_weight", None)
    style_guidance = config["model"].get("style_guidance", None)

    def fmt_tag(prefix: str, value) -> str:
        if isinstance(value, bool):
            return f"{prefix}{int(value)}"
        try:
            return f"{prefix}{str(float(value)).replace('.', 'p')}"
        except (TypeError, ValueError):
            return f"{prefix}{str(value)}"

    tag_parts = []
    if style_weight is not None:
        tag_parts.append(fmt_tag("w", style_weight))
    if style_guidance is not None:
        tag_parts.append(fmt_tag("g", style_guidance))

    style_tag = "_".join(tag_parts) if tag_parts else "default"
    return os.path.join(
        config["result_dir"],
        config["_cli"]["ref_motion_id"],
        style_tag,
        slug(caption),
    )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config()
    set_seed(config["random_seed"])

    model = load_model(config, device)
    style_cfg = config["dataset_style"]
    ds_style = Dataset100STYLE(style_cfg)

    ref_i = find_index_by_motion_id(ds_style, config["_cli"]["ref_motion_id"])
    default_caption, window, motion_length, _ = get_eval_window(ds_style, ref_i)

    caption = config["_cli"]["caption"] if config["_cli"]["caption"] is not None else default_caption
    output_dir = build_output_dir(config, caption)
    reset_dir(output_dir)

    num_samples = config["_cli"]["num_samples"]
    output_length = config["_cli"]["output_length"]

    motions = window[:motion_length].unsqueeze(0).to(device)
    style_lengths = torch.tensor([motion_length], dtype=torch.long, device=device)

    motions = motions.repeat(num_samples, 1, 1)
    style_lengths = style_lengths.repeat(num_samples)
    captions = [caption] * num_samples
    gen_lengths = torch.full((num_samples,), output_length, dtype=torch.long, device=device)

    stylized, captions_out = model.generate(motions, captions, gen_lengths, style_lengths)

    mean = torch.tensor(np.load(style_cfg["mean_path"]), dtype=torch.float32, device=device)
    std = torch.tensor(np.load(style_cfg["std_path"]), dtype=torch.float32, device=device)

    stylized = stylized * std + mean
    reference = motions * std + mean

    joints_stylized = recover_from_ric(stylized, 22).detach().cpu().numpy()
    joints_reference = recover_from_ric(reference, 22).detach().cpu().numpy()

    kinematic_tree = [
        [0, 2, 5, 8, 11],
        [0, 1, 4, 7, 10],
        [0, 3, 6, 9, 12, 15],
        [9, 14, 17, 19, 21],
        [9, 13, 16, 18, 20],
    ]

    ref_length = int(gen_lengths[0].item())
    xyz_ref = joints_reference[0][:ref_length].astype(np.float32)
    ref_path = os.path.join(output_dir, "sample00_rep00.mp4")
    plot_3d_motion(ref_path, kinematic_tree, xyz_ref, title="", dataset="humanml", fps=20)

    lengths = gen_lengths.detach().cpu().numpy().astype(int)
    num_repetitions = 1
    max_length = int(lengths.max())
    all_motions = np.zeros((num_samples, 22, 3, max_length), dtype=np.float32)
    all_lengths = lengths.astype(np.int32)
    all_text = ["[REF]"] + [str(captions_out[i]) for i in range(1, num_samples)]

    all_motions[0, :, :, :ref_length] = np.transpose(xyz_ref, (1, 2, 0))
    all_lengths[0] = ref_length

    for sample_i in range(1, num_samples):
        sample_length = int(all_lengths[sample_i])
        xyz_gen = joints_stylized[sample_i][:sample_length].astype(np.float32)
        all_motions[sample_i, :, :, :sample_length] = np.transpose(xyz_gen, (1, 2, 0))

        gen_path = os.path.join(output_dir, f"sample{sample_i:02d}_rep00.mp4")
        plot_3d_motion(gen_path, kinematic_tree, xyz_gen, title="", dataset="humanml", fps=20)

    np.save(
        os.path.join(output_dir, "results.npy"),
        {
            "motion": all_motions,
            "text": all_text,
            "lengths": all_lengths,
            "num_samples": num_samples,
            "num_repetitions": num_repetitions,
        },
    )
