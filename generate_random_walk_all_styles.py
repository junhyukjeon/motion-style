import argparse
import os
import random
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from guidance_test_utils import (
    build_style_dataset,
    denormalize_motion,
    ensure_dir,
    load_config,
    load_model,
    load_style_stats,
    motion_to_joints,
    save_json,
    save_motion_video,
    set_seed,
    slug,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate 'a person walks forward' from one random reference motion per style."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config file (YAML).")
    parser.add_argument(
        "--caption",
        type=str,
        default="a person walks forward",
        help="Caption to generate for every sampled style.",
    )
    parser.add_argument(
        "--output_length",
        type=int,
        default=140,
        help="Generated motion length in frames.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of styles to generate per batch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for sampling one motion per style.",
    )
    parser.add_argument(
        "--output_subdir",
        type=str,
        default="random_style_walk_forward",
        help="Subdirectory under result_dir for this export.",
    )
    return parser.parse_args()


def build_style_groups(dataset) -> Dict[int, List[Dict]]:
    groups = defaultdict(list)
    for item in dataset.items:
        groups[int(item["style_idx"])].append(item)
    return groups


def choose_one_motion_per_style(dataset, seed: int):
    rng = random.Random(seed)
    style_groups = build_style_groups(dataset)
    selections = []

    for style_idx in sorted(style_groups.keys()):
        chosen = rng.choice(style_groups[style_idx])
        selections.append(
            {
                "style_idx": style_idx,
                "style_name": dataset.idx_to_style[style_idx],
                "motion_id": str(chosen["motion_id"]),
                "style_length": int(chosen["length"]),
            }
        )
    return selections


def make_run_dir(config: Dict, args) -> str:
    caption_tag = slug(args.caption, maxlen=60)
    run_dir = os.path.join(
        config["result_dir"],
        args.output_subdir,
        f"{caption_tag}_seed{args.seed}_len{args.output_length}",
    )
    ensure_dir(run_dir)
    return run_dir


def load_motion_batch(dataset, selections, device):
    motions = []
    style_lengths = []

    for item in selections:
        motion = dataset.motion_cache[item["motion_id"]][: item["style_length"]]
        motions.append(motion)
        style_lengths.append(item["style_length"])

    motion_batch = pad_sequence(motions, batch_first=True).to(device)
    style_lengths = torch.tensor(style_lengths, dtype=torch.long, device=device)
    return motion_batch, style_lengths


def save_sample_outputs(out_dir, caption, sample_meta, style_motion_real, generated_motion_real):
    style_dir = os.path.join(
        out_dir,
        f"{sample_meta['style_idx']:03d}_{slug(sample_meta['style_name'], maxlen=40)}_{sample_meta['motion_id']}",
    )
    ensure_dir(style_dir)

    ref_joints = motion_to_joints(style_motion_real.unsqueeze(0)).detach().cpu().numpy()[0]
    gen_joints = motion_to_joints(generated_motion_real.unsqueeze(0)).detach().cpu().numpy()[0]

    save_motion_video(
        os.path.join(style_dir, "reference_style.mp4"),
        ref_joints.astype(np.float32),
        title=f"reference style {sample_meta['style_name']} ({sample_meta['motion_id']})",
        fps=20,
    )
    save_motion_video(
        os.path.join(style_dir, "generated_walk_forward.mp4"),
        gen_joints.astype(np.float32),
        title=f"{sample_meta['style_name']} | {caption}",
        fps=20,
    )

    np.save(os.path.join(style_dir, "reference_style.npy"), style_motion_real.detach().cpu().numpy())
    np.save(os.path.join(style_dir, "generated_walk_forward.npy"), generated_motion_real.detach().cpu().numpy())
    save_json(
        os.path.join(style_dir, "metadata.json"),
        {
            "caption": caption,
            "style_idx": sample_meta["style_idx"],
            "style_name": sample_meta["style_name"],
            "ref_motion_id": sample_meta["motion_id"],
            "reference_length": int(style_motion_real.shape[0]),
            "generated_length": int(generated_motion_real.shape[0]),
        },
    )


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config(args.config)
    set_seed(args.seed)

    dataset = build_style_dataset(config)
    model = load_model(config, device)
    mean, std = load_style_stats(config["dataset_style"], device)

    selections = choose_one_motion_per_style(dataset, args.seed)
    run_dir = make_run_dir(config, args)

    save_json(
        os.path.join(run_dir, "run_metadata.json"),
        {
            "config": args.config,
            "caption": args.caption,
            "seed": args.seed,
            "output_length": args.output_length,
            "batch_size": args.batch_size,
            "num_styles": len(selections),
        },
    )

    summary = []

    for start in tqdm(range(0, len(selections), args.batch_size), desc="Generating by style"):
        batch_meta = selections[start : start + args.batch_size]
        motion_batch, style_lengths = load_motion_batch(dataset, batch_meta, device)
        captions = [args.caption] * len(batch_meta)
        output_lengths = torch.full(
            (len(batch_meta),),
            int(args.output_length),
            dtype=torch.long,
            device=device,
        )

        with torch.no_grad():
            generated_norm, _ = model.generate(
                motion_batch,
                captions,
                output_lengths,
                style_lengths,
            )

        generated_real = denormalize_motion(generated_norm, mean, std)
        reference_real = denormalize_motion(motion_batch, mean, std)

        for idx, sample_meta in enumerate(batch_meta):
            ref_len = int(style_lengths[idx].item())
            gen_len = int(output_lengths[idx].item())

            style_motion_real = reference_real[idx, :ref_len]
            generated_motion_real = generated_real[idx, :gen_len]
            save_sample_outputs(
                out_dir=run_dir,
                caption=args.caption,
                sample_meta=sample_meta,
                style_motion_real=style_motion_real,
                generated_motion_real=generated_motion_real,
            )
            summary.append(
                {
                    "style_idx": sample_meta["style_idx"],
                    "style_name": sample_meta["style_name"],
                    "ref_motion_id": sample_meta["motion_id"],
                    "reference_length": ref_len,
                    "generated_length": gen_len,
                }
            )

    save_json(os.path.join(run_dir, "summary.json"), {"samples": summary})
    print(f"Saved {len(summary)} styled walk-forward generations to: {run_dir}")


if __name__ == "__main__":
    main()
