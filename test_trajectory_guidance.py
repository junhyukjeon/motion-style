import argparse
import os

import numpy as np
import torch

from guidance_test_utils import (
    build_style_dataset,
    denormalize_motion,
    ensure_dir,
    get_full_motion,
    load_config,
    load_model,
    load_style_stats,
    motion_to_joints,
    motion_to_root_xz,
    plot_root_trajectory,
    repeat_batch,
    save_json,
    save_motion_video,
    set_seed,
    slug,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Test trajectory guidance for Text2StylizedMotion.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML.")
    parser.add_argument("--ref_motion_id", type=str, required=True, help="100STYLE motion id used as style reference.")
    parser.add_argument("--caption", type=str, default="a person walks in a circle", help="Content prompt.")
    parser.add_argument("--output_length", type=int, default=140, help="Generated motion length in frames.")
    parser.add_argument("--num_samples", type=int, default=4, help="How many samples to draw with the same constraint.")
    parser.add_argument("--radius", type=float, default=1.0, help="Circle radius in world-space root coordinates.")
    parser.add_argument("--turns", type=float, default=1.0, help="How many turns the target circle should complete.")
    parser.add_argument(
        "--run_tag",
        type=str,
        default=None,
        help="Optional suffix for the output folder. If omitted, a tag is built from the guidance settings.",
    )
    parser.add_argument("--trajectory_weight", type=float, default=5.0, help="Trajectory guidance weight.")
    parser.add_argument("--trajectory_start_frac", type=float, default=0.0, help="Trajectory guidance start fraction in denoising.")
    parser.add_argument("--trajectory_end_frac", type=float, default=1.0, help="Trajectory guidance end fraction in denoising.")
    parser.add_argument(
        "--trajectory_schedule",
        type=str,
        default="constant",
        choices=["constant", "linear_ramp", "linear_decay", "cosine_ramp", "cosine_decay", "bell"],
        help="Trajectory guidance schedule within its active timestep window.",
    )
    parser.add_argument(
        "--style_guidance_weight",
        type=float,
        default=None,
        help="Optional override for style guidance weight during trajectory tests. Use 0 to disable.",
    )
    parser.add_argument("--style_start_frac", type=float, default=0.0, help="Style guidance start fraction in denoising.")
    parser.add_argument("--style_end_frac", type=float, default=1.0, help="Style guidance end fraction in denoising.")
    parser.add_argument(
        "--style_schedule",
        type=str,
        default="constant",
        choices=["constant", "linear_ramp", "linear_decay", "cosine_ramp", "cosine_decay", "bell"],
        help="Style guidance schedule within its active timestep window.",
    )
    parser.add_argument(
        "--optimize_initial_noise_only",
        action="store_true",
        help="Optimize only the starting z_T against the final trajectory loss, then sample without per-step guidance.",
    )
    parser.add_argument("--noise_opt_steps", type=int, default=10, help="Outer optimization steps for z_T-only optimization.")
    parser.add_argument("--noise_opt_lr", type=float, default=0.05, help="Learning rate for z_T-only optimization.")
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of DDIM denoising steps used for sampling and z_T-only optimization rollouts.",
    )
    parser.add_argument("--guidance_steps", type=int, default=1, help="Inner guidance steps per diffusion step.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def make_circular_trajectory(length: int, radius: float, turns: float, device: torch.device) -> torch.Tensor:
    theta = torch.linspace(0.0, 2.0 * np.pi * turns, steps=length, device=device, dtype=torch.float32)
    x = radius * (torch.cos(theta) - 1.0)
    z = radius * torch.sin(theta)
    return torch.stack([x, z], dim=-1)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config(args.config)
    dataset = build_style_dataset(config)
    model = load_model(config, device)
    mean, std = load_style_stats(config["dataset_style"], device)

    style_motion, style_length = get_full_motion(dataset, args.ref_motion_id, device)
    style_batch = repeat_batch(style_motion, args.num_samples)
    style_lengths = torch.full((args.num_samples,), style_length, dtype=torch.long, device=device)
    output_lengths = torch.full((args.num_samples,), args.output_length, dtype=torch.long, device=device)
    captions = [args.caption] * args.num_samples

    target_traj = make_circular_trajectory(args.output_length, args.radius, args.turns, device)
    target_traj_batch = target_traj.unsqueeze(0).repeat(args.num_samples, 1, 1)
    target_mask = torch.ones(args.num_samples, args.output_length, dtype=torch.bool, device=device)

    guidance = {
        "steps": args.guidance_steps,
        "num_inference_steps": args.num_inference_steps,
        "style": {
            "start_frac": args.style_start_frac,
            "end_frac": args.style_end_frac,
            "schedule": args.style_schedule,
        },
        "trajectory": {
            "mode": "root_xz",
            "weight": args.trajectory_weight,
            "target": target_traj_batch,
            "mask": target_mask,
            "start_frac": args.trajectory_start_frac,
            "end_frac": args.trajectory_end_frac,
            "schedule": args.trajectory_schedule,
        },
    }
    if args.style_guidance_weight is not None:
        guidance["style"]["weight"] = args.style_guidance_weight

    opt_info = None
    if args.optimize_initial_noise_only:
        stylized_norm, captions_out, opt_info = model.generate_with_optimized_initial_noise(
            style_batch,
            captions,
            output_lengths,
            style_lengths,
            guidance=guidance,
            noise_opt_steps=args.noise_opt_steps,
            noise_opt_lr=args.noise_opt_lr,
            num_inference_steps=args.num_inference_steps,
        )
    else:
        stylized_norm, captions_out = model.generate(
            style_batch,
            captions,
            output_lengths,
            style_lengths,
            guidance=guidance,
        )

    stylized_real = denormalize_motion(stylized_norm, mean, std)
    reference_real = denormalize_motion(style_batch, mean, std)
    joints_stylized = motion_to_joints(stylized_real).detach().cpu().numpy()
    joints_reference = motion_to_joints(reference_real).detach().cpu().numpy()
    root_xz = motion_to_root_xz(stylized_real).detach().cpu().numpy()
    target_xz_np = target_traj.detach().cpu().numpy()

    opt_tag = ""
    if args.optimize_initial_noise_only:
        opt_tag = f"_opt{args.noise_opt_steps}_lr{str(args.noise_opt_lr).replace('.', 'p')}"

    auto_tag = (
        f"mode-{'zTonly' if args.optimize_initial_noise_only else 'step'}"
        f"_n{args.num_inference_steps}"
        f"{opt_tag}"
        f"traj-{args.trajectory_schedule}"
        f"_s{str(args.trajectory_start_frac).replace('.', 'p')}"
        f"_e{str(args.trajectory_end_frac).replace('.', 'p')}"
        f"_w{str(args.trajectory_weight).replace('.', 'p')}"
        f"_style-{str(args.style_guidance_weight if args.style_guidance_weight is not None else 'cfg').replace('.', 'p')}"
        f"_g{args.guidance_steps}"
    )
    run_tag = slug(args.run_tag if args.run_tag else auto_tag, maxlen=120)

    out_dir = os.path.join(
        config["result_dir"],
        "guidance_tests",
        "trajectory",
        f"{args.ref_motion_id}_{slug(args.caption)}_r{str(args.radius).replace('.', 'p')}_{run_tag}",
    )
    ensure_dir(out_dir)

    save_json(
        os.path.join(out_dir, "metadata.json"),
        {
            "config": args.config,
            "ref_motion_id": args.ref_motion_id,
            "caption": args.caption,
            "output_length": args.output_length,
            "num_samples": args.num_samples,
            "radius": args.radius,
            "turns": args.turns,
            "trajectory_weight": args.trajectory_weight,
            "trajectory_start_frac": args.trajectory_start_frac,
            "trajectory_end_frac": args.trajectory_end_frac,
            "trajectory_schedule": args.trajectory_schedule,
            "style_guidance_weight": args.style_guidance_weight,
            "style_start_frac": args.style_start_frac,
            "style_end_frac": args.style_end_frac,
            "style_schedule": args.style_schedule,
            "optimize_initial_noise_only": bool(args.optimize_initial_noise_only),
            "noise_opt_steps": args.noise_opt_steps,
            "noise_opt_lr": args.noise_opt_lr,
            "num_inference_steps": args.num_inference_steps,
            "guidance_steps": args.guidance_steps,
            "seed": args.seed,
            "optimization_history": None if opt_info is None else opt_info["history"],
        },
    )
    np.save(os.path.join(out_dir, "target_trajectory.npy"), target_xz_np)
    np.save(os.path.join(out_dir, "generated_root_xz.npy"), root_xz)
    np.save(os.path.join(out_dir, "stylized_motion.npy"), stylized_real.detach().cpu().numpy())

    ref_len = min(style_length, style_motion.shape[0])
    save_motion_video(
        os.path.join(out_dir, "reference_style.mp4"),
        joints_reference[0][:ref_len].astype(np.float32),
        title=f"reference style {args.ref_motion_id}",
        fps=20,
    )

    for idx in range(args.num_samples):
        sample_len = int(output_lengths[idx].item())
        title = captions_out[idx] if isinstance(captions_out, (list, tuple)) else args.caption
        save_motion_video(
            os.path.join(out_dir, f"sample_{idx:02d}.mp4"),
            joints_stylized[idx][:sample_len].astype(np.float32),
            title=title,
            fps=20,
        )
        plot_root_trajectory(
            os.path.join(out_dir, f"sample_{idx:02d}_trajectory.png"),
            target_xz_np[:sample_len],
            root_xz[idx][:sample_len],
            title=f"sample {idx:02d}",
        )


if __name__ == "__main__":
    main()
