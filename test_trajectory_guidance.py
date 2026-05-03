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
    parser = argparse.ArgumentParser(description="Test step-guided trajectory guidance for Text2StylizedMotion.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML.")
    parser.add_argument("--ref_motion_id", type=str, required=True, help="100STYLE motion id used as style reference.")
    parser.add_argument("--caption", type=str, default="a person runs", help="Content prompt.")
    parser.add_argument("--output_length", type=int, default=140, help="Generated motion length in frames.")
    parser.add_argument("--num_samples", type=int, default=4, help="How many samples to draw with the same constraint.")
    parser.add_argument(
        "--trajectory_shape",
        type=str,
        default="s_curve",
        choices=["circle", "s_curve"],
        help="Target trajectory shape used for guidance.",
    )
    parser.add_argument("--radius", type=float, default=1.0, help="Trajectory radius or lateral amplitude in world-space root coordinates.")
    parser.add_argument("--turns", type=float, default=1.0, help="How many turns or S-curve oscillations the target should complete.")
    parser.add_argument(
        "--forward_length",
        type=float,
        default=4.0,
        help="Forward distance covered by the S-curve along the root z axis. Ignored for circles.",
    )
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
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of DDIM denoising steps used for step-guided sampling.",
    )
    parser.add_argument("--guidance_steps", type=int, default=1, help="Inner latent update steps per diffusion step.")
    parser.add_argument("--style_guidance_steps", type=int, default=1, help="Style-only inner latent update steps per diffusion step.")
    parser.add_argument("--trajectory_guidance_steps", type=int, default=1, help="Trajectory/motion inner latent update steps per diffusion step.")
    parser.add_argument(
        "--guidance_inner_mode",
        type=str,
        default="separate",
        choices=["combined", "separate", "style_fixed_then_motion_recompute"],
        help="Use one combined update, separate fixed-v style then motion passes, or a staged style-fixed then motion-recompute update per diffusion step.",
    )
    parser.add_argument(
        "--guidance_order",
        type=str,
        default="style_then_motion",
        choices=["style_then_motion", "motion_then_style"],
        help="Order of separate style and motion inner updates when guidance_inner_mode=separate.",
    )
    parser.add_argument(
        "--recompute_guided_v_pred",
        action="store_true",
        help="After latent guidance, recompute v_pred from the guided latent before the DDIM step.",
    )
    parser.add_argument(
        "--print_step_trajectory_loss",
        action="store_true",
        help="Print and save the batch-mean trajectory loss estimated at each diffusion timestep.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def make_circular_trajectory(length: int, radius: float, turns: float, device: torch.device) -> torch.Tensor:
    theta = torch.linspace(0.0, 2.0 * np.pi * turns, steps=length, device=device, dtype=torch.float32)
    x = radius * (torch.cos(theta) - 1.0)
    z = radius * torch.sin(theta)
    return torch.stack([x, z], dim=-1)


def make_s_curve_trajectory(length: int, amplitude: float, turns: float, forward_length: float, device: torch.device) -> torch.Tensor:
    u = torch.linspace(0.0, 1.0, steps=length, device=device, dtype=torch.float32)
    x = amplitude * torch.sin(2.0 * np.pi * turns * u)
    z = forward_length * u
    return torch.stack([x, z], dim=-1)


def make_target_trajectory(
    shape: str,
    length: int,
    radius: float,
    turns: float,
    forward_length: float,
    device: torch.device,
) -> torch.Tensor:
    if shape == "circle":
        return make_circular_trajectory(length=length, radius=radius, turns=turns, device=device)
    if shape == "s_curve":
        return make_s_curve_trajectory(
            length=length,
            amplitude=radius,
            turns=turns,
            forward_length=forward_length,
            device=device,
        )
    raise ValueError(f"Unsupported trajectory shape: {shape}")


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

    target_traj = make_target_trajectory(
        shape=args.trajectory_shape,
        length=args.output_length,
        radius=args.radius,
        turns=args.turns,
        forward_length=args.forward_length,
        device=device,
    )
    target_traj_batch = target_traj.unsqueeze(0).repeat(args.num_samples, 1, 1)
    target_mask = torch.ones(args.num_samples, args.output_length, dtype=torch.bool, device=device)

    guidance = {
        "steps": args.guidance_steps,
        "style_steps": args.style_guidance_steps,
        "trajectory_steps": args.trajectory_guidance_steps,
        "inner_mode": args.guidance_inner_mode,
        "guidance_order": args.guidance_order,
        "num_inference_steps": args.num_inference_steps,
        "record_step_trajectory_loss": bool(args.print_step_trajectory_loss),
        "recompute_v_guided": bool(args.recompute_guided_v_pred),
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
            "relative_to_start": True,
            "start_frac": args.trajectory_start_frac,
            "end_frac": args.trajectory_end_frac,
            "schedule": args.trajectory_schedule,
        },
    }
    if args.style_guidance_weight is not None:
        guidance["style"]["weight"] = args.style_guidance_weight

    stylized_norm, captions_out = model.generate(
        style_batch,
        captions,
        output_lengths,
        style_lengths,
        guidance=guidance,
    )
    debug_info = model.get_last_debug_info()

    stylized_real = denormalize_motion(stylized_norm, mean, std)
    reference_real = denormalize_motion(style_batch, mean, std)
    joints_stylized = motion_to_joints(stylized_real).detach().cpu().numpy()
    joints_reference = motion_to_joints(reference_real).detach().cpu().numpy()
    root_xz = motion_to_root_xz(stylized_real).detach().cpu().numpy()
    target_xz_np = target_traj.detach().cpu().numpy()

    auto_tag = (
        f"step_n{args.num_inference_steps}"
        f"_{args.trajectory_shape}"
        f"traj-{args.trajectory_schedule}"
        f"_s{str(args.trajectory_start_frac).replace('.', 'p')}"
        f"_e{str(args.trajectory_end_frac).replace('.', 'p')}"
        f"_w{str(args.trajectory_weight).replace('.', 'p')}"
        f"_style-{str(args.style_guidance_weight if args.style_guidance_weight is not None else 'cfg').replace('.', 'p')}"
        f"_g{args.guidance_steps}"
        f"_sg{args.style_guidance_steps}"
        f"_tg{args.trajectory_guidance_steps}"
        f"_im-{args.guidance_inner_mode}"
        f"_go-{args.guidance_order}"
        f"_rv{1 if args.recompute_guided_v_pred else 0}"
    )
    run_tag = slug(args.run_tag if args.run_tag else auto_tag, maxlen=120)

    out_dir = os.path.join(
        config["result_dir"],
        "guidance_tests",
        "trajectory",
        (
            f"{args.ref_motion_id}_{slug(args.caption)}_{slug(args.trajectory_shape)}"
            f"_r{str(args.radius).replace('.', 'p')}_{run_tag}"
        ),
    )
    ensure_dir(out_dir)

    save_json(
        os.path.join(out_dir, "metadata.json"),
        {
            "config": args.config,
            "guidance_mode": "step_latent",
            "ref_motion_id": args.ref_motion_id,
            "caption": args.caption,
            "trajectory_shape": args.trajectory_shape,
            "output_length": args.output_length,
            "num_samples": args.num_samples,
            "radius": args.radius,
            "turns": args.turns,
            "forward_length": args.forward_length,
            "trajectory_weight": args.trajectory_weight,
            "trajectory_relative_to_start": True,
            "trajectory_start_frac": args.trajectory_start_frac,
            "trajectory_end_frac": args.trajectory_end_frac,
            "trajectory_schedule": args.trajectory_schedule,
            "style_guidance_weight": args.style_guidance_weight,
            "style_start_frac": args.style_start_frac,
            "style_end_frac": args.style_end_frac,
            "style_schedule": args.style_schedule,
            "num_inference_steps": args.num_inference_steps,
            "guidance_steps": args.guidance_steps,
            "style_guidance_steps": args.style_guidance_steps,
            "trajectory_guidance_steps": args.trajectory_guidance_steps,
            "guidance_inner_mode": args.guidance_inner_mode,
            "guidance_order": args.guidance_order,
            "recompute_guided_v_pred": bool(args.recompute_guided_v_pred),
            "print_step_trajectory_loss": bool(args.print_step_trajectory_loss),
            "seed": args.seed,
        },
    )
    if args.print_step_trajectory_loss:
        step_losses = debug_info.get("step_trajectory_losses", [])
        save_json(os.path.join(out_dir, "step_trajectory_losses.json"), step_losses)
        print("\nPer-timestep guidance losses:")
        for row in step_losses:
            step_idx = row.get("step_idx")
            step_idx_str = "?" if step_idx is None else str(step_idx)
            step_frac = row.get("step_frac")
            step_frac_str = "?" if step_frac is None else f"{step_frac:.3f}"
            style_loss = row.get("style_loss")
            style_loss_str = "None" if style_loss is None else f"{style_loss:.6f}"
            traj_loss = row.get("trajectory_loss")
            traj_loss_str = "None" if traj_loss is None else f"{traj_loss:.6f}"
            style_weight = row.get("style_weight")
            style_weight_str = "" if style_weight is None else f" style_w={style_weight:.6f}"
            weight = row.get("trajectory_weight")
            weight_str = "" if weight is None else f" weight={weight:.6f}"
            print(
                f"  step={step_idx_str:>2} timestep={row['timestep']:>4} "
                f"frac={step_frac_str} style_loss={style_loss_str}{style_weight_str} "
                f"traj_loss={traj_loss_str}{weight_str}"
            )
    np.save(os.path.join(out_dir, "target_trajectory.npy"), target_xz_np)
    np.save(os.path.join(out_dir, "generated_root_xz.npy"), root_xz)
    np.save(os.path.join(out_dir, "stylized_motion.npy"), stylized_real.detach().cpu().numpy())

    ref_len = min(style_length, style_motion.shape[0])
    np.save(
        os.path.join(out_dir, "reference_style.npy"),
        reference_real[0, :ref_len].detach().cpu().numpy(),
    )
    save_motion_video(
        os.path.join(out_dir, "reference_style.mp4"),
        joints_reference[0][:ref_len].astype(np.float32),
        title=f"reference style {args.ref_motion_id}",
        fps=20,
    )

    for idx in range(args.num_samples):
        sample_len = int(output_lengths[idx].item())
        title = captions_out[idx] if isinstance(captions_out, (list, tuple)) else args.caption
        aligned_target_xz = target_xz_np[:sample_len] + root_xz[idx : idx + 1, :1, :]
        save_motion_video(
            os.path.join(out_dir, f"sample_{idx:02d}.mp4"),
            joints_stylized[idx][:sample_len].astype(np.float32),
            title=title,
            fps=20,
        )
        plot_root_trajectory(
            os.path.join(out_dir, f"sample_{idx:02d}_trajectory.png"),
            aligned_target_xz[0],
            root_xz[idx][:sample_len],
            title=f"sample {idx:02d}",
        )


if __name__ == "__main__":
    main()
