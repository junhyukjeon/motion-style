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
    repeat_batch,
    resolve_joint_names,
    save_json,
    save_motion_video,
    set_seed,
    slug,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Test joint-space keyframe guidance for Text2StylizedMotion.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML.")
    parser.add_argument("--ref_motion_id", type=str, required=True, help="100STYLE motion id used as style reference.")
    parser.add_argument("--caption", type=str, default="a person walks forward", help="Content prompt.")
    parser.add_argument("--output_length", type=int, default=140, help="Generated motion length in frames.")
    parser.add_argument("--num_samples", type=int, default=4, help="How many samples to draw with the same constraints.")
    parser.add_argument("--num_keyframes", type=int, default=3, help="How many keyframes to constrain.")
    parser.add_argument(
        "--joint_names",
        type=str,
        default="pelvis,left_wrist,right_wrist,left_foot,right_foot",
        help="Comma-separated joint names to constrain.",
    )
    parser.add_argument(
        "--source_motion_ids",
        type=str,
        default="",
        help="Optional comma-separated source motion ids to pull keyframes from. If omitted, keyframes are taken from the style reference motion.",
    )
    parser.add_argument("--keyframe_weight", type=float, default=8.0, help="Keyframe guidance weight.")
    parser.add_argument("--keyframe_start_frac", type=float, default=0.0, help="Keyframe guidance start fraction in denoising.")
    parser.add_argument("--keyframe_end_frac", type=float, default=1.0, help="Keyframe guidance end fraction in denoising.")
    parser.add_argument(
        "--keyframe_schedule",
        type=str,
        default="constant",
        choices=["constant", "linear_ramp", "linear_decay", "cosine_ramp", "cosine_decay", "bell"],
        help="Keyframe guidance schedule within its active timestep window.",
    )
    parser.add_argument(
        "--style_guidance_weight",
        type=float,
        default=None,
        help="Optional override for style guidance weight during keyframe tests. Use 0 to disable.",
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
    parser.add_argument("--guidance_steps", type=int, default=1, help="Inner guidance steps per diffusion step.")
    parser.add_argument(
        "--recompute_guided_v_pred",
        action="store_true",
        help="After latent guidance, recompute v_pred from the guided latent before the DDIM step.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def choose_source_motion_ids(ref_motion_id: str, requested_ids, num_keyframes: int):
    if requested_ids:
        return [str(mid) for mid in requested_ids], {"mode": "manual"}
    return [str(ref_motion_id)] * num_keyframes, {"mode": "style_reference"}


def build_keyframe_targets(dataset, source_motion_ids, output_length, joint_indices, device, mean, std):
    num_keyframes = len(source_motion_ids)
    frame_positions = np.linspace(0.2, 0.8, num=num_keyframes, endpoint=True)
    out_frames = np.round(frame_positions * max(output_length - 1, 1)).astype(np.int64)

    target = torch.zeros(1, num_keyframes, 22, 3, dtype=torch.float32, device=device)
    mask = torch.zeros(1, num_keyframes, 22, dtype=torch.bool, device=device)
    source_records = []
    source_videos = []
    source_motions = []

    for k, motion_id in enumerate(source_motion_ids):
        src_motion_norm, src_len = get_full_motion(dataset, motion_id, device)
        src_motion_real = denormalize_motion(src_motion_norm.unsqueeze(0), mean, std)
        src_joints = motion_to_joints(src_motion_real)[0]

        ratio = float(out_frames[k]) / float(max(output_length - 1, 1))
        src_frame = int(round(ratio * max(src_len - 1, 0)))
        src_frame = max(0, min(src_frame, src_len - 1))

        target[0, k, joint_indices] = src_joints[src_frame, joint_indices]
        mask[0, k, joint_indices] = True
        source_records.append(
            {
                "motion_id": str(motion_id),
                "source_frame": int(src_frame),
                "output_frame": int(out_frames[k]),
            }
        )
        source_videos.append((str(motion_id), src_joints[:src_len].detach().cpu().numpy()))
        source_motions.append(
            {
                "motion_id": str(motion_id),
                "motion_real": src_motion_real[0, :src_len].detach().cpu().numpy(),
                "joints": src_joints[:src_len].detach().cpu().numpy(),
            }
        )

    return (
        torch.tensor(out_frames, dtype=torch.long, device=device),
        target,
        mask,
        source_records,
        source_videos,
        source_motions,
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config(args.config)
    dataset = build_style_dataset(config)
    model = load_model(config, device)
    mean, std = load_style_stats(config["dataset_style"], device)

    joint_names = [name.strip() for name in args.joint_names.split(",") if name.strip()]
    joint_indices, canonical_names = resolve_joint_names(joint_names)

    style_motion, style_length = get_full_motion(dataset, args.ref_motion_id, device)
    style_batch = repeat_batch(style_motion, args.num_samples)
    style_lengths = torch.full((args.num_samples,), style_length, dtype=torch.long, device=device)
    output_lengths = torch.full((args.num_samples,), args.output_length, dtype=torch.long, device=device)
    captions = [args.caption] * args.num_samples

    requested_ids = [s.strip() for s in args.source_motion_ids.split(",") if s.strip()]
    source_motion_ids, source_selection = choose_source_motion_ids(
        args.ref_motion_id,
        requested_ids,
        args.num_keyframes,
    )
    key_frames, key_targets, key_mask, source_records, source_videos, source_motions = build_keyframe_targets(
        dataset=dataset,
        source_motion_ids=source_motion_ids,
        output_length=args.output_length,
        joint_indices=joint_indices,
        device=device,
        mean=mean,
        std=std,
    )

    guidance = {
        "steps": args.guidance_steps,
        "recompute_v_guided": bool(args.recompute_guided_v_pred),
        "style": {
            "start_frac": args.style_start_frac,
            "end_frac": args.style_end_frac,
            "schedule": args.style_schedule,
        },
        "keyframe": {
            "mode": "joints",
            "weight": args.keyframe_weight,
            "frames": key_frames.unsqueeze(0).repeat(args.num_samples, 1),
            "target": key_targets.repeat(args.num_samples, 1, 1, 1),
            "mask": key_mask.repeat(args.num_samples, 1, 1),
            "start_frac": args.keyframe_start_frac,
            "end_frac": args.keyframe_end_frac,
            "schedule": args.keyframe_schedule,
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

    stylized_real = denormalize_motion(stylized_norm, mean, std)
    reference_real = denormalize_motion(style_batch, mean, std)
    joints_stylized = motion_to_joints(stylized_real).detach().cpu().numpy()
    joints_reference = motion_to_joints(reference_real).detach().cpu().numpy()

    out_dir = os.path.join(
        config["result_dir"],
        "guidance_tests",
        "keyframe",
        f"{args.ref_motion_id}_{slug(args.caption)}",
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
            "num_keyframes": args.num_keyframes,
            "joint_names": canonical_names,
            "keyframe_weight": args.keyframe_weight,
            "keyframe_start_frac": args.keyframe_start_frac,
            "keyframe_end_frac": args.keyframe_end_frac,
            "keyframe_schedule": args.keyframe_schedule,
            "style_guidance_weight": args.style_guidance_weight,
            "style_start_frac": args.style_start_frac,
            "style_end_frac": args.style_end_frac,
            "style_schedule": args.style_schedule,
            "guidance_steps": args.guidance_steps,
            "recompute_guided_v_pred": bool(args.recompute_guided_v_pred),
            "seed": args.seed,
            "source_selection": source_selection,
            "source_records": source_records,
        },
    )
    np.save(os.path.join(out_dir, "keyframe_frames.npy"), key_frames.detach().cpu().numpy())
    np.save(os.path.join(out_dir, "keyframe_target_joints.npy"), key_targets.detach().cpu().numpy())
    np.save(os.path.join(out_dir, "keyframe_mask.npy"), key_mask.detach().cpu().numpy())
    np.save(os.path.join(out_dir, "stylized_motion.npy"), stylized_real.detach().cpu().numpy())
    keyframe_motion = np.full((args.output_length, 22, 3), np.nan, dtype=np.float32)
    keyframe_motion_mask = np.zeros((args.output_length, 22), dtype=bool)
    keyframe_frames_np = key_frames.detach().cpu().numpy()
    keyframe_target_np = key_targets[0].detach().cpu().numpy()
    keyframe_mask_np = key_mask[0].detach().cpu().numpy()
    for i, frame_idx in enumerate(keyframe_frames_np):
        keyframe_motion[frame_idx, keyframe_mask_np[i]] = keyframe_target_np[i, keyframe_mask_np[i]]
        keyframe_motion_mask[frame_idx] = keyframe_mask_np[i]
    np.save(os.path.join(out_dir, "keyframe_motion.npy"), keyframe_motion)
    np.save(os.path.join(out_dir, "keyframe_motion_mask.npy"), keyframe_motion_mask)

    ref_len = min(style_length, style_motion.shape[0])
    save_motion_video(
        os.path.join(out_dir, "reference_style.mp4"),
        joints_reference[0][:ref_len].astype(np.float32),
        title=f"reference style {args.ref_motion_id}",
        fps=20,
    )

    sources_dir = os.path.join(out_dir, "source_keyframes")
    ensure_dir(sources_dir)
    for motion_id, joints in source_videos:
        save_motion_video(
            os.path.join(sources_dir, f"{motion_id}.mp4"),
            joints.astype(np.float32),
            title=f"source {motion_id}",
            fps=20,
        )
    for source_motion in source_motions:
        motion_id = source_motion["motion_id"]
        np.save(os.path.join(sources_dir, f"{motion_id}_motion.npy"), source_motion["motion_real"].astype(np.float32))
        np.save(os.path.join(sources_dir, f"{motion_id}_joints.npy"), source_motion["joints"].astype(np.float32))

    for idx in range(args.num_samples):
        sample_len = int(output_lengths[idx].item())
        title = captions_out[idx] if isinstance(captions_out, (list, tuple)) else args.caption
        save_motion_video(
            os.path.join(out_dir, f"sample_{idx:02d}.mp4"),
            joints_stylized[idx][:sample_len].astype(np.float32),
            title=title,
            fps=20,
        )


if __name__ == "__main__":
    main()
