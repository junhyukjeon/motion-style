import argparse
import os
from math import ceil

import numpy as np
import torch

from guidance_test_utils import (
    build_humanml_dataset,
    build_style_dataset,
    denormalize_motion,
    ensure_dir,
    get_full_motion,
    get_full_motion_with_caption,
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
    parser.add_argument(
        "--caption",
        type=str,
        default="",
        help="Content prompt. Leave empty to reuse the first keyframe-source caption when available.",
    )
    parser.add_argument("--output_length", type=int, default=140, help="Generated motion length in frames.")
    parser.add_argument("--num_samples", type=int, default=4, help="How many samples to draw with the same constraints.")
    parser.add_argument("--num_keyframes", type=int, default=3, help="How many keyframes to constrain.")
    parser.add_argument(
        "--joint_names",
        type=str,
        default="all",
        help="Comma-separated joint names to constrain, or 'all' for the full body.",
    )
    parser.add_argument(
        "--source_dataset",
        type=str,
        default="humanml3d",
        choices=["100style", "humanml3d"],
        help="Dataset used to source key poses.",
    )
    parser.add_argument(
        "--source_motion_ids",
        type=str,
        default="",
        help="Optional comma-separated source motion ids to pull keyframes from. A single motion id will be reused to sample multiple target poses.",
    )
    parser.add_argument(
        "--use_source_caption",
        action="store_true",
        help="Use the first keyframe-source caption as the generation caption when available.",
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
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of DDIM denoising steps used for keyframe-guided sampling.",
    )
    parser.add_argument("--guidance_steps", type=int, default=1, help="Inner guidance steps per diffusion step.")
    parser.add_argument("--style_guidance_steps", type=int, default=1, help="Style-only latent update steps per diffusion step.")
    parser.add_argument("--keyframe_guidance_steps", type=int, default=1, help="Keyframe latent update steps per diffusion step.")
    parser.add_argument(
        "--guidance_inner_mode",
        type=str,
        default="separate",
        choices=["combined", "separate", "style_fixed_then_motion_recompute"],
        help="Guidance update mode. 'separate' uses fixed-v style updates followed by recomputed-v keyframe updates.",
    )
    parser.add_argument(
        "--guidance_order",
        type=str,
        default="style_then_motion",
        choices=["style_then_motion", "motion_then_style"],
        help="Order of the separate style and keyframe motion updates.",
    )
    parser.add_argument(
        "--keyframe_match_mode",
        type=str,
        default="windowed_ordered",
        choices=["fixed_frames", "any_frame", "windowed_ordered"],
        help="Match target key poses at fixed frames, anywhere in the motion, or inside ordered time windows.",
    )
    parser.add_argument(
        "--keyframe_root_relative",
        action="store_true",
        help="Compare key poses in a root-relative joint space instead of absolute global pose.",
    )
    parser.add_argument(
        "--recompute_guided_v_pred",
        action="store_true",
        help="After latent guidance, recompute v_pred from the guided latent before the DDIM step.",
    )
    parser.add_argument(
        "--print_step_keyframe_loss",
        action="store_true",
        help="Print and save the batch-mean style/keyframe losses estimated at each diffusion timestep.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def choose_source_motion_ids(ref_motion_id: str, requested_ids, num_keyframes: int, source_dataset_name: str, source_dataset):
    if requested_ids:
        source_motion_ids = [str(mid) for mid in requested_ids]
        if len(source_motion_ids) < num_keyframes:
            repeats = int(ceil(float(num_keyframes) / float(len(source_motion_ids))))
            source_motion_ids = (source_motion_ids * repeats)[:num_keyframes]
        else:
            source_motion_ids = source_motion_ids[:num_keyframes]
        return source_motion_ids, {"mode": "manual", "requested_ids": [str(mid) for mid in requested_ids], "source_dataset": source_dataset_name}

    if source_dataset_name == "100style":
        return [str(ref_motion_id)] * num_keyframes, {"mode": "style_reference", "source_dataset": source_dataset_name}

    sampled_motion_id = str(np.random.choice([item["motion_id"] for item in source_dataset.items]))
    return [sampled_motion_id] * num_keyframes, {"mode": "random_source", "source_dataset": source_dataset_name, "sampled_motion_id": sampled_motion_id}


def evenly_spaced_frame_indices(length: int, count: int) -> np.ndarray:
    length = max(1, int(length))
    count = max(1, int(count))

    if count == 1:
        return np.array([0], dtype=np.int64)

    span = length - 1
    base_step, remainder = divmod(span, count - 1)
    indices = np.zeros(count, dtype=np.int64)

    cursor = 0
    for idx in range(1, count):
        cursor += base_step + (1 if idx <= remainder else 0)
        indices[idx] = cursor

    return indices


def build_keyframe_targets(dataset, source_motion_ids, output_length, joint_indices, device, mean, std):
    num_keyframes = len(source_motion_ids)
    display_frames = evenly_spaced_frame_indices(output_length, num_keyframes)

    target = torch.zeros(1, num_keyframes, 22, 3, dtype=torch.float32, device=device)
    mask = torch.zeros(1, num_keyframes, 22, dtype=torch.bool, device=device)
    source_records = []
    source_videos = []
    source_motions = []
    source_captions = []

    for k, motion_id in enumerate(source_motion_ids):
        src_motion_norm, src_len, src_caption = get_full_motion_with_caption(dataset, motion_id, device)
        src_motion_real = denormalize_motion(src_motion_norm.unsqueeze(0), mean, std)
        src_joints = motion_to_joints(src_motion_real)[0]
        effective_src_len = min(int(src_len), int(output_length))
        src_motion_real = src_motion_real[:, :effective_src_len]
        src_joints = src_joints[:effective_src_len]
        source_frames = evenly_spaced_frame_indices(effective_src_len, num_keyframes)

        src_frame = int(source_frames[k])

        target[0, k, joint_indices] = src_joints[src_frame, joint_indices]
        mask[0, k, joint_indices] = True
        source_records.append(
            {
                "motion_id": str(motion_id),
                "source_frame": int(src_frame),
                "display_frame": int(display_frames[k]),
                "caption": src_caption,
                "source_length": int(src_len),
                "effective_length": int(effective_src_len),
            }
        )
        source_videos.append((str(motion_id), src_joints.detach().cpu().numpy()))
        source_captions.append(src_caption)
        source_motions.append(
            {
                "motion_id": str(motion_id),
                "caption": src_caption,
                "motion_real": src_motion_real[0].detach().cpu().numpy(),
                "joints": src_joints.detach().cpu().numpy(),
            }
        )

    return (
        torch.tensor(display_frames, dtype=torch.long, device=device),
        target,
        mask,
        source_records,
        source_videos,
        source_motions,
        source_captions,
    )


def build_keyframe_windows(display_frames, output_length, device):
    display_frames = torch.as_tensor(display_frames, dtype=torch.long, device=device)
    num_keyframes = int(display_frames.numel())
    windows = torch.zeros(num_keyframes, 2, dtype=torch.long, device=device)
    if num_keyframes == 0:
        return windows

    boundaries = torch.zeros(num_keyframes + 1, dtype=torch.long, device=device)
    boundaries[0] = 0
    boundaries[-1] = max(int(output_length) - 1, 0)
    if num_keyframes > 1:
        boundaries[1:-1] = torch.div(display_frames[:-1] + display_frames[1:], 2, rounding_mode="floor")

    for key_idx in range(num_keyframes):
        start_idx = int(boundaries[key_idx].item())
        end_idx = int(boundaries[key_idx + 1].item())
        windows[key_idx, 0] = start_idx
        windows[key_idx, 1] = max(start_idx, end_idx)

    return windows


def find_best_match_frames(pred_joints, target_joints, key_mask, frame_windows=None, root_relative=False):
    num_samples, seq_len = pred_joints.shape[:2]
    num_keyframes = target_joints.shape[0]
    best_frames = np.full((num_samples, num_keyframes), -1, dtype=np.int64)
    best_losses = np.full((num_samples, num_keyframes), np.nan, dtype=np.float32)

    if root_relative:
        pred_joints = pred_joints - pred_joints[:, :, :1, :]
        target_joints = target_joints - target_joints[:, :1, :]

    for sample_idx in range(num_samples):
        pred_seq = pred_joints[sample_idx]
        for key_idx in range(num_keyframes):
            joint_mask = key_mask[key_idx]
            if not np.any(joint_mask):
                continue
            if frame_windows is None:
                search_start, search_end = 0, seq_len - 1
            else:
                search_start = max(0, min(int(frame_windows[key_idx, 0]), seq_len - 1))
                search_end = max(search_start, min(int(frame_windows[key_idx, 1]), seq_len - 1))
            diff = pred_seq[search_start : search_end + 1, joint_mask] - target_joints[key_idx, joint_mask]
            frame_loss = np.mean(diff * diff, axis=(1, 2))
            best_offset = int(np.argmin(frame_loss))
            best_frame = search_start + best_offset
            best_frames[sample_idx, key_idx] = best_frame
            best_losses[sample_idx, key_idx] = float(frame_loss[best_offset])

    return best_frames, best_losses


def build_pose_clip(poses: np.ndarray, hold_frames: int = 20) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim != 3:
        raise ValueError(f"Expected poses with shape (K, J, 3), got {poses.shape}")
    if poses.shape[0] == 0:
        raise ValueError("Expected at least one pose to visualize.")
    hold_frames = max(1, int(hold_frames))
    return np.repeat(poses, hold_frames, axis=0)


def build_visual_keyframe_poses(source_motions, source_records) -> np.ndarray:
    joints_by_motion = {item["motion_id"]: item["joints"] for item in source_motions}
    poses = []
    for record in source_records:
        motion_id = record["motion_id"]
        source_frame = int(record["source_frame"])
        poses.append(joints_by_motion[motion_id][source_frame])
    return np.asarray(poses, dtype=np.float32)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config(args.config)
    dataset = build_style_dataset(config)
    source_dataset = build_style_dataset(config) if args.source_dataset == "100style" else build_humanml_dataset(config, train=False)
    model = load_model(config, device)
    mean, std = load_style_stats(config["dataset_style"], device)

    joint_names = [name.strip() for name in args.joint_names.split(",") if name.strip()]
    joint_indices, canonical_names = resolve_joint_names(joint_names)

    style_motion, style_length = get_full_motion(dataset, args.ref_motion_id, device)
    style_batch = repeat_batch(style_motion, args.num_samples)
    style_lengths = torch.full((args.num_samples,), style_length, dtype=torch.long, device=device)
    output_lengths = torch.full((args.num_samples,), args.output_length, dtype=torch.long, device=device)
    requested_ids = [s.strip() for s in args.source_motion_ids.split(",") if s.strip()]
    source_motion_ids, source_selection = choose_source_motion_ids(
        args.ref_motion_id,
        requested_ids,
        args.num_keyframes,
        args.source_dataset,
        source_dataset,
    )
    key_frames, key_targets, key_mask, source_records, source_videos, source_motions, source_captions = build_keyframe_targets(
        dataset=source_dataset,
        source_motion_ids=source_motion_ids,
        output_length=args.output_length,
        joint_indices=joint_indices,
        device=device,
        mean=mean,
        std=std,
    )
    generation_caption = args.caption.strip()
    if args.use_source_caption and source_captions:
        generation_caption = source_captions[0]
    if not generation_caption:
        generation_caption = source_captions[0] if source_captions else "a person is moving"
    captions = [generation_caption] * args.num_samples
    keyframe_windows = build_keyframe_windows(key_frames, args.output_length, device=device)

    guidance = {
        "steps": args.guidance_steps,
        "style_steps": args.style_guidance_steps,
        "keyframe_steps": args.keyframe_guidance_steps,
        "inner_mode": args.guidance_inner_mode,
        "guidance_order": args.guidance_order,
        "num_inference_steps": args.num_inference_steps,
        "recompute_v_guided": bool(args.recompute_guided_v_pred),
        "record_step_trajectory_loss": bool(args.print_step_keyframe_loss),
        "style": {
            "start_frac": args.style_start_frac,
            "end_frac": args.style_end_frac,
            "schedule": args.style_schedule,
        },
        "keyframe": {
            "mode": "joints",
            "weight": args.keyframe_weight,
            "frames": key_frames.unsqueeze(0).repeat(args.num_samples, 1),
            "frame_windows": keyframe_windows.unsqueeze(0).repeat(args.num_samples, 1, 1),
            "target": key_targets.repeat(args.num_samples, 1, 1, 1),
            "mask": key_mask.repeat(args.num_samples, 1, 1),
            "match_mode": args.keyframe_match_mode,
            "root_relative": bool(args.keyframe_root_relative),
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
    debug_info = model.get_last_debug_info()

    stylized_real = denormalize_motion(stylized_norm, mean, std)
    reference_real = denormalize_motion(style_batch, mean, std)
    joints_stylized = motion_to_joints(stylized_real).detach().cpu().numpy()
    joints_reference = motion_to_joints(reference_real).detach().cpu().numpy()
    matched_frames, matched_losses = find_best_match_frames(
        pred_joints=joints_stylized[:, : args.output_length],
        target_joints=key_targets[0].detach().cpu().numpy(),
        key_mask=key_mask[0].detach().cpu().numpy(),
        frame_windows=keyframe_windows.detach().cpu().numpy() if args.keyframe_match_mode == "windowed_ordered" else None,
        root_relative=bool(args.keyframe_root_relative),
    )

    out_dir = os.path.join(
        config["result_dir"],
        "guidance_tests",
        "keyframe",
        f"{args.ref_motion_id}_{slug(generation_caption)}",
    )
    ensure_dir(out_dir)

    save_json(
        os.path.join(out_dir, "metadata.json"),
        {
            "config": args.config,
            "ref_motion_id": args.ref_motion_id,
            "caption": args.caption,
            "generation_caption": generation_caption,
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
            "num_inference_steps": args.num_inference_steps,
            "guidance_steps": args.guidance_steps,
            "style_guidance_steps": args.style_guidance_steps,
            "keyframe_guidance_steps": args.keyframe_guidance_steps,
            "guidance_inner_mode": args.guidance_inner_mode,
            "guidance_order": args.guidance_order,
            "keyframe_match_mode": args.keyframe_match_mode,
            "keyframe_root_relative": bool(args.keyframe_root_relative),
            "recompute_guided_v_pred": bool(args.recompute_guided_v_pred),
            "print_step_keyframe_loss": bool(args.print_step_keyframe_loss),
            "seed": args.seed,
            "source_selection": source_selection,
            "source_dataset": args.source_dataset,
            "source_motion_ids": source_motion_ids,
            "source_captions": source_captions,
            "source_records": source_records,
            "keyframe_windows": keyframe_windows.detach().cpu().tolist(),
            "best_match_frames": matched_frames.tolist(),
            "best_match_losses": matched_losses.tolist(),
        },
    )
    if args.print_step_keyframe_loss:
        step_losses = debug_info.get("step_trajectory_losses", [])
        save_json(os.path.join(out_dir, "step_keyframe_losses.json"), step_losses)
        print("\nPer-timestep guidance losses:")
        for row in step_losses:
            step_idx = row.get("step_idx")
            step_idx_str = "?" if step_idx is None else str(step_idx)
            step_frac = row.get("step_frac")
            step_frac_str = "?" if step_frac is None else f"{step_frac:.3f}"
            style_loss = row.get("style_loss")
            style_loss_str = "None" if style_loss is None else f"{style_loss:.6f}"
            keyframe_loss = row.get("keyframe_loss")
            keyframe_loss_str = "None" if keyframe_loss is None else f"{keyframe_loss:.6f}"
            style_weight = row.get("style_weight")
            style_weight_str = "" if style_weight is None else f" style_w={style_weight:.6f}"
            keyframe_weight = row.get("keyframe_weight")
            keyframe_weight_str = "" if keyframe_weight is None else f" keyframe_w={keyframe_weight:.6f}"
            print(
                f"  step={step_idx_str:>2} timestep={row['timestep']:>4} "
                f"frac={step_frac_str} style_loss={style_loss_str}{style_weight_str} "
                f"keyframe_loss={keyframe_loss_str}{keyframe_weight_str}"
            )
    np.save(os.path.join(out_dir, "keyframe_frames.npy"), key_frames.detach().cpu().numpy())
    np.save(os.path.join(out_dir, "keyframe_windows.npy"), keyframe_windows.detach().cpu().numpy())
    np.save(os.path.join(out_dir, "keyframe_target_joints.npy"), key_targets.detach().cpu().numpy())
    np.save(os.path.join(out_dir, "keyframe_mask.npy"), key_mask.detach().cpu().numpy())
    np.save(os.path.join(out_dir, "stylized_motion.npy"), stylized_real.detach().cpu().numpy())
    np.save(os.path.join(out_dir, "best_match_frames.npy"), matched_frames)
    np.save(os.path.join(out_dir, "best_match_losses.npy"), matched_losses)
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

    target_keyframe_clip = build_pose_clip(
        build_visual_keyframe_poses(source_motions, source_records),
        hold_frames=20,
    )
    np.save(
        os.path.join(out_dir, "reference_keyframe_sequence.npy"),
        target_keyframe_clip.astype(np.float32),
    )
    save_motion_video(
        os.path.join(out_dir, "target_keyframes.mp4"),
        target_keyframe_clip.astype(np.float32),
        title="target keyframe poses",
        fps=20,
    )

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
        matched_idx = np.clip(matched_frames[idx], 0, max(sample_len - 1, 0))
        matched_pose_clip = build_pose_clip(
            joints_stylized[idx][matched_idx],
            hold_frames=20,
        )
        save_motion_video(
            os.path.join(out_dir, f"sample_{idx:02d}_matched_keyframes.mp4"),
            matched_pose_clip.astype(np.float32),
            title=f"matched keyframes sample {idx:02d}",
            fps=20,
        )


if __name__ == "__main__":
    main()
