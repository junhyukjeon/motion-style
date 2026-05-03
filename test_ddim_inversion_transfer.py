import argparse
import os

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
    save_json,
    save_motion_video,
    set_seed,
    slug,
)


def crop_eval_window(
    motion: torch.Tensor,
    length: int,
    unit_length: int,
    max_frames: int | None,
    align: str = "center",
):
    length = max(1, min(int(length), int(motion.shape[0])))

    if unit_length > 0:
        cropped_length = max(1, length // unit_length) * unit_length
    else:
        cropped_length = length

    if max_frames is not None:
        cropped_length = min(cropped_length, int(max_frames))
    cropped_length = max(1, min(cropped_length, length))

    if align == "start":
        start = 0
    elif align == "center":
        start = max(0, (length - cropped_length) // 2)
    else:
        raise ValueError(f"Unsupported crop alignment: {align}")
    window = motion[start : start + cropped_length]
    return window, int(cropped_length), int(start)


def parse_args():
    parser = argparse.ArgumentParser(
        description="DDIM-invert a source motion, then edit it with text and style conditioning."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML.")
    parser.add_argument("--source_motion_id", type=str, required=True, help="Motion id used as the inversion source.")
    parser.add_argument(
        "--source_dataset",
        type=str,
        default="100style",
        choices=["100style", "humanml3d"],
        help="Dataset used for the inversion source motion.",
    )
    parser.add_argument("--style_motion_id", type=str, required=True, help="Motion id used as the style reference.")
    parser.add_argument("--caption", type=str, required=True, help="Target content prompt.")
    parser.add_argument(
        "--output_length",
        type=int,
        default=None,
        help="Edited motion length in frames. Defaults to the source motion length. Must match for DDIM inversion.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of DDIM steps used for both inversion and forward denoising.",
    )
    parser.add_argument(
        "--inversion_conditioning",
        type=str,
        default="text_style",
        choices=["uncond", "text", "text_style"],
        help="Conditioning used during DDIM inversion before the forward text+style edit.",
    )
    parser.add_argument(
        "--style_guidance_weight",
        type=float,
        default=None,
        help="Optional extra step-guidance style weight during the forward denoising pass.",
    )
    parser.add_argument(
        "--cfg_text_weight_inversion",
        type=float,
        default=None,
        help="Optional override for the CFG text weight used only during DDIM inversion.",
    )
    parser.add_argument(
        "--cfg_text_weight_denoising",
        type=float,
        default=None,
        help="Optional override for the CFG text weight used only during reverse denoising.",
    )
    parser.add_argument(
        "--cfg_style_weight_inversion",
        type=float,
        default=None,
        help="Optional override for the CFG style weight used only during DDIM inversion.",
    )
    parser.add_argument(
        "--cfg_style_weight_denoising",
        type=float,
        default=None,
        help="Optional override for the CFG style weight used only during reverse denoising.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config(args.config)
    style_dataset = build_style_dataset(config)
    source_dataset = style_dataset if args.source_dataset == "100style" else build_humanml_dataset(config, train=False)
    model = load_model(config, device)
    default_cfg_text_weight = float(model.config["text_weight"])
    cfg_text_weight_inversion = default_cfg_text_weight
    cfg_text_weight_denoising = default_cfg_text_weight
    if args.cfg_text_weight_inversion is not None:
        cfg_text_weight_inversion = float(args.cfg_text_weight_inversion)
    if args.cfg_text_weight_denoising is not None:
        cfg_text_weight_denoising = float(args.cfg_text_weight_denoising)

    default_cfg_style_weight = float(model.config["style_weight"])
    cfg_style_weight_inversion = default_cfg_style_weight
    cfg_style_weight_denoising = default_cfg_style_weight

    if args.cfg_style_weight_inversion is not None:
        cfg_style_weight_inversion = float(args.cfg_style_weight_inversion)
    if args.cfg_style_weight_denoising is not None:
        cfg_style_weight_denoising = float(args.cfg_style_weight_denoising)

    model.config["text_weight"] = cfg_text_weight_denoising
    model.config["style_weight"] = cfg_style_weight_denoising
    mean, std = load_style_stats(config["dataset_style"], device)
    style_cfg = config["dataset_style"]
    unit_length = int(style_cfg.get("unit_length", 4))
    max_frames = style_cfg.get("max_frames")

    source_motion_full, source_length_raw, source_caption = get_full_motion_with_caption(
        source_dataset,
        args.source_motion_id,
        device,
    )
    style_motion_full, style_length_raw = get_full_motion(style_dataset, args.style_motion_id, device)
    requested_output_length = None if args.output_length is None else int(args.output_length)

    source_max_frames = requested_output_length if requested_output_length is not None else max_frames
    source_align = "start" if requested_output_length is not None else "center"
    style_max_frames = max_frames
    style_align = "center"
    source_motion, source_length, source_start = crop_eval_window(
        source_motion_full,
        source_length_raw,
        unit_length=unit_length,
        max_frames=source_max_frames,
        align=source_align,
    )
    style_motion, style_length, style_start = crop_eval_window(
        style_motion_full,
        style_length_raw,
        unit_length=unit_length,
        max_frames=style_max_frames,
        align=style_align,
    )

    output_length = source_length

    source_batch = repeat_batch(source_motion, 1)
    style_batch = repeat_batch(style_motion, 1)
    source_lengths = torch.tensor([source_length], dtype=torch.long, device=device)
    style_lengths = torch.tensor([style_length], dtype=torch.long, device=device)
    output_lengths = torch.tensor([output_length], dtype=torch.long, device=device)
    captions = [args.caption]

    guidance = {
        "num_inference_steps": args.num_inference_steps,
        "inversion": {
            "conditioning": args.inversion_conditioning,
            "text_weight": cfg_text_weight_inversion,
            "style_weight": cfg_style_weight_inversion,
        },
    }
    if args.style_guidance_weight is not None:
        guidance["style"] = {
            "weight": args.style_guidance_weight,
            "start_frac": 0.0,
            "end_frac": 1.0,
            "schedule": "constant",
        }

    sampling_ctx = model._prepare_sampling_context(
        motion=style_batch,
        text=captions,
        lengths=output_lengths,
        style_lengths=style_lengths,
        num_inference_steps=args.num_inference_steps,
    )
    inversion_ctx = model._prepare_sampling_context(
        motion=source_batch,
        text=captions,
        lengths=source_lengths,
        style_lengths=source_lengths,
        num_inference_steps=args.num_inference_steps,
    )
    z = model._invert_source_motion(
        motion=source_batch,
        lengths=source_lengths,
        ctx=inversion_ctx,
        inversion_cfg=guidance.get("inversion"),
    )
    stylized_norm, debug_info = model._generate_from_noise_with_step_guidance(z, sampling_ctx, guidance)
    debug_info["used_ddim_inversion"] = True
    debug_info["inversion"] = {
        "conditioning": str(guidance.get("inversion", {}).get("conditioning", "uncond")).lower(),
        "num_inference_steps": int(args.num_inference_steps),
        "text_source": "target_caption",
        "style_source": "source_motion",
    }
    model.last_debug_info = debug_info
    captions_out = captions

    stylized_real = denormalize_motion(stylized_norm, mean, std)
    source_real = denormalize_motion(source_batch, mean, std)
    style_real = denormalize_motion(style_batch, mean, std)

    joints_stylized = motion_to_joints(stylized_real).detach().cpu().numpy()
    joints_source = motion_to_joints(source_real).detach().cpu().numpy()
    joints_style = motion_to_joints(style_real).detach().cpu().numpy()

    out_dir = os.path.join(
        config["result_dir"],
        "guidance_tests",
        "ddim_inversion_transfer",
        f"{args.source_motion_id}_to_{args.style_motion_id}_{slug(args.caption)}",
    )
    ensure_dir(out_dir)

    save_json(
        os.path.join(out_dir, "metadata.json"),
        {
            "config": args.config,
            "source_motion_id": args.source_motion_id,
            "source_dataset": args.source_dataset,
            "style_motion_id": args.style_motion_id,
            "caption": args.caption,
            "source_caption": source_caption,
            "requested_output_length": requested_output_length,
            "output_length": output_length,
            "source_length_raw": source_length_raw,
            "source_length": source_length,
            "source_start_frame": source_start,
            "source_crop_align": source_align,
            "style_length_raw": style_length_raw,
            "style_length": style_length,
            "style_start_frame": style_start,
            "style_crop_align": style_align,
            "unit_length": unit_length,
            "max_frames": max_frames,
            "num_inference_steps": args.num_inference_steps,
            "inversion_conditioning": args.inversion_conditioning,
            "cfg_text_weight_inversion": cfg_text_weight_inversion,
            "cfg_text_weight_denoising": cfg_text_weight_denoising,
            "cfg_style_weight_inversion": cfg_style_weight_inversion,
            "cfg_style_weight_denoising": cfg_style_weight_denoising,
            "style_guidance_weight": args.style_guidance_weight,
            "seed": args.seed,
            "model_debug": debug_info,
        },
    )

    np.save(os.path.join(out_dir, "source_motion.npy"), source_real.detach().cpu().numpy())
    np.save(os.path.join(out_dir, "reference_style.npy"), style_real.detach().cpu().numpy())
    np.save(os.path.join(out_dir, "edited_motion.npy"), stylized_real.detach().cpu().numpy())

    save_motion_video(
        os.path.join(out_dir, "source_motion.mp4"),
        joints_source[0][:source_length].astype(np.float32),
        title=f"source {args.source_dataset} {args.source_motion_id}",
        fps=20,
    )
    save_motion_video(
        os.path.join(out_dir, "reference_style.mp4"),
        joints_style[0][:style_length].astype(np.float32),
        title=f"style {args.style_motion_id}",
        fps=20,
    )
    save_motion_video(
        os.path.join(out_dir, "edited_motion.mp4"),
        joints_stylized[0][:output_length].astype(np.float32),
        title=captions_out[0] if isinstance(captions_out, (list, tuple)) else args.caption,
        fps=20,
    )


if __name__ == "__main__":
    main()
