#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Override any of these from the shell, for example:
#   SOURCE_MOTION_ID=036222 STYLE_MOTION_ID=036201 CAPTION="a person is jogging" \
#   bash style-salad/test_ddim_inversion_transfer.sh
CONFIG="${CONFIG:-$SCRIPT_DIR/configs/final_final/ours.yaml}"
SOURCE_MOTION_ID="${SOURCE_MOTION_ID:-031947}"
SOURCE_DATASET="${SOURCE_DATASET:-100style}"
STYLE_MOTION_ID="${STYLE_MOTION_ID:-031662}"
CAPTION="${CAPTION:-"a person walks forward"}"
# Leave empty to reuse the cropped source length, or set a frame count to trim
# the source clip first. Inversion and output will both use that trimmed length.
OUTPUT_LENGTH="${OUTPUT_LENGTH:-88}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
INVERSION_CONDITIONING="${INVERSION_CONDITIONING:-text_style}"
CFG_TEXT_WEIGHT_INVERSION="${CFG_TEXT_WEIGHT_INVERSION:-}"
CFG_TEXT_WEIGHT_DENOISING="${CFG_TEXT_WEIGHT_DENOISING:-}"
CFG_STYLE_WEIGHT_INVERSION="${CFG_STYLE_WEIGHT_INVERSION:-0}"
CFG_STYLE_WEIGHT_DENOISING="${CFG_STYLE_WEIGHT_DENOISING:-1.5}"
STYLE_GUIDANCE_WEIGHT="${STYLE_GUIDANCE_WEIGHT:-0.75}"
SEED="${SEED:-42}"

extra_args=("$@")

config_stem="$(basename "${CONFIG%.*}")"
results_root="$SCRIPT_DIR/results/$config_stem/guidance_tests/ddim_inversion_transfer"

cmd=(
  python "$SCRIPT_DIR/test_ddim_inversion_transfer.py"
  --config "$CONFIG"
  --source_motion_id "$SOURCE_MOTION_ID"
  --source_dataset "$SOURCE_DATASET"
  --style_motion_id "$STYLE_MOTION_ID"
  --caption "$CAPTION"
  --num_inference_steps "$NUM_INFERENCE_STEPS"
  --inversion_conditioning "$INVERSION_CONDITIONING"
  --seed "$SEED"
)

if [[ -n "$OUTPUT_LENGTH" ]]; then
  cmd+=(--output_length "$OUTPUT_LENGTH")
fi

if [[ -n "$CFG_TEXT_WEIGHT_INVERSION" ]]; then
  cmd+=(--cfg_text_weight_inversion "$CFG_TEXT_WEIGHT_INVERSION")
fi

if [[ -n "$CFG_TEXT_WEIGHT_DENOISING" ]]; then
  cmd+=(--cfg_text_weight_denoising "$CFG_TEXT_WEIGHT_DENOISING")
fi

if [[ -n "$CFG_STYLE_WEIGHT_INVERSION" ]]; then
  cmd+=(--cfg_style_weight_inversion "$CFG_STYLE_WEIGHT_INVERSION")
fi

if [[ -n "$CFG_STYLE_WEIGHT_DENOISING" ]]; then
  cmd+=(--cfg_style_weight_denoising "$CFG_STYLE_WEIGHT_DENOISING")
fi

if [[ -n "$STYLE_GUIDANCE_WEIGHT" ]]; then
  cmd+=(--style_guidance_weight "$STYLE_GUIDANCE_WEIGHT")
fi

if [[ "${#extra_args[@]}" -gt 0 ]]; then
  cmd+=("${extra_args[@]}")
fi

echo
echo "=== Running DDIM inversion motion style transfer ==="
echo "config:                $CONFIG"
echo "source_motion_id:      $SOURCE_MOTION_ID"
echo "source_dataset:        $SOURCE_DATASET"
echo "style_motion_id:       $STYLE_MOTION_ID"
echo "caption:               $CAPTION"
echo "requested_length:      ${OUTPUT_LENGTH:-<cropped source length>}"
echo "num_inference_steps:   $NUM_INFERENCE_STEPS"
echo "inversion_conditioning:$INVERSION_CONDITIONING"
echo "cfg_text_weight_inv:   ${CFG_TEXT_WEIGHT_INVERSION:-<config default>}"
echo "cfg_text_weight_den:   ${CFG_TEXT_WEIGHT_DENOISING:-<config default>}"
echo "cfg_style_weight_inv:  ${CFG_STYLE_WEIGHT_INVERSION:-<config default>}"
echo "cfg_style_weight_den:  ${CFG_STYLE_WEIGHT_DENOISING:-<config default>}"
echo "style_guidance_weight: ${STYLE_GUIDANCE_WEIGHT:-<config default>}"
echo "seed:                  $SEED"
echo

"${cmd[@]}"

run_dir="$(
  python - "$SCRIPT_DIR" "$config_stem" "$SOURCE_MOTION_ID" "$STYLE_MOTION_ID" "$CAPTION" <<'PY'
import os
import sys

script_dir, config_stem, source_id, style_id, caption = sys.argv[1:]

def slug(s: str, maxlen: int = 80) -> str:
    s = "".join(c if (c.isalnum() or c in " _-.,()[]{}") else "_" for c in s.strip())
    s = "_".join(s.split())
    return s[:maxlen].rstrip("_") or "sample"

print(
    os.path.join(
        script_dir,
        "results",
        config_stem,
        "guidance_tests",
        "ddim_inversion_transfer",
        f"{source_id}_to_{style_id}_{slug(caption)}",
    )
)
PY
)"

python - "$SCRIPT_DIR" "$run_dir" <<'PY'
import json
import os
import sys

import numpy as np
import torch

script_dir, run_dir = sys.argv[1:]
sys.path.insert(0, script_dir)

from utils.motion import recover_root_rot_pos


def path_progress(path: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if path.shape[0] <= 1:
        return torch.zeros(path.shape[0], device=path.device, dtype=path.dtype)

    deltas = path[1:] - path[:-1]
    seg_lengths = torch.linalg.norm(deltas, dim=-1)
    total_length = seg_lengths.sum()
    if float(total_length.detach().item()) <= eps:
        return torch.linspace(0.0, 1.0, path.shape[0], device=path.device, dtype=path.dtype)

    progress = torch.zeros(path.shape[0], device=path.device, dtype=path.dtype)
    progress[1:] = torch.cumsum(seg_lengths, dim=0) / total_length.clamp_min(eps)
    return progress


def resample_path_by_progress(
    path: torch.Tensor,
    progress: torch.Tensor,
    sample_progress: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    if path.shape[0] == 1:
        return path.expand(sample_progress.shape[0], -1)

    sample_idx = torch.searchsorted(progress, sample_progress, right=True)
    upper_idx = sample_idx.clamp(1, path.shape[0] - 1)
    lower_idx = upper_idx - 1

    lower_progress = progress[lower_idx]
    upper_progress = progress[upper_idx]
    alpha = ((sample_progress - lower_progress) / (upper_progress - lower_progress).clamp_min(eps)).unsqueeze(-1)
    return path[lower_idx] + alpha * (path[upper_idx] - path[lower_idx])


def trajectory_shape_loss(pred_path: torch.Tensor, target_path: torch.Tensor) -> float:
    pred_len = pred_path.shape[0]
    target_len = target_path.shape[0]
    if pred_len == 0 or target_len == 0:
        return 0.0

    if pred_len == 1 or target_len == 1:
        shared = min(pred_len, target_len)
        return float(torch.mean((pred_path[:shared] - target_path[:shared]) ** 2).item())

    num_shape_samples = max(2, min(pred_len, target_len))
    sample_progress = torch.linspace(0.0, 1.0, num_shape_samples, device=pred_path.device, dtype=pred_path.dtype)

    pred_progress = path_progress(pred_path)
    target_progress = path_progress(target_path)
    pred_shape = resample_path_by_progress(pred_path, pred_progress, sample_progress)
    target_shape = resample_path_by_progress(target_path, target_progress, sample_progress)
    return float(torch.mean((pred_shape - target_shape) ** 2).item())


metadata_path = os.path.join(run_dir, "metadata.json")
source_path = os.path.join(run_dir, "source_motion.npy")
edited_path = os.path.join(run_dir, "edited_motion.npy")

metadata = {}
if os.path.exists(metadata_path):
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

source = torch.tensor(np.load(source_path), dtype=torch.float32)
edited = torch.tensor(np.load(edited_path), dtype=torch.float32)

source_len = int(metadata.get("source_length", source.shape[1]))
edited_len = int(metadata.get("output_length", edited.shape[1]))
shared_len = min(source_len, edited_len, int(source.shape[1]), int(edited.shape[1]))

source_root = recover_root_rot_pos(source[:, :shared_len])[1][0, :, [0, 2]]
edited_root = recover_root_rot_pos(edited[:, :shared_len])[1][0, :, [0, 2]]

abs_loss = trajectory_shape_loss(edited_root, source_root)
source_rel = source_root - source_root[:1]
edited_rel = edited_root - edited_root[:1]
rel_loss = trajectory_shape_loss(edited_rel, source_rel)
start_l2 = float(torch.linalg.norm(edited_root[0] - source_root[0]).item())

metrics = {
    "trajectory_loss_root_xz_abs": abs_loss,
    "trajectory_loss_root_xz_relative": rel_loss,
    "root_start_l2": start_l2,
    "shared_length": shared_len,
}

metrics_path = os.path.join(run_dir, "trajectory_metrics.json")
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print()
print("Trajectory metrics:")
print(f"  abs_root_xz_shape_mse: {abs_loss:.6f}")
print(f"  rel_root_xz_shape_mse: {rel_loss:.6f}")
print(f"  root_start_l2:         {start_l2:.6f}")
print(f"  saved:                 {metrics_path}")
PY

echo
echo "DDIM inversion transfer run finished."
echo "Outputs are saved under:"
echo "  $results_root"
