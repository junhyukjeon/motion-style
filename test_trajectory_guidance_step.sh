#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIG="${CONFIG:-$SCRIPT_DIR/configs/final_final/ours.yaml}"
REF_MOTION_ID="${REF_MOTION_ID:-031617}"
CAPTION="${CAPTION:-a person runs}"
OUTPUT_LENGTH="${OUTPUT_LENGTH:-140}"
RADIUS="${RADIUS:-1.0}"
TURNS="${TURNS:-1.0}"
NUM_SAMPLES="${NUM_SAMPLES:-4}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
GUIDANCE_STEPS="${GUIDANCE_STEPS:-1}"
RECOMPUTE_GUIDED_V_PRED="${RECOMPUTE_GUIDED_V_PRED:-0}"
PRINT_STEP_TRAJECTORY_LOSS="${PRINT_STEP_TRAJECTORY_LOSS:-1}"
SEED="${SEED:-42}"

run_case() {
  local run_tag="$1"
  local traj_weight="$2"
  local traj_start="$3"
  local traj_end="$4"
  local traj_schedule="$5"
  local style_weight="$6"
  local style_start="$7"
  local style_end="$8"
  local style_schedule="$9"

  echo
  echo "=== Running step-guided trajectory case: ${run_tag} ==="

  local extra_args=()
  if [[ "$PRINT_STEP_TRAJECTORY_LOSS" == "1" ]]; then
    extra_args+=(--print_step_trajectory_loss)
  fi
  if [[ "$RECOMPUTE_GUIDED_V_PRED" == "1" ]]; then
    extra_args+=(--recompute_guided_v_pred)
  fi

  python "$SCRIPT_DIR/test_trajectory_guidance.py" \
    --config "$CONFIG" \
    --ref_motion_id "$REF_MOTION_ID" \
    --caption "$CAPTION" \
    --output_length "$OUTPUT_LENGTH" \
    --radius "$RADIUS" \
    --turns "$TURNS" \
    --num_samples "$NUM_SAMPLES" \
    --num_inference_steps "$NUM_INFERENCE_STEPS" \
    --trajectory_weight "$traj_weight" \
    --trajectory_start_frac "$traj_start" \
    --trajectory_end_frac "$traj_end" \
    --trajectory_schedule "$traj_schedule" \
    --style_guidance_weight "$style_weight" \
    --style_start_frac "$style_start" \
    --style_end_frac "$style_end" \
    --style_schedule "$style_schedule" \
    --guidance_steps "$GUIDANCE_STEPS" \
    --seed "$SEED" \
    --run_tag "$run_tag" \
    "${extra_args[@]}"
}

# Baseline: trajectory guidance only, active across the full diffusion chain.
run_case \
  "step_const_full_style0" \
  "5.0" "0.0" "1.0" "constant" \
  "0.0" "0.0" "1.0" "constant"

# Emphasize earlier denoising steps, then fade out.
run_case \
  "step_early_linear_decay_style0" \
  "5.0" "0.0" "0.7" "linear_decay" \
  "0.0" "0.0" "1.0" "constant"

# Concentrate pressure in the early-middle portion of diffusion.
run_case \
  "step_earlymid_bell_style0" \
  "5.0" "0.0" "0.8" "bell" \
  "0.0" "0.0" "1.0" "constant"

# Smooth early emphasis with cosine decay.
run_case \
  "step_early_cosine_decay_style0" \
  "5.0" "0.0" "0.6" "cosine_decay" \
  "0.0" "0.0" "1.0" "constant"

# Keep some style guidance alive in the later half for comparison.
run_case \
  "step_early_decay_style_late" \
  "5.0" "0.0" "0.7" "linear_decay" \
  "0.75" "0.4" "1.0" "linear_ramp"

echo
echo "Step-guided trajectory sweep finished."
echo "Outputs are saved under:"
echo "  $SCRIPT_DIR/results/ours/guidance_tests/trajectory/"
