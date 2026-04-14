#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIG="${CONFIG:-$SCRIPT_DIR/configs/final_final/ours.yaml}"
REF_MOTION_ID="${REF_MOTION_ID:-031617}"
CAPTION="${CAPTION:-a person walks}"
OUTPUT_LENGTH="${OUTPUT_LENGTH:-140}"
RADIUS="${RADIUS:-1.0}"
TURNS="${TURNS:-1.0}"
NUM_SAMPLES="${NUM_SAMPLES:-4}"
GUIDANCE_STEPS="${GUIDANCE_STEPS:-1}"
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
  echo "=== Running trajectory sweep case: ${run_tag} ==="

  python "$SCRIPT_DIR/test_trajectory_guidance.py" \
    --config "$CONFIG" \
    --ref_motion_id "$REF_MOTION_ID" \
    --caption "$CAPTION" \
    --output_length "$OUTPUT_LENGTH" \
    --radius "$RADIUS" \
    --turns "$TURNS" \
    --num_samples "$NUM_SAMPLES" \
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
    --run_tag "$run_tag"
}

run_zt_case() {
  local run_tag="$1"
  local traj_weight="$2"
  local noise_opt_steps="$3"
  local noise_opt_lr="$4"
  local num_inference_steps="$5"

  echo
  echo "=== Running initial-z_T optimization case: ${run_tag} ==="

  python "$SCRIPT_DIR/test_trajectory_guidance.py" \
    --config "$CONFIG" \
    --ref_motion_id "$REF_MOTION_ID" \
    --caption "$CAPTION" \
    --output_length "$OUTPUT_LENGTH" \
    --radius "$RADIUS" \
    --turns "$TURNS" \
    --num_samples "$NUM_SAMPLES" \
    --trajectory_weight "$traj_weight" \
    --style_guidance_weight "0.0" \
    --num_inference_steps "$num_inference_steps" \
    --optimize_initial_noise_only \
    --noise_opt_steps "$noise_opt_steps" \
    --noise_opt_lr "$noise_opt_lr" \
    --seed "$SEED" \
    --run_tag "$run_tag"
}

# Baseline: no style guidance, constant trajectory pressure everywhere.
run_case \
  "baseline_const_full_style0" \
  "5.0" "0.0" "1.0" "constant" \
  "0.0" "0.0" "1.0" "constant"

# Strong global push early, fades later.
run_case \
  "traj_early_linear_decay_style0" \
  "5.0" "0.0" "0.7" "linear_decay" \
  "0.0" "0.0" "1.0" "constant"

# Global guidance concentrated in early-mid timesteps.
run_case \
  "traj_earlymid_bell_style0" \
  "5.0" "0.0" "0.8" "bell" \
  "0.0" "0.0" "1.0" "constant"

# Early guidance with a smoother decay.
run_case \
  "traj_early_cosine_decay_style0" \
  "5.0" "0.0" "0.6" "cosine_decay" \
  "0.0" "0.0" "1.0" "constant"

# Compare against keeping style guidance on in the later half.
run_case \
  "traj_early_decay_style_late" \
  "5.0" "0.0" "0.7" "linear_decay" \
  "0.75" "0.4" "1.0" "linear_ramp"

# Optimize only the starting z_T against the final trajectory loss.
run_zt_case \
  "zt_only_opt10_lr5em2_n50" \
  "5.0" "10" "0.05" "50"

# Same idea, but a bit more optimization pressure.
run_zt_case \
  "zt_only_opt20_lr2em2_n50" \
  "5.0" "20" "0.02" "50"

echo
echo "Trajectory sweep finished."
echo "Outputs are saved under:"
echo "  $SCRIPT_DIR/results/ours/guidance_tests/trajectory/"
