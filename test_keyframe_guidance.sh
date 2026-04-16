#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIG="${CONFIG:-$SCRIPT_DIR/configs/final_final/ours.yaml}"
REF_MOTION_ID="${REF_MOTION_ID:-031617}"
CAPTION="${CAPTION:-a person walks forward}"
OUTPUT_LENGTH="${OUTPUT_LENGTH:-140}"
NUM_SAMPLES="${NUM_SAMPLES:-4}"
NUM_KEYFRAMES="${NUM_KEYFRAMES:-3}"
JOINT_NAMES="${JOINT_NAMES:-pelvis,left_wrist,right_wrist,left_foot,right_foot}"
SOURCE_MOTION_IDS="${SOURCE_MOTION_IDS:-}"
GUIDANCE_STEPS="${GUIDANCE_STEPS:-1}"
RECOMPUTE_GUIDED_V_PRED="${RECOMPUTE_GUIDED_V_PRED:-0}"
SEED="${SEED:-42}"

run_case() {
  local run_tag="$1"
  local keyframe_weight="$2"
  local keyframe_start="$3"
  local keyframe_end="$4"
  local keyframe_schedule="$5"
  local style_weight="$6"
  local style_start="$7"
  local style_end="$8"
  local style_schedule="$9"

  echo
  echo "=== Running keyframe guidance case: ${run_tag} ==="

  local extra_args=()
  if [[ -n "$SOURCE_MOTION_IDS" ]]; then
    extra_args+=(--source_motion_ids "$SOURCE_MOTION_IDS")
  fi
  if [[ "$RECOMPUTE_GUIDED_V_PRED" == "1" ]]; then
    extra_args+=(--recompute_guided_v_pred)
  fi

  python "$SCRIPT_DIR/test_keyframe_guidance.py" \
    --config "$CONFIG" \
    --ref_motion_id "$REF_MOTION_ID" \
    --caption "$CAPTION" \
    --output_length "$OUTPUT_LENGTH" \
    --num_samples "$NUM_SAMPLES" \
    --num_keyframes "$NUM_KEYFRAMES" \
    --joint_names "$JOINT_NAMES" \
    --keyframe_weight "$keyframe_weight" \
    --keyframe_start_frac "$keyframe_start" \
    --keyframe_end_frac "$keyframe_end" \
    --keyframe_schedule "$keyframe_schedule" \
    --style_guidance_weight "$style_weight" \
    --style_start_frac "$style_start" \
    --style_end_frac "$style_end" \
    --style_schedule "$style_schedule" \
    --guidance_steps "$GUIDANCE_STEPS" \
    --seed "$SEED" \
    "${extra_args[@]}"
}

# Baseline: keyframe guidance only, active across the full diffusion chain.
run_case \
  "key_const_full_style0" \
  "8.0" "0.0" "1.0" "constant" \
  "0.0" "0.0" "1.0" "constant"

# Stronger push earlier in denoising, then fade out.
run_case \
  "key_early_linear_decay_style0" \
  "8.0" "0.0" "0.7" "linear_decay" \
  "0.0" "0.0" "1.0" "constant"

# Concentrate keyframe pressure around the early-middle steps.
run_case \
  "key_earlymid_bell_style0" \
  "8.0" "0.0" "0.8" "bell" \
  "0.0" "0.0" "1.0" "constant"

# Smooth early emphasis with cosine decay.
run_case \
  "key_early_cosine_decay_style0" \
  "8.0" "0.0" "0.6" "cosine_decay" \
  "0.0" "0.0" "1.0" "constant"

# Compare against preserving some style guidance late in the chain.
run_case \
  "key_early_decay_style_late" \
  "8.0" "0.0" "0.7" "linear_decay" \
  "0.75" "0.4" "1.0" "linear_ramp"

echo
echo "Keyframe guidance sweep finished."
echo "Outputs are saved under:"
echo "  $SCRIPT_DIR/results/ours/guidance_tests/keyframe/"
