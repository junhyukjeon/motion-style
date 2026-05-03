#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo
echo "=== Running constant keyframe guidance case ==="

extra_args=()
extra_args+=(--print_step_keyframe_loss)

python "$SCRIPT_DIR/test_keyframe_guidance.py" \
  --config "$SCRIPT_DIR/configs/final_final/ours.yaml" \
  --ref_motion_id "036201" \
  --caption "a person is jogging" \
  --source_dataset "100style" \
  --source_motion_ids "036222" \
  --output_length "100" \
  --num_samples "4" \
  --num_keyframes "5" \
  --joint_names "all" \
  --keyframe_weight "0.20" \
  --keyframe_start_frac "0.0" \
  --keyframe_end_frac "1.0" \
  --keyframe_schedule "constant" \
  --style_guidance_weight "0.75" \
  --style_start_frac "0.0" \
  --style_end_frac "1.0" \
  --style_schedule "constant" \
  --num_inference_steps "50" \
  --guidance_steps "1" \
  --style_guidance_steps "1" \
  --keyframe_guidance_steps "10" \
  --guidance_inner_mode "separate" \
  --guidance_order "style_then_motion" \
  --keyframe_match_mode "windowed_ordered" \
  --seed "44" \
  "${extra_args[@]}"

echo
echo "Constant keyframe guidance run finished."
echo "Outputs are saved under:"
echo "  $SCRIPT_DIR/results/ours/guidance_tests/keyframe/"
