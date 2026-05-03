#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo
echo "=== Running constant step-guided trajectory case ==="

extra_args=()
extra_args+=(--print_step_trajectory_loss)

python "$SCRIPT_DIR/test_trajectory_guidance.py" \
  --config "$SCRIPT_DIR/configs/final_final/ours.yaml" \
  --ref_motion_id "035391" \
  --caption "a person is walking" \
  --trajectory_shape "s_curve" \
  --output_length "200" \
  --radius "0.75" \
  --turns "1" \
  --forward_length "2.5" \
  --num_samples "4" \
  --num_inference_steps "50" \
  --trajectory_weight "0.50" \
  --trajectory_start_frac "0" \
  --trajectory_end_frac "1.0" \
  --trajectory_schedule "constant" \
  --style_guidance_weight "0.75" \
  --style_start_frac "0.0" \
  --style_end_frac "1.0" \
  --style_schedule "constant" \
  --guidance_steps "1" \
  --style_guidance_steps "1" \
  --trajectory_guidance_steps "3" \
  --guidance_inner_mode "separate" \
  --guidance_order "style_then_motion" \
  --seed "42" \
  --run_tag "step_const_full_style0" \
  "${extra_args[@]}"

echo
echo "Constant step-guided trajectory run finished."
echo "Outputs are saved under:"
echo "  $SCRIPT_DIR/results/ours/guidance_tests/trajectory/"
