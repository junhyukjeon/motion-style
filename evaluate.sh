export PYTHONPATH="$PYTHONPATH:$(pwd)/eval:$(pwd)/salad"
RECOMPUTE_GUIDED_V_PRED="${RECOMPUTE_GUIDED_V_PRED:-0}"

extra_args=()
if [[ "$RECOMPUTE_GUIDED_V_PRED" == "1" ]]; then
  extra_args+=(--recompute_guided_v_pred)
fi

python eval/evaluate.py \
  --config configs/final_final/ours.yaml \
  --style_weight 1.5 \
  --style_guidance 0.75 \
  --csv_name 0416.csv \
  --style_name_dict_path ./dataset/100style/100STYLE_name_dict_Filter.txt \
  --recompute_guided_v_pred \
  "${extra_args[@]}"
