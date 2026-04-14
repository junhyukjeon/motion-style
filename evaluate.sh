export PYTHONPATH="$PYTHONPATH:$(pwd)/eval:$(pwd)/salad"

python eval/evaluate.py \
  --config configs/final_final/ours.yaml \
  --style_weight 1.5 \
  --style_guidance 0.75 \
  --csv_name 0406.csv \
  --style_name_dict_path ./dataset/100style/100STYLE_name_dict_Filter.txt