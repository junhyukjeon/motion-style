export PYTHONPATH="$PYTHONPATH:$(pwd)/eval2:$(pwd)/salad"

python eval2/evaluate_mix.py --config configs/mix/1.yaml  --style_weight 1.5 --csv_name 1119.csv