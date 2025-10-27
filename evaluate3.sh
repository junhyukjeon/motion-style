export PYTHONPATH="$PYTHONPATH:$(pwd)/eval2:$(pwd)/salad"

python eval2/evaluate.py --config configs/2//loss/0.yaml  --style_weight 2.5 --csv_name 1027.csv
python eval2/evaluate.py --config configs/2//loss/0.yaml  --style_weight 3.5 --csv_name 1027.csv