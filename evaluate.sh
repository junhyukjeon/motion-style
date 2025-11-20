export PYTHONPATH="$PYTHONPATH:$(pwd)/eval2:$(pwd)/salad"

python eval2/evaluate.py --config configs/new/4.yaml  --style_weight 1.5 --csv_name 1120.csv
python eval2/evaluate.py --config configs/new/4.yaml  --style_weight 2.5 --csv_name 1120.csv
python eval2/evaluate.py --config configs/new/4.yaml  --style_weight 3.5 --csv_name 1120.csv
python eval2/evaluate.py --config configs/new/4.yaml  --style_weight 4.5 --csv_name 1120.csv
python eval2/evaluate.py --config configs/new/4.yaml  --style_weight 5.5 --csv_name 1120.csv
python eval2/evaluate.py --config configs/new/4.yaml  --style_weight 6.5 --csv_name 1120.csv
python eval2/evaluate.py --config configs/new/4.yaml  --style_weight 7.5 --csv_name 1120.csv