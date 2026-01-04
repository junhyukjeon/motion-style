export PYTHONPATH="$PYTHONPATH:$(pwd)/eval2:$(pwd)/salad"

python eval2/evaluate.py --config configs/new/19.yaml  --style_weight 2 --style_guidance 0.5 --csv_name 1224.csv
python eval2/evaluate.py --config configs/new/19.yaml  --style_weight 2 --style_guidance 1.0 --csv_name 1224.csv
# python eval2/evaluate.py --config configs/new/19.yaml  --style_weight 2.5 --style_guidance 0.5 --csv_name 1224.csv
# python eval2/evaluate.py --config configs/new/19.yaml  --style_weight 2.5 --style_guidance 1.0 --csv_name 1224.csv
# python eval2/evaluate.py --config configs/new/19.yaml  --style_weight 2.5 --style_guidance 2.0 --csv_name 1224.csv

python eval2/evaluate.py --config configs/new/20.yaml  --style_weight 2 --style_guidance 0.5 --csv_name 1224.csv
python eval2/evaluate.py --config configs/new/20.yaml  --style_weight 2 --style_guidance 1.0 --csv_name 1224.csv
# python eval2/evaluate.py --config configs/new/20.yaml  --style_weight 2.5 --style_guidance 0.5 --csv_name 1224.csv
# python eval2/evaluate.py --config configs/new/20.yaml  --style_weight 2.5 --style_guidance 0.5 --csv_name 1224.csv
# python eval2/evaluate.py --config configs/new/20.yaml  --style_weight 2.5 --style_guidance 1.0 --csv_name 1224.csv
# python eval2/evaluate.py --config configs/new/20.yaml  --style_weight 2.5 --style_guidance 2.0 --csv_name 1224.csv