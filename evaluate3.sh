export PYTHONPATH="$PYTHONPATH:$(pwd)/eval2:$(pwd)/salad"

python eval2/evaluate2.py --config configs/2//loss/0.yaml  --style_weight 5.5 --csv_name 1028.csv
python eval2/evaluate2.py --config configs/2//loss/0.yaml  --style_weight 6.5 --csv_name 1028.csv
python eval2/evaluate2.py --config configs/2//loss/0.yaml  --style_weight 7.5 --csv_name 1028.csv