export PYTHONPATH="$PYTHONPATH:$(pwd)/eval2:$(pwd)/salad"

python eval2/evaluate.py --config configs/final/75.yaml  --style_weight 1.5 --style_guidance 1.0 --csv_name please.csv
python eval2/evaluate.py --config configs/final/75.yaml  --style_weight 1.5 --style_guidance 0.0 --csv_name please.csv
# python eval2/evaluate.py --config configs/final/75.yaml  --style_weight 1.5 --style_guidance 1.0 --csv_name please.csv

# python eval2/evaluate.py --config configs/final/random2_supcon.yaml  --style_weight 1.5 --style_guidance 1.0 --csv_name ablation5.csv
# python eval2/evaluate.py --config configs/final/random3_supcon.yaml  --style_weight 1.5 --style_guidance 1.0 --csv_name ablation5.csv

# python eval2/evaluate.py --config configs/final/ours.yaml  --style_weight 1.5 --style_guidance 1.0 --csv_name ablation3.csv
# python eval2/evaluate.py --config configs/final/ours.yaml  --style_weight 2.0 --style_guidance 1.0 --csv_name ablation3.csv
# python eval2/evaluate.py --config configs/final/ours.yaml  --style_weight 2.5 --style_guidance 1.0 --csv_name ablation3.csv
# python eval2/evaluate.py --config configs/final/ours.yaml  --style_weight 1.0 --style_guidance 1.0 --csv_name ablation3.csv

# python eval2/evaluate.py --config configs/final/supcon.yaml  --style_weight 1.5 --style_guidance 1.0 --csv_name ablation3.csv
# python eval2/evaluate.py --config configs/final/supcon.yaml  --style_weight 2.0 --style_guidance 1.0 --csv_name ablation3.csv
# python eval2/evaluate.py --config configs/final/supcon.yaml  --style_weight 2.5 --style_guidance 1.0 --csv_name ablation3.csv
# python eval2/evaluate.py --config configs/final/supcon.yaml  --style_weight 1.0 --style_guidance 1.0 --csv_name ablation3.csv

# python eval2/evaluate.py --config configs/final/ours.yaml  --style_weight 1.5 --style_guidance 0.0 --csv_name ablation3.csv
# python eval2/evaluate.py --config configs/final/ours.yaml  --style_weight 2.0 --style_guidance 0.0 --csv_name ablation3.csv
# python eval2/evaluate.py --config configs/final/ours.yaml  --style_weight 2.5 --style_guidance 0.0 --csv_name ablation3.csv
# python eval2/evaluate.py --config configs/final/ours.yaml  --style_weight 1.0 --style_guidance 0.0 --csv_name ablation3.csv

# python eval2/evaluate.py --config configs/final/ours.yaml  --style_weight 1.5 --style_guidance 1.0 --csv_name ablation.csv

# python eval2/evaluate.py --config configs/new/31.yaml  --style_weight 1.5 --style_guidance 1.0 --csv_name 0112.csv
# python eval2/evaluate.py --config configs/new/32.yaml  --style_weight 1.5 --style_guidance 1.0 --csv_name 0112.csv
# python eval2/evaluate.py --config configs/new/31.yaml  --style_weight 2.0 --style_guidance 1.0 --csv_name 0112.csv
# python eval2/evaluate.py --config configs/new/32.yaml  --style_weight 2.0 --style_guidance 1.0 --csv_name 0112.csv
# python eval2/evaluate.py --config configs/new/31.yaml  --style_weight 2.5 --style_guidance 1.0 --csv_name 0112.csv
# python eval2/evaluate.py --config configs/new/32.yaml  --style_weight 2.5 --style_guidance 1.0 --csv_name 0112.csv

# python eval2/evaluate.py --config configs/new/31.yaml  --style_weight 1.5 --style_guidance 0.0 --csv_name 0112.csv
# python eval2/evaluate.py --config configs/new/32.yaml  --style_weight 1.5 --style_guidance 0.0 --csv_name 0112.csv
# python eval2/evaluate.py --config configs/new/31.yaml  --style_weight 2.0 --style_guidance 0.0 --csv_name 0112.csv
# python eval2/evaluate.py --config configs/new/32.yaml  --style_weight 2.0 --style_guidance 0.0 --csv_name 0112.csv
# python eval2/evaluate.py --config configs/new/31.yaml  --style_weight 2.5 --style_guidance 0.0 --csv_name 0112.csv
# python eval2/evaluate.py --config configs/new/32.yaml  --style_weight 2.5 --style_guidance 0.0 --csv_name 0112.csv

# python eval2/evaluate.py --config configs/new/27.yaml  --style_weight 1.5 --style_guidance 1.0 --csv_name 0111.csv
# python eval2/evaluate.py --config configs/new/28.yaml  --style_weight 1.5 --style_guidance 1.0 --csv_name 0111.csv
# python eval2/evaluate.py --config configs/new/29.yaml  --style_weight 1.5 --style_guidance 1.0 --csv_name 0111.csv
# python eval2/evaluate.py --config configs/new/30.yaml  --style_weight 1.5 --style_guidance 1.0 --csv_name 0111.csv

# python eval2/evaluate.py --config configs/new/27.yaml  --style_weight 2.0 --style_guidance 1.0 --csv_name 0111.csv
# python eval2/evaluate.py --config configs/new/28.yaml  --style_weight 2.0 --style_guidance 1.0 --csv_name 0111.csv
# python eval2/evaluate.py --config configs/new/29.yaml  --style_weight 2.0 --style_guidance 1.0 --csv_name 0111.csv
# python eval2/evaluate.py --config configs/new/30.yaml  --style_weight 2.0 --style_guidance 1.0 --csv_name 0111.csv

# python eval2/evaluate.py --config configs/new/27.yaml  --style_weight 2.5 --style_guidance 1.0 --csv_name 0111.csv
# python eval2/evaluate.py --config configs/new/28.yaml  --style_weight 2.5 --style_guidance 1.0 --csv_name 0111.csv
# python eval2/evaluate.py --config configs/new/29.yaml  --style_weight 2.5 --style_guidance 1.0 --csv_name 0111.csv
# python eval2/evaluate.py --config configs/new/30.yaml  --style_weight 2.5 --style_guidance 1.0 --csv_name 0111.csv