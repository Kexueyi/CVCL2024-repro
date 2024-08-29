# Baseline, no attribute, no baby_vocab
# python trial.py --model cvcl_resx --top_k 1 --map_file neuron_concept/cvcl_broden_broden.csv --trial_path datasets/trials/object_5_3_42.json
# python trial.py --model clip --top_k 1 --map_file neuron_concept/cvcl_broden_broden.csv

python trial.py --model cvcl_resx --top_k 3 --map_file neuron_concept/cvcl_broden_broden.csv --trial_path datasets/trials/object_5_3_42.json
# python trial.py --model clip --top_k 3 --layers --map_file neuron_concept/cvcl_broden_broden.csv


python trial.py --model cvcl_resx --top_k 1 --map_file neuron_concept/cvcl_konk_baby.csv --trial_path datasets/trials/object_5_3_42.json
# python trial.py --model clip --top_k 1 --map_file neuron_concept/cvcl_konk_baby.csv

python trial.py --model cvcl_resx --top_k 3 --map_file neuron_concept/cvcl_konk_baby.csv --trial_path datasets/trials/object_5_3_42.json
# python trial.py --model clip --top_k 3 --map_file neuron_concept/cvcl_konk_baby.csv

python trial.py --model cvcl_resx --trial_path datasets/trials/object_5_3_42.json
# python trial.py --model clip 
