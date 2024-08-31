#!/bin/bash
DATA_ROOT_DIR='/home/Dataset/xueyi/KonkLab/17-objects'
# CVCL Trials
python trial.py --seed 39 --data_dir $DATA_ROOT_DIR
python trial.py --seed 40 --data_dir $DATA_ROOT_DIR
python trial.py --seed 41 --data_dir $DATA_ROOT_DIR
python trial.py --seed 42 --data_dir $DATA_ROOT_DIR
python trial.py --seed 43 --data_dir $DATA_ROOT_DIR

# CLIP Trials
python trial.py --seed 39 --data_dir $DATA_ROOT_DIR --model clip
python trial.py --seed 40 --data_dir $DATA_ROOT_DIR --model clip
python trial.py --seed 41 --data_dir $DATA_ROOT_DIR --model clip
python trial.py --seed 42 --data_dir $DATA_ROOT_DIR --model clip
python trial.py --seed 43 --data_dir $DATA_ROOT_DIR --model clip