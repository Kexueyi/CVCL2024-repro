import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import random
import data_utils 
import utils
import os
import argparse
import datetime
import json

parser = argparse.ArgumentParser(description='CVCL-ZeroShot')

parser.add_argument('--model', type=str, default='cvcl_res', help='Model name')
parser.add_argument('--dataset', type=str, default='awa2', help='Dataset name')
parser.add_argument('--use_attr', type=bool, default=False, help='Use attributes')
parser.add_argument('--device', type=str, default='cuda:1', help='Device')
parser.add_argument('--seed', type=int, default=42, help='Seed')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--result_dir', type=str, default='results', help='Result directory')

parser.parse_args()

if __name__ == "__main__":
    args = parser.parse_args()
    
    utils.set_seed(args.seed)
    device = args.device
    
    model_name = args.model 
    model, preprocess = data_utils.get_model(model_name, device)

    dataset_name = args.dataset
    data = data_utils.get_data(dataset_name, preprocess, get_attr=False)
    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=False)
    
    clean_cls, class_names = data_utils.clean_class_names(dataset_name, data)

    similarities, predictions, labels = utils.zs_predict(model_name, model, dataloader, clean_cls, device)

    top1_acc = utils.avg_accuracy(predictions, labels)
    class_acc = utils.per_class_accuracy(predictions, labels)

    utils.save_results(args.model, args.dataset, args.use_attr, top1_acc, class_acc, args.result_dir, class_names, clean_cls)