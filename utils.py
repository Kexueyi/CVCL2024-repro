import os
import torch
import numpy as np
import pandas as pd
import datetime
import json
import re
import random

vocab_cache = None

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def save_trial_results(args_dict=None, accuracy=None, cls_accuracy=None):
    if args_dict is None:
        args_dict = {}
    
    results = {
        'args': args_dict,           
        'overall_accuracy': accuracy,  
        'class_accuracy': cls_accuracy 
    }
    seed = args_dict.get('seed', 0)
    filename = f"trial_{seed}_{datetime.datetime.now().strftime('%m-%d_%H:%M')}.json"
      
    results_dir ='results'
    os.makedirs(results_dir, exist_ok=True)

    filepath = os.path.join(results_dir, filename)

    with open(filepath, 'w') as file:
        json.dump(results, file, indent=4)

    print(f"Results saved successfully to {filepath}")
    
    return results
    
def load_baby_vocab():
    global vocab_cache
    if vocab_cache is not None:
        return vocab_cache
    with open("multimodal/vocab.json", 'r') as f:
        vocab_cache = set(json.load(f).keys())
    return vocab_cache

def vocab_class_filter(class_names, vocab_set, match_type='full'):
    if match_type == 'partial':
        return list({class_name for class_name in class_names if set(re.compile(r'\W+').split(class_name)) & vocab_set})
    elif match_type == 'full':
        return list({class_name for class_name in class_names if class_name in vocab_set})

def get_baby_filter_class(class_names):
    vocab = load_baby_vocab()
    return vocab_class_filter(class_names, vocab, match_type='full')

def get_class_names(data_root_dir):
    subfolders = [name for name in os.listdir(data_root_dir)
                  if os.path.isdir(os.path.join(data_root_dir, name))]
    return subfolders