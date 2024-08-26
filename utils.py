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
    
def get_activation(outputs, mode):
    if mode == 'avg':
        def hook(model, input, output):
            if len(output.shape) == 4: # CNN layers
                activation = output.mean(dim=[2, 3]).detach()
            elif len(output.shape) == 3: # ViT
                activation = output[:, 0].clone()
            elif len(output.shape) == 2: # FC layers
                activation = output.detach()
            outputs.append(activation)
    elif mode == 'max':
        def hook(model, input, output):
            if len(output.shape) == 4: # CNN layers
                activation = output.amax(dim=[2, 3]).detach()
            elif len(output.shape) == 3: # ViT
                activation = output[:, 0].clone()
            elif len(output.shape) == 2: # FC layers
                activation = output.detach()
            outputs.append(activation)
    return hook

def register_hooks(model, layers, mode = 'avg'):
    activations = {layer: [] for layer in layers}
    hooks = {}

    # register forward hook
    for layer in layers:
        module = dict(model.named_modules()).get(layer)
        if module:
            hooks[layer] = module.register_forward_hook(get_activation(activations[layer], mode))
            # print(f"Hook registered for layer: {layer}")
        # else:
            # print(f"Warning: Layer '{layer}' does not exist in the model.")

    return activations, hooks

def remove_hooks(hooks):
    for layer in hooks:
        hooks[layer].remove()
        # print(f"Hook removed for layer: {layer}")
    
    # torch.cuda.empty_cache()

def calculate_accuracy(predictions, labels):
    predictions = predictions.cpu()
    labels = labels.cpu()

    correct = (predictions == labels).sum().item()
    accuracy = correct / labels.size(0)

    unique_labels = torch.unique(labels)
    # print(f"Number of unique labels: {len(unique_labels)}")
    # print(f"Unique labels: {unique_labels}")

    class_acc = {label.item(): 0.0 for label in unique_labels} # initialize with float
    
    for label in unique_labels:   
        idx = (labels == label) # boolean tensor, same size as labels
        if idx.any():
            class_correct = (predictions[idx] == labels[idx]).sum().item()
            class_total = idx.sum().item()
            class_acc[label.item()] = class_correct / class_total

    return accuracy, class_acc

def save_attr_results(args_dict, predictions, labels, similarities, class_names, clean_cls, text_combinations):
    """
    Save overall results and per-class accuracies to separate CSV files, 

    Args:
    model (str): Model name.
    dataset (str): Dataset name.
    use_attr (bool): Indicates whether attributes were used in the model.
    top1_acc (float): Overall Top-1 accuracy.
    class_acc (np.array): Array of accuracies for each class.
    result_dir (str): Base directory for saving results files.
    args_dict (dict): Dictionary of command-line arguments used.
    """
    model = args_dict.get('model', 'unknown_model')
    dataset = args_dict.get('dataset', 'unknown_data')
    result_dir = args_dict.get('result_dir', './results')
    use_attr = args_dict.get('use_attr', False)
    class_file = args_dict.get('class_file', 'classes.txt')
    
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_script_dir)  # Moving up one directory level
    result_dir = os.path.join(parent_dir, result_dir)  # Joining the parent directory with the result_dir


    os.makedirs(result_dir, exist_ok=True)
    accuracy, class_acc = calculate_accuracy(predictions, labels)
    save_path = os.path.join(result_dir, f"{model}_{dataset}_{datetime.datetime.now().strftime('%m-%d_%H-%M')}")
    os.makedirs(save_path, exist_ok=True)

    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if isinstance(similarities, torch.Tensor):
        similarities = similarities.cpu().numpy()
    
    df_overall = pd.DataFrame({
        "model": [model],
        "dataset": [dataset],
        "class_file": [class_file],
        "filer_baby": [args_dict.get('baby_vocab', False)],
        "use_attr": [use_attr],
        "num_classes": [len(class_names)],
        "top_n_desc": [args_dict.get('top_n_desc', 0)],
        "top1_acc": [accuracy],
        "time": [datetime.datetime.now().strftime('%m-%d %H:%M')]
    })
    overall_file = os.path.join(result_dir, "top-1-acc.csv")
    df_overall.to_csv(overall_file, mode='a', header=not os.path.exists(overall_file), index=False)
    
    # per-class accuracies
    class_ids = list(class_acc.keys())
    top1_accs = list(class_acc.values())

    df_per_class = pd.DataFrame({
        "class_id": class_ids, # original class indices
        "class_name": class_names,
        "clean_class_name": clean_cls,
        "top1_acc": top1_accs,
        "class_desc": text_combinations
    })
    per_class_file = os.path.join(save_path, "class_acc.csv")
    df_per_class.to_csv(per_class_file, mode='a', header=not os.path.exists(per_class_file), index=False)
    
    # Save predictions with additional details
    df_predictions = pd.DataFrame({
        "id": np.arange(len(predictions)),
        "class_id": labels,
        "predic_id": predictions,
        "similarity": similarities
    })
    predictions_file = os.path.join(save_path, "predictions.csv")
    df_predictions.to_csv(predictions_file, mode='a', header=not os.path.exists(predictions_file), index=False)

    # Save command-line arguments to JSON
    args_json_file = os.path.join(save_path, "args.json")
    with open(args_json_file, 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)

    print(f"Results and args saved to {save_path}")
    
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