from tqdm import tqdm
import clip
import torch
import random
import numpy as np
import pandas as pd
import os 
import datetime
import json

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_activation(outputs, mode):
    '''
    Extracts activations with specified pooling mode (avg or max).
    Handles different layer types (CNN, ViT, FC) appropriately.
    
    Parameters:
        outputs (list): Storage for the pooled activations.
        mode (str): Pooling mode, one of 'avg' or 'max'.
    '''
    if mode not in ['avg', 'max']:
        raise ValueError("Unsupported mode. Choose 'avg' or 'max'")
    
    def hook(model, input, output):
        if len(output.shape) == 4:  # CNN layers
            pooled_output = output.mean(dim=[2, 3]) if mode == 'avg' else output.amax(dim=[2, 3])
        elif len(output.shape) == 3:  # ViT
            pooled_output = output[:, 0]
        elif len(output.shape) == 2:  # FC layers
            pooled_output = output
        else:
            raise ValueError("Unsupported output shape")

        outputs.append(pooled_output.detach())

    return hook

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


def save_results(args_dict, predictions, labels, similarities, class_names, clean_cls, text_combinations):
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
    
