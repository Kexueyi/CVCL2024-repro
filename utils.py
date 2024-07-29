from tqdm import tqdm
import clip
import torch
import random
import numpy as np
import pandas as pd
import os 
import datetime

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def avg_accuracy(predictions, labels):
    correct = (predictions == labels).sum()
    total = labels.size
    accuracy = correct / total
    return accuracy

def per_class_accuracy(predictions, labels):
    num_classes = labels.max() + 1
    class_acc = np.zeros(num_classes)
    for i in range(num_classes):
        idx = (labels == i)
        if idx.any():
            class_correct = np.sum(predictions[idx] == labels[idx])
            class_total = idx.sum()
            class_acc[i] = class_correct / class_total
    return class_acc

def save_results(model, dataset, use_attr, top1_acc, class_acc, result_dir, class_names, clean_cls):
    """
    Save overall results and per-class accuracies to separate CSV files, 

    Args:
    model (str): Model name.
    dataset (str): Dataset name.
    use_attr (bool): Indicates whether attributes were used in the model.
    top1_acc (float): Overall Top-1 accuracy.
    class_acc (np.array): Array of accuracies for each class.
    result_dir (str): Base directory for saving results files.
    """
    results = {
        "model": model,
        "dataset": dataset,
        "use_attr": use_attr,
        "top1_acc": top1_acc
    }

    save_path = os.path.join(result_dir, f"{model}_{dataset}_{datetime.datetime.now().strftime('%m-%d_%H-%M')}")
    os.makedirs(save_path, exist_ok=True)

    # overall results
    df_results = pd.DataFrame([results])
    results_file = os.path.join(save_path, "overall_results.csv")
    df_results.to_csv(results_file, mode='a', header=not os.path.exists(results_file), index=False)
    
    # per-class accuracies
    df_per_class = pd.DataFrame({
        "Class_ID": np.arange(len(class_acc)),
        "Class_Name": class_names,
        "Clean_Class_Name": clean_cls,
        "Accuracy": class_acc
    })
    per_class_file = os.path.join(save_path, "per_class_accuracy.csv")
    df_per_class.to_csv(per_class_file, mode='a', header=not os.path.exists(per_class_file), index=False)


def zs_predict(model_name, model, dataloader, class_names, device):
    """
    Set model to eval and evaluate zero-shot classification acc,
    return predic labels based on given class_names
    """
    model.eval()
    all_values = [] # similarity values
    all_preds = [] # predicted labels
    all_labels = [] # ground truth labels

    with torch.no_grad():
        if "cvcl" in model_name:
            txt_tokens = [model.tokenize(f"{c}") for c in class_names]
            txt_input  = torch.cat([txt[0] for txt in txt_tokens]).to(device)
            txt_len = torch.cat([txt[1] for txt in txt_tokens]).to(device)
            txt_feature = model.encode_text(txt_input, txt_len)
        
        elif "clip" in model_name:
            txt_input = torch.cat([clip.tokenize(f"a photo of {c}") for c in class_names]).to(device)
            txt_feature = model.encode_text(txt_input)
            
        txt_feature /= txt_feature.norm(dim=-1, keepdim=True)

        for img, label in tqdm(dataloader, desc="Evaluating"):
            imgs = img.to(device)
            labels = label.to(device)

            img_feature = model.encode_image(imgs)
            img_feature /= img_feature.norm(dim=-1, keepdim=True)
            txt_feature /= txt_feature.norm(dim=-1, keepdim=True)
            similarity = (100.0 * img_feature @ txt_feature.T).softmax(dim=-1)
            indices = similarity.argmax(dim=-1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(indices.cpu().numpy())

    return all_values, all_preds, all_labels
