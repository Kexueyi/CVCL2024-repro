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

def calculate_accuracy(predictions, labels):
    predictions = predictions.cpu()
    labels = labels.cpu()

    correct = (predictions == labels).sum().item()
    accuracy = correct / labels.size(0)

    num_classes = labels.max().item() + 1
    class_acc = torch.zeros(num_classes)
    for i in range(num_classes):
        idx = (labels == i)
        if idx.any():
            class_correct = (predictions[idx] == labels[idx]).sum().item()
            class_total = idx.sum().item()
            class_acc[i] = class_correct / class_total

    return accuracy, class_acc


def save_results(args_dict, predictions, labels, similarities, class_names, clean_cls):
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
    
    accuracy, class_acc = calculate_accuracy(predictions, labels)
    # save_path = os.path.join(result_dir, f"{model}_{dataset}_{datetime.datetime.now().strftime('%m-%d_%H-%M')}")
    # os.makedirs(save_path, exist_ok=True)

    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if isinstance(similarities, torch.Tensor):
        similarities = similarities.cpu().numpy()
    
    df_overall = pd.DataFrame({
        "model": [model],
        "dataset": [dataset],
        "filer_baby": [args_dict.get('baby_vocab', False)],
        "use_attr": [use_attr],
        "top_n_desc": [args_dict.get('top_n_desc', 0)],
        "top1_acc": [accuracy],
        "time": [datetime.datetime.now().strftime('%m-%d %H:%M')]
    })
    overall_file = os.path.join(result_dir, "top-1-acc.csv")
    df_overall.to_csv(overall_file, mode='a', header=not os.path.exists(overall_file), index=False)
    
    # # per-class accuracies
    # df_per_class = pd.DataFrame({
    #     "class_id": np.arange(len(class_acc)),
    #     "class_name": class_names,
    #     "clean_class_name": clean_cls,
    #     "top1_acc": class_acc
    # })
    # per_class_file = os.path.join(save_path, "per_class_acc.csv")
    # df_per_class.to_csv(per_class_file, mode='a', header=not os.path.exists(per_class_file), index=False)
    
    # # Save predictions with additional details
    # df_predictions = pd.DataFrame({
    #     "id": np.arange(len(predictions)),
    #     "class_id": labels,
    #     "predic_id": predictions,
    #     "similarity": similarities
    # })
    # predictions_file = os.path.join(save_path, "predictions.csv")
    # df_predictions.to_csv(predictions_file, mode='a', header=not os.path.exists(predictions_file), index=False)

    # # Save command-line arguments to JSON
    # args_json_file = os.path.join(save_path, "args.json")
    # with open(args_json_file, 'w') as json_file:
    #     json.dump(args_dict, json_file, indent=4)

    # print(f"Results and args saved to {save_path}")

class ZeroShotClassifier:
    def __init__(self, model_name, model, device):
        self.model_name = model_name
        self.model = model
        self.device = device
        self.model.eval()

    def get_txt_feature(self, clean_cls_name, cls_desc=None, prefix="", use_attr=False):
        if use_attr:
            if cls_desc is None:
                raise ValueError("cls_desc must be provided when use_attr is True.")            
            combined_texts = [f"{prefix}{name}, {desc}" for name, desc in zip(clean_cls_name, cls_desc.values())]
            if "cvcl" in self.model_name:
                text_tokens = [self.model.tokenize(text) for text in combined_texts]
            elif "clip" in self.model_name:
                text_inputs = clip.tokenize(combined_texts).to(self.device)
        else:
            texts = [f"{prefix}{c}" for c in clean_cls_name]
            if "cvcl" in self.model_name:
                text_tokens = [self.model.tokenize(c) for c in texts]
            elif "clip" in self.model_name:
                text_inputs = clip.tokenize(texts).to(self.device)

        if "cvcl" in self.model_name:
            text_inputs = torch.cat([txt[0] for txt in text_tokens]).to(self.device)
            text_lens = torch.cat([txt[1] for txt in text_tokens]).to(self.device)
            text_features = self.model.encode_text(text_inputs, text_lens)
        elif "clip" in self.model_name:
            text_features = self.model.encode_text(text_inputs)

        normalized_text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return normalized_text_features
        
    def get_img_feature(self, dataloader):
        img_features_list = []
        all_labels_list = []
        for data in tqdm(dataloader, desc="Encoding Images"):
            img, label = data[:2] # can return more than 2 items
            img = img.to(self.device)
            img_features = self.model.encode_image(img)
            img_features_list.append(img_features)
            all_labels_list.append(label.to(self.device)) 
        img_features_tensor = torch.cat(img_features_list)
        norm_img_features = img_features_tensor / img_features_tensor.norm(dim=-1, keepdim=True)
        all_labels_tensor = torch.cat(all_labels_list)
        return norm_img_features, all_labels_tensor
    
    def compute_similarity(self, img_features, text_features):
        similarity = (100.0 * img_features @ text_features.T).softmax(dim=-1)
        return similarity

    def predict_labels(self, similarity, index_map):
        preds = similarity.argmax(dim=-1)  
        mapped_preds = [index_map[pred.item()] for pred in preds]
        return torch.tensor(mapped_preds, device=preds.device)
    
    def predict(self, dataloader, prefix=None, use_attr=False):
        with torch.no_grad():
            text_features = self.get_txt_feature(dataloader.dataset.clean_cls_names, dataloader.dataset.class_descriptions, prefix, use_attr)
            img_features, all_labels = self.get_img_feature(dataloader)
            similarity = self.compute_similarity(img_features, text_features)
            index_map = dataloader.dataset.index_map
            all_preds = self.predict_labels(similarity, index_map)  # Pass the full_class_indices here
            similarities = similarity.max(dim=1)[0].cpu().numpy()  # max value of each row
        return similarities, all_preds, all_labels