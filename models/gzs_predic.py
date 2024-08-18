import torch
import clip
from tqdm import tqdm


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
            text_combinations = [f"{prefix}{name}, {desc}" for name, desc in zip(clean_cls_name, cls_desc.values())] # cls_desc dict value
            if "cvcl" in self.model_name:
                text_tokens = [self.model.tokenize(text) for text in text_combinations]
            elif "clip" in self.model_name:
                text_inputs = clip.tokenize(text_combinations).to(self.device)
        else:
            text_combinations = [f"{prefix}{name}" for name in clean_cls_name]
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
        return normalized_text_features, text_combinations
        
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
        mapped_preds = [index_map[pred.item()] for pred in preds] # original class indices
        return torch.tensor(mapped_preds, device=preds.device)
    
    def predict(self, dataloader, prefix=None, use_attr=False):
        with torch.no_grad():
            text_features, text_combinations = self.get_txt_feature(dataloader.dataset.clean_cls_names, dataloader.dataset.class_descriptions, prefix, use_attr)
            img_features, all_labels = self.get_img_feature(dataloader) # return full_class_index
            similarity = self.compute_similarity(img_features, text_features)
            index_map = dataloader.dataset.index_map
            all_preds = self.predict_labels(similarity, index_map)  # Pass the full_class_indices here, return original class predictions
            similarities = similarity.max(dim=1)[0].cpu().numpy()  # max value of each row
        return similarities, all_preds, all_labels, text_combinations