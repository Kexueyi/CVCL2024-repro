import torch
from utils import register_hooks, remove_hooks
import clip

class FeatureExtractor:
    def __init__(self, model_name, model, device):
        self.model_name = model_name
        self.model = model
        self.device = device
        self.model.eval()

    def get_txt_feature(self, label):
        if "cvcl" in self.model_name:
            tokens, token_len = self.model.tokenize(label)  # Separate the tokenization from the device transfer
            tokens = tokens.to(self.device)
            if isinstance(token_len, torch.Tensor):
                token_len = token_len.to(self.device)
            txt_features = self.model.encode_text(tokens, token_len)
            
        elif "clip" in self.model_name:
            # label = label.squeeze(0)  originated from CVCL repo
            tokens = clip.tokenize(label).to(self.device)
            txt_features = self.model.encode_text(tokens)
        return txt_features
    
    def get_concept_feature(self, concepts): # pass composed concepts
        concepts = concepts.to(self.device)
        concept_features = self.get_txt_feature(concepts)
        return concept_features
        
    def get_img_feature(self, imgs):
        imgs = imgs.to(self.device)  
        img_features = self.model.encode_image(imgs)
        return img_features
    
    def norm_features(self, features):
        return features / features.norm(dim=-1, keepdim=True)
