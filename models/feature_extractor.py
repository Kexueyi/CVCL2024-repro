import torch
from utils import register_hooks, remove_hooks
import clip

class FeatureExtractor:
    def __init__(self, model_name, model, device):
        self.model_name = model_name
        self.model = model
        self.device = device
        self.model.eval()

    def get_txt_feature(self, label, without_eossos=False):
        if "cvcl" in self.model_name:
            if without_eossos:
                tokens, token_len = self.model.tokenize_without_eos_sos(label)  # Separate the tokenization from the device transfer
            else:
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
    
    def get_concept_features(self, concepts):
        batch_size = len(concepts) 
        img_per_trial = len(concepts[0])  

        batch_trial_features = []

        # over same img_idx within trials in a batch
        for img_idx in range(img_per_trial):
            img_idx_concepts = [concepts[batch_index][img_idx] for batch_index in range(batch_size)]

            img_idx_features = self.get_txt_feature(img_idx_concepts)
            
            batch_trial_features.append(img_idx_features)

        concept_features = torch.stack(batch_trial_features).permute(1, 0, 2)

        return concept_features
    
    def get_img_feature(self, imgs):
        imgs = imgs.to(self.device)  
        img_features = self.model.encode_image(imgs)
        return img_features
    
    def norm_features(self, features):
        # norm txt img feature on feature dim
        return features / features.norm(dim=-1, keepdim=True)  
