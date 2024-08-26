from collections import defaultdict
import torch
import clip
from tqdm import tqdm
from utils import register_hooks, remove_hooks


class ZSTrialPredic:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
        self.model = feature_extractor.model
        self.device = feature_extractor.device

    def predict(self, dataloader):
        correct_pred = 0
        total_pred = 0
        correct_cls_pred = defaultdict(int)
        total_cls_predic = defaultdict(int)

        with torch.no_grad():
            for imgs, label in tqdm(dataloader, desc="Evaluating"):
                batch_size, per_trial_img_num, channels, height, width = imgs.size()
                imgs = imgs.view(-1, channels, height, width)  # Flatten the trials into the batch dimension

                img_features = self.feature_extractor.get_img_feature(imgs)  # [batch_size*4, 512]
                img_features = img_features.view(batch_size, per_trial_img_num, -1)  
                img_features = self.feature_extractor.norm_features(img_features) # [batch_size, 4, 512]

                txt_features = self.feature_extractor.get_txt_feature(label)  # [batch_size, 512]
                txt_features = self.feature_extractor.norm_features(txt_features) 
                txt_features = txt_features.unsqueeze(1) # [batch_size, 1, 512]
                
                # Calculate the cosine similarity
                similarity = (100.0 * img_features @ txt_features.transpose(-2, -1)).softmax(dim=-2)  # [batch_size, 4, 1]
                similarity = similarity.squeeze(-1) # Remove the last dimension
                
                for i in range(batch_size):
                    simil = similarity[i]  # Get the similarity scores for the i-th item in the batch
                    predic_idx = simil.argmax().item()  # Find the index of the max similarity score for each trial

                    if predic_idx == 0:  # gt is the first image
                        correct_pred += 1
                        correct_cls_pred[label[i]] += 1
                    total_pred += 1
                    total_cls_predic[label[i]] += 1

        # Calculate overall accuracy
        acc = correct_pred / total_pred if total_pred > 0 else 0

        # Calculate per-class accuracy
        cls_acc = {cls: (correct_cls_pred[cls] / total_cls_predic[cls] if total_cls_predic[cls] > 0 else 0) for cls in total_cls_predic}
        return acc, cls_acc
    
    def predict_with_concepts(self, dataloader, layers, concept_mapper, top_k):
        correct_pred = 0
        total_pred = 0
        correct_cls_pred = defaultdict(int)
        total_cls_predic = defaultdict(int)
        
        with torch.no_grad():
            for imgs, label in tqdm(dataloader, desc="ConceptZS Predicting"):
                # register hooks
                activations, hooks = register_hooks(self.model, layers, mode='avg')
                
                # get img features
                batch_size, per_trial_img_num, channels, height, width = imgs.size()
                imgs = imgs.view(-1, channels, height, width)  # Flatten the trials into the batch dimension
                img_features = self.feature_extractor.get_img_feature(imgs)  # [batch_size*4, 512]               
                img_features = img_features.view(batch_size, per_trial_img_num, -1)  
                img_features = self.feature_extractor.norm_features(img_features) # [batch_size, 4, 512]
                
                # get activated concepts
                concept_info = concept_mapper.get_concepts(activations, top_k=top_k, thres_param=None, mode='per_layer')
                
                # remove img hooks
                remove_hooks(hooks)
                
                # get txt features
                txt_features = self.feature_extractor.get_txt_feature(label)  # [batch_size, 512]
                txt_features = self.feature_extractor.norm_features(txt_features) 
                txt_features = txt_features.unsqueeze(1) # [batch_size, 1, 512]
                
                # get concept features
                concepts = concept_mapper.compose_concepts(concept_info)
                concepts = ', '.join(concepts)
                concept_features = self.feature_extractor.get_txt_feature(concepts)
                concept_features = self.feature_extractor.norm_features(concept_features)
                
                # Calculate the cosine similarity
                similarity = (100.0 * img_features @ txt_features.transpose(-2, -1)).softmax(dim=-2)  # [batch_size, 4, 1]
                similarity = similarity.squeeze(-1) # [batch_size, 4]
                
                for i in range(batch_size):
                    simil = similarity[i]  # Get the similarity scores for the i-th item in the batch
                    predic_idx = simil.argmax().item()  # Find the index of the max similarity score for each trial

                    if predic_idx == 0:  # gt is the first image
                        correct_pred += 1
                        correct_cls_pred[label[i]] += 1
                    total_pred += 1
                    total_cls_predic[label[i]] += 1

        # Calculate overall accuracy
        acc = correct_pred / total_pred if total_pred > 0 else 0

        # Calculate per-class accuracy
        cls_acc = {cls: (correct_cls_pred[cls] / total_cls_predic[cls] if total_cls_predic[cls] > 0 else 0) for cls in total_cls_predic}
        return acc, cls_acc, concept_info

            
