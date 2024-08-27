from collections import defaultdict
import torch
from tqdm import tqdm
from utils import register_hooks, remove_hooks


class ZSTrialPredic:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
        self.model = self.feature_extractor.model
        self.device = self.feature_extractor.device

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
                txt_features = txt_features.unsqueeze(1) # [batch_size, 1, 512] to match trial img feature shape
                
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
    
    def predict_with_concepts(self, dataloader, layers, mapper, top_k):
        correct_pred = 0
        total_pred = 0
        correct_cls_pred = defaultdict(int)
        total_cls_predic = defaultdict(int)
        all_concept_info = {}
        with torch.no_grad():
            for batch_idx, (imgs, labels) in enumerate(tqdm(dataloader)):
                activations, hooks = register_hooks(self.model, layers, mode='avg', keep_trial_dim=True) 
                batch_size, per_trial_img_num, channels, height, width = imgs.size()
                imgs = imgs.view(-1, channels, height, width)

                img_features = self.feature_extractor.get_img_feature(imgs)
                img_features = img_features.view(batch_size, per_trial_img_num, -1) 
                # img_features = self.feature_extractor.norm_features(img_features)
                
                for layer in layers:
                    if all(item.shape == activations[layer][0].shape for item in activations[layer]):
                        activations[layer] = torch.stack(activations[layer])
                        activations[layer] = torch.squeeze(activations[layer], 0)  # Squeeze the first dimension        
                concept_info = mapper.get_concepts(activations, top_k=2, thres_param=None)
                remove_hooks(hooks)

                all_concept_info[batch_idx] = concept_info
                concepts = mapper.aggregate_concepts(concept_info)

                concept_features = self.feature_extractor.get_concept_features(concepts)
                # print(concept_features)
                img_cpt_features = torch.add(concept_features, img_features)
                img_cpt_features = self.feature_extractor.norm_features(img_cpt_features)

                txt_features = self.feature_extractor.get_txt_feature(labels)  # [batch_size, 512]
                txt_features = self.feature_extractor.norm_features(txt_features) 
                txt_features = txt_features.unsqueeze(1) # [batch_size, 1, 512]

                similarity = (100.0 * img_cpt_features @ txt_features.transpose(-2, -1)).softmax(dim=-2)  # [batch_size, 4, 1]
                similarity = similarity.squeeze(-1) # [batch_size, 4]
                
                for i in range(batch_size):
                    simil = similarity[i]  # Get the similarity scores for the i-th item in the batch
                    predic_idx = simil.argmax().item()  # Find the index of the max similarity score for each trial

                    if predic_idx == 0:  # gt is the first image
                        correct_pred += 1
                        correct_cls_pred[labels[i]] += 1
                    total_pred += 1
                    total_cls_predic[labels[i]] += 1

        # Calculate overall accuracy
        acc = correct_pred / total_pred if total_pred > 0 else 0

        # Calculate per-class accuracy
        cls_acc = {cls: (correct_cls_pred[cls] / total_cls_predic[cls] if total_cls_predic[cls] > 0 else 0) for cls in total_cls_predic}

        return acc, cls_acc, all_concept_info

            
