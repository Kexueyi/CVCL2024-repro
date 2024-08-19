from collections import defaultdict
import torch
import clip
from tqdm import tqdm


class ZSTrialClassifier:
    def __init__(self, model_name, model, device):
        self.model_name = model_name
        self.model = model
        self.device = device
        self.model.eval()

    def get_txt_feature(self, label):
        if "cvcl" in self.model_name:
            tokens, token_len = self.model.tokenize(label)  # Separate the tokenization from the device transfer
            tokens = tokens.to(self.device)
            # Ensure token_len is a tensor before moving it; this depends on the model's tokenize implementation
            if isinstance(token_len, torch.Tensor):
                token_len = token_len.to(self.device)
            txt_features = self.model.encode_text(tokens, token_len)
            
        elif "clip" in self.model_name:
            # label = label.squeeze(0)  originated from CVCL repo
            tokens = clip.tokenize(label).to(self.device)
            txt_features = self.model.encode_text(tokens)
         # norm_text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return txt_features


    def get_img_feature(self, imgs): # input trial group tensor
        imgs = imgs.to(self.device)  
        img_features = self.model.encode_image(imgs)
        # norm_img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        return img_features
    
    def norm_features(self, features):
        return features / features.norm(dim=-1, keepdim=True)

    def predict(self, dataloader):
        correct_pred = 0
        total_pred = 0
        # correct_cls_pred, total_cls_predic= {classname: 0 for classname in dataloader.dataset.class_names}
        correct_cls_pred = defaultdict(int)
        total_cls_predic = defaultdict(int)

        with torch.no_grad():
            for imgs, label in tqdm(dataloader, desc="Evaluating"):
                batch_size, per_trial_img_num, channels, height, width = imgs.size()
                imgs = imgs.view(-1, channels, height, width)  # Flatten the trials into the batch dimension

                img_features = self.get_img_feature(imgs)  # Get image features
                img_features = img_features.view(batch_size, per_trial_img_num, -1)  # Reshape back to separate trials
                img_features = self.norm_features(img_features) 

                txt_features = self.get_txt_feature(label)  # Get text features for each label
                txt_features = txt_features.unsqueeze(1).expand(-1, per_trial_img_num, -1)  # Expand text features across trials
                txt_features = self.norm_features(txt_features) 

                # Calculate the cosine similarity
                similarity = (100.0 * img_features @ txt_features.transpose(-2, -1)).softmax(dim=-1)  # Calculate softmax over the trial dimension

                for i in range(batch_size):
                    simil = similarity[i]  # Get the similarity scores for the i-th item in the batch
                    predic_idx = simil.argmax(dim=-1)  # Find the index of the max similarity score for each trial

                    if predic_idx == 0:  # Assuming the target correct image is at index 0
                        correct_pred += 1
                        correct_cls_pred[label[i]] += 1
                    total_pred += 1
                    total_cls_predic[label[i]] += 1

                
        # Calculate overall accuracy
        acc = correct_pred / total_pred if total_pred > 0 else 0

        # Calculate per-class accuracy
        cls_acc = {cls: (correct_cls_pred[cls] / total_cls_predic[cls] if total_cls_predic[cls] > 0 else 0) for cls in total_cls_predic}

        return acc, cls_acc

            
