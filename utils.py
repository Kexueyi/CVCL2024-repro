from tqdm import tqdm
import clip
import torch

def evaluate_model(model_name, model, dataloader, class_names, device):
    """set model to eval and evaluate zero-shot classification acc"""
    model.eval()
    all_preds = []
    all_labels = []

    if "cvcl" in model_name:
        texts, texts_len = model.tokenize(class_names)
        texts, texts_len = texts.to(device), texts_len.to(device)
        text_features = model.encode_text(texts, texts_len)
    
    elif "clip" in model_name:
        texts = clip.tokenize(class_names).to(device)
        text_features = model.encode_text(texts)
        
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            image_features = model.encode_image(inputs)

            # compare image and text features
            similarity = torch.matmul(image_features, text_features.T)
            preds = similarity.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels