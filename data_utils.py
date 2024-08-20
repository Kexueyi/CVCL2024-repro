import os
from huggingface_hub import hf_hub_download
import clip

from torchvision import datasets
from torch.utils.data import ConcatDataset
from multimodal.multimodal_lit import MultiModalLitModel
from datasets.object_dataset import KonkTrialDataset

def get_model(model_name, device):
    """returns target model and its transform function"""
    if "cvcl" in model_name:  
        print("Loading CVCL...")
        if "res" in model_name:
            backbone = "resnext50" # ResNeXt-50 32x4d 
        elif "vit" in model_name:
            backbone = "vit" #  ViT-B/14 
        else:
            print("Invalid backbone, set to resnext50")
            backbone = "resnext50"
        checkpoint_name = f"cvcl_s_dino_{backbone}_embedding" 
        checkpoint = hf_hub_download(repo_id="wkvong/"+checkpoint_name, filename=checkpoint_name+".ckpt")
        model, transform = MultiModalLitModel.load_model(model_name="cvcl")
        print(f"Successfully loaded CVCL-{backbone}")
        model.to(device)
        
    elif "clip" in model_name:
        print("Loading CLIP...")
        backbone = "ViT-L/14" # source: CVCL Supplementary Materials
        model, transform = clip.load(f"{backbone}", device=device)
        print(f"Successfully loaded CLIP-{backbone}")
    
    elif "resnet" in model_name:
        model_name_cap = model_name.replace("resnet", "ResNet")
        weights = eval("models.{}_Weights.IMAGENET1K_V1".format(model_name_cap))
        transform = weights.transforms()
        model = eval("models.{}(weights=weights).to(device)".format(model_name))
    
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    model.eval()
    
    return model, transform

def get_dataset(dataset_name, **kwargs):
    # Print loading message
    print(f"Loading dataset: {dataset_name}")
    
    # Dictionary of dataset configurations
    dataset_configs = {
        'object-trial': (KonkTrialDataset, {**kwargs}),
    }

    # Fetch dataset class and kwargs
    dataset_class, kwargs = dataset_configs.get(dataset_name, (None, None))

    # Handle different types of datasets
    if dataset_class:
        if isinstance(kwargs, list):  # For concatenating datasets
            return ConcatDataset([dataset_class(**config) for config in kwargs])
        else:
            return dataset_class(**kwargs)
    
    # Handle unsupported datasets
    raise ValueError(f"Unsupported dataset: {dataset_name}")
