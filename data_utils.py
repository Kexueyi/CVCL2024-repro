import os
from huggingface_hub import hf_hub_download
import clip

from torchvision import datasets
from torch.utils.data import ConcatDataset
from multimodal.multimodal_lit import MultiModalLitModel
from datasets.awa2_dataset import AnimalDataset
from datasets.cub_dataset import CUBDataset
from datasets.object_dataset import KonkTrialDataset

data_root = os.getenv("DATA_ROOT", default="/home/Dataset/xueyi")

DATASET_ROOTS = {
    "imagenet_val": os.path.join(data_root, "ImageNet100/val"),
    "broden": os.path.join(data_root, "broden1_224/images"),
    "cub": os.path.join(data_root, "CUB/CUB_200_2011"),
    "awa2": os.path.join(data_root, "Animals_with_Attributes2"),
    "konk": os.path.join(data_root, "KonkleLab/17-objects"),
}


def get_model(model_name, device):
    """returns target model and its transform function"""
    if "cvcl" in model_name:  
        print("Loading CVCL...")
        if "res" in model_name:
            backbone = "resnext50" # ResNeXt-50 32x4d 
        elif "vit" in model_name:
            backbone = "vit" #  ViT-B/14 
        else:
            print("Unknown backbone, set to default resnext50")
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
        'awa2': (AnimalDataset, {'root_dir': DATASET_ROOTS['awa2'], **kwargs}),
        'cub': (CUBDataset, {'root_dir': DATASET_ROOTS['cub'], **kwargs}),
        'object-trial': (KonkTrialDataset, {**kwargs}),
        'imagenet_broden': (ConcatDataset, [
            {'root_dir': DATASET_ROOTS['imagenet_val'], **kwargs},
            {'root_dir': DATASET_ROOTS['broden'], **kwargs}
        ]),
        'cifar100_train': (datasets.CIFAR100, {'root': os.path.expanduser("~/.cache"), 'download': True, 'train': True, **kwargs}),
        'cifar100_val': (datasets.CIFAR100, {'root': os.path.expanduser("~/.cache"), 'download': True, 'train': False, **kwargs}),
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
