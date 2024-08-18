import os
import json
from torchvision import datasets
from torch.utils.data import ConcatDataset
from huggingface_hub import hf_hub_download
import clip

from multimodal.multimodal_lit import MultiModalLitModel
from ..datasets.awa2_dataset import AnimalDataset
from ..datasets.cub_dataset import CUBDataset

data_root = os.getenv("DATA_ROOT", default="/home/Dataset/xueyi")

DATASET_ROOTS = {
    "imagenet_val": os.path.join(data_root, "ImageNet100/val"),
    "broden": os.path.join(data_root, "broden1_224/images"),
    "cub": os.path.join(data_root, "CUB/CUB_200_2011"),
    "awa2": os.path.join(data_root, "Animals_with_Attributes2"),
    "konk": os.path.join(data_root, "KonkleLab/17-objects"),
}

def get_model(model_name, device):
    """returns target model and its preprocess function"""
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
        model, preprocess = MultiModalLitModel.load_model(model_name="cvcl")
        print(f"Successfully loaded CVCL-{backbone}")
        model.to(device)
        
    elif "clip" in model_name:
        print("Loading CLIP...")
        backbone = "ViT-L/14" # source: CVCL Supplementary Materials
        model, preprocess = clip.load(f"{backbone}", device=device)
        print(f"Successfully loaded CLIP-{backbone}")
    
    elif "resnet" in model_name:
        model_name_cap = model_name.replace("resnet", "ResNet")
        weights = eval("models.{}_Weights.IMAGENET1K_V1".format(model_name_cap))
        preprocess = weights.transforms()
        model = eval("models.{}(weights=weights).to(device)".format(model_name))
    
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    model.eval()
    
    return model, preprocess

def get_dataset(dataset_name, preprocess=None, class_file_path='classes.txt',baby_vocab=False, get_attr=False, top_n=None):
    # dictionary of dataset configurations
    print(f"Loading dataset: {dataset_name}")
    dataset_configs = {
        'awa2': (AnimalDataset, {'root_dir': DATASET_ROOTS['awa2'], 'transform': preprocess, 'class_file': class_file_path, 'baby_vocab': baby_vocab, 'use_attr': get_attr, 'top_n':top_n}),
        'cub': (CUBDataset, {'root_dir': DATASET_ROOTS['cub'], 'transform': preprocess, 'class_file': class_file_path, 'use_attr': get_attr}),
        'imagenet_broden': (ConcatDataset, [
            {'root_dir': DATASET_ROOTS['imagenet_val'], 'transform': preprocess},
            {'root_dir': DATASET_ROOTS['broden'], 'transform': preprocess}
        ]),
        'cifar100_train': (datasets.CIFAR100, {'root': os.path.expanduser("~/.cache"), 'download': True, 'train': True, 'transform': preprocess}),
        'cifar100_val': (datasets.CIFAR100, {'root': os.path.expanduser("~/.cache"), 'download': True, 'train': False, 'transform': preprocess})
    }

    dataset_class, kwargs = dataset_configs.get(dataset_name, (None, None))

    if dataset_class:
        if isinstance(kwargs, list):  # 如果是列表，代表需要ConcatDataset
            return ConcatDataset([dataset_class(**config) for config in kwargs])
        else:
            return dataset_class(**kwargs)
    
    raise ValueError(f"Unsupported dataset: {dataset_name}")

def clean_class_names(dataset_name, data):
    cleaners = {
        'cub': lambda names: [name.split(".")[1] for name in names], # _ still remains
        'awa2': lambda names: [name.replace("+", " ") for name in names], 
        # 'awa2': lambda names: [re.sub(r'^.*\+', '', name) for name in names],
    }
    cleaner = cleaners.get(dataset_name, lambda names: names)
    clean_cls = cleaner(data.classes['class_name'].tolist())
    return clean_cls, data.classes['class_name'].tolist()

def get_class_names(data_root_dir):
    subfolders = [name for name in os.listdir(folder_path)
                  if os.path.isdir(os.path.join(folder_path, name))]
    return subfolders