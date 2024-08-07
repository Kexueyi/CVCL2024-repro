import os
import torch
import pandas as pd
import numpy as np
from glob import glob
from PIL import Image
from pathlib import Path
import json

import torch
from torchvision import datasets
from torch.utils.data import Dataset, ConcatDataset

from multimodal.multimodal_lit import MultiModalLitModel
from huggingface_hub import hf_hub_download
import clip

# dataset path
DATASET_ROOTS = {"imagenet_val": "/home/Dataset/xueyi/ImageNet100/val",
                "broden": "/home/xke001/demo/NetDissect-Lite/dataset/broden1_224/images",
                "cub": "/home/project/12003885/data/CUB/CUB_200_2011",
                "awa2": "/home/Dataset/xueyi/Animals_with_Attributes2",
                "konk": "/home/project/12003885/data/17-objects",
                "konk_example": "/home/xke001/demo/CLIP-dissect/data/toy_example_dataset_konka"}


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


class CUBDataset(Dataset):
    """This contains all train/test classes in CUB dataset."""
    def __init__(self, root_dir,  transform=None, use_attr=False):
        self.root_dir = root_dir
        self.transform = transform 
        self.use_attr = use_attr  
        # Load datasets
        self.data_frame = pd.read_csv(os.path.join(root_dir, 'image_class_labels.txt'), sep=' ', header=None, names=['image_id', 'class_id'])

        self.classes = pd.read_csv(os.path.join(root_dir, 'classes.txt'), sep=' ', header=None, names=['class_id', 'class_name'], index_col=0)
        self.images = pd.read_csv(os.path.join(root_dir, 'images.txt'), sep=' ', header=None, names=['image_index', 'image_path'], index_col=0)

        # Merge with classes to get class names
        self.data_frame = self.data_frame.merge(self.classes, on='class_id')
        # Merge with images to get paths
        self.data_frame = self.data_frame.merge(self.images, left_on='image_id', right_on='image_index')

        if use_attr:
            #TODO: img_attr_mat weight txt file implementation
            self.attributes = pd.read_csv(os.path.join(root_dir, 'attributes/image_attribute_labels_clean.txt'), sep='\s+', header=None, names=['image_id', 'attribute_id', 'is_present', 'certainty_id', 'time'], engine='python', on_bad_lines='warn')
            num_attributes = self.attributes['attribute_id'].max()
            self.img_attr_tensor = torch.zeros((len(self.images)+1, num_attributes+1, 2), dtype=torch.int8)
            
            # Initialize the attribute matrices with an additional row to account for 1-based indexing
            # self.image_attribute_matrices = np.zeros((len(self.images) + 1, num_attributes + 1, 2))
            """ Given 3 images with 2 attributes, the matrix:
            [
                [[1, 3], [0, 2]],
                [[0, 1], [1, 4]],
                [[1, 3], [1, 2]]
            ]
            for 1st img, 1st attr, is_present=1, certainty=3
            note that time column is not used
            dtype=torch.int8 is used to save memory
            """ 

            for _, row in self.attributes.iterrows():
                image_id = int(row['image_id'])
                attribute_id = int(row['attribute_id'])
                self.img_attr_tensor[image_id, attribute_id, 0] = int(row['is_present'])
                self.img_attr_tensor[image_id, attribute_id, 1] = int(row['certainty_id'])


    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        image_rel_path = self.data_frame.iloc[idx]['image_path']
        img_path = os.path.join(self.root_dir, 'images', image_rel_path)
        
        image = Image.open(img_path)
        label_id = self.data_frame.iloc[idx]['class_id'] - 1 # 0-based index
        
        if image.mode == 'L':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        if self.use_attr:
            image_id = self.data_frame.iloc[idx]['image_id']
            attributes = self.img_attr_tensor[image_id].to(torch.int8)
            return image, label_id, attributes
        else:
            return image, label_id
  
class AnimalDataset(Dataset):
    def __init__(self, root_dir, transform=None, class_file=None, baby_vocab=False, use_attr=False, continuous=True, top_n=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.class_file = class_file
        
        self.use_attr = use_attr
        self.continuous = continuous
        self.top_n = top_n
        # load all classes
        self.full_classes = self.load_full_class_info()

        self.attribute_file = self.load_attribute_file()
        self.attribute_matrix = self.load_cls_attr_matrix()
        
        self.baby_vocab = baby_vocab # boolean flag for baby_vocab
        if self.baby_vocab:
            self.vocab = self.load_vocab()
            self.filter_vocab() # updae full_classes, attribute_file, attribute_matrix
        else:
            self.vocab = []

        # Filter classes and prepare internal mappings
        self.classes, self.index_map = self.filter_classes_and_attributes() # {filtered index: original class index}
        self.class_names = self.classes['class_name'].tolist()
        self.clean_cls_names = self.clean_class_names()  # fed to zero-shot text embedding
        self.class_descriptions = self.generate_class_descriptions()

        # Load images and labels
        self.img_paths, self.img_labels = self.load_images()

    def load_vocab(self):
        with open("multimodal/vocab.json", 'r') as f:
            return list(json.load(f).keys()) # return list of baby vocabularies
    
    def filter_vocab(self):
        self.attribute_file = self.attribute_file[self.attribute_file['attribute_name'].isin(self.vocab)]
        baby_attr_indices = self.attribute_file.index.tolist()

        self.attribute_matrix = self.attribute_matrix[:, baby_attr_indices]
        # Filter classes based on the vocab
        baby_class_names = {
            class_name for class_name in self.full_classes['class_name']
            if any(
                sub_name in self.vocab for sub_name in class_name.split('+')  # "siamese+cat" -> ["siamese", "cat"]
            )
        }
        self.full_classes = self.full_classes[self.full_classes['class_name'].isin(baby_class_names)]

    def load_full_class_info(self):
        class_path = self.root_dir / 'classes.txt'
        return pd.read_csv(class_path, sep='\t', header=None, names=['class_index', 'class_name'])

    def load_attribute_file(self):
        attr_path = self.root_dir / 'predicates.txt'
        attr_data = pd.read_csv(attr_path, sep='\t', header=None, names=['attribute_index', 'attribute_name'])
        return attr_data.reset_index(drop=True)

    def load_cls_attr_matrix(self):
        matrix_file = 'predicate-matrix-continuous.txt' if self.continuous else 'predicate-matrix-binary.txt'
        return np.genfromtxt(self.root_dir / matrix_file, dtype='float' if self.continuous else 'int')

    def clean_class_names(self):
        cleaned_class_names = []
        for name in self.classes['class_name'].tolist():
            cleaned_name = name.replace('+', ' ')
            if self.baby_vocab:   # Remove subwords not in the baby vocab
                subwords = cleaned_name.split()
                # ['siamese+cat', 'persian+cat'] -> ['cat', 'cat'] but different indices
                cleaned_name = ' '.join(word for word in subwords if word in self.vocab) 
            cleaned_class_names.append(cleaned_name)
        return cleaned_class_names
    
    def filter_classes_and_attributes(self):
        subset_path = self.root_dir / self.class_file
        subset_classes = pd.read_csv(subset_path, sep='\t', header=None, names=['class_name'])
        filtered_classes = self.full_classes[self.full_classes['class_name'].isin(subset_classes['class_name'])]
        return filtered_classes.reset_index(drop=True), {i: idx for i, idx in enumerate(filtered_classes['class_index'])}
    
    def generate_class_descriptions(self):
        descriptions = {}
        for idx, row in self.classes.iterrows():
            full_index = row['class_index']
            attr_vector = self.attribute_matrix[full_index - 1]
            descriptions[full_index] = ', '.join(self.attributes_to_text(attr_vector, self.top_n))
        return descriptions

    def attributes_to_text(self, attributes_vector, top_n):
        valid_indices = [i for i, name in enumerate(self.attribute_file['attribute_name']) if not self.vocab or name in self.vocab]
        filtered_attributes = attributes_vector[valid_indices]
        filtered_names = [self.attribute_file['attribute_name'].iloc[i] for i in valid_indices]

        if self.continuous:
            top_indices = np.argsort(filtered_attributes)[-top_n:]
            return [filtered_names[i] for i in reversed(top_indices)]
        else:
            return [name for attr, name in zip(filtered_attributes, filtered_names) if attr == 1]

    def load_images(self):
        img_paths = []
        img_labels = []
        for idx, row in self.classes.iterrows():
            class_folder = self.root_dir / 'JPEGImages' / row['class_name']
            class_images = glob(str(class_folder / '*.jpg'))
            img_paths.extend(class_images)
            img_labels.extend([row['class_index']] * len(class_images))
        return img_paths, img_labels

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        full_class_index = self.img_labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        if self.use_attr:
            reverse_index_map = {v: k for k, v in self.index_map.items()}
            continuous_index = reverse_index_map.get(full_class_index, -1)
            attributes = self.attribute_matrix[continuous_index] if continuous_index != -1 else None
            description = self.class_descriptions.get(full_class_index, "")
            return image, full_class_index, description, attributes

        return image, full_class_index

    def __len__(self):
        return len(self.img_paths)