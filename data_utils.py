import os
import torch
import pandas as pd
import numpy as np
from glob import glob
from PIL import Image
from pathlib import Path
import re

import torch
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from multimodal.multimodal_lit import MultiModalLitModel
from huggingface_hub import hf_hub_download
import clip

# dataset path
DATASET_ROOTS = {"imagenet_val": "/home/Dataset/xueyi/ImageNet100/val",
                "broden": "/home/xke001/demo/NetDissect-Lite/dataset/broden1_224/images",
                "cub": "/home/xke001/demo/zero-shot/data/CUB/CUB_200_2011",
                "awa2": "/home/Dataset/xueyi/Animals_with_Attributes2",
                "konk": "/home/Dataset/xueyi/Konklab/17-objects",
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

def get_dataset(dataset_name, preprocess=None, class_file_path='classes.txt', get_attr=False):
    # dictionary of dataset configurations
    print(f"Loading dataset: {dataset_name}")
    dataset_configs = {
        'awa2': (AnimalDataset, {'root_dir': DATASET_ROOTS['awa2'], 'transform': preprocess, 'class_file': class_file_path, 'use_attr': get_attr}),
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
    """
    Dataset for animal classes in AWA2, suitable for zero-shot and generalized zero-shot learning.
    Handles datasets with optional attributes and supports data augmentations.

    Attributes:
        root_dir (Path): The root directory of the dataset.
        transform (callable): A function/transform that takes in a PIL image and returns a transformed version.
        class_file (str): File path to the class file, which lists all classes.
        use_attr (bool): Whether to use attributes associated with classes.
        continuous (bool): Whether the attributes/class matrix are continuous.
    """
    def __init__(self, root_dir, transform=None, class_file=None, use_attr=False, continuous=False):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.class_file = class_file
        self.use_attr = use_attr
        self.continuous = continuous
        self.attribute_file = self.load_attribute_file()

        self.classes = self.load_class_file()
        self.cls_attr_mat = self.load_cls_attr_matrix(continuous) if use_attr else None
        self.class_descriptions = self.generate_class_description() if use_attr else None
        self.img_paths, self.img_indexes = self.load_images()

    def load_class_file(self):
        class_path = self.root_dir / self.class_file
        class_data = pd.read_csv(class_path, sep='\t', header=None)
        if class_data.shape[1] == 1:
            classes = pd.read_csv(class_path, header=None, names=['class_name'])
            classes['class_index'] = range(1, len(classes) + 1)
        else:
            classes = class_data
            classes.columns = ['class_index', 'class_name']
        return classes

    def load_attribute_file(self):
        attr_path = self.root_dir / 'predicates.txt' 
        return pd.read_csv(attr_path, sep='\t', header=None, names=['attribute_index', 'attribute_name'])
    
    def load_cls_attr_matrix(self, continuous):
        dtype = 'float' if continuous else 'int'
        attr_file = 'predicate-matrix-continuous.txt' if continuous else 'predicate-matrix-binary.txt'
        return np.array(np.genfromtxt(self.root_dir / attr_file, dtype=dtype))

    def generate_class_description(self):
        def attributes_to_text(attributes_vector):
            if self.continuous:
                threshold = np.mean(attributes_vector) # threshold for continuous attributes
                descriptions = [desc for attr, desc in zip(attributes_vector, self.attribute_file['attribute_name']) if attr > threshold]
            else:
                descriptions = [desc for attr, desc in zip(attributes_vector, self.attribute_file['attribute_name']) if attr == 1]
            return descriptions

        return [attributes_to_text(attr) for attr in self.cls_attr_mat]
    
    def load_images(self):
        img_paths = []
        img_indexes = []
        for _, row in self.classes.iterrows():
            class_index = row['class_index']
            class_name = row['class_name'].replace('+', ' ')
            folder_dir = self.root_dir / 'JPEGImages' / class_name
            files = glob(str(folder_dir / '*.jpg'))
            img_paths.extend(files)
            img_indexes.extend([class_index] * len(files))
        return img_paths, img_indexes

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img_index = self.img_indexes[idx] - 1  # Convert 1-based index to 0-based

        image = Image.open(img_path)
        if image.mode == 'L':
            image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        if self.use_attr:
            im_attr = self.cls_attr_mat[img_index]
            class_description = ', '.join(self.class_descriptions[img_index])
            return image, img_index, class_description, im_attr
        return image, img_index

    def __len__(self):
        return len(self.img_paths)
    

#         if train:
#             self.data_frame = self.data_frame[self.split['is_train'] == 1]
#         else:
#             self.data_frame = self.data_frame[self.split['is_train'] == 0]
#         # Merge with classes to get class names
#         self.data_frame = self.data_frame.merge(self.classes, on='class_id')
#         # Merge with images to get paths
#         self.data_frame = self.data_frame.merge(self.images, left_on='image_id', right_on='image_index')