import os
import torch
import pandas as pd
import numpy as np
from glob import glob
from PIL import Image

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
        print("Loading CVCL")
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
        print(f"Successfully load CVCL-{backbone}")
        model.to(device)

    elif "clip" in model_name:
        backbone = "ViT-L/14" # source: CVCL Supplementary Materials
        model, preprocess = clip.load(f"{backbone}", device=device)
        print(f"Successfully load CLIP-{backbone}")
    
    elif "resnet" in model_name:
        model_name_cap = model_name.replace("resnet", "ResNet")
        weights = eval("models.{}_Weights.IMAGENET1K_V1".format(model_name_cap))
        preprocess = weights.transforms()
        model = eval("models.{}(weights=weights).to(device)".format(model_name))
    
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    model.eval()
    
    return model, preprocess

def get_data(dataset_name, preprocess=None, get_attr=False):  
    if dataset_name == "awa2":
        data = AnimalDataset(root_dir=DATASET_ROOTS['awa2'],transform=preprocess, use_attr=get_attr)

    elif dataset_name == "cub":
        data = CUBDataset(root_dir=DATASET_ROOTS['cub'], transform=preprocess, use_attr=get_attr)
        
    elif dataset_name in DATASET_ROOTS.keys():
        data = datasets.ImageFolder(DATASET_ROOTS[dataset_name], preprocess)

    # elif dataset_name == "cub":
    #     data = ConcatDataset([CUBDataset(root_dir=DATASET_ROOTS["cub"], train=True, transform=preprocess, use_attr=get_attr), 
    #                           CUBDataset(root_dir=DATASET_ROOTS['cub'], train=False, transform=preprocess, use_attr=get_attr)])        
        
    # elif dataset_name == "imagenet_broden":
    #     data = ConcatDataset([datasets.ImageFolder(DATASET_ROOTS["imagenet_val"], preprocess), 
    #                                                  datasets.ImageFolder(DATASET_ROOTS["broden"], preprocess)])
    # elif dataset_name == "cifar100_train":
    #     data = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=True,
    #                                transform=preprocess)

    # elif dataset_name == "cifar100_val":
    #     data = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False, 
    #                                transform=preprocess)
    
    return data

def clean_class_names(dataset_name, data):
    class_names = data.classes['class_name'].tolist()
    if dataset_name == "cub":
        # generalized zero-shot learning
        #TODO: maybe not that cleaned due to "_"
        clean_cls = [name.split(".")[1] for name in class_names]
    elif dataset_name == "awa2":
        clean_cls = [name.replace("+", " ") for name in class_names]
    else:
        clean_cls = class_names
    return clean_cls, class_names

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
  
class CUBSplitDataset(Dataset):
    """This is for train/test split in CUB dataset."""
    def __init__(self, root_dir, train=True, transform=None, use_attr=False):
        self.root_dir = root_dir
        self.transform = transform 
        self.use_attr = use_attr  
        # Load datasets
        self.data_frame = pd.read_csv(os.path.join(root_dir, 'image_class_labels.txt'), sep=' ', header=None, names=['image_id', 'class_id'])
        # for GZSL, all classes
        self.classes = pd.read_csv(os.path.join(root_dir, 'classes.txt'), sep=' ', header=None, names=['class_id', 'class_name'])
        self.images = pd.read_csv(os.path.join(root_dir, 'images.txt'), sep=' ', header=None, names=['image_index', 'image_path'], index_col=0)
        self.split = pd.read_csv(os.path.join(root_dir, 'train_test_split.txt'), sep=' ', header=None, names=['image_id', 'is_train'])
        # Filter by train or test
        if train:
            self.data_frame = self.data_frame[self.split['is_train'] == 1]
        else:
            self.data_frame = self.data_frame[self.split['is_train'] == 0]
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
        label_id = self.data_frame.iloc[idx]['class_id']
        
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
  """This contains all train/test class in AWA2 dataset."""
  def __init__(self, root_dir, transform=None, use_attr=False, continuous=False):
    self.root_dir = root_dir
    self.use_attr = use_attr     
    self.transform = transform
    # for GZSL, all classes
    self.classes = pd.read_csv(os.path.join(root_dir, 'classes.txt'), sep='\t', header=None, names=['class_index', 'class_name'])

    if use_attr:
        if continuous:
            self.cls_attr_mat = np.array(np.genfromtxt(os.path.join(root_dir,'predicate-matrix-continuous.txt', dtype='float')))
        else:
            self.cls_attr_mat = np.array(np.genfromtxt(os.path.join(root_dir,'predicate-matrix-binary.txt', dtype='int')))
    
    img_names = []
    img_index = []
    for _, row in self.classes.iterrows():
        class_index = row['class_index']  # 1-based index
        class_name = row['class_name']
        FOLDER_DIR = os.path.join(root_dir, 'JPEGImages', class_name) 
        file_descriptor = os.path.join(FOLDER_DIR, '*.jpg')
        files = glob(file_descriptor)

        for file_name in files:
            img_names.append(file_name)
            img_index.append(class_index)

    self.img_names = img_names
    self.img_index = img_index

  def __getitem__(self, idx):
    im = Image.open(self.img_names[idx])
    if im.getbands()[0] == 'L':
      im = im.convert('RGB')
    if self.transform:
      im = self.transform(im)

    im_index = self.img_index[idx] -1 # 0-based index
    if self.use_attr:
        im_attr = self.cls_attr_mat[im_index,:]
        return im, im_index, im_attr
    else:
        return im, im_index

  def __len__(self):
    return len(self.img_names)

class AWASplitDataset(Dataset):
  """This is for train/test split in AWA2 dataset."""
  def __init__(self, root_dir, train=True, transform=None, use_attr=False, continuous=False):
    self.root_dir = root_dir
    self.use_attr = use_attr     
    self.transform = transform
    # for GZSL, all classes
    self.classes = pd.read_csv(os.path.join(root_dir, 'classes.txt'), sep='\t', header=None, names=['class_index', 'class_name'])

    if use_attr:
        if continuous:
            self.cls_attr_mat = np.array(np.genfromtxt(os.path.join(root_dir,'predicate-matrix-continuous.txt', dtype='float')))
        else:
            self.cls_attr_mat = np.array(np.genfromtxt(os.path.join(root_dir,'predicate-matrix-binary.txt', dtype='int')))

    if train:
        classes_file = 'trainclasses.txt'
    else:
        classes_file = 'testclasses.txt'
    
    img_names = []
    img_index = []
    with open(os.path.join(root_dir, classes_file)) as f:
      for line in f:
        class_name = line.strip()
        FOLDER_DIR = os.path.join(root_dir, 'JPEGImages', class_name)
        file_descriptor = os.path.join(FOLDER_DIR, '*.jpg')
        files = glob(file_descriptor)

        class_index = self.classes['class_name']
        for file_name in files:
          img_names.append(file_name)
          img_index.append(class_index)
    self.img_names = img_names
    self.img_index = img_index

  def __getitem__(self, idx):
    im = Image.open(self.img_names[idx])
    if im.getbands()[0] == 'L':
      im = im.convert('RGB')
    if self.transform:
      im = self.transform(im)

    im_index = self.img_index[idx] # class index
    if self.use_attr:
        im_predicate = self.cls_attr_mat[im_index,:]
        return im, im_predicate, im_index
    else:
        return im, im_index

  def __len__(self):
    return len(self.img_names)

# def get_places_id_to_broden_label():
#     with open("data/categories_places365.txt", "r") as f:
#         places365_classes = f.read().split("\n")
    
#     broden_scenes = pd.read_csv('/home/Dataset/xueyi/broden1_227/c_scene.csv')
#     id_to_broden_label = {}
#     for i, cls in enumerate(places365_classes):
#         name = cls[3:].split(' ')[0] 
#         name = name.replace('/', '-')
        
#         found = (name+'-s' in broden_scenes['name'].values)
        
#         if found:
#             id_to_broden_label[i] = name.replace('-', '/')+'-s'
#         if not found:
#             id_to_broden_label[i] = None
#     return id_to_broden_label
    
# def get_cifar_superclass():
#     cifar100_has_superclass = [i for i in range(7)]
#     cifar100_has_superclass.extend([i for i in range(33, 69)])
#     cifar100_has_superclass.append(70)
#     cifar100_has_superclass.extend([i for i in range(72, 78)])
#     cifar100_has_superclass.extend([101, 104, 110, 111, 113, 114])
#     cifar100_has_superclass.extend([i for i in range(118, 126)])
#     cifar100_has_superclass.extend([i for i in range(147, 151)])
#     cifar100_has_superclass.extend([i for i in range(269, 281)])
#     cifar100_has_superclass.extend([i for i in range(286, 298)])
#     cifar100_has_superclass.extend([i for i in range(300, 308)])
#     cifar100_has_superclass.extend([309, 314])
#     cifar100_has_superclass.extend([i for i in range(321, 327)])
#     cifar100_has_superclass.extend([i for i in range(330, 339)])
#     cifar100_has_superclass.extend([345, 354, 355, 360, 361])
#     cifar100_has_superclass.extend([i for i in range(385, 398)])
#     cifar100_has_superclass.extend([409, 438, 440, 441, 455, 463, 466, 483, 487])
#     cifar100_doesnt_have_superclass = [i for i in range(500) if (i not in cifar100_has_superclass)]
    
#     return cifar100_has_superclass, cifar100_doesnt_have_superclass

