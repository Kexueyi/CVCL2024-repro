from torch.utils.data import Dataset
import os
import torch
import pandas as pd
from PIL import Image

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
  