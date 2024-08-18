from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from pathlib import Path
import json
import numpy as np
from glob import glob

class AnimalDataset(Dataset):
    def __init__(self, root_dir, transform=None, class_file=None, baby_vocab=False, use_attr=False, top_n=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.class_file = class_file
        
        self.use_attr = use_attr
        self.top_n = top_n  # top_n attributes with class descriptions
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

    def load_cls_attr_matrix(self, continuous=True):
        matrix_file = 'predicate-matrix-continuous.txt' if continuous else 'predicate-matrix-binary.txt'
        return np.genfromtxt(self.root_dir / matrix_file, dtype='float' if continuous else 'int')

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