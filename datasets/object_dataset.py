from torch.utils.data import Dataset
from torchvision.io import read_image
import torch
from PIL import Image
from pathlib import Path
import json

from utils import get_class_names, get_baby_filter_class

class KonkObjectDataset(Dataset):
    def __init__(self, root_dir, transform=None, use_baby_vocab_filter=False):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.use_baby_vocab_filter = use_baby_vocab_filter  
        
        # Loading class names
        self.class_names = get_class_names(self.root_dir)
        
        # Filter class names based on a baby vocabulary if needed
        if self.use_baby_vocab_filter:
            self.class_names = get_baby_filter_class(self.class_names)            
        
        # TODO: data loading
        self.data = []
        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            for img_path in class_dir.iterdir():
                self.data.append((img_path, class_name))
    
    def __getitem__(self, idx):
        img_path, class_label = self.data[idx]
        img = read_image(str(img_path))  # Load image
        label = self.class_names.index(class_label)  # Convert label name to an index
        
        if self.transform:
            img = self.transform(img)  # Apply transformation if any
        
        return img, label

    def __len__(self):
        return len(self.data)
    
class KonkTrialDataset(Dataset):
    def __init__(self, trials_file, transform=None):
        with open(trials_file, 'r') as f:
            self.data = json.load(f)
        self.transform = transform
        self.class_names = list(set([trial["target_category"] for trial in self.data]))

    def __getitem__(self, idx):
        trial = self.data[idx]
        imgs = []
        
        # supply 50% transform
        # resize_transform = transforms.Resize((lambda size: (int(size[0] * 0.5), int(size[1] * 0.5))), interpolation=transforms.InterpolationMode.BICUBIC)
        
        # Load and transform the target and foil images
        for filename in [trial["target_img_filename"]] + trial["foil_img_filenames"]:
            # ["target.jpg", "foil1.jpg", "foil2.jpg", "foil3.jpg"]
            try:
                with Image.open(filename).convert("RGB") as img:
                    # img = resize_transform(img)
                    if self.transform:
                        img = self.transform(img)
                    imgs.append(img)
            except IOError:
                print(f"Error: Could not open image {filename}. Skipping.")
                continue

        # if len(imgs) != 4:
        #     raise ValueError(f"Expected 4 images for trial index {idx}, got {len(imgs)}. Check generated trials' paths.")

        # Stack images into a single tensor
        imgs = torch.stack(imgs) #[batch, trial_img, channel, height, width]

        label = trial["target_category"]
        return imgs, label

    def __len__(self):
        return len(self.data)