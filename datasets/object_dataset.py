from torch.utils.data import Dataset
import torch
from PIL import Image
from pathlib import Path
import json


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

        if len(imgs) != 4:
            raise ValueError(f"Expected 4 images for trial index {idx}, got {len(imgs)}. Check generated trials' paths.")

        # Stack images into a single tensor
        imgs = torch.stack(imgs) #[batch_size, number_of_trial_imgs, channel, height, width]

        label = trial["target_category"]
        return imgs, label

    def __len__(self):
        return len(self.data)