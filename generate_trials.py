import os
import random
import json
import pandas as pd
from tqdm import tqdm

from utils import load_baby_vocab, vocab_class_filter, set_seed

class TrialGenerator:
    def __init__(self, root_dir, seed, num_foils=3, num_trials_per_image=5):
        self.root_dir = root_dir
        self.seed = seed
        self.num_foils = num_foils
        self.num_trials_per_image = num_trials_per_image
        set_seed(self.seed)
        self.trials_file_path = os.path.join(self.root_dir, 'datasets', 'trials', f'object_{self.num_trials_per_image}_{self.num_foils}_{self.seed}.json')

    def get_trials(self):
        """Check if trials file exists and generate if not, then return trials path and data."""
        if os.path.exists(self.trials_file_path):
            print(f"Trial file already exists: {self.trials_file_path}, skipping generation.")
            return self.trials_file_path

        print("Generating trials...")
        filtered_classes = self.filter_classes()
        filtered_imgs = self.get_all_class_images(filtered_classes)
        trials = self.generate_trials(filtered_imgs)
        
        self.save_json(trials, self.trials_file_path)
        return trials, self.trials_file_path

    def filter_classes(self):
        """Filter classes based on the category list and vocabulary."""
        df = pd.read_excel(os.path.join(self.root_dir, 'MM2-Ranks.xls'))
        class_names = df['Category'].tolist()
        vocab_set = set(load_baby_vocab())
        return vocab_class_filter(class_names, vocab_set, match_type='full')

    def get_all_class_images(self, class_names):
        """Collect images for each class and return a dictionary mapping classes to image paths."""
        all_images = {}
        for class_name in tqdm(class_names, desc="Collecting class images"):
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_dir):
                all_images[class_name] = self.get_images_from_directory(class_dir)
        return all_images

    def get_images_from_directory(self, directory, extension='.jpg'):
        """Collect all images from the specified directory with the given extension."""
        image_paths = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(extension):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def generate_trials(self, all_images):
        """Generate trials for each image by selecting foils from other classes."""
        trials = []
        all_classes = list(all_images.keys())

        for class_name in tqdm(all_classes, desc="Generating trials"):
            images = all_images[class_name]
            for image in images:
                for _ in range(self.num_trials_per_image):
                    foil_classes = random.sample([cls for cls in all_classes if cls != class_name and len(all_images[cls]) > 0], self.num_foils)
                    foil_images = [random.choice(all_images[cls]) for cls in foil_classes]
                    trials.append({
                        'target_img_filename': image,
                        'target_category': class_name,
                        'foil_img_filenames': foil_images
                    })
        return trials

    def save_json(self, data, file_path):
        """Save data to a JSON file."""
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Data saved to {file_path}")
