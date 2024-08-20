import os
import random
import json
import pandas as pd
from tqdm import tqdm

from utils import load_baby_vocab, vocab_class_filter, set_seed

SEED = 43
# NUM_TRIALS = 200
NUM_FOILS = 3
NUM_TRIALS_PER_IMAGE = 5

set_seed(SEED)

# os.chdir('..') # Change working directory to the root directory of the project

def collect_images_from_directory(directory):
    """Recursively collects all JPEG images from the directory and subdirectories."""
    image_paths = []
    for root, dirs, files in os.walk(directory): #  all files incuding testiterms 
        for file in files:
            if file.lower().endswith('.jpg'):
                image_paths.append(os.path.join(root, file))
    return image_paths

def collect_all_class_images(data_root_dir, class_names):
    all_images = {}
    for class_name in tqdm(class_names, desc="Collecting class images"):
        class_dir = os.path.join(data_root_dir, class_name)
        if os.path.isdir(class_dir):
            all_images[class_name] = collect_images_from_directory(class_dir)
    return all_images

def save_image_paths(all_images, file_path):
    with open(file_path, 'w') as f:
        json.dump(all_images, f, indent=4)


def load_image_paths(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# def generate_trials(all_images, num_trials=NUM_TRIALS, num_foils=NUM_FOILS):
#     '''Generates trials directly while collecting image paths.
#     Args:
#         num_trials: Number of trials to generate for each class.
#         num_foils: Number of foil images to select in each trial.
    
#     For exmaple, num_trials=100, num_foils=3 means that for each class, 100 trials will be generated, each trials have 4 images, 1 target image and 3 foil images.

#     Returns:
#         A list of trial dictionaries.
#     '''
#     trials = []
#     all_classes = list(all_images.keys())

#     for class_name in tqdm(all_classes, desc="Generating trials"):
#         images = all_images[class_name]
#         for _ in range(num_trials):
#             if len(images) < 1:
#                 continue
#             target_image = random.choice(images)
#             foil_classes = random.sample([cls for cls in all_classes if cls != class_name and len(all_images[cls]) > 0], num_foils)
#             foil_images = [random.choice(all_images[cls]) for cls in foil_classes]
#             trials.append({
#                 'target_img_filename': target_image,
#                 'target_category': class_name,
#                 'foil_img_filenames': foil_images
#             })

#     return trials

def generate_trials(all_images, num_trials_per_image=NUM_TRIALS_PER_IMAGE, num_foils=NUM_FOILS):
    '''Generates trials for each image in each class.
    Args:
        num_trials_per_image: Number of trials to generate for each image.
        num_foils: Number of foil images to select in each trial.
    
    Returns:
        A list of trial dictionaries.
    '''
    trials = []
    all_classes = list(all_images.keys())

    for class_name in tqdm(all_classes, desc="Generating trials"):
        images = all_images[class_name]
        for image in images:
            for _ in range(num_trials_per_image):
                foil_classes = random.sample([cls for cls in all_classes if cls != class_name and len(all_images[cls]) > 0], num_foils)
                foil_images = [random.choice(all_images[cls]) for cls in foil_classes]
                trials.append({
                    'target_img_filename': image,
                    'target_category': class_name,
                    'foil_img_filenames': foil_images
                })

    return trials

def save_trials(trials, file_path):
    with open(file_path, 'w') as f:
        json.dump(trials, f, indent=4)


data_root_dir = '/home/Dataset/xueyi/KonkLab/17-objects'
df = pd.read_excel('/home/Dataset/xueyi/KonkLab/17-objects/MM2-Ranks.xls')
class_names = df['Category'].tolist()

vocab_set = set(load_baby_vocab())
filtered_classes = vocab_class_filter(class_names, vocab_set, match_type='full')

all_images = collect_all_class_images(data_root_dir, filtered_classes)
# save_image_paths(all_images, 'object_images.json')
trials = generate_trials(all_images)
save_trials(trials, f'datasets/trials/object_{NUM_TRIALS_PER_IMAGE}_{NUM_FOILS}_{SEED}.json')
# save_trials(trials, f'datasets/trials/object_{NUM_TRIALS}_{NUM_FOILS}_{SEED}.json')