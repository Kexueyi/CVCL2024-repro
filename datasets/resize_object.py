import os
from PIL import Image
import shutil
from tqdm import tqdm

IMAGE_H = 224
IMAGE_W = 224

def resize_object_img(img):
    img = img.resize((int(IMAGE_W / 2), int(IMAGE_H / 2)), Image.BICUBIC)
    new_img = Image.new('RGB', (IMAGE_W, IMAGE_H), 'white')
    new_img.paste(img, (int(IMAGE_W / 4), int(IMAGE_H / 4)))
    return new_img

def resize_object_folder(source_folder, target_folder):
    files_to_process = []
    for root, dirs, files in os.walk(source_folder):
        # copy all dirs
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            target_dir_path = dir_path.replace(source_folder, target_folder)
            if not os.path.exists(target_dir_path):
                os.makedirs(target_dir_path)

        # collect all img files
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                files_to_process.append((file_path, file_path.replace(source_folder, target_folder)))

    # process all img files with a progress bar
    for file_path, target_file_path in tqdm(files_to_process, desc="Processing Images"):
        img = Image.open(file_path)
        new_img = resize_object_img(img)
        new_img.save(target_file_path)

source_folder = '/home/Dataset/xueyi/KonkLab/17-objects'
target_folder = '/home/Dataset/xueyi/KonkLab/resized-objects'

shutil.copytree(source_folder, target_folder, dirs_exist_ok=True, ignore=shutil.ignore_patterns('*.*'))

resize_object_folder(source_folder, target_folder)