from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from pathlib import Path
import json
import numpy as np
from glob import glob

from ..utils.data_utils import my_function


class ObjectCategoriesEvalDataset(Dataset):
    def __init__(self, root_dir, transform, use_baby_vocab_filter=False, eval_include_sos_eos=False):
        self.root_dir = Path(root_dir)
        self.transform = transform
        if use_baby_vocab_filter:
            self.vocab = self.load_vocab()
            self.filter_vocab() # updae full_classes, attribute_file, attribute_matrix
        else:
            self.vocab = []
        self.eval_include_sos_eos = eval_include_sos_eos



    def __getitem__(self, idx):
        # read trial information
        trial = self.data[idx]

        # read in images (target and foils)
        # target image is always the first index
        n_imgs = len(trial["foil_img_filenames"]) + 1
        imgs = torch.zeros((n_imgs, 3, IMAGE_H, IMAGE_W))

        target_img_filename = trial["target_img_filename"]
        imgs[0] = self.transform(Image.open(
            target_img_filename).convert("RGB"))

        for i, foil_img_filename in enumerate(trial["foil_img_filenames"]):
            imgs[i +
                 1] = self.transform(Image.open(foil_img_filename).convert("RGB"))

        # get target category index from vocab as a single utterance
        raw_label = trial["target_category"]

        if not self.clip_eval:
            # use SAYCam vocab/tokenizer
            label = [self.vocab[raw_label]]
            if self.eval_include_sos_eos:
                # label is [<sos>, label, <eos>] to match LM training
                label = [SOS_TOKEN_ID] + label + [EOS_TOKEN_ID]

            label = torch.LongTensor(label)
            label_len = len(label)
        else:
            # use CLIP tokenizer
            label = clip.tokenize(raw_label)
            label_len = len(label)

        return imgs, label, label_len, [raw_label]

    def __len__(self):
        return len(self.data)