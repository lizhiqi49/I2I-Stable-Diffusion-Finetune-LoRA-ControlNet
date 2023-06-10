import os
from PIL import Image
import imageio.v3 as imageio
import json
import glob
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np
import cv2
import pandas as pd


# default_prompt = ", a detailed high-quality professional 3D render, without background"
default_prompt = "a painting"


def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img

class LoRADataset(Dataset):
    """
    Customed dataset for cldm3d 
    """
    def __init__(
        self, 
        root, 
        default_prompt=default_prompt, 
        reso=512,
        split="train", 
        length=None
    ):
        self.dir = os.path.join(root, split)
        self.img_names = os.listdir(self.dir)
        self.default_prompt = default_prompt
        self.reso = reso

        # Read prompts as dataframe
        prompt_f = open(os.path.join(self.dir, 'prompt.json'), 'r')
        self.prompt_dict = json.load(prompt_f)
        prompt_f.close()
        # prompt_df = pd.read_csv(os.path.join(self.dir, 'prompts.csv'), names=['image_name', 'prompt'])
        # self.prompt_dict = dict(zip(prompt_df.image_name.values, prompt_df.prompt.values))

        np.random.shuffle(self.img_names)
        if length:
            self.img_names = self.img_names[:length]

        

    def __len__(self):
        return len(self.img_names)


    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img = imageio.imread(os.path.join(self.dir, img_name))
        img = resize_image(img, self.reso)
        img = img.astype(np.float32) / 127.5 - 1.   # Normalize to [-1, 1]
        img = torch.from_numpy(img).permute(2,0,1)

        prompt = self.prompt_dict[img_name]

        p = np.random.rand()
        if p >= 0.5:
            prompt = default_prompt

        return {'image': img, 'prompt': prompt}

