import os
from torch.utils import data
from utils import load_image
import numpy as np
import random
import cv2
from utils import randomFlipRotation
import glob

class LOLLoader(data.Dataset):
    def __init__(self, root_folder, split="train", is_train=True, transforms=None, patch_size=48):
        assert split in ["train", "test"]
        self.root_folder = root_folder
        self.split = split
        self.transforms = transforms
        self.patch_size = patch_size
        self.is_train = is_train

        img_dir_path = os.path.join(root_folder, split, "low")
        self.img_list = [os.path.basename(x) for x in glob.glob(os.path.join(img_dir_path, "*"))]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        filename = self.img_list[index]
        filename_high = os.path.join(self.root_folder, self.split, "high", filename)
        filename_low = os.path.join(self.root_folder, self.split, "low", filename)
        img_high = np.ascontiguousarray(load_image(filename_high))
        img_low = np.ascontiguousarray(load_image(filename_low))

        # apply data augmentation
        if self.is_train:
            # random crop
            H, W, _ = img_high.shape
            start_x = random.randint(0, W - self.patch_size)
            start_y = random.randint(0, H - self.patch_size)
            img_high = img_high[start_y:start_y+self.patch_size, start_x:start_x+self.patch_size, :]
            img_low = img_low[start_y:start_y+self.patch_size, start_x:start_x+self.patch_size, :]

            # random flip and rotate
            rand_mode = random.randint(0, 7)
            img_high = randomFlipRotation(img_high, rand_mode)
            img_low = randomFlipRotation(img_low, rand_mode)

        if self.transforms is not None:
            img_high = self.transforms(img_high)
            img_low = self.transforms(img_low)
        return img_high, img_low