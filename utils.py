import math
import warnings
from collections import Counter
from bisect import bisect_right

import cv2
import numpy as np

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler


def randomFlipRotation(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)
        return np.flipud(image)


def gradient(input_tensor, direction):
    channel_num = input_tensor.shape[1]
    smooth_kernel_x = torch.Tensor(np.array([[0, 0], [-1, 1]]))
    smooth_kernel_x = smooth_kernel_x.expand(channel_num, channel_num, 2, 2)
    smooth_kernel_y = smooth_kernel_x.permute(0, 1, 3, 2)
    kernel = smooth_kernel_x
    if direction == "y":
        kernel = smooth_kernel_y
    input_tensor = F.pad(input_tensor, (0, 1, 0, 1))
    gradient_orig = torch.abs(F.conv2d(input_tensor, kernel))
    grad_min = torch.min(gradient_orig)
    grad_max = torch.max(gradient_orig)
    grad_norm = torch.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))
    return grad_norm


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_image(path):
    # load an image
    img = cv2.imread(path)
    img = img[:, :, ::-1]  # BGR -> RGB
    return img


def save_image(path, img):
    img = img.copy()[:, :, ::-1]
    return cv2.imwrite(path, img)


def save_tensor_to_image(path, tensor):
    img = tensor.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = img * 255
    img = img.astype(np.uint8)
    save_image(path, img)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

