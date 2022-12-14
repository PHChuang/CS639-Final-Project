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


def gradient(input_tensor, direction, device):
    channel_num = input_tensor.shape[1]
    if direction == "x":
        kernel = torch.Tensor(np.array([[0, 0], [-1, 1]])).expand(channel_num, channel_num, 2, 2)
    elif direction == "y":
        kernel = torch.Tensor(np.array([[0, -1], [0, 1]])).expand(channel_num, channel_num, 2, 2)
    kernel = kernel.to(device)

    input_tensor = F.pad(input_tensor, (0, 1, 0, 1))
    gradient_orig = torch.abs(F.conv2d(input_tensor, kernel))
    grad_min = torch.min(gradient_orig)
    grad_max = torch.max(gradient_orig)
    grad_norm = torch.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))
    return grad_norm


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pad_to(x, stride):
    """https://stackoverflow.com/questions/66028743/how-to-handle-odd-resolutions-in-unet-architecture-pytorch
    """
    h, w = x.shape[-2:]

    if h % stride > 0:
        new_h = h + stride - h % stride
    else:
        new_h = h
    if w % stride > 0:
        new_w = w + stride - w % stride
    else:
        new_w = w
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pads = (lw, uw, lh, uh)

    # zero-padding by default.
    # See others at https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.pad
    out = F.pad(x, pads, "constant", 0)

    return out, pads


def unpad(x, pad):
    """https://stackoverflow.com/questions/66028743/how-to-handle-odd-resolutions-in-unet-architecture-pytorch
    """
    if pad[2]+pad[3] > 0:
        x = x[:,:,pad[2]:-pad[3],:]
    if pad[0]+pad[1] > 0:
        x = x[:,:,:,pad[0]:-pad[1]]
    return x


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

