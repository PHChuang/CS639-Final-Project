import numpy as np
import torch
import random
import cv2
import collections
import math


# default list of interpolations
_DEFAULT_INTERPOLATIONS = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC]

class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> Compose([
        >>>     Scale(320),
        >>>     RandomSizedCrop(224),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        repr_str = ""
        for t in self.transforms:
            repr_str += t.__repr__() + "\n"
        return repr_str

class ToTensor(object):
    """Convert a ``numpy.ndarray`` image pair to tensor.

    Converts a numpy.ndarray (H x W x C) image in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, img):
        assert isinstance(img, np.ndarray)
        # make ndarray normal
        img = img.copy()
        # convert image to tensor
        if img.ndim == 2:
            img = img[:, :, None]

        tensor_img = torch.from_numpy(img.transpose((2, 0, 1)))
        # backward compatibility
        if isinstance(tensor_img, torch.ByteTensor):
            return tensor_img.float().div(255)
        else:
            return tensor_img

    def __repr__(self):
        return "To Tensor()"
