import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import rgb_to_grayscale

from torchmetrics import StructuralSimilarityIndexMeasure

from utils import gradient

class RestorationLoss(nn.Module):
    """
        R_low: output of restoration net
        R_high: reflectance map of high exposire from output of decomposition net
    """
    def __init__(self, device):
        super().__init__()
        self.ssim = StructuralSimilarityIndexMeasure()
        self.device = device

    def grad_loss(self, R_low, R_high):
        R_low_gray = rgb_to_grayscale(R_low)
        R_high_gray = rgb_to_grayscale(R_high)
        x_loss = torch.square(gradient(R_low_gray, 'x', self.device) - gradient(R_high_gray, 'x', self.device))
        y_loss = torch.square(gradient(R_low_gray, 'y', self.device) - gradient(R_high_gray, 'y', self.device))
        grad_loss_all = torch.mean(x_loss + y_loss)
        return grad_loss_all

    def forward(self, R_low, R_high):
        loss_ssim = 1 - self.ssim(R_low, R_high)
        loss_grad = self.grad_loss(R_low, R_high)
        loss_square = torch.mean(torch.square(R_low - R_high))
        loss_restoration = loss_ssim + loss_grad + loss_square

        return loss_restoration