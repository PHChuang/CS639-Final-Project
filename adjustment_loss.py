import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import rgb_to_grayscale

from torchmetrics import StructuralSimilarityIndexMeasure

from utils import gradient

class AdjustmentLoss(nn.Module):
    """
        A_low: output of adjustment net
        A_high: ilumination map of high exposire from output of decomposition net
    """
    def __init__(self, device):
        super().__init__()
        self.ssim = StructuralSimilarityIndexMeasure()
        self.device = device

    def grad_loss(self, A_low, A_high):
        x_loss = torch.square(gradient(A_low, 'x', self.device) - gradient(A_high, 'x', self.device))
        y_loss = torch.square(gradient(A_low, 'y', self.device) - gradient(A_high, 'y', self.device))
        grad_loss_all = torch.mean(x_loss + y_loss)
        return grad_loss_all

    def forward(self, A_low, A_high):
        loss_grad = self.grad_loss(A_low, A_high)
        loss_square = torch.mean(torch.square(A_low - A_high))
        loss_adjust = loss_grad + loss_square
        return loss_adjust