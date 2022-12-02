import torch
import torch.nn as nn
from torchvision.transforms.functional import rgb_to_grayscale
from utils import gradient

class DecompositionLoss(nn.Module):
    """
        Im: input image
        R: reflectance
        I: illumination
    """
    def __init__(self, device):
        super().__init__()
        self.device = device

    def reflectance_similarity(self, R_low, R_high):
        return torch.mean(torch.abs(R_low - R_high))

    def illumination_smoothness(self, I, Im):
        Im_gray = rgb_to_grayscale(Im)
        Im_gradient_x = gradient(Im_gray, "x", self.device)
        I_gradient_x = gradient(I, "x", self.device)
        epsilon = 0.01 * torch.ones_like(Im_gradient_x)
        x_loss = torch.abs(torch.div(I_gradient_x, torch.max(Im_gradient_x, epsilon)))
        Im_gradient_y = gradient(Im_gray, "y", self.device)
        I_gradient_y = gradient(I, "y", self.device)
        y_loss = torch.abs(torch.div(I_gradient_y, torch.max(Im_gradient_y, epsilon)))
        illu_smooth_loss = torch.mean(x_loss + y_loss)
        return illu_smooth_loss

    def mutual_consistency(self, I_low, I_high):
        low_gradient_x = gradient(I_low, "x", self.device)
        high_gradient_x = gradient(I_high, "x", self.device)
        m_gradient_x = low_gradient_x + high_gradient_x
        x_loss = m_gradient_x * torch.exp(-10 * m_gradient_x)
        low_gradient_y = gradient(I_low, "y", self.device)
        high_gradient_y = gradient(I_high, "y", self.device)
        m_gradient_y = low_gradient_y + high_gradient_y
        y_loss = m_gradient_y * torch.exp(-10 * m_gradient_y)
        mutual_loss = torch.mean(x_loss + y_loss)
        return mutual_loss

    def reconstruction_error(self, Im_low, Im_high, R_low, R_high, I_low, I_high):
        I_low_3 = torch.cat([I_low, I_low, I_low], dim=1)
        I_high_3 = torch.cat([I_high, I_high, I_high], dim=1)
        recon_loss_low = torch.mean(torch.abs(R_low * I_low_3 -  Im_low))
        recon_loss_high = torch.mean(torch.abs(R_high * I_high_3 - Im_high))
        return recon_loss_high + recon_loss_low

    def forward(self, R_low, R_high, I_low, I_high, Im_low, Im_high):
        reconstruction_loss = self.reconstruction_error(Im_low, Im_high, R_low, R_high, I_low, I_high)
        reflectance_similarity_loss = self.reflectance_similarity(R_low, R_high)
        mutual_consistency_loss = self.mutual_consistency(I_low, I_high)
        illumination_smoothness_loss = self.illumination_smoothness(I_low, Im_low) + self.illumination_smoothness(I_high, Im_high)

        # decomposition_loss = (
        #     1 * reconstruction_loss +
        #     0.01 * reflectance_similarity_loss +
        #     0.2 * mutual_consistency_loss +
        #     0.15 * illumination_smoothness_loss
        # )

        decomposition_loss = (
            1 * reconstruction_loss +
            0.01 * reflectance_similarity_loss +
            0.2 * mutual_consistency_loss +
            0.03 * illumination_smoothness_loss
        )

        return decomposition_loss