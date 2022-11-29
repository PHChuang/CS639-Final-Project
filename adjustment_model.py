import torch
import torch.nn as nn


class AdjustmentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_r, input_i):
        concat1 = torch.cat([input_r, input_i], dim=1)
        conv1 = self.lrelu(self.conv1(concat1))
        conv2 = self.lrelu(self.conv2(conv1))
        conv3 = self.lrelu(self.conv3(conv2))
        conv4 = self.conv4(conv3)
        adjustment = self.sigmoid(conv4)
        return adjustment