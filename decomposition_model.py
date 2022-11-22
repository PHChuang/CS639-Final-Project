import torch
import torch.nn as nn
import torch.nn.functional as F


class DecompositionNet(nn.Module):
    def __init__(self):
        super().__init__()

        # reflectance map path
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.up5 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

        # illumination map path
        self.conv7 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        # self.conv8 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0) # author's implementation use 1x1 kernel to replace 3x3 kernel
        self.conv8 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1) # try original 3x3 kernel

        self.lrelu = nn.LeakyReLU(0.2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # reflectance map path
        conv1 = self.lrelu(self.conv1(x))
        down1 = self.maxpool(conv1)
        conv2 = self.lrelu(self.conv2(down1))
        down2 = self.maxpool(conv2)
        conv3 = self.conv3(down2)
        up4 = torch.cat([self.up4(conv3), conv2], dim=1)
        conv4 = self.lrelu(self.conv4(up4))
        up5 = torch.cat([self.up5(conv4), conv1], dim=1)
        conv5 = self.lrelu(self.conv5(up5))
        conv6 = self.conv6(conv5)
        decomp_reflectance = self.sigmoid(conv6)

        # illumination map path
        illu_layer2 = self.lrelu(self.conv7(conv1))
        illu_layer3 = torch.cat([illu_layer2, conv5], dim=1)
        illu_layer4 = self.conv8(illu_layer3)
        decomp_illumination = self.sigmoid(illu_layer4)

        return decomp_reflectance, decomp_illumination
