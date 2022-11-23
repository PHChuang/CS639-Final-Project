import torch
import torch.nn as nn


class RestorationNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1_1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)

        self.conv6_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)

        self.conv7_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.conv8_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)

        self.conv9_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.conv10 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

        self.lrelu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_r, input_i):
        concat1 = torch.cat([input_r, input_i], dim=1)
        conv1_1 = self.lrelu(self.conv1_1(concat1))
        conv1_2 = self.lrelu(self.conv1_2(conv1_1))
        pool1 = self.maxpool1(conv1_2)

        conv2_1 = self.lrelu(self.conv2_1(pool1))
        conv2_2 = self.lrelu(self.conv2_2(conv2_1))
        pool2 = self.maxpool2(conv2_2)

        conv3_1 = self.lrelu(self.conv3_1(pool2))
        conv3_2 = self.lrelu(self.conv3_2(conv3_1))
        pool3 = self.maxpool3(conv3_2)

        conv4_1 = self.lrelu(self.conv4_1(pool3))
        conv4_2 = self.lrelu(self.conv4_2(conv4_1))
        pool4 = self.maxpool4(conv4_2)

        conv5_1 = self.lrelu(self.conv5_1(pool4))
        conv5_2 = self.lrelu(self.conv5_2(conv5_1))

        up1 = self.up1(conv5_2)
        concat2 = torch.cat([up1, conv4_2], dim=1)
        conv6_1 = self.lrelu(self.conv6_1(concat2))
        conv6_2 = self.lrelu(self.conv6_2(conv6_1))

        up2 = self.up2(conv6_2)
        concat3 = torch.cat([up2, conv3_2], dim=1)
        conv7_1 = self.lrelu(self.conv7_1(concat3))
        conv7_2 = self.lrelu(self.conv7_2(conv7_1))

        up3 = self.up3(conv7_2)
        concat4 = torch.cat([up3, conv2_2], dim=1)
        conv8_1 = self.lrelu(self.conv8_1(concat4))
        conv8_2 = self.lrelu(self.conv8_2(conv8_1))

        up4 = self.up4(conv8_2)
        concat5 = torch.cat([up4, conv1_2], dim=1)
        conv9_1 = self.lrelu(self.conv9_1(concat5))
        conv9_2 = self.lrelu(self.conv9_2(conv9_1))

        conv10 = self.conv10(conv9_2)
        reflectance = self.sigmoid(conv10)

        return reflectance