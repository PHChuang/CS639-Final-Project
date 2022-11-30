import torch
import torch.nn as nn


class AdjustmentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def cancat_fix(self, x, y):
        _, _, x_h, x_w = x.size()
        _, _, y_h, y_w = y.size()
        diff_x = x_w - y_w
        diff_y = x_h - y_h
        y = nn.functional.pad(y, (diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2))
        return torch.cat((x, y), dim=1)
    
    def forward(self, input_r, input_i):
        with torch.no_grad():
            ratio = torch.ones_like(input_r) * input_i
        concat1 = self.cancat_fix(input_r, ratio)
        conv1 = self.lrelu(self.conv1(concat1))
        conv2 = self.lrelu(self.conv2(conv1))
        conv3 = self.lrelu(self.conv3(conv2))
        conv4 = self.conv4(conv3)
        adjustment = self.sigmoid(conv4)
        return adjustment