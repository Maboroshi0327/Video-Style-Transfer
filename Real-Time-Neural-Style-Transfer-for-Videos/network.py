from typing import Union

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, activation: Union[nn.Module, None] = None):
        super().__init__()
        pad = int(np.floor(kernel_size / 2))
        self.pad = torch.nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=False)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.activation = activation

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class Res(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = Conv(in_channels, out_channels, 3, 1, nn.ReLU())
        self.conv2 = Conv(out_channels, out_channels, 3, 1, None)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)

        if residual.shape[1] != x.shape[1]:
            pad = (0, 0, 0, 0, 0, x.shape[1] - residual.shape[1])
            residual = F.pad(residual, pad, "constant", 0)

        x = x + residual
        return x


class Deconv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, activation: Union[nn.Module, None] = None):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=1, output_padding=1, bias=False)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.activation = activation

    def forward(self, x):
        x = self.deconv(x)
        x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class StylizingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv(3, 16, 3, 1, nn.ReLU())
        self.conv2 = Conv(16, 32, 3, 2, nn.ReLU())
        self.conv3 = Conv(32, 48, 3, 2, nn.ReLU())
        self.res1 = Res(48, 48)
        self.res2 = Res(48, 48)
        self.res3 = Res(48, 48)
        self.res4 = Res(48, 48)
        self.res5 = Res(48, 48)
        self.deconv1 = Deconv(48, 32, 3, 2, nn.ReLU())
        self.deconv2 = Deconv(32, 16, 3, 2, nn.ReLU())
        self.conv4 = Conv(16, 3, 3, 1, nn.Tanh())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.conv4(x)
        x = (x + 1) / 2 * 255
        return x


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(1, 3, 256, 256).to(device)
    model = StylizingNetwork().to(device)
    y = model(x)
    print(y.shape)
