import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np


class Conv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()
        pad = int(np.floor(kernel_size / 2))
        self.pad = torch.nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        return x


class ConvReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()
        self.conv = Conv(in_channels, out_channels, kernel_size, stride)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class ConvTanh(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()
        self.conv = Conv(in_channels, out_channels, kernel_size, stride)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = self.tanh(x)
        x = (x + 1) / 2 * 255
        return x


class ConvReluInterpolate(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, scale_factor: float):
        super().__init__()
        self.conv = Conv(in_channels, out_channels, kernel_size, stride)
        self.relu = nn.ReLU()
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
        return x


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvReLU(3, 64, kernel_size=3, stride=1)
        self.conv2 = ConvReLU(64, 128, kernel_size=3, stride=2)
        self.conv3 = ConvReLU(128, 256, kernel_size=3, stride=2)
        self.conv4 = ConvReLU(256, 512, kernel_size=3, stride=2)
        self.conv5 = ConvReLU(512, 512, kernel_size=3, stride=2)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x_1)
        x_3 = self.conv3(x_2)
        x_4 = self.conv4(x_3)
        x_5 = self.conv5(x_4)

        return [x_1, x_2, x_3, x_4, x_5]


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv1 = ConvReluInterpolate(512, 512, kernel_size=3, stride=1, scale_factor=2)
        self.deconv2 = ConvReluInterpolate(512, 256, kernel_size=3, stride=1, scale_factor=2)
        self.deconv3 = ConvReluInterpolate(256, 128, kernel_size=3, stride=1, scale_factor=2)
        self.deconv4 = ConvReluInterpolate(128, 64, kernel_size=3, stride=1, scale_factor=2)
        self.conv = ConvTanh(64, 3, kernel_size=3, stride=1)

    def forward(self, x5, x4, x3):
        x = self.deconv1(x5)
        x = self.deconv2(x + x4)
        x = self.deconv3(x + x3)
        x = self.deconv4(x)
        x = self.conv(x)
        return x


class AdaAttN(nn.Module):
    def __init__(self, v_dim, qk_dim):
        super().__init__()
        self.f = nn.Conv2d(qk_dim, qk_dim, 1)
        self.g = nn.Conv2d(qk_dim, qk_dim, 1)
        self.h = nn.Conv2d(v_dim, v_dim, 1)
        self.norm_q = nn.InstanceNorm2d(qk_dim, affine=True)
        self.norm_k = nn.InstanceNorm2d(qk_dim, affine=True)
        self.norm_v = nn.InstanceNorm2d(v_dim, affine=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, c_x, s_x, c_1x, s_1x):
        # Q^T
        Q = self.f(self.norm_q(c_1x))
        b, _, h, w = Q.size()
        Q = Q.view(b, -1, h * w).permute(0, 2, 1)

        # K
        K = self.g(self.norm_k(s_1x))
        b, _, h, w = K.size()
        K = K.view(b, -1, h * w)

        # V^T
        V = self.h(s_x)
        b, _, h, w = V.size()
        V = V.view(b, -1, h * w).permute(0, 2, 1)

        # A * V^T
        A = self.softmax(torch.bmm(Q, K))
        M = torch.bmm(A, V)

        # S
        Var = torch.bmm(A, V**2) - M**2
        S = torch.sqrt(Var.clamp(min=1e-6))

        # Reshape M and S
        b, _, h, w = c_x.size()
        M = M.view(b, h, w, -1).permute(0, 3, 1, 2)
        S = S.view(b, h, w, -1).permute(0, 3, 1, 2)

        return S * self.norm_v(c_x) + M


class StylizingNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()

        self.adaattn = nn.ModuleList(
            [
                AdaAttN(256, 64 + 128 + 256),
                AdaAttN(512, 64 + 128 + 256 + 512),
                AdaAttN(512, 64 + 128 + 256 + 512 + 512),
            ]
        )

        self.decoder = Decoder()

    def feature_down_sample(self, feat, last_feat_idx):
        size = feat[last_feat_idx].shape[-2:]

        # Downsample the features
        result = list()
        for i in range(last_feat_idx):
            down = F.interpolate(feat[i], size=size, mode="bilinear", align_corners=False)
            result.append(down)
        result.append(feat[last_feat_idx])

        # Concatenate the features
        return torch.cat(result, dim=1)

    def forward(self, c, s):
        # c_1~5
        c = self.encoder(c)

        # s_1~5
        s = self.encoder(s)

        # cs_3~5
        adaattn = list()
        for i in range(3):
            idx_feat = i + 2
            c_1x = self.feature_down_sample(c, idx_feat)
            s_1x = self.feature_down_sample(s, idx_feat)
            adaattn.append(self.adaattn[i](c[idx_feat], s[idx_feat], c_1x, s_1x))

        # decode
        cs = self.decoder(adaattn[2], adaattn[1], adaattn[0])
        return adaattn, cs


if __name__ == "__main__":
    # Test the network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StylizingNetwork().to(device)
    x = torch.randn(1, 3, 256, 256).to(device)
    adaattn, cs = model(x, x)
    print(cs.shape)
    for a in adaattn:
        print(a.shape)
