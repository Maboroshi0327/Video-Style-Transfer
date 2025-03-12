import torch
import torch.nn as nn
from torch import linalg as LA
from torch.nn import functional as F

import numpy as np

from utilities import feature_down_sample


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


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvReLU(512, 512, kernel_size=3, stride=1)
        self.conv2 = ConvReLU(512, 256, kernel_size=3, stride=1)
        self.conv3 = nn.Sequential(
            ConvReLU(512, 256, kernel_size=3, stride=1),
            ConvReLU(256, 256, kernel_size=3, stride=1),
            ConvReLU(256, 256, kernel_size=3, stride=1),
        )
        self.conv4 = ConvReLU(256, 128, kernel_size=3, stride=1)
        self.conv5 = ConvReLU(128, 128, kernel_size=3, stride=1)
        self.conv6 = ConvReLU(128, 64, kernel_size=3, stride=1)
        self.conv7 = ConvReLU(64, 64, kernel_size=3, stride=1)
        self.conv8 = Conv(64, 3, kernel_size=3, stride=1)

    def forward(self, x5, x4, x3):
        x = F.interpolate(x5, scale_factor=2, mode="bilinear", align_corners=False)
        x = x + x4
        x = self.conv1(x)

        x = self.conv2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        x = torch.cat([x, x3], dim=1)
        x = self.conv3(x)
        x = self.conv4(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        x = self.conv5(x)
        x = self.conv6(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        x = self.conv7(x)
        x = self.conv8(x)

        return x


class Softmax(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k):
        return self.softmax(torch.bmm(q, k))


class CosineSimilarity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k):
        """
        q:   (b, d, c)
        k:   (b, c, d)
        out: (b, d, d)
        """
        q_norm = LA.vector_norm(q, dim=-1, keepdim=True)
        k_norm = LA.vector_norm(k, dim=1, keepdim=True)
        s = torch.bmm(q, k) / (q_norm * k_norm) + 1
        a = s / s.sum(dim=-1, keepdim=True)
        return a


class AdaAttnNoConv(nn.Module):
    def __init__(self, v_dim, qk_dim, activation="softmax"):
        super().__init__()
        self.norm_q = nn.InstanceNorm2d(qk_dim, affine=False)
        self.norm_k = nn.InstanceNorm2d(qk_dim, affine=False)
        self.norm_v = nn.InstanceNorm2d(v_dim, affine=False)

        if activation == "softmax":
            self.activation = Softmax()
        elif activation == "cosine":
            self.activation = CosineSimilarity()
        else:
            raise ValueError(f"Unknown activation function: {activation}")

    def forward(self, c_x, s_x, c_1x, s_1x):
        # Q^T
        Q = self.norm_q(c_1x)
        b, _, h, w = Q.size()
        Q = Q.view(b, -1, h * w).permute(0, 2, 1)

        # K
        K = self.norm_k(s_1x)
        b, _, h, w = K.size()
        K = K.view(b, -1, h * w)

        # V^T
        V = s_x
        b, _, h, w = V.size()
        V = V.view(b, -1, h * w).permute(0, 2, 1)

        # A * V^T
        A = self.activation(Q, K)
        M = torch.bmm(A, V)

        # S
        Var = torch.bmm(A, V**2) - M**2
        S = torch.sqrt(Var.clamp(min=1e-6))

        # Reshape M and S
        b, _, h, w = c_x.size()
        M = M.view(b, h, w, -1).permute(0, 3, 1, 2)
        S = S.view(b, h, w, -1).permute(0, 3, 1, 2)

        return S * self.norm_v(c_x) + M


class AdaAttN(nn.Module):
    def __init__(self, v_dim, qk_dim, activation="softmax"):
        super().__init__()
        self.f = nn.Conv2d(qk_dim, qk_dim, 1)
        self.g = nn.Conv2d(qk_dim, qk_dim, 1)
        self.h = nn.Conv2d(v_dim, v_dim, 1)
        self.norm_q = nn.InstanceNorm2d(qk_dim, affine=False)
        self.norm_k = nn.InstanceNorm2d(qk_dim, affine=False)
        self.norm_v = nn.InstanceNorm2d(v_dim, affine=False)

        if activation == "softmax":
            self.activation = Softmax()
        elif activation == "cosine":
            self.activation = CosineSimilarity()
        else:
            raise ValueError(f"Unknown activation function: {activation}")

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
        A = self.activation(Q, K)
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
    def __init__(self, activation="softmax"):
        super().__init__()

        self.adaattn = nn.ModuleList(
            [
                AdaAttN(256, 64 + 128 + 256, activation=activation),
                AdaAttN(512, 64 + 128 + 256 + 512, activation=activation),
                AdaAttN(512, 64 + 128 + 256 + 512 + 512, activation=activation),
            ]
        )

        self.decoder = Decoder()

    def forward(self, fc, fs):
        fc = list(fc.values())
        fs = list(fs.values())

        # fcs_3~5
        adaattn = list()
        for i in range(3):
            idx_feat = i + 2
            c_1x = feature_down_sample(fc, idx_feat)
            s_1x = feature_down_sample(fs, idx_feat)
            adaattn.append(self.adaattn[i](fc[idx_feat], fs[idx_feat], c_1x, s_1x))

        # decode
        cs = self.decoder(adaattn[2], adaattn[1], adaattn[0])
        return cs


if __name__ == "__main__":
    from vgg19 import VGG19

    # Test the network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StylizingNetwork(activation="cosine").to(device)
    vgg19 = VGG19().to(device)
    x = torch.randn(1, 3, 256, 256).to(device)
    x = vgg19(x)
    cs = model(x, x)
    print(cs.shape)
