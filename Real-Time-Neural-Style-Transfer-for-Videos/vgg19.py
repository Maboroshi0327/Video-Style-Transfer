import torch
import torch.nn as nn
from torchvision.models import vgg19

from utilities import vgg_normalize

class VGG19(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(weights="VGG19_Weights.IMAGENET1K_V1").features

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()

        # Relu1_2
        for x in range(4):
            self.slice1.add_module(str(x), vgg[x])

        # Relu2_2
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg[x])

        # Relu3_2
        for x in range(9, 14):
            self.slice3.add_module(str(x), vgg[x])

        # Relu4_2
        for x in range(14, 23):
            self.slice4.add_module(str(x), vgg[x])

        # Freeze all VGG parameters by setting requires_grad to False
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = vgg_normalize(x)
        x = self.slice1(x)
        relu1_2 = x
        x = self.slice2(x)
        relu2_2 = x
        x = self.slice3(x)
        relu3_2 = x
        x = self.slice4(x)
        relu4_2 = x

        features = {
            "relu1_2": relu1_2,
            "relu2_2": relu2_2,
            "relu3_2": relu3_2,
            "relu4_2": relu4_2,
        }
        return features


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(1, 3, 224, 224).to(device)
    model = VGG19().to(device)
    features = model(x)
    for key, value in features.items():
        print(key, value.shape)
