import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from PIL import Image
from tqdm import tqdm
from collections import OrderedDict

from vgg19 import VGG19
from network import StylizingNetwork
from datasets import Videvo
from utilities import gram_matrix, vgg_normalize, toTensor255, warp


device = "cuda" if torch.cuda.is_available() else "cpu"
epoch_start = 1
epoch_end = 10
batch_size = 2
LR = 1e-3
ALPHA = 1
BETA = 10
GAMMA = 1e-3
LAMBDA = 1e4
IMG_SIZE = (640, 360)


def train():
    # Datasets and model
    dataloader = DataLoader(
        Videvo("./Videvo-jpg"),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        prefetch_factor=2,
    )
    model = StylizingNetwork().to(device)

    # Optimizer and loss
    adam = optim.Adam(model.parameters(), lr=LR)
    L2distance = nn.MSELoss(reduction="mean")
    L2distanceMatrix = nn.MSELoss(reduction="none")
    vgg19 = VGG19().to(device)

    # Style image
    style_img_path = "./styles/starry-night.jpg"
    style = Image.open(style_img_path).convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
    style = toTensor255(style).unsqueeze(0).to(device)

    # Style image Gram Matrix
    style_features = vgg19(vgg_normalize(style))
    style_GM = [gram_matrix(f) for f in style_features.values()]

    # Training loop
    for epoch in range(epoch_start, epoch_end + 1):
        model.train()

        batch_iterator = tqdm(dataloader, desc=f"Epoch {epoch}/{epoch_end}", leave=True)
        for img1, img2, flow, mask in batch_iterator:
            img1 = img1.to(device)
            img2 = img2.to(device)
            mask = mask.to(device)
            flow = flow.to(device)

        # Zero gradients
        adam.zero_grad()

        # Forward pass
        styled_img1 = model(img1)
        styled_img2 = model(img2)

        # Normalize and use VGG19 to get features
        styled_features = vgg19(styled_img2)
        content_features = vgg19(img2)["relu4_2"]

        # Content Loss
        content_loss = L2distance(styled_features["relu4_2"], content_features)
        content_loss *= ALPHA

        # Style Loss
        style_loss = 0
        for gram_s, feature in zip(style_GM, styled_features.values()):
            gram_f = gram_matrix(feature)
            style_loss += L2distance(gram_f, gram_s.expand(gram_f.shape[0], -1, -1))
        style_loss *= BETA

        # Regularization Loss
        reg1 = torch.square(styled_img2[:, :, :-1, 1:] - styled_img2[:, :, :-1, :-1])
        reg2 = torch.square(styled_img2[:, :, 1:, :-1] - styled_img2[:, :, :-1, :-1])
        reg_loss = torch.sqrt(reg1 + reg2).sum()
        reg_loss *= GAMMA

        # Temporal Loss
        warped_style = warp(styled_img1, flow)
        temporal_loss = mask * L2distanceMatrix(styled_img2, warped_style)
        temporal_loss = LAMBDA * temporal_loss.mean()

        # Total Loss
        loss = content_loss + style_loss + reg_loss + temporal_loss

        # Backward pass
        loss.backward()
        adam.step()

        # Use OrderedDict to set suffix information
        postfix = OrderedDict(
            [
                ("loss", loss.item()),
                ("SL", style_loss.item()),
                ("CL", content_loss.item()),
                ("RL", reg_loss.item()),
                ("TL", temporal_loss.item()),
            ]
        )

        # Update progress bar
        batch_iterator.set_postfix(postfix)

    # Save model
    torch.save(model.state_dict(), f"./models/RTNSTV_epoch_{epoch}_batchSize_{batch_size}.pth")


if __name__ == "__main__":
    print(f"Using {device} device")
    train()
