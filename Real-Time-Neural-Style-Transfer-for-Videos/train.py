import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import os
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use("Agg")

from vgg19 import VGG19
from network import StylizingNetwork
from datasets import Videvo, FlyingThings3D_Monkaa
from utilities import gram_matrix, toTensor255, warp


device = "cuda" if torch.cuda.is_available() else "cpu"
epoch_start = 1
epoch_end = 10
batch_size = 1
LR = 1e-3
# ALPHA = 1
# BETA = 10
# GAMMA = 1e-1
# LAMBDA = 1e-1
ALPHA = 1e7
BETA = 1e7
GAMMA = 1e-1
LAMBDA = 1e5
IMG_SIZE = (640, 360)


def spatial_loss(content, styled, style_GM, vgg19):
    L2distance = nn.MSELoss(reduction="mean")

    # Normalize and use VGG19 to get features
    content_features = vgg19(content)["relu4_2"]
    styled_features = vgg19(styled)

    # Content Loss
    content_loss = L2distance(content_features, styled_features["relu4_2"])
    content_loss *= ALPHA

    # Style Loss
    style_loss = 0
    for gram_s, feature in zip(style_GM, styled_features.values()):
        gram_f = gram_matrix(feature)
        style_loss += L2distance(gram_f, gram_s.expand(gram_f.shape[0], -1, -1))
    style_loss *= BETA

    # Regularization Loss
    reg1 = torch.square(styled[:, :, :-1, 1:] - styled[:, :, :-1, :-1])
    reg2 = torch.square(styled[:, :, 1:, :-1] - styled[:, :, :-1, :-1])
    reg_loss = torch.sqrt((reg1 + reg2).clamp(min=1e-8)).mean()
    reg_loss *= GAMMA

    return content_loss, style_loss, reg_loss


def train():
    # Datasets and model
    # dataloader = DataLoader(
    #     Videvo("./Videvo"),
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=4,
    #     prefetch_factor=2,
    # )
    dataloader = DataLoader(
        FlyingThings3D_Monkaa("../datasets/SceneFlowDatasets"),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        prefetch_factor=2,
    )
    model = StylizingNetwork().to(device)

    # Optimizer and loss
    adam = optim.Adam(model.parameters(), lr=LR)
    L2distanceMatrix = nn.MSELoss(reduction="none")
    vgg19 = VGG19().to(device)

    # Style image
    style_img_path = "./styles/candy.jpg"
    style = Image.open(style_img_path).convert("RGB")
    style = toTensor255(style).unsqueeze(0).to(device)

    # Style image Gram Matrix
    style_features = vgg19(style)
    style_GM = [gram_matrix(f) for f in style_features.values()]

    # Training loop
    for epoch in range(epoch_start, epoch_end + 1):
        model.train()

        loss_c = list()
        loss_s = list()
        loss_r = list()
        loss_t = list()
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

            # Spatial Loss
            content_loss_1, style_loss_1, reg_loss_1 = spatial_loss(img1, styled_img1, style_GM, vgg19)
            content_loss_2, style_loss_2, reg_loss_2 = spatial_loss(img2, styled_img2, style_GM, vgg19)
            content_loss = content_loss_1 + content_loss_2
            style_loss = style_loss_1 + style_loss_2
            reg_loss = reg_loss_1 + reg_loss_2

            # Temporal Loss
            aaa = mask.clone()
            mask = mask.unsqueeze(1)
            mask = mask.expand(-1, styled_img2.shape[1], -1, -1)
            non_zero_count = mask.sum() + 1e-8
            warped_style = warp(styled_img1, flow)
            temporal_loss = mask * L2distanceMatrix(styled_img2, warped_style)
            temporal_loss = temporal_loss.sum() / non_zero_count
            temporal_loss *= LAMBDA

            # Total Loss
            loss_c.append(content_loss.item())
            loss_s.append(style_loss.item())
            loss_r.append(reg_loss.item())
            loss_t.append(temporal_loss.item())
            loss = content_loss + style_loss + reg_loss + temporal_loss

            # if torch.isnan(loss):
            #     print("Loss is NaN")
            #     print(f"Content Loss: {content_loss.item()}")
            #     print(f"Style Loss: {style_loss.item()}")
            #     print(f"Reg Loss: {reg_loss.item()}")
            #     print(f"Temporal Loss: {temporal_loss.item()}")
            #     print(non_zero_count)

            #     print("Saving images")
            #     os.makedirs("./nan_images", exist_ok=True)
            #     img1 = toPil(img1[0].byte())
            #     img1.save(f"./nan_images/img1_epoch_{epoch}_batchSize_{batch_size}.png")
            #     styled_img1 = toPil(styled_img1[0].byte())
            #     styled_img1.save(f"./nan_images/styled_img1_epoch_{epoch}_batchSize_{batch_size}.png")
            #     img2 = toPil(img2[0].byte())
            #     img2.save(f"./nan_images/img2_epoch_{epoch}_batchSize_{batch_size}.png")
            #     styled_img2 = toPil(styled_img2[0].byte())
            #     styled_img2.save(f"./nan_images/styled_img2_epoch_{epoch}_batchSize_{batch_size}.png")
            #     mask = toPil(aaa)
            #     mask.save(f"./nan_images/mask_epoch_{epoch}_batchSize_{batch_size}.png")
            #     flow_rgb = visualize_flow(flow.squeeze(0))
            #     cv2.imwrite(f"./nan_images/flow_epoch_{epoch}_batchSize_{batch_size}.png", flow_rgb)
            #     exit()

            # Backward pass
            loss.backward()
            adam.step()

            # Use OrderedDict to set suffix information
            postfix = OrderedDict(
                [
                    ("loss", loss.item()),
                    ("CL", content_loss.item()),
                    ("SL", style_loss.item()),
                    ("RL", reg_loss.item()),
                    ("TL", temporal_loss.item()),
                ]
            )

            # Update progress bar
            batch_iterator.set_postfix(postfix)

        # Save model
        torch.save(model.state_dict(), f"./models/Candy_epoch_{epoch}_batchSize_{batch_size}.pth")

        # Save loss plots
        os.makedirs("./loss_plots", exist_ok=True)
        plt.figure()
        iterations = range(1, len(loss_c) + 1)
        plt.plot(iterations[1000:], loss_c[1000:], label="Content Loss")
        plt.plot(iterations[1000:], loss_s[1000:], label="Style Loss")
        plt.plot(iterations[1000:], loss_r[1000:], label="Regularization Loss")
        plt.plot(iterations[1000:], loss_t[1000:], label="Temporal Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title(f"Losses for Epoch {epoch}")
        plt.legend()
        plt.savefig(f"./loss_plots/Candy_epoch_{epoch}_loss.png")
        plt.close()


if __name__ == "__main__":
    print(f"Using {device} device")
    train()
