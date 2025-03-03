import sys
sys.path.append("/root/project")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from PIL import Image
from tqdm import tqdm
from collections import OrderedDict

from datasets import FlyingThings3D_Monkaa, toTensor255
from network import ReCoNet, Vgg16
from utilities import gram_matrix, vgg_normalize, warp


device = "cuda" if torch.cuda.is_available() else "cpu"
epoch_start = 1
epoch_end = 6
batch_size = 2
input_frame_num = 1
LR = 1e-3
ALPHA = 1e5
BETA = 1e11
GAMMA = 1e-2
LAMBDA_F = 1e12
LAMBDA_O = 1e7
IMG_SIZE = (640, 360)


def train():
    # Datasets and model
    dataloader = DataLoader(
        FlyingThings3D_Monkaa("../datasets/SceneFlowDatasets", IMG_SIZE, input_frame_num),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        prefetch_factor=2,
    )
    model = ReCoNet(input_frame_num).to(device)

    # Optimizer and loss
    adam = optim.Adam(model.parameters(), lr=LR)
    L2distance = nn.MSELoss(reduction="mean")
    L2distanceMatrix = nn.MSELoss(reduction="none")
    vgg16 = Vgg16().to(device)

    # Style image
    style_img_path = "./styles/starry-night.jpg"
    style = Image.open(style_img_path).convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
    style = toTensor255(style).unsqueeze(0).to(device)

    # Style image Gram Matrix
    style_features = vgg16(vgg_normalize(style))
    style_GM = [gram_matrix(f) for f in style_features]

    # warp index
    index = (input_frame_num - 1) * 3
    index = [i for i in range(index, index + 3)]

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
            _, feature_map1, styled_img1 = model(img1)
            _, feature_map2, styled_img2 = model(img2)

            # Normalize and use VGG16 to get features
            styled_img1 = vgg_normalize(styled_img1)
            styled_img2 = vgg_normalize(styled_img2)
            img1 = vgg_normalize(img1[:, index])
            img2 = vgg_normalize(img2[:, index])
            styled_features1 = vgg16(styled_img1)
            styled_features2 = vgg16(styled_img2)
            content_features1 = vgg16(img1)
            content_features2 = vgg16(img2)

            # Warp feature maps
            feature_flow = nn.functional.interpolate(flow, size=feature_map1.shape[2:], mode="bilinear")
            feature_flow[:, 0] *= float(feature_map1.shape[3]) / flow.shape[3]
            feature_flow[:, 1] *= float(feature_map1.shape[2]) / flow.shape[2]
            warped_fmap = warp(feature_map1, feature_flow)

            # Create feature mask
            feature_mask = nn.functional.interpolate(mask.unsqueeze(1), size=feature_map1.shape[2:], mode="bilinear").squeeze(1)
            feature_mask = (feature_mask > 0).float()
            feature_mask = feature_mask.unsqueeze(1)
            feature_mask = feature_mask.expand(-1, feature_map1.shape[1], -1, -1)

            # Feature-Map-Level Temporal Loss
            f_temporal_loss = torch.sum(feature_mask * L2distanceMatrix(feature_map2, warped_fmap))
            non_zero_count = torch.nonzero(feature_mask).shape[0]
            f_temporal_loss *= 1 / non_zero_count
            f_temporal_loss *= LAMBDA_F

            # Output-Level Temporal Loss
            warped_style = warp(styled_img1, flow)
            warped_image = warp(img1, flow)
            output_term = styled_img2 - warped_style
            input_term = img2 - warped_image

            input_term = 0.2126 * input_term[:, 0] + 0.7152 * input_term[:, 1] + 0.0722 * input_term[:, 2]
            input_term = input_term.unsqueeze(1).expand(-1, img2.shape[1], -1, -1)

            mask = mask.unsqueeze(1)
            mask = mask.expand(-1, img2.shape[1], -1, -1)

            o_temporal_loss = torch.sum(mask * (L2distanceMatrix(output_term, input_term)))
            non_zero_count = torch.nonzero(mask).shape[0]
            o_temporal_loss *= 1 / non_zero_count
            o_temporal_loss *= LAMBDA_O

            # Content Loss
            content_loss = 0
            content_loss += L2distance(styled_features1[2], content_features1[2])
            content_loss += L2distance(styled_features2[2], content_features2[2])
            content_loss *= ALPHA

            # Style Loss
            style_loss = 0
            for i, gram_s in enumerate(style_GM):
                gram_img1 = gram_matrix(styled_features1[i])
                gram_img2 = gram_matrix(styled_features2[i])
                style_loss += L2distance(gram_img1, gram_s.expand(gram_img1.shape[0], -1, -1))
                style_loss += L2distance(gram_img2, gram_s.expand(gram_img1.shape[0], -1, -1))
            style_loss *= BETA

            # Regularization Loss
            reg1 = torch.square(styled_img1[:, :, :-1, 1:] - styled_img1[:, :, :-1, :-1])
            reg2 = torch.square(styled_img1[:, :, 1:, :-1] - styled_img1[:, :, :-1, :-1])
            reg3 = torch.square(styled_img2[:, :, :-1, 1:] - styled_img2[:, :, :-1, :-1])
            reg4 = torch.square(styled_img2[:, :, 1:, :-1] - styled_img2[:, :, :-1, :-1])
            reg_loss = GAMMA * torch.sum(reg1 + reg2 + reg3 + reg4)

            # Total Loss
            loss = f_temporal_loss + o_temporal_loss + content_loss + style_loss + reg_loss

            # Backward pass
            loss.backward()
            adam.step()

            # Use OrderedDict to set suffix information
            postfix = OrderedDict(
                [
                    ("loss", loss.item()),
                    ("CL", content_loss.item()),
                    ("SL", style_loss.item()),
                    ("FTL", f_temporal_loss.item()),
                    ("OTL", o_temporal_loss.item()),
                    ("RL", reg_loss.item()),
                ]
            )

            # Update progress bar
            batch_iterator.set_postfix(postfix)

        # Save model
        torch.save(model.state_dict(), f"./models/Flow_input_{input_frame_num}_epoch_{epoch}_batchSize_{batch_size}.pth")


if __name__ == "__main__":
    train()
