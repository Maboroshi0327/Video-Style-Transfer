import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
from collections import OrderedDict

from vgg19 import VGG19
from datasets import CocoWikiArt
from utilities import feature_down_sample
from network import StylizingNetwork, AdaAttnNoConv
from lossfn import global_stylized_loss, local_feature_loss


EPOCH_START = 1
EPOCH_END = 5
BATCH_SIZE = 8
LR = 1e-4
LAMBDA_G = 10
LAMBDA_L = 3
ACTIAVTION = "softmax"


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Datasets
    dataloader = DataLoader(
        CocoWikiArt(),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        prefetch_factor=2,
    )

    # Model
    model = StylizingNetwork(activation=ACTIAVTION).to(device)
    model.train()

    # AdaAttN for calculating local feature loss
    adaattn_no_conv = nn.ModuleList(
        [
            AdaAttnNoConv(256, 64 + 128 + 256, ACTIAVTION),
            AdaAttnNoConv(512, 64 + 128 + 256 + 512, ACTIAVTION),
            AdaAttnNoConv(512, 64 + 128 + 256 + 512 + 512, ACTIAVTION),
        ]
    )
    adaattn_no_conv = adaattn_no_conv.to(device)
    adaattn_no_conv.eval()

    # VGG19 as feature extractor (encoder)
    vgg19 = VGG19().to(device)
    vgg19.eval()

    # Loss function
    mse = torch.nn.MSELoss(reduction="mean")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Training loop
    for epoch in range(EPOCH_START, EPOCH_END + 1):

        # Batch iterator
        batch_iterator = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCH_END}", leave=True)

        # Training
        for content, style in batch_iterator:
            content = content.to(device)
            style = style.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # VGG19 encoder
            fc = vgg19(content)
            fs = vgg19(style)

            # Forward pass
            cs = model(fc, fs)
            fcs = vgg19(cs)

            # Global stylized loss
            loss_gs = 0
            loss_gs += global_stylized_loss(fcs["relu2_1"], fs["relu2_1"], mse)
            loss_gs += global_stylized_loss(fcs["relu3_1"], fs["relu3_1"], mse)
            loss_gs += global_stylized_loss(fcs["relu4_1"], fs["relu4_1"], mse)
            loss_gs += global_stylized_loss(fcs["relu5_1"], fs["relu5_1"], mse)
            loss_gs *= LAMBDA_G

            # Local feature loss
            fc = list(fc.values())
            fs = list(fs.values())

            loss_lf = 0
            for i in range(3):
                idx_feat = i + 2
                c_1x = feature_down_sample(fc, idx_feat)
                s_1x = feature_down_sample(fs, idx_feat)
                adaattn = adaattn_no_conv[i](fc[idx_feat], fs[idx_feat], c_1x, s_1x)
                loss_lf += local_feature_loss(fcs[f"relu{i + 3}_1"], adaattn, mse)
            loss_lf *= LAMBDA_L

            # Loss
            loss = loss_gs + loss_lf

            # Backward pass
            loss.backward()
            optimizer.step()

            # Use OrderedDict to set suffix information
            postfix = OrderedDict(
                [
                    ("loss", loss.item()),
                    ("loss_gs", loss_gs.item()),
                    ("loss_lf", loss_lf.item()),
                ]
            )

            # Update progress bar
            batch_iterator.set_postfix(postfix)

        # Save model
        torch.save(model.state_dict(), f"./models/AdaAttN-image_epoch_{epoch}_batchSize_{BATCH_SIZE}.pth")


if __name__ == "__main__":
    train()
