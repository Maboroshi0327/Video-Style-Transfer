import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
from collections import OrderedDict

from vgg19 import VGG19
from datasets import Coco, WikiArt
from network import StylizingNetwork


EPOCH_START = 1
EPOCH_END = 10
BATCH_SIZE = 8
IMG_SIZE = (256, 256)
LR = 1e-3
LAMBDA_G = 1
LAMBDA_L = 1e3


def global_stylized_loss(fcs, fs):
    # Mean distance
    mean_dist = fcs.mean(dim=(2, 3)) - fs.mean(dim=(2, 3))
    mean_dist_norm = torch.linalg.vector_norm(mean_dist, ord=2, dim=None, keepdim=False)
    mean_dist_norm = mean_dist_norm / mean_dist.numel()

    # Standard deviation distance
    std_dist = fcs.std(dim=(2, 3)) - fs.std(dim=(2, 3))
    std_dist_norm = torch.linalg.vector_norm(std_dist, ord=2, dim=None, keepdim=False)
    std_dist_norm = std_dist_norm / std_dist.numel()

    # Loss for each ReLU_x_1 layer
    return mean_dist_norm + std_dist_norm


def local_feature_loss(fcs, adaattn):
    dist = fcs - adaattn
    dist_norm = torch.linalg.vector_norm(dist, ord=2, dim=None, keepdim=False)
    dist_norm = dist_norm / dist.numel()
    return dist_norm


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Datasets
    dataloader_coco = DataLoader(
        Coco("../datasets/coco", IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        prefetch_factor=2,
    )
    dataloader_wikiart = DataLoader(
        WikiArt("../datasets/WikiArt", IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        prefetch_factor=2,
    )
    data_length = min(len(dataloader_coco), len(dataloader_wikiart))

    # Model
    model = StylizingNetwork().to(device)
    model.train()

    # VGG19 for perceptual loss
    vgg19 = VGG19().to(device)
    vgg19.eval()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Training loop
    for epoch in range(EPOCH_START, EPOCH_END + 1):

        # Batch iterator
        dataloader_zip = zip(dataloader_coco, dataloader_wikiart)
        batch_iterator = tqdm(dataloader_zip, desc=f"Epoch {epoch}/{EPOCH_END}", total=data_length, leave=True)

        # Training
        for (content, _), (style, _) in batch_iterator:
            bc, *_ = content.size()
            bs, *_ = style.size()
            b_min = min(bc, bs)

            content = content[:b_min,].to(device)
            style = style[:b_min,].to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            adaattn, cs = model(content, style)
            fs_dict = vgg19(style)
            fcs_dict = vgg19(cs)

            # Global stylized loss
            loss_gs = 0
            loss_gs += global_stylized_loss(fcs_dict["relu2_1"], fs_dict["relu2_1"])
            loss_gs += global_stylized_loss(fcs_dict["relu3_1"], fs_dict["relu3_1"])
            loss_gs += global_stylized_loss(fcs_dict["relu4_1"], fs_dict["relu4_1"])
            loss_gs += global_stylized_loss(fcs_dict["relu5_1"], fs_dict["relu5_1"])
            loss_gs *= LAMBDA_G

            # Local feature loss
            loss_lf = 0
            loss_lf += local_feature_loss(fcs_dict["relu3_1"], adaattn[0])
            loss_lf += local_feature_loss(fcs_dict["relu4_1"], adaattn[1])
            loss_lf += local_feature_loss(fcs_dict["relu5_1"], adaattn[2])
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
        torch.save(model.state_dict(), f"./models/AdaAttN-test_epoch_{epoch}_batchSize_{BATCH_SIZE}.pth")


if __name__ == "__main__":
    train()
