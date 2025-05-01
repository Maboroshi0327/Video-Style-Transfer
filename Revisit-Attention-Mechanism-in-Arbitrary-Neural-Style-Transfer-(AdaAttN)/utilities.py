import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import os
import numpy as np
from typing import Union

import cv2


toTensor255 = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
    ]
)
toTensor = transforms.ToTensor()
toPil = transforms.ToPILImage()
raftTransforms = transforms.Compose(
    [
        # min–max → [0,1]
        transforms.Lambda(lambda x: x.div(255.0)),
        # map [0, 1] into [-1, 1]
        transforms.Normalize(mean=0.5, std=0.5),
    ]
)


def toTensorCrop(size_resize: tuple = (512, 512), size_crop: tuple = (256, 256)):
    """
    size_resize: (height, width) \\
    size_crop: (height, width)
    """
    transform = transforms.Compose(
        [
            transforms.Resize(size_resize),
            transforms.RandomCrop(size=size_crop),
            toTensor255,
        ]
    )
    return transform


def cv2_to_tensor(img: np.ndarray, resize: Union[tuple, None] = None) -> torch.Tensor:
    """
    resize: (width, height)
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if resize is not None:
        img = cv2.resize(img, resize, interpolation=cv2.INTER_AREA)

    return toTensor255(img)


def list_files(directory):
    files = [f.path for f in os.scandir(directory) if f.is_file()]
    return sorted(files)


def list_folders(directory):
    folders = [f.path for f in os.scandir(directory) if f.is_dir()]
    return sorted(folders)


def mkdir(directory, delete_existing_files=False):
    # create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # delete old data
    if delete_existing_files:
        files = list_files(directory)
        for f in files:
            os.remove(f)


def vgg_normalize(batch: torch.Tensor):
    # normalize using imagenet mean and std
    batch = batch.float()
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(batch.device)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(batch.device)
    normalized_batch = (batch / 255.0 - mean) / std
    return normalized_batch


def print_parameters(model):
    for name, _ in model.named_parameters():
        print(name)


def print_state_dict(state_dict):
    for name, _ in state_dict.items():
        print(name)


def feature_down_sample(feat, last_feat_idx):
    size = feat[last_feat_idx].shape[-2:]

    # Downsample the features
    result = list()
    for i in range(last_feat_idx):
        down = F.interpolate(feat[i], size=size, mode="bilinear", align_corners=False)
        result.append(down)
    result.append(feat[last_feat_idx])

    # Concatenate the features
    return torch.cat(result, dim=1)


def warp(x, flo, padding_mode="zeros"):
    B, C, H, W = x.size()

    # Mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo

    # Scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid, mode="bilinear", padding_mode=padding_mode, align_corners=False)
    return output


def flow_warp_mask(flo01, flo10, padding_mode="zeros", threshold=2):
    flo01 = flo01.unsqueeze(0)
    flo10 = flo10.unsqueeze(0)
    B, C, H, W = flo01.size()

    # Mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    if flo01.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo10
    flo01 = grid + flo01

    # Scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    flow_warp = F.grid_sample(flo01, vgrid, mode="bilinear", padding_mode=padding_mode, align_corners=False)

    # create mask
    flow_warp = flow_warp.squeeze(0)
    grid = grid.squeeze(0)
    warp_error = torch.abs(flow_warp - grid)
    warp_error = torch.sum(warp_error, dim=0)
    mask = warp_error < threshold
    mask = mask.float()

    return mask
