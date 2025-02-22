import os
import struct
from typing import Union

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

toTensor255 = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
    ]
)
toTensor = transforms.ToTensor()
toPil = transforms.ToPILImage()
gaussianBlur = transforms.GaussianBlur(kernel_size=3, sigma=1.0)


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


def visualize_flow(flow: Union[torch.Tensor, np.ndarray]):
    if isinstance(flow, torch.Tensor):
        flow = flow.cpu().numpy()

    hsv = np.zeros((flow.shape[1], flow.shape[2], 3), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[0], flow[1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return rgb


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


def read_sintel_flow(filename):
    TAG_FLOAT = 202021.25  # Check for this when reading the file

    with open(filename, "rb") as stream:
        # Read the tag
        tag_data = stream.read(4)
        tag = struct.unpack("<f", tag_data)[0]
        if tag != TAG_FLOAT:
            raise ValueError(f"ReadFlowFile({filename}): wrong tag (possibly due to big-endian machine?)")

        # Read width and height
        width_data = stream.read(4)
        height_data = stream.read(4)
        width = struct.unpack("<i", width_data)[0]
        height = struct.unpack("<i", height_data)[0]

        if width < 1 or width > 99999:
            raise ValueError(f"ReadFlowFile({filename}): illegal width {width}")
        if height < 1 or height > 99999:
            raise ValueError(f"ReadFlowFile({filename}): illegal height {height}")

        # Prepare to read flow data
        n_bands = 2
        flow_data = np.zeros((height, width, n_bands), dtype=np.float32)

        # Read the flow data
        for y in range(height):
            row_data = stream.read(width * n_bands * 4)
            if len(row_data) != width * n_bands * 4:
                raise ValueError(f"ReadFlowFile({filename}): file is too short")

            row_data = np.frombuffer(row_data, dtype=np.float32)
            flow_data[y, :, 0] = row_data[::2]
            flow_data[y, :, 1] = row_data[1::2]

        # Check for extra data
        if stream.read(1):
            raise ValueError(f"ReadFlowFile({filename}): file is too long")

    return flow_data


def gram_matrix(y: torch.Tensor):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (h * w)
    return gram


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


def cvframe_to_tensor(frame, resize: Union[tuple, None] = None):
    """
    resize: (width, height)
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if resize is not None:
        frame = cv2.resize(frame, resize, interpolation=cv2.INTER_LINEAR)

    return toTensor255(frame)


def temporal_errors_sintel(model_class, model_path: str, scene: str, device: str = "cuda") -> Union[float, int]:
    frames_folder = f"../datasets/MPI-Sintel-complete/training/final/{scene}"
    mask_folder = f"../datasets/MPI-Sintel-complete/training/occlusions/{scene}"
    flow_folder = f"../datasets/MPI-Sintel-complete/training/flow/{scene}"
    frames_fies = list_files(frames_folder)
    mask_files = list_files(mask_folder)
    flow_files = list_files(flow_folder)

    model = model_class().to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True), strict=True)
    model.eval()

    L2distanceMatrix = nn.MSELoss(reduction="none")

    error = 0
    for idx in range(len(flow_files)):
        img0 = cv2.imread(frames_fies[idx])
        img1 = cv2.imread(frames_fies[idx + 1])
        img0 = cvframe_to_tensor(img0).to(device)
        img1 = cvframe_to_tensor(img1).to(device)
        img0 = img0.unsqueeze(0)
        img1 = img1.unsqueeze(0)

        # styled images
        with torch.no_grad():
            styled_img0 = model(img0)
            styled_img1 = model(img1)

        # flow
        flow = read_sintel_flow(flow_files[idx])
        flow = toTensor(flow).to(device)

        # mask
        mask = cv2.imread(mask_files[idx], cv2.IMREAD_GRAYSCALE)
        mask = (mask == 0).astype(np.uint8)
        mask = toTensor(mask).to(device)

        # Temporal Loss
        mask = mask.unsqueeze(1)
        mask = mask.expand(-1, img0.shape[1], -1, -1)
        warped_style = warp(styled_img1, flow)
        temporal_loss = mask * L2distanceMatrix(styled_img0, warped_style)
        error += temporal_loss.mean().item()

    error /= len(flow_files)
    error = np.sqrt(error)
    return error


def calculate_mse(model_class, input_frame_num: int, model_path: str, video_path: str, device: str = "cuda"):
    model = model_class(input_frame_num).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True), strict=True)
    cap = cv2.VideoCapture(video_path)
    mse = nn.MSELoss(reduction="mean")
    content_styled_imgs = list()
    count = 0
    loss = 0

    # Read the few frames to match the input_frame_num
    imgs = list()
    for _ in range(input_frame_num):
        _, frame = cap.read()
        imgs.append(cvframe_to_tensor(frame, resize=(640, 360)))

    # Start calculating the MSE
    while True:
        # Pass the input tensor through the model
        with torch.no_grad():
            input_tensor = torch.cat(imgs, dim=0).unsqueeze(0).to(device)
            *_, output_tensor = model(input_tensor)
            output_tensor = output_tensor.clamp(0, 255)

        content_img = imgs[-1].unsqueeze(0).to(device)
        styled_img = output_tensor
        content_styled_imgs.append([content_img, styled_img])

        if len(content_styled_imgs) == 2:
            x_t = content_styled_imgs[0][0]
            x_t1 = content_styled_imgs[1][0]
            y_t = content_styled_imgs[0][1]
            y_t1 = content_styled_imgs[1][1]

            x = x_t1 - x_t
            y = y_t1 - y_t
            loss += mse(x, y).item()

            count += 1
            content_styled_imgs.pop(0)

        # Read the next frame
        ret, frame = cap.read()
        if not ret:
            break

        # Update the input tensor list
        imgs.pop(0)
        imgs.append(cvframe_to_tensor(frame, resize=(640, 360)))

    cap.release()
    return loss / count


class Inference:
    def __init__(
        self,
        model_class,
        model_path: str,
        video_path: str,
        device: str = "cuda",
    ):
        self.model = model_class().to(device)
        self.model.load_state_dict(torch.load(model_path, weights_only=True), strict=True)
        self.video_path = video_path
        self.device = device

        self.cap = cv2.VideoCapture(video_path)

    def __del__(self):
        self.cap.release()

    def __iter__(self):
        # Start the video style transfer
        while True:
            # Read the next frame
            ret, frame = self.cap.read()
            if not ret:
                break

            # Pass the input tensor through the model
            with torch.no_grad():
                input_tensor = cvframe_to_tensor(frame, resize=(640, 360)).unsqueeze(0).to(self.device)
                output_tensor = self.model(input_tensor)

            # Convert output tensor back to image format
            output_image = output_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
            output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
            output_image = output_image.astype("uint8")

            yield output_image
