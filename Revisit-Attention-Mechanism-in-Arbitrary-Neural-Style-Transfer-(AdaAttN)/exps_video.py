import torch
import torch.nn as nn
from torchvision.models.optical_flow import raft_large

import cv2
from PIL import Image
from tqdm import tqdm

from utilities import toTensor255, raftTransforms, cv2_to_tensor, warp, flow_warp_mask
from vgg19 import VGG19
from network import StylizingNetwork


# MODEL_PATH = "./models/AdaAttN-image_epoch_5_batchSize_8.pth"
# ACTIAVTION = "softmax"

MODEL_PATH = "./models/AdaAttN-video_epoch_5_batchSize_4.pth"
ACTIAVTION = "cosine"

VIDEO_PATH = "../datasets/Videvo/67.mp4"
STYLE_PATH = "./styles/Udnie.png"

IMAGE_SIZE1 = (256, 256)
IMAGE_SIZE2 = (256, 512)
NUM_LAYERS = 3
NUM_HEADS = 8
HIDDEN_DIM = 512


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model
    model = StylizingNetwork(activation=ACTIAVTION).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True), strict=True)
    model.eval()

    vgg19 = VGG19().to(device)
    vgg19.eval()

    # Load optical flow model
    raft = raft_large(weights="Raft_Large_Weights.C_T_SKHT_V2").to(device)
    raft = raft.eval()

    # Load style image
    s = Image.open(STYLE_PATH).convert("RGB").resize((IMAGE_SIZE1[1], IMAGE_SIZE1[0]), Image.BILINEAR)
    s = toTensor255(s).unsqueeze(0).to(device)
    with torch.no_grad():
        fs = vgg19(s)

    # Count for optical flow loss
    count = 0
    optical_loss = 0
    mseMatrix = nn.MSELoss(reduction="none")

    # Load video
    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Total frames: {total_frames}", f"FPS: {fps}")

    # Progress bar
    bar = tqdm(total=total_frames, desc="Processing video", unit="frame")

    # First frame
    frames = list()
    ret, frame = cap.read()
    frames.append(frame)
    bar.update(1)

    while True:
        ret, frame = cap.read()
        frames.append(frame)
        if not ret:
            break

        with torch.no_grad():
            # Convert frame to tensor
            c1 = cv2_to_tensor(frame[0], resize=(IMAGE_SIZE2[1], IMAGE_SIZE2[0])).unsqueeze(0).to(device)
            c2 = cv2_to_tensor(frame[1], resize=(IMAGE_SIZE2[1], IMAGE_SIZE2[0])).unsqueeze(0).to(device)

            # Forward pass
            fc1 = vgg19(c1)
            fc2 = vgg19(c2)
            cs1 = model(fc1, fs)
            cs2 = model(fc2, fs)
            cs1 = cs1.clamp(0, 255)
            cs2 = cs2.clamp(0, 255)

            # Calculate optical flow
            c1 = raftTransforms(c1)
            c2 = raftTransforms(c2)
            flow_into_future = raft(c1, c2)[-1].squeeze(0)
            flow_into_past = raft(c2, c1)[-1].squeeze(0)

            # create mask
            mask = flow_warp_mask(flow_into_future, flow_into_past).unsqueeze(0)

            # Optical Flow Loss
            warped_cs1 = warp(cs1, flow_into_past)
            mask = mask.unsqueeze(1)
            mask = mask.expand(-1, cs1.shape[1], -1, -1)
            loss = torch.sum(mask * mseMatrix(cs2, warped_cs1)) / (cs1.shape[1] * cs1.shape[2] * cs1.shape[3])
            optical_loss += loss
            count += 1

            # Pop the frame 1
            frames.pop(0)

            # Print loss
            loss_temp = torch.sqrt(optical_loss).item() / count
            bar.set_postfix(optical_loss=loss_temp)
            bar.update(1)

    bar.close()
    cap.release()
    optical_loss = torch.sqrt(optical_loss).item() / count
    print(f"Optical Flow Loss: {optical_loss}")
