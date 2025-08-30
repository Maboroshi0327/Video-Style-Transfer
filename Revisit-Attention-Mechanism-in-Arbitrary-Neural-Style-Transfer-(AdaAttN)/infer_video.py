import torch

import os
import cv2
import imageio
import numpy as np
from PIL import Image

from vgg19 import VGG19
from network import StylizingNetwork
from utilities import toTensor255, cv2_to_tensor, mkdir


# MODE = "Original"
MODE = "Stylized"

MODEL_PATH = "./models/AdaAttN-video_epoch_10_batchSize_4.pth"
ACTIAVTION = "cosine"

# VIDEO_PATH = "../datasets/Videvo/19.mp4"
# STYLE_PATH = "./styles/Sketch.jpg"

# VIDEO_PATH = "../datasets/Videvo/98.mp4"
# STYLE_PATH = "./styles/The-Scream.jpg"

VIDEO_PATH = "../datasets/Videvo/38.mp4"
STYLE_PATH = "./styles/Untitled-1964.jpg"


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vgg19 = VGG19().to(device)
    vgg19.eval()

    model = StylizingNetwork(activation=ACTIAVTION).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True), strict=True)
    model.eval()

    s = Image.open(STYLE_PATH).convert("RGB").resize((512, 256), Image.BILINEAR)
    s = toTensor255(s).unsqueeze(0).to(device)
    fs = vgg19(s)

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {fps}")

    frames = list()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        with torch.no_grad():
            # Convert frame to tensor
            c = cv2_to_tensor(frame, resize=(512, 256))
            c = c.unsqueeze(0).to(device)
            fc = vgg19(c)

            # Forward pass
            cs = model(fc, fs)
            if MODE == "Stylized":
                cs = cs.clamp(0, 255)
            else:
                cs = c.clamp(0, 255)

        # Convert output tensor back to image format
        cs = cs.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        frames.append(cs)
        cs = cv2.cvtColor(cs, cv2.COLOR_RGB2BGR)
        cs = cs.astype("uint8")

        # Display the frame
        cv2.imshow("Frames", cs)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
    
    # Create output directory if it doesn't exist
    SAVE_PATH = "./results/Video"
    mkdir(SAVE_PATH, delete_existing_files=True)

    # Save the frames as images
    for i in range(len(frames)):
        imageio.imwrite(os.path.join(SAVE_PATH, f"{MODE}_{i}.jpg"), frames[i])

    # Save the frames as a video
    imageio.mimsave(os.path.join(SAVE_PATH, f"{MODE}.mp4"), frames, fps=fps)
