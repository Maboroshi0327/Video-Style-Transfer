import sys
sys.path.append("/root/project")

import torch
from network import ReCoNet, ReCoNetSD1, ReCoNetSD2
from utilities import calculate_mse

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mse = calculate_mse(ReCoNet, 1, "./models/starry-night-2.pth", "../datasets/video3.mp4", device)
    print(f"MSE: {mse}")