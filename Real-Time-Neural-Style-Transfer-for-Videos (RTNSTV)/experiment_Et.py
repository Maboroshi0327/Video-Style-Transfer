import torch

from network import StylizingNetwork
from utilities import temporal_errors_sintel


device = "cuda" if torch.cuda.is_available() else "cpu"
# error = temporal_errors_sintel(StylizingNetwork, "./models/Candy_epoch_1_batchSize_2.pth", "alley_1", device)
# print(error)

for i in range(1, 11):
    error = temporal_errors_sintel(StylizingNetwork, f"./models/Candy_epoch_{i}_batchSize_2.pth", "alley_1", device)
    print(f"Epoch {i}: {error}")