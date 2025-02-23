import torch

from network import StylizingNetwork
from utilities import temporal_errors_sintel


device = "cuda" if torch.cuda.is_available() else "cpu"
error = temporal_errors_sintel(StylizingNetwork, "./models/Candy_epoch_2_batchSize_2.pth", "alley_1", device)
print(error)
