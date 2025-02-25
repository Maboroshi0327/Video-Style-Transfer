import cv2
import torch

from utilities import Inference
from network import StylizingNetwork

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inference = Inference(StylizingNetwork, "./models/Candy_epoch_1_batchSize_1.pth", "../datasets/Videvo/48.mp4", device)
    for output in inference:
        cv2.imshow("Frames", output)
        if cv2.waitKey(15) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
