import cv2
import torch

from utilities import Inference
from network import StylizingNetwork

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inference = Inference(StylizingNetwork, "./models/RTNSTV_epoch_2_batchSize_2.pth", "../datasets/video3.mp4", device)
    for output in inference:
        cv2.imshow("Frames", output)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
