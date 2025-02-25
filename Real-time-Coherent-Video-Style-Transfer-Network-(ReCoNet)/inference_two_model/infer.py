import sys
sys.path.append("/root/project")

import cv2
import torch
from network import ReCoNet
from utilities import Inference


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_1_infer = Inference(ReCoNet, 1, "./models/starry-night-1.pth", "../datasets/video3.mp4", device)
    model_4_infer = Inference(ReCoNet, 1, "./models/starry-night-2.pth", "../datasets/video3.mp4", device)

    for output_image_1, output_image_4 in zip(model_1_infer, model_4_infer):
        cv2.imshow("Input Frames =  1", output_image_1)
        cv2.imshow("Input Frames =  4", output_image_4)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
