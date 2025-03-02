import sys
sys.path.append("/root/project")

import cv2
import torch
from network import ReCoNet
from utilities import Inference


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_10_infer = Inference(ReCoNet, 1, "./models/Flow_input_1_epoch_10_batchSize_2.pth", "../datasets/video3.mp4", device)
    model_4_infer = Inference(ReCoNet, 1, "./models/Flow_input_1_epoch_4_batchSize_2.pth", "../datasets/video3.mp4", device)

    for output_image_10, output_image_4 in zip(model_10_infer, model_4_infer):
        cv2.imshow("10", output_image_10)
        cv2.imshow("4", output_image_4)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
