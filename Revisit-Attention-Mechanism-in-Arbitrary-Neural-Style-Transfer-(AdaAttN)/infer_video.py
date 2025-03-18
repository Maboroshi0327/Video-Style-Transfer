import torch

import cv2
from PIL import Image

from vgg19 import VGG19
from network import StylizingNetwork
from utilities import toTensor255, cvframe_to_tensor


MODEL_PATH = "./models/AdaAttN-video_epoch_1_batchSize_4.pth"
STYLE_PATH = "./styles/candy.jpg"
VIDEO_PATH = "../datasets/Videvo/20.mp4"
ACTIAVTION = "cosine"
# ACTIAVTION = "softmax"


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vgg19 = VGG19().to(device)
    vgg19.eval()

    model = StylizingNetwork(activation=ACTIAVTION).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True), strict=True)
    model.eval()

    s = Image.open(STYLE_PATH).convert("RGB").resize((512, 256), Image.BILINEAR)
    s = toTensor255(s).unsqueeze(0).to(device)
    fs = vgg19(s)

    cap = cv2.VideoCapture(VIDEO_PATH)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to tensor
        c = cvframe_to_tensor(frame, resize=(512, 256))
        c = c.unsqueeze(0).to(device)
        fc = vgg19(c)

        # Forward pass
        with torch.no_grad():
            cs = model(fc, fs)
            cs = cs.clamp(0, 255)

        # Convert output tensor back to image format
        cs = cs.squeeze(0).cpu().permute(1, 2, 0).numpy()
        cs = cv2.cvtColor(cs, cv2.COLOR_RGB2BGR)
        cs = cs.astype("uint8")

        # Display the frame
        cv2.imshow("Frames", cs)
        if cv2.waitKey(15) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
