import torch

from PIL import Image

from vgg19 import VGG19
from network import StylizingNetwork
from utilities import toTensor255, toPil, list_files


MODEL_EPOCH = 10
BATCH_SIZE = 8
MODEL_PATH = f"./models/AdaAttN-image_epoch_{MODEL_EPOCH}_batchSize_{BATCH_SIZE}.pth"

IMAGE_SIZE = (512, 512)
ACTIAVTION = "softmax"


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model
    vgg19 = VGG19().to(device)
    model = StylizingNetwork(ACTIAVTION).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True), strict=True)

    vgg19.eval()
    model.eval()

    # Load images
    print("Loading images...")
    c = list()
    s = list()
    for content_path in list_files("./contents"):
        img = Image.open(content_path).convert("RGB").resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.BILINEAR)
        img = toTensor255(img).unsqueeze(0).to(device)
        c.append(img)

    for style_path in list_files("./styles"):
        img = Image.open(style_path).convert("RGB").resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.BILINEAR)
        img = toTensor255(img).unsqueeze(0).to(device)
        s.append(img)

    # Model inference
    for i, content in enumerate(c):
        for j, style in enumerate(s):
            print(f"Processing content {i + 1} and style {j + 1}...")

            # Model inference
            with torch.no_grad():
                fc = vgg19(content)
                fs = vgg19(style)
                cs = model(fc, fs)
                cs = cs.clamp(0, 255)

            # Save the results
            save_path = f"./results/content_{i + 1}_style_{j + 1}.jpg"
            toPil(cs[0].byte()).save(save_path)
