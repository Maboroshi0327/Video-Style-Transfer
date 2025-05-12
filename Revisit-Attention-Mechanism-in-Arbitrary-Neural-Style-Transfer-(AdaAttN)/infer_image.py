import torch

from PIL import Image

from vgg19 import VGG19
from datasets import CocoWikiArt
from network import StylizingNetwork
from utilities import toTensor255, toPil


MODEL_EPOCH = 5
BATCH_SIZE = 8
MODEL_PATH = f"./models/AdaAttN-image_epoch_{MODEL_EPOCH}_batchSize_{BATCH_SIZE}.pth"

CONTENT_IDX = 66666
CONTENT_PATH = None
STYLE_PATH = None

CONTENT_PATH = "./contents/Chair.jpg"
STYLE_PATH = "./styles/Candy.jpg"

IMAGE_SIZE = (256, 256)
ACTIAVTION = "softmax"


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model
    vgg19 = VGG19().to(device)
    model = StylizingNetwork(ACTIAVTION).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True), strict=True)

    vgg19.eval()
    model.eval()

    # Load dataset
    dataset = CocoWikiArt()
    coco, wikiart = dataset[CONTENT_IDX]

    # Use COCO as content image if CONTENT_PATH is None
    if CONTENT_PATH is not None:
        c = Image.open(CONTENT_PATH).convert("RGB").resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.BILINEAR)
        c = toTensor255(c).unsqueeze(0).to(device)
    else:
        c = coco.unsqueeze(0).to(device)

    # Use wikiart as style image if STYLE_PATH is None
    if STYLE_PATH is not None:
        s = Image.open(STYLE_PATH).convert("RGB").resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.BILINEAR)
        s = toTensor255(s).unsqueeze(0).to(device)
    else:
        s = wikiart.unsqueeze(0).to(device)

    # Model inference
    with torch.no_grad():
        fc = vgg19(c)
        fs = vgg19(s)
        cs = model(fc, fs)
        cs = cs.clamp(0, 255)
        print(cs[0, 0].min(), cs[0, 0].max())
        print(cs[0, 1].min(), cs[0, 1].max())
        print(cs[0, 2].min(), cs[0, 2].max())

    # Save images
    toPil(c.squeeze(0).byte()).save("./results/content.png")
    toPil(s.squeeze(0).byte()).save("./results/style.png")
    toPil(cs.squeeze(0).byte()).save("./results/stylized.png")
