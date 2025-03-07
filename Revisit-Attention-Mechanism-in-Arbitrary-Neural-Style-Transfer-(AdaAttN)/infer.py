import torch

from utilities import toPil
from datasets import Coco, WikiArt
from vgg19 import VGG19
from network import StylizingNetwork


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    coco = Coco()
    wikiart = WikiArt()
    print(len(coco), len(wikiart))

    vgg19 = VGG19().to(device)
    vgg19.eval()

    model = StylizingNetwork().to(device)
    model.load_state_dict(torch.load("./models/AdaAttN-test_epoch_1_batchSize_8.pth", weights_only=True), strict=True)
    model.eval()

    idx = 12345
    c, s = coco[idx], wikiart[idx]
    c, s = c[0].unsqueeze(0).to(device), s[0].unsqueeze(0).to(device)
    fc = vgg19(c)
    fs = vgg19(s)

    with torch.no_grad():
        _, cs = model(fc, fs)
        cs = cs.squeeze(0)
        cs = toPil(cs.byte())
        cs.save(f"./{idx}.png")
