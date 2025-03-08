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

    idx = 0
    c, s = coco[idx], wikiart[idx]
    c, s = c[0], s[0]
    toPil(c.byte()).save("./content.png")
    toPil(s.byte()).save("./style.png")
    c, s = c.unsqueeze(0).to(device), s.unsqueeze(0).to(device)

    fc = vgg19(c)
    fs = vgg19(c.clone())

    with torch.no_grad():
        _, cs = model(fc, fs)

        # min_v = cs.min()
        # max_v = cs.max()
        # cs = (cs - min_v) / (max_v - min_v) * 255
        # print(min_v, max_v)

        cs = cs.squeeze(0)
        cs = toPil(cs.byte())
        cs.save("./stylized.png")
