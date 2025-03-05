import torch

from network import StylizingNetwork
from utilities import toPil
from datasets import Coco, WikiArt


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    coco = Coco()
    wikiart = WikiArt()
    print(len(coco), len(wikiart))

    model = StylizingNetwork().to(device)
    model.load_state_dict(torch.load("./models/AdaAttN-test_epoch_6_batchSize_8.pth", weights_only=True), strict=True)
    model.eval()

    idx = 0
    c, s = coco[idx], wikiart[idx]
    c, s = c[0].unsqueeze(0).to(device), s[0].unsqueeze(0).to(device)

    with torch.no_grad():
        _, cs = model(c, s)
        cs = cs.squeeze(0)
        cs = toPil(cs.byte())
        cs.save(f"./{idx}.png")
