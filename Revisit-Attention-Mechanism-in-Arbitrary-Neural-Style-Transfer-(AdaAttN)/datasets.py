from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

import random

from utilities import toTensorCrop


def Coco(path="../datasets/coco"):
    dataset = ImageFolder(root=path, transform=toTensorCrop())
    return dataset


def WikiArt(path="../datasets/WikiArt"):
    dataset = ImageFolder(root=path, transform=toTensorCrop())
    return dataset


class CocoWikiArt(Dataset):
    def __init__(self, coco_path="../datasets/coco", wikiart_path="../datasets/WikiArt"):
        self.coco = Coco(coco_path)
        self.wikiart = WikiArt(wikiart_path)
        self.coco_len = len(self.coco)
        self.wikiart_len = len(self.wikiart)

    def __len__(self):
        return self.coco_len

    def __getitem__(self, idx):
        wikiart_idx = random.randint(0, self.wikiart_len - 1)
        return self.coco[idx][0], self.wikiart[wikiart_idx][0]


if __name__ == "__main__":
    dataset = CocoWikiArt()
    c, s = dataset[123]
    print("CocoWikiArt dataset")
    print("dataset length:", len(dataset))

    from utilities import toPil
    toPil(c.byte()).save("coco.png")
    toPil(s.byte()).save("wikiart.png")
    print("Saved coco.png and wikiart.png")
