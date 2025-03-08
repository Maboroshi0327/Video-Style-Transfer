from torchvision.datasets import ImageFolder

from utilities import toTensorCrop


def Coco(path="../datasets/coco"):
    dataset = ImageFolder(root=path, transform=toTensorCrop())
    return dataset


def WikiArt(path="../datasets/WikiArt"):
    dataset = ImageFolder(root=path, transform=toTensorCrop())
    return dataset


if __name__ == "__main__":
    dataset = Coco()
    img = dataset[0][0]
    print(len(dataset))
    print(img.shape)
    print(img.min(), img.max())

    from utilities import toPil
    img = toPil(img.byte())
    img.save("test.png")

