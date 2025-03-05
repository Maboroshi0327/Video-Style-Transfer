from torchvision.datasets import ImageFolder

from utilities import toTensorCrop


def Coco(path="../datasets/coco", img_size=(256, 256)):
    dataset = ImageFolder(root=path, transform=toTensorCrop(img_size))
    return dataset


def WikiArt(path="../datasets/WikiArt", img_size=(256, 256)):
    dataset = ImageFolder(root=path, transform=toTensorCrop(img_size))
    return dataset


if __name__ == "__main__":
    dataset = Coco()
    img = dataset[0][0]
    print(len(dataset))
    print(img.shape)
    print(img.min(), img.max())
