import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.models.optical_flow import raft_large

import os
import random
from PIL import Image

import cv2
from tqdm import tqdm

from utilities import toTensor255, toPil, toTensorCrop, raftTransforms, list_files, list_folders, mkdir, flow_warp_mask


def Coco(path="../datasets/coco", size_crop: tuple = (256, 256)):
    """
    size_crop: (height, width)
    """
    dataset = ImageFolder(root=path, transform=toTensorCrop(size_crop=size_crop))
    return dataset


def WikiArt(path="../datasets/WikiArt", size_crop: tuple = (256, 256)):
    """
    size_crop: (height, width)
    """
    dataset = ImageFolder(root=path, transform=toTensorCrop(size_crop=size_crop))
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


class Sintel(Dataset):
    def __init__(self, image_size: tuple = (256, 512), path="../datasets/MPI-Sintel-complete", scene="all", device: str = "cuda"):
        if scene == "all":
            path = os.path.join(path, "training/final")
        else:
            path = os.path.join(path, "training/final", scene)
        assert os.path.exists(path), f"Path {path} does not exist."

        self.path = path
        self.device = device
        self.image_size = image_size
        self.resolution = (image_size[1], image_size[0])

        self.frame = list()
        if scene == "all":
            for folder in list_folders(path):
                files = list_files(folder)
                for i in range(len(files) - 1):
                    self.frame.append(files[i : i + 2])
        else:
            files = list_files(path)
            for i in range(len(files) - 1):
                self.frame.append(files[i : i + 2])

        self.length = len(self.frame)

        self.raft = raft_large(weights="Raft_Large_Weights.C_T_SKHT_V2").to(device)
        self.raft = self.raft.eval()

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        """
        idx -> Index of the image pair, optical flow and mask to be returned.
        """
        with torch.no_grad():
            # read image
            img1 = Image.open(self.frame[idx][0]).convert("RGB").resize(self.resolution, Image.BILINEAR)
            img2 = Image.open(self.frame[idx][1]).convert("RGB").resize(self.resolution, Image.BILINEAR)
            img1 = toTensor255(img1).to(self.device)
            img2 = toTensor255(img2).to(self.device)

            # Calculate optical flow
            img1_batch = img1.unsqueeze(0)
            img2_batch = img2.unsqueeze(0)
            img1_batch = raftTransforms(img1_batch)
            img2_batch = raftTransforms(img2_batch)
            flow_into_future = self.raft(img1_batch, img2_batch)[-1].squeeze(0)
            flow_into_past = self.raft(img2_batch, img1_batch)[-1].squeeze(0)

            # create mask
            mask = flow_warp_mask(flow_into_future, flow_into_past)

        return img1, img2, flow_into_past, mask


def get_frames(video_path="../datasets/Videvo", img_size=(512, 256)):
    files = list_files(video_path)

    # progress bar
    pbar = tqdm(desc="Extracting frames", total=len(files))

    videos_idx = 0
    for file in files:
        # create directory if it doesn't exist
        save_dir = f"./Videvo/frames/{videos_idx:05d}/"
        mkdir(save_dir)

        # read video and save frames
        cap = cv2.VideoCapture(file)
        frames_idx = 0
        while True:
            # read video frame
            ret, frame = cap.read()
            if not ret:
                break

            # resize frame and save it
            frame = cv2.resize(frame, img_size, interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(save_dir, f"{frames_idx:05d}.jpg"), frame)
            frames_idx += 1

        cap.release()
        videos_idx += 1

        pbar.update(1)


class Videvo(Dataset):
    def __init__(self, path: str = "./Videvo", frame_num: int = 1):
        super().__init__()
        path_frame = os.path.join(path, "frames")

        assert os.path.exists(path_frame), f"Path {path_frame} does not exist."
        assert 1 <= frame_num, "Frame number must be equal or greater than 1."

        self.frames = list()
        for folder in list_folders(path_frame):
            files = list_files(folder)
            for i in range(len(files) - frame_num):
                self.frames.append(files[i : i + frame_num + 1])

        self.frame_num = frame_num
        self.length = len(self.frames)
        print(f"Videvo total data: {self.length}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        # read image
        imgs_gray = list()
        imgs_tensor = list()
        for path in self.frames[idx]:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            imgs_gray.append(img_gray)
            imgs_tensor.append(toTensor255(img))

        img1 = torch.cat(imgs_tensor[0 : self.frame_num], dim=0)
        img2 = torch.cat(imgs_tensor[1 : self.frame_num + 1], dim=0)
        return img1, img2


class VidevoWikiArt(Dataset):
    def __init__(self, videvo_path="./Videvo", wikiart_path="../datasets/WikiArt"):
        self.videvo = Videvo(videvo_path)
        self.wikiart = WikiArt(wikiart_path, size_crop=(256, 512))
        self.videvo_len = len(self.videvo)
        self.wikiart_len = len(self.wikiart)

    def __len__(self):
        return self.videvo_len

    def __getitem__(self, idx):
        wikiart_idx = random.randint(0, self.wikiart_len - 1)
        return self.videvo[idx][0], self.videvo[idx][1], self.wikiart[wikiart_idx][0]


if __name__ == "__main__":
    dataset = CocoWikiArt()
    c, s = dataset[123]
    print("CocoWikiArt dataset")
    print("dataset length:", len(dataset))

    from utilities import toPil

    toPil(c.byte()).save("coco.png")
    toPil(s.byte()).save("wikiart.png")
    print("Saved coco.png and wikiart.png")

    # get_frames()
    # dataset = VidevoWikiArt()
    # c1, c2, s = dataset[10000]
    # toPil(c1.byte()).save("c1.png")
    # toPil(c2.byte()).save("c2.png")
    # toPil(s.byte()).save("s.png")
