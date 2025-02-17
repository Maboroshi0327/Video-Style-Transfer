import os
from typing import Union

import cv2
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from utilities import list_files, list_folders, mkdir, toTensor255, toTensor, toPil, warp, flow_warp_mask


def get_frames(video_path, img_size=(640, 360)):
    files = list_files(video_path)

    # progress bar
    pbar = tqdm(desc="Extracting frames", total=len(files))

    videos_idx = 0
    for file in files:
        # create directory if it doesn't exist
        save_dir = f"./Videvo/frames/{videos_idx:05d}/"
        os.makedirs(save_dir, exist_ok=True)

        # delete old images
        files = list_files(save_dir)
        for f in files:
            os.remove(f)

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


def calculate_optical_flow(frame_path="./Videvo/frames", flow_path="./Videvo/flow/"):
    data_num = 0
    for folder in list_folders(frame_path):
        data_num += len(list_files(folder)) - 1

    # progress bar
    pbar = tqdm(desc="Calculating optical flow", total=data_num)

    for folder in sorted(os.listdir(frame_path)):
        front_dir = os.path.join(flow_path, folder, "front")
        back_dir = os.path.join(flow_path, folder, "back")
        mkdir(front_dir, True)
        mkdir(back_dir, True)

        # calculate optical flow
        deepflow = cv2.optflow.createOptFlow_DeepFlow()
        files = list_files(os.path.join(frame_path, folder))
        for i in range(len(files) - 1):
            img1 = cv2.imread(files[i])
            img2 = cv2.imread(files[i + 1])
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            flow_into_future = deepflow.calc(img1, img2, None)
            flow_into_past = deepflow.calc(img2, img1, None)
            flow_into_future = toTensor(flow_into_future)
            flow_into_past = toTensor(flow_into_past)

            # save flow data
            torch.save(flow_into_future, os.path.join(front_dir, f"{i:05d}_01.pt"))
            torch.save(flow_into_past, os.path.join(back_dir, f"{i+1:05d}_10.pt"))

            pbar.update(1)


def test_Videvo():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = "./Videvo-jpg"
    dataset = Videvo(path)
    img1, img2, flow, mask = dataset[726]
    img1 = img1.to(device)
    img2 = img2.to(device)
    mask = mask.to(device)
    flow = flow.to(device)
    print(img1.shape, img2.shape, flow.shape, mask.shape)

    # warp image & visualize flow
    next_img = warp(img1.unsqueeze(0), flow.unsqueeze(0)).squeeze(0)
    warp_mask = mask * next_img

    img1 = toPil(img1.byte())
    img2 = toPil(img2.byte())
    next_img = toPil(next_img.byte())
    warp_mask = toPil(warp_mask.byte())

    img1.save("img1.jpg")
    img2.save("img2.jpg")
    next_img.save("next_img.jpg")
    warp_mask.save("warp_mask.jpg")


class Videvo(Dataset):
    def __init__(self, path: str, frame_num: int = 1):
        super().__init__()
        path_frame = os.path.join(path, "frames")
        path_flow = os.path.join(path, "flow")

        assert os.path.exists(path_frame), f"Path {path_frame} does not exist."
        assert os.path.exists(path_flow), f"Path {path_flow} does not exist."
        assert 1 <= frame_num, "Frame number must be equal or greater than 1."

        self.frames = list()
        for folder in list_folders(path_frame):
            files = list_files(folder)
            for i in range(len(files) - frame_num):
                self.frames.append(files[i : i + frame_num + 1])

        self.flow = list()
        for folder in list_folders(path_flow):
            front_dir = os.path.join(folder, "front")
            back_dir = os.path.join(folder, "back")
            front_files = list_files(front_dir)
            back_files = list_files(back_dir)
            for i in range(len(front_files)):
                self.flow.append((front_files[i + frame_num - 1], back_files[i + frame_num - 1]))

        self.deepflow = cv2.optflow.createOptFlow_DeepFlow()
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

        # read optical flow
        flow_into_future = torch.load(self.flow[idx][0], weights_only=True)
        flow_into_past = torch.load(self.flow[idx][1], weights_only=True)

        # create mask
        mask = flow_warp_mask(flow_into_future, flow_into_past)

        return img1, img2, flow_into_past, mask


if __name__ == "__main__":
    # get_frames(video_path="../datasets/Videvo")
    calculate_optical_flow()
    # dataset = Videvo(path="./Videvo", frame_num=1)
