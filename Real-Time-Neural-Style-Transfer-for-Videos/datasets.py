import os
from typing import Union

import cv2

import torch
from torch.utils.data import Dataset

from utilities import list_files, list_folders, toTensor255, toTensor, toPil, warp, flow_warp_mask


def get_frames(video_path, img_size=(640, 360)):
    files = list_files(video_path)

    videos_idx = 0
    for file in files:
        # create directory if it doesn't exist
        save_dir = f"./Videvo-jpg/{videos_idx:05d}/"
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


class Videvo(Dataset):
    def __init__(self, path: str, frame_num: int = 1):
        super().__init__()
        assert 1 <= frame_num, "Frame number must be equal or greater than 1."

        self.paths = list()
        for folder in list_folders(path):
            files = list_files(folder)
            for i in range(len(files) - frame_num):
                self.paths.append(files[i : i + frame_num + 1])

        self.deepflow = cv2.optflow.createOptFlow_DeepFlow()
        self.frame_num = frame_num
        self.length = len(self.paths)
        print(f"Videvo total data: {self.length}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        # read image
        imgs_gray = list()
        imgs_tensor = list()
        for path in self.paths[idx]:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            imgs_gray.append(img_gray)
            imgs_tensor.append(toTensor255(img))

        img1 = torch.cat(imgs_tensor[0 : self.frame_num], dim=0)
        img2 = torch.cat(imgs_tensor[1 : self.frame_num + 1], dim=0)

        # compute optical flow
        flow_into_future = self.deepflow.calc(imgs_gray[-2], imgs_gray[-1], None)
        flow_into_past = self.deepflow.calc(imgs_gray[-1], imgs_gray[-2], None)
        flow_into_future = toTensor(flow_into_future)
        flow_into_past = toTensor(flow_into_past)

        # create mask
        mask = flow_warp_mask(flow_into_future, flow_into_past)

        return img1, img2, flow_into_past, mask


if __name__ == "__main__":
    path = "./Videvo-jpg"
    dataset = Videvo(path)
    img1, img2, flow_into_past, mask = dataset[726]
    print(img1.shape, img2.shape, flow_into_past.shape, mask.shape)

    # warp image & visualize flow
    next_img = warp(img1.unsqueeze(0), flow_into_past.unsqueeze(0)).squeeze(0)
    warp_mask = mask * next_img

    img1 = toPil(img1.byte())
    img2 = toPil(img2.byte())
    next_img = toPil(next_img.byte())
    warp_mask = toPil(warp_mask.byte())

    img1.save("img1.jpg")
    img2.save("img2.jpg")
    next_img.save("next_img.jpg")
    warp_mask.save("warp_mask.jpg")
