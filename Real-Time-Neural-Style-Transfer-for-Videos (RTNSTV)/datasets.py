import os
from typing import Union

import cv2
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from flowlib import read
from utilities import list_files, list_folders, mkdir, toTensor255, toTensor, toPil, warp, flow_warp_mask, visualize_flow


def get_frames(video_path="../datasets/Videvo", img_size=(640, 360)):
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


def test_warp(frame_path="./Videvo/frames", flow_path="./Videvo/flow/", warp_path="./Videvo/warp", device="cuda"):
    frame_folders = list_folders(frame_path)
    flow_folders = list_folders(flow_path)

    data_num = 0
    for folder in frame_folders:
        data_num += len(list_files(folder)) - 1

    # progress bar
    pbar = tqdm(desc="Calculating optical flow", total=data_num)

    for i in range(len(flow_folders)):
        # read frames and flows for each video
        frame_files = list_files(frame_folders[i])
        flows_01 = list_files(os.path.join(flow_folders[i], "front"))
        flows_10 = list_files(os.path.join(flow_folders[i], "back"))

        # create directory for saving warped images
        save_path = os.path.join(warp_path, f"{i:05d}")
        mkdir(save_path, True)

        for j in range(len(flows_10)):
            # read image
            img = cv2.imread(frame_files[j])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = toTensor255(img).unsqueeze(0).to(device)

            # read optical flow
            flow_into_future = torch.load(flows_01[j], weights_only=True)
            flow_into_past = torch.load(flows_10[j], weights_only=True)
            flow_into_future = flow_into_future.to(device)
            flow_into_past = flow_into_past.to(device)

            # create mask
            mask = flow_warp_mask(flow_into_future, flow_into_past)
            mask = mask.expand(1, img.shape[1], -1, -1)
            mask = mask.to(device)

            # warp image
            warped_img = mask * warp(img, flow_into_past)

            # save image
            warped_img = toPil(warped_img.squeeze(0).byte())
            warped_img.save(os.path.join(save_path, f"{j:05d}.jpg"))

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


class FlyingThings3D(Dataset):
    def __init__(self, path: str, resolution: tuple = (640, 360), frame_num: int = 1):
        """
        path -> Path to the location where the "frames_finalpass", "optical_flow" and "motion_boundaries" folders are kept inside the FlyingThings3D folder. \\
        resolution -> Resolution of the images to be returned. Width first, then height. \\
        frame_num -> Number of frames to be returned. Must be between 1 and 9.
        """
        super().__init__()
        path_frame = os.path.join(path, "frames_finalpass/TRAIN")
        path_flow = os.path.join(path, "optical_flow/TRAIN")
        path_motion = os.path.join(path, "motion_boundaries/TRAIN")

        assert os.path.exists(path_frame), f"Path {path_frame} does not exist."
        assert os.path.exists(path_flow), f"Path {path_flow} does not exist."
        assert os.path.exists(path_motion), f"Path {path_motion} does not exist."
        assert len(resolution) == 2 and isinstance(resolution, tuple), "Resolution must be a tuple of 2 integers."
        assert 1 <= frame_num and frame_num <= 9, "Frame number must be between 1 and 9."

        self.path = path
        self.frame = list()
        self.flow = list()
        self.motion = list()

        # progress bar
        pbar = tqdm(desc="Initial FlyingThings3D", total=2239 * (10 - frame_num) * 3)

        # frames_finalpass
        for abcpath in ["A", "B", "C"]:
            for folder in os.listdir(os.path.join(path_frame, abcpath)):
                files = list_files(os.path.join(path_frame, abcpath, folder, "left"))
                for i in range(10 - frame_num):
                    self.frame.append(files[i : i + frame_num + 1])
                    pbar.update(1)

        # optical_flow
        for abcpath in ["A", "B", "C"]:
            for folder in os.listdir(os.path.join(path_flow, abcpath)):
                files_into_future = list_files(os.path.join(path_flow, abcpath, folder, "into_future", "left"))
                files_into_past = list_files(os.path.join(path_flow, abcpath, folder, "into_past", "left"))
                for i in range(10 - frame_num):
                    self.flow.append((files_into_future[i + frame_num - 1], files_into_past[i + frame_num]))
                    pbar.update(1)

        # mask
        for abcpath in ["A", "B", "C"]:
            for folder in os.listdir(os.path.join(path_motion, abcpath)):
                files = list_files(os.path.join(path_motion, abcpath, folder, "into_future", "left"))
                for i in range(10 - frame_num):
                    self.motion.append(files[i + frame_num])
                    pbar.update(1)

        self.length = len(self.frame)
        self.resolution = resolution
        self.frame_num = frame_num

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> Union[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        idx -> Index of the image pair, optical flow and mask to be returned.
        """
        # read image
        imgs = list()
        for path in self.frame[idx]:
            img = Image.open(path).convert("RGB").resize(self.resolution, Image.BILINEAR)
            img = toTensor255(img)
            imgs.append(img)
        img1 = torch.cat(imgs[0 : self.frame_num], dim=0)
        img2 = torch.cat(imgs[1 : self.frame_num + 1], dim=0)

        # read flow
        flow_into_future = toTensor(read(self.flow[idx][0]).copy())[:-1]
        flow_into_past = toTensor(read(self.flow[idx][1]).copy())[:-1]
        originalflowshape = flow_into_past.shape

        flow_into_past = F.interpolate(
            flow_into_past.unsqueeze(0),
            size=(self.resolution[1], self.resolution[0]),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        flow_into_future = F.interpolate(
            flow_into_future.unsqueeze(0),
            size=(self.resolution[1], self.resolution[0]),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        flow_into_future[0] *= flow_into_future.shape[1] / originalflowshape[1]
        flow_into_future[1] *= flow_into_future.shape[2] / originalflowshape[2]
        flow_into_past[0] *= flow_into_past.shape[1] / originalflowshape[1]
        flow_into_past[1] *= flow_into_past.shape[2] / originalflowshape[2]

        # read motion boundaries
        motion = Image.open(self.motion[idx]).resize(self.resolution, Image.BILINEAR)
        motion = toTensor(motion).squeeze(0)
        motion[motion != 0] = 1
        motion = 1 - motion

        # create mask
        mask = flow_warp_mask(flow_into_future, flow_into_past)
        mask = mask * motion

        return img1, img2, flow_into_past, mask


class Monkaa(Dataset):
    def __init__(self, path: str, resolution: tuple = (640, 360), frame_num: int = 1):
        """
        path -> Path to the location where the "frames_finalpass", "optical_flow" and "motion_boundaries" folders are kept inside the Monkaa folder. \\
        resolution -> Resolution of the images to be returned. Width first, then height. \\
        frame_num -> Number of frames to be returned. Must be between 1 and 9.
        """
        super().__init__()
        path_frame = os.path.join(path, "frames_finalpass")
        path_flow = os.path.join(path, "optical_flow")
        path_motion = os.path.join(path, "motion_boundaries")

        assert os.path.exists(path_frame), f"Path {path_frame} does not exist."
        assert os.path.exists(path_flow), f"Path {path_flow} does not exist."
        assert os.path.exists(path_motion), f"Path {path_motion} does not exist."
        assert len(resolution) == 2 and isinstance(resolution, tuple), "Resolution must be a tuple of 2 integers."
        assert 1 <= frame_num and frame_num <= 9, "Frame number must be between 1 and 9."

        self.path = path
        self.frame = list()
        self.flow = list()
        self.motion = list()

        # count number of data
        data_num = 0
        for folder in os.listdir(path_frame):
            files = list_files(os.path.join(path_frame, folder, "left"))
            data_num += len(files) - frame_num

        # progress bar
        pbar = tqdm(desc="Initial Monkaa", total=data_num * 3)

        for folder in os.listdir(path_frame):
            files = list_files(os.path.join(path_frame, folder, "left"))
            for i in range(len(files) - frame_num):
                self.frame.append(files[i : i + frame_num + 1])
                pbar.update(1)

        for folder in os.listdir(path_flow):
            files_into_future = list_files(os.path.join(path_flow, folder, "into_future", "left"))
            files_into_past = list_files(os.path.join(path_flow, folder, "into_past", "left"))
            for i in range(len(files_into_future) - frame_num):
                self.flow.append((files_into_future[i + frame_num - 1], files_into_past[i + frame_num]))
                pbar.update(1)

        for folder in os.listdir(path_motion):
            files = list_files(os.path.join(path_motion, folder, "into_future", "left"))
            for i in range(len(files) - frame_num):
                self.motion.append(files[i + frame_num])
                pbar.update(1)

        self.length = len(self.frame)
        self.resolution = resolution
        self.frame_num = frame_num

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> Union[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        idx -> Index of the image pair, optical flow and mask to be returned.
        """
        # read image
        imgs = list()
        for path in self.frame[idx]:
            img = Image.open(path).convert("RGB").resize(self.resolution, Image.BILINEAR)
            img = toTensor255(img)
            imgs.append(img)
        img1 = torch.cat(imgs[0 : self.frame_num], dim=0)
        img2 = torch.cat(imgs[1 : self.frame_num + 1], dim=0)

        # read flow
        flow_into_future = toTensor(read(self.flow[idx][0]).copy())[:-1]
        flow_into_past = toTensor(read(self.flow[idx][1]).copy())[:-1]
        originalflowshape = flow_into_past.shape

        flow_into_past = F.interpolate(
            flow_into_past.unsqueeze(0),
            size=(self.resolution[1], self.resolution[0]),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        flow_into_future = F.interpolate(
            flow_into_future.unsqueeze(0),
            size=(self.resolution[1], self.resolution[0]),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        flow_into_future[0] *= flow_into_future.shape[1] / originalflowshape[1]
        flow_into_future[1] *= flow_into_future.shape[2] / originalflowshape[2]
        flow_into_past[0] *= flow_into_past.shape[1] / originalflowshape[1]
        flow_into_past[1] *= flow_into_past.shape[2] / originalflowshape[2]

        # read motion boundaries
        motion = Image.open(self.motion[idx]).resize(self.resolution, Image.BILINEAR)
        motion = toTensor(motion).squeeze(0)
        motion[motion != 0] = 1
        motion = 1 - motion

        # create mask
        mask = flow_warp_mask(flow_into_future, flow_into_past)
        mask = mask * motion

        return img1, img2, flow_into_past, mask


class FlyingThings3D_Monkaa(Dataset):
    def __init__(self, path: Union[str, list], resolution: tuple = (640, 360), frame_num: int = 1):
        """
        path -> Path to the location where the "monkaa" and "flyingthings3d" folders are kept.
                If path is a list, then the first element is the path to the "monkaa" folder and the second element is the path to the "flyingthings3d" folder.
        resolution -> Resolution of the images to be returned. Width first, then height.
        """
        super().__init__()

        if isinstance(path, str):
            self.monkaa = Monkaa(os.path.join(path, "monkaa"), resolution, frame_num)
            self.flyingthings3d = FlyingThings3D(os.path.join(path, "flyingthings3d"), resolution, frame_num)
        elif isinstance(path, list):
            self.monkaa = Monkaa(path[0], resolution, frame_num)
            self.flyingthings3d = FlyingThings3D(path[1], resolution, frame_num)
        else:  # pragma: no cover
            raise ValueError("Path must be a string or a list of strings.")

        self.length = len(self.monkaa) + len(self.flyingthings3d)

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> Union[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if idx < len(self.monkaa):
            return self.monkaa[idx]
        else:
            return self.flyingthings3d[idx - len(self.monkaa)]


if __name__ == "__main__":
    # get_frames(video_path="../datasets/Videvo")
    # calculate_optical_flow()
    # test_warp()
    # dataset = Videvo(path="./Videvo", frame_num=1)
    # img1, img2, flow, mask = dataset[0]
    # print(img1.shape, img2.shape, flow.shape, mask.shape)

    dataset = FlyingThings3D_Monkaa("../datasets/SceneFlowDatasets")
    img1, img2, flow, mask = dataset[12908]
    print(img1.shape, img2.shape, flow.shape, mask.shape)
    img1 = toPil(img1.byte())
    img2 = toPil(img2.byte())
    mask = toPil(mask)
    flow_rgb = visualize_flow(flow)
    # img1.save("img1.jpg")
    # img2.save("img2.jpg")
    # mask.save("mask.jpg")
    # cv2.imwrite("flow.jpg", flow_rgb)