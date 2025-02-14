import os
import cv2

from utilities import list_files


IMG_SIZE = (640, 360)


def get_frames(video_path):
    files = list_files(video_path)

    videos_idx = 0
    for file in files:
        # create directory if it doesn't exist
        save_dir = f"./Videvo-jpg/{videos_idx}/"
        os.makedirs(save_dir, exist_ok=True)

        # delete old images
        files = list_files(save_dir)
        for f in files:
            os.remove(f)

        # read video and save frames
        cap = cv2.VideoCapture(file)
        frames_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, IMG_SIZE, interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(save_dir, f"{frames_idx}.jpg"), frame)
            frames_idx += 1

        cap.release()

        videos_idx += 1


if __name__ == "__main__":
    video_path = "../datasets/Videvo"
    # get_frames(video_path)
