import os
import cv2
import random
import numpy as np

import torch
# import torchvision

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.MP4', '.MOV', '.AVI')

def read_frame_from_videos(frame_root):
    if frame_root.endswith(VIDEO_EXTENSIONS):  # Video file path
        video_name = os.path.splitext(os.path.basename(frame_root))[0]

        cap = cv2.VideoCapture(frame_root)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {frame_root}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            fps = 24

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # OpenCV reads BGR; MatAnyone2 expects RGB
            frame = frame[..., [2, 1, 0]]
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            raise RuntimeError(f"No frames decoded from video: {frame_root}")

        frames = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()

    else:
        video_name = os.path.basename(frame_root)
        frames = []

        fr_lst = sorted(os.listdir(frame_root))
        fr_lst = [f for f in fr_lst if f.endswith(IMAGE_EXTENSIONS)]

        for fr in fr_lst:
            frame_path = os.path.join(frame_root, fr)
            frame = cv2.imread(frame_path)

            if frame is None:
                raise RuntimeError(f"Could not read frame: {frame_path}")

            frame = frame[..., [2, 1, 0]]  # RGB, HWC
            frames.append(frame)

        if len(frames) == 0:
            raise RuntimeError(f"No image frames found in folder: {frame_root}")

        fps = 24  # default
        frames = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()

    length = frames.shape[0]
    return frames, fps, length, video_name
def get_video_paths(input_root):
    video_paths = []
    for root, _, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith(VIDEO_EXTENSIONS):
                video_paths.append(os.path.join(root, file))
    return sorted(video_paths)

def str_to_list(value):
    return list(map(int, value.split(',')))

def gen_dilate(alpha, min_kernel_size, max_kernel_size): 
    kernel_size = random.randint(min_kernel_size, max_kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
    fg_and_unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
    dilate = cv2.dilate(fg_and_unknown, kernel, iterations=1)*255
    return dilate.astype(np.float32)

def gen_erosion(alpha, min_kernel_size, max_kernel_size): 
    kernel_size = random.randint(min_kernel_size, max_kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
    fg = np.array(np.equal(alpha, 255).astype(np.float32))
    erode = cv2.erode(fg, kernel, iterations=1)*255
    return erode.astype(np.float32)