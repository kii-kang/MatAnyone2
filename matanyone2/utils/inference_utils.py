import os
import cv2
import random
import numpy as np

import torch
# import torchvision

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.MP4', '.MOV', '.AVI')


def _get_frame_file_list(frame_root):
    fr_lst = sorted(os.listdir(frame_root))
    return [f for f in fr_lst if f.endswith(IMAGE_EXTENSIONS)]


def get_input_metadata(frame_root):
    if frame_root.endswith(VIDEO_EXTENSIONS):
        video_name = os.path.splitext(os.path.basename(frame_root))[0]

        cap = cv2.VideoCapture(frame_root)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {frame_root}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            fps = 24

        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if length <= 0:
            length = 0
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                length += 1

        cap.release()

        if length == 0:
            raise RuntimeError(f"No frames decoded from video: {frame_root}")

        return fps, length, video_name

    video_name = os.path.basename(frame_root)
    fr_lst = _get_frame_file_list(frame_root)
    if len(fr_lst) == 0:
        raise RuntimeError(f"No image frames found in folder: {frame_root}")

    fps = 24
    length = len(fr_lst)
    return fps, length, video_name


def iter_frames_from_videos(frame_root, start_frame=0):
    start_frame = int(start_frame)
    if start_frame < 0:
        raise ValueError(f"start_frame must be non-negative, got {start_frame}.")

    if frame_root.endswith(VIDEO_EXTENSIONS):  # Video file path
        cap = cv2.VideoCapture(frame_root)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {frame_root}")

        skipped = 0
        while skipped < start_frame:
            ret, _ = cap.read()
            if not ret:
                cap.release()
                raise ValueError(f"start_frame {start_frame} exceeds decoded video length for {frame_root}.")
            skipped += 1

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # OpenCV reads BGR; MatAnyone2 expects RGB
            frame = frame[..., [2, 1, 0]]
            yield torch.from_numpy(frame).permute(2, 0, 1).contiguous()

        cap.release()

    else:
        fr_lst = _get_frame_file_list(frame_root)
        if len(fr_lst) == 0:
            raise RuntimeError(f"No image frames found in folder: {frame_root}")
        if start_frame >= len(fr_lst):
            raise ValueError(f"start_frame must be in [0, {len(fr_lst) - 1}], got {start_frame}.")

        for fr in fr_lst[start_frame:]:
            frame_path = os.path.join(frame_root, fr)
            frame = cv2.imread(frame_path)

            if frame is None:
                raise RuntimeError(f"Could not read frame: {frame_path}")

            frame = frame[..., [2, 1, 0]]  # RGB, HWC
            yield torch.from_numpy(frame).permute(2, 0, 1).contiguous()


def read_frame_from_videos(frame_root):
    fps, length, video_name = get_input_metadata(frame_root)
    frames = list(iter_frames_from_videos(frame_root))
    frames = torch.stack(frames, dim=0)
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
