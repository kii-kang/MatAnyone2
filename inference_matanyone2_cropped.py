import argparse
import json
import math
import os
from pathlib import Path
import warnings

import cv2
import imageio
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F

from matanyone2.utils.download_util import load_file_from_url
from matanyone2.utils.inference_utils import (
    gen_dilate,
    gen_erosion,
    get_input_metadata,
    iter_frames_from_videos,
)
from matanyone2.inference.inference_core import InferenceCore
from matanyone2.utils.get_default_model import get_matanyone2_model
from matanyone2.utils.device import get_default_device, safe_autocast_decorator


device = get_default_device()
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi")


def load_crop_plan(plan_path):
    entries = []
    with open(plan_path, "r") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    if not entries:
        raise RuntimeError(f"No crop entries found in {plan_path}")

    entries = sorted(entries, key=lambda e: e["frame"])
    return {int(e["frame"]): e for e in entries}


def is_video_input_path(input_path):
    return Path(input_path).suffix.lower() in VIDEO_EXTENSIONS


def list_frame_paths(frame_root):
    frame_root = Path(frame_root)
    frame_paths = [
        path for path in sorted(frame_root.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not frame_paths:
        raise RuntimeError(f"No image frames found in folder: {frame_root}")

    return frame_paths


def get_frame_source_info(input_path, frame_idx, frame_paths=None):
    input_path = Path(input_path)
    info = {
        "input_type": "video" if is_video_input_path(input_path) else "frame_folder",
        "input_path": str(input_path),
        "input_frame_index": int(frame_idx),
        "input_frame_path": None,
        "input_frame_name": None,
        "input_video_path": None,
    }

    if info["input_type"] == "video":
        info["input_video_path"] = str(input_path)
        return info

    if frame_paths is None:
        frame_paths = list_frame_paths(input_path)

    try:
        frame_path = frame_paths[frame_idx]
    except IndexError as exc:
        raise RuntimeError(f"Frame index {frame_idx} is out of range for {input_path}") from exc

    info["input_frame_path"] = str(frame_path)
    info["input_frame_name"] = frame_path.name
    return info


def load_bbox_entries(bbox_json_path):
    with open(bbox_json_path, "r") as f:
        entries = json.load(f)

    if not isinstance(entries, list) or not entries:
        raise RuntimeError(f"No bbox entries found in {bbox_json_path}")

    plan_entries = {}
    for entry in entries:
        mask_path = entry.get("mask_path")
        bbox_xyxy = entry.get("bbox_xyxy")

        if not mask_path:
            raise RuntimeError(f"bbox entry is missing mask_path in {bbox_json_path}: {entry}")

        if bbox_xyxy is None:
            continue

        frame_idx = parse_frame_index(mask_path)
        if frame_idx in plan_entries:
            warnings.warn(f"Duplicate bbox entry for frame {frame_idx}; keeping the last one.")

        plan_entries[frame_idx] = entry

    if not plan_entries:
        raise RuntimeError(f"No non-empty bbox entries found in {bbox_json_path}")

    return dict(sorted(plan_entries.items()))


def parse_frame_index(mask_path):
    stem = Path(mask_path).stem
    try:
        return int(stem)
    except ValueError as exc:
        raise RuntimeError(f"Could not infer frame index from mask path: {mask_path}") from exc


def clamp_crop_box(box, full_w, full_h):
    if full_w <= 0 or full_h <= 0:
        raise RuntimeError(f"Invalid full-frame size: {full_w}x{full_h}")

    x0, y0, x1, y1 = [int(v) for v in box]

    x0 = min(max(x0, 0), full_w - 1)
    y0 = min(max(y0, 0), full_h - 1)
    x1 = min(max(x1, x0 + 1), full_w)
    y1 = min(max(y1, y0 + 1), full_h)

    return [x0, y0, x1, y1]


def expand_crop_box(box, full_w, full_h, expand_ratio=0.0, expand_pixels=0):
    x0, y0, x1, y1 = [int(v) for v in box]
    pad_x = int(round((x1 - x0) * float(expand_ratio))) + int(expand_pixels)
    pad_y = int(round((y1 - y0) * float(expand_ratio))) + int(expand_pixels)

    return clamp_crop_box(
        [x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y],
        full_w=full_w,
        full_h=full_h,
    )


def bbox_xyxy_to_crop_box(bbox_xyxy, src_w, src_h, full_w, full_h):
    if src_w <= 0 or src_h <= 0:
        raise RuntimeError(f"Invalid bbox source size: {src_w}x{src_h}")

    x0, y0, x1, y1 = [int(v) for v in bbox_xyxy]

    crop_x0 = int(math.floor(x0 * full_w / src_w))
    crop_y0 = int(math.floor(y0 * full_h / src_h))
    crop_x1 = int(math.ceil((x1 + 1) * full_w / src_w))
    crop_y1 = int(math.ceil((y1 + 1) * full_h / src_h))

    return clamp_crop_box([crop_x0, crop_y0, crop_x1, crop_y1], full_w, full_h)


def build_crop_plan_from_bbox_entries(
    bbox_entries,
    full_w,
    full_h,
    expand_ratio=0.0,
    expand_pixels=0,
):
    plan = {}

    for frame_idx, entry in bbox_entries.items():
        bbox_xyxy = entry.get("bbox_xyxy")
        if bbox_xyxy is None:
            continue

        mask_w = int(entry["width"])
        mask_h = int(entry["height"])

        crop_box = bbox_xyxy_to_crop_box(
            bbox_xyxy,
            src_w=mask_w,
            src_h=mask_h,
            full_w=full_w,
            full_h=full_h,
        )

        crop_box = expand_crop_box(
            crop_box,
            full_w=full_w,
            full_h=full_h,
            expand_ratio=expand_ratio,
            expand_pixels=expand_pixels,
        )

        plan[frame_idx] = {
            "frame": frame_idx,
            "crop_bbox_original": crop_box,
            "mask_path": entry["mask_path"],
            "bbox_xyxy_mask_space": bbox_xyxy,
            "bbox_source_size": [mask_w, mask_h],
        }

    if not plan:
        raise RuntimeError("No usable crop boxes could be built from bbox JSON.")

    return dict(sorted(plan.items()))


def get_mask_path(mask_dir, frame_idx, digits=4):
    mask_dir = Path(mask_dir)

    candidates = [
        mask_dir / f"{frame_idx:0{digits}d}.png",
        mask_dir / f"{frame_idx}.png",
    ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(f"No mask found for frame {frame_idx} in {mask_dir}")


def read_binary_mask(mask_path, full_w, full_h, threshold=8):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Could not read mask: {mask_path}")

    # Resize low-res mask back to original full-frame coordinates.
    if mask.shape[1] != full_w or mask.shape[0] != full_h:
        mask = cv2.resize(mask, (full_w, full_h), interpolation=cv2.INTER_NEAREST)

    mask = (mask > threshold).astype(np.uint8) * 255
    return mask


def resolve_first_mask_path(mask_dir, plan_entry, frame_idx, digits=4):
    if mask_dir:
        return get_mask_path(mask_dir, frame_idx, digits=digits)

    mask_path = plan_entry.get("mask_path")
    if not mask_path:
        raise RuntimeError("mask_dir is required unless bbox JSON provides a mask_path for the start frame.")

    mask_path = Path(mask_path)
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask path from bbox JSON does not exist: {mask_path}")

    return mask_path


def image_chw_to_rgb_u8(image_chw):
    return np.round(
        np.clip(image_chw.permute(1, 2, 0).cpu().numpy(), 0, 255)
    ).astype(np.uint8)


def build_intrinsics_matrix(fx, fy, cx, cy):
    return [
        [float(fx), 0.0, float(cx)],
        [0.0, float(fy), float(cy)],
        [0.0, 0.0, 1.0],
    ]


def compute_crop_intrinsics(fx, fy, cx, cy, meta):
    x0, y0, _, _ = meta["box"]
    pad_x, pad_y = meta["pad"]
    scale = float(meta["scale"])

    crop_fx = scale * float(fx)
    crop_fy = scale * float(fy)
    crop_cx = scale * (float(cx) - float(x0)) + float(pad_x)
    crop_cy = scale * (float(cy) - float(y0)) + float(pad_y)

    return build_intrinsics_matrix(crop_fx, crop_fy, crop_cx, crop_cy)


def interpolate_image_chw(img_chw, size_hw):
    mode = "area" if size_hw[0] < img_chw.shape[-2] or size_hw[1] < img_chw.shape[-1] else "bilinear"

    if mode == "bilinear":
        return F.interpolate(
            img_chw.unsqueeze(0),
            size=size_hw,
            mode=mode,
            align_corners=False,
        )[0]

    return F.interpolate(
        img_chw.unsqueeze(0),
        size=size_hw,
        mode=mode,
    )[0]


def crop_and_letterbox_image_chw(img_chw, box, target_h, target_w):
    """
    img_chw: torch tensor C,H,W, values 0..255
    box: [x0, y0, x1, y1] in original full-frame coordinates
    """
    x0, y0, x1, y1 = [int(v) for v in box]

    crop = img_chw[:, y0:y1, x0:x1].float()
    crop_h, crop_w = crop.shape[-2:]

    if crop_h <= 0 or crop_w <= 0:
        raise RuntimeError(f"Invalid crop box: {box}")

    scale = min(target_w / crop_w, target_h / crop_h)

    new_w = max(1, int(round(crop_w * scale)))
    new_h = max(1, int(round(crop_h * scale)))

    resized = interpolate_image_chw(crop, (new_h, new_w))

    # Fill padding with average crop color, not black, because hard black borders are dumb noise.
    fill = crop.reshape(crop.shape[0], -1).mean(dim=1).view(-1, 1, 1)
    canvas = fill.expand(crop.shape[0], target_h, target_w).clone()

    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2

    canvas[:, pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    meta = {
        "box": [x0, y0, x1, y1],
        "crop_size": [crop_w, crop_h],
        "target_size": [target_w, target_h],
        "scale": float(scale),
        "resized_size": [new_w, new_h],
        "pad": [pad_x, pad_y],
    }

    return canvas, meta


def crop_and_letterbox_mask_np(mask_full, box, target_h, target_w):
    x0, y0, x1, y1 = [int(v) for v in box]

    crop = mask_full[y0:y1, x0:x1]
    crop_h, crop_w = crop.shape[:2]

    if crop_h <= 0 or crop_w <= 0:
        raise RuntimeError(f"Invalid mask crop box: {box}")

    scale = min(target_w / crop_w, target_h / crop_h)

    new_w = max(1, int(round(crop_w * scale)))
    new_h = max(1, int(round(crop_h * scale)))

    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    canvas = np.zeros((target_h, target_w), dtype=np.uint8)

    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2

    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    return canvas


def unletterbox_mask_to_full(mask_target, meta, full_h, full_w):
    """
    mask_target: uint8 H,W in target crop coordinates
    returns: uint8 full-frame alpha mask
    """
    x0, y0, x1, y1 = meta["box"]
    crop_w, crop_h = meta["crop_size"]
    new_w, new_h = meta["resized_size"]
    pad_x, pad_y = meta["pad"]

    roi = mask_target[pad_y:pad_y + new_h, pad_x:pad_x + new_w]

    crop_mask = cv2.resize(
        roi,
        (crop_w, crop_h),
        interpolation=cv2.INTER_LINEAR,
    )

    full = np.zeros((full_h, full_w), dtype=np.uint8)
    full[y0:y1, x0:x1] = crop_mask

    return full


def mask_to_composite(image_chw, mask_torch):
    """
    image_chw: crop-space tensor C,H,W, values 0..255, RGB
    mask_torch: H,W, values 0/1-ish
    """
    image_np = image_chw.permute(1, 2, 0).cpu().numpy().astype(np.float32)

    pha = mask_torch.unsqueeze(2).cpu().numpy().astype(np.float32)

    green = (np.array([120, 255, 155], dtype=np.float32) / 255.0).reshape((1, 1, 3))

    com_np = image_np / 255.0 * pha + green * (1.0 - pha)

    com_np = np.round(np.clip(com_np * 255.0, 0, 255)).astype(np.uint8)
    pha_u8 = np.round(np.clip(pha[..., 0] * 255.0, 0, 255)).astype(np.uint8)

    return com_np, pha_u8


@torch.inference_mode()
@safe_autocast_decorator()
def main(
    input_path,
    output_path,
    mask_dir=None,
    crop_plan=None,
    bbox_json=None,
    ckpt_path="pretrained_models/matanyone2.pth",
    start_frame=None,
    end_frame=None,
    target_size=1024,
    target_width=None,
    target_height=None,
    n_warmup=10,
    r_erode=10,
    r_dilate=10,
    mask_threshold=8,
    save_crop_images=False,
    save_full_pha=True,
    save_full_video=False,
    digits=4,
    bbox_expand_ratio=0.10,
    bbox_expand_pixels=0,
    save_crop_rgb=True,
    fx=None,
    fy=None,
    cx=None,
    cy=None,
):
    os.makedirs(output_path, exist_ok=True)

    target_h = int(target_height or target_size)
    target_w = int(target_width or target_size)

    if bool(crop_plan) == bool(bbox_json):
        raise ValueError("Specify exactly one of --crop_plan or --bbox_json.")

    if bbox_expand_ratio < 0:
        raise ValueError("bbox_expand_ratio must be non-negative")
    if bbox_expand_pixels < 0:
        raise ValueError("bbox_expand_pixels must be non-negative")

    intrinsics = [fx, fy, cx, cy]
    has_intrinsics = any(value is not None for value in intrinsics)
    if has_intrinsics and not all(value is not None for value in intrinsics):
        raise ValueError("Provide all of fx, fy, cx, cy together, or none of them.")

    if crop_plan:
        plan_source = load_crop_plan(crop_plan)
        plan_source_type = "crop_plan"
    else:
        plan_source = load_bbox_entries(bbox_json)
        plan_source_type = "bbox_json"

    plan = None
    available_frames = sorted(plan_source.keys())

    if start_frame is None:
        start_frame = available_frames[0]
    if end_frame is None:
        end_frame = available_frames[-1]

    start_frame = int(start_frame)
    end_frame = int(end_frame)

    if start_frame > end_frame:
        raise ValueError("start_frame must be <= end_frame")

    fps, video_length, video_name = get_input_metadata(input_path)

    if start_frame < 0 or start_frame >= video_length:
        raise ValueError(f"start_frame must be inside video range [0, {video_length - 1}]")

    end_frame = min(end_frame, video_length - 1)

    crop_pha_dir = Path(output_path) / video_name / "pha_crop"
    crop_fgr_dir = Path(output_path) / video_name / "fgr_crop"
    crop_rgb_dir = Path(output_path) / video_name / "rgb_crop"
    full_pha_dir = Path(output_path) / video_name / "pha_full"
    meta_dir = Path(output_path) / video_name

    crop_pha_dir.mkdir(parents=True, exist_ok=True)
    crop_fgr_dir.mkdir(parents=True, exist_ok=True)
    if save_crop_rgb:
        crop_rgb_dir.mkdir(parents=True, exist_ok=True)

    if save_full_pha:
        full_pha_dir.mkdir(parents=True, exist_ok=True)

    # Download checkpoint only if the requested path is unavailable locally.
    if not ckpt_path or not Path(ckpt_path).exists():
        pretrain_model_url = "https://github.com/pq-yang/MatAnyone2/releases/download/v1.0.0/matanyone2.pth"
        ckpt_path = load_file_from_url(pretrain_model_url, "pretrained_models")

    matanyone2 = get_matanyone2_model(ckpt_path, device)
    processor = InferenceCore(matanyone2, cfg=matanyone2.cfg)

    r_erode = int(r_erode)
    r_dilate = int(r_dilate)
    n_warmup = int(n_warmup)

    frame_iter = iter_frames_from_videos(input_path, start_frame=start_frame)

    try:
        reference_frame = next(frame_iter).float()
    except StopIteration as exc:
        raise RuntimeError(f"No frames available from start_frame {start_frame}") from exc

    full_h, full_w = reference_frame.shape[-2:]

    if plan_source_type == "crop_plan":
        plan = plan_source
    else:
        plan = build_crop_plan_from_bbox_entries(
            plan_source,
            full_w=full_w,
            full_h=full_h,
            expand_ratio=bbox_expand_ratio,
            expand_pixels=bbox_expand_pixels,
        )

    frame_paths = None
    if not is_video_input_path(input_path):
        frame_paths = list_frame_paths(input_path)

    if start_frame not in plan:
        raise RuntimeError(f"Start frame {start_frame} is not present in crop plan")

    first_box = plan[start_frame]["crop_bbox_original"]

    first_image_crop, first_meta = crop_and_letterbox_image_chw(
        reference_frame,
        first_box,
        target_h,
        target_w,
    )

    first_mask_path = resolve_first_mask_path(
        mask_dir,
        plan[start_frame],
        start_frame,
        digits=digits,
    )
    first_mask_full = read_binary_mask(
        first_mask_path,
        full_w=full_w,
        full_h=full_h,
        threshold=mask_threshold,
    )

    first_mask_crop = crop_and_letterbox_mask_np(
        first_mask_full,
        first_box,
        target_h,
        target_w,
    )

    if r_dilate > 0:
        first_mask_crop = gen_dilate(first_mask_crop, r_dilate, r_dilate)
    if r_erode > 0:
        first_mask_crop = gen_erosion(first_mask_crop, r_erode, r_erode)

    first_mask_torch = torch.from_numpy(first_mask_crop).float().to(device)

    objects = [1]

    image_torch = (first_image_crop / 255.0).float().to(device)

    # Encode initial mask.
    output_prob = processor.step(image_torch, first_mask_torch, objects=objects)

    # Warm up / stabilize first-frame prediction.
    for _ in tqdm(range(n_warmup), desc="Warmup"):
        output_prob = processor.step(image_torch, first_frame_pred=True)

    crop_fgr_writer = imageio.get_writer(
        str(Path(output_path) / f"{video_name}_crop_fgr.mp4"),
        fps=fps,
        quality=7,
    )
    crop_pha_writer = imageio.get_writer(
        str(Path(output_path) / f"{video_name}_crop_pha.mp4"),
        fps=fps,
        quality=7,
    )

    full_pha_writer = None
    if save_full_video:
        full_pha_writer = imageio.get_writer(
            str(Path(output_path) / f"{video_name}_full_pha.mp4"),
            fps=fps,
            quality=7,
        )

    meta_jsonl_path = meta_dir / "crop_inference_meta.jsonl"

    last_plan_entry = plan[start_frame]

    try:
        with open(meta_jsonl_path, "w") as meta_f:
            for frame_idx in tqdm(range(start_frame, end_frame + 1), desc="Cropped MatAnyone2 inference"):

                if frame_idx == start_frame:
                    entry = plan[start_frame]
                    frame = reference_frame
                    image_crop = first_image_crop
                    meta = first_meta
                else:
                    try:
                        frame = next(frame_iter).float()
                    except StopIteration:
                        print(f"Stopped early at frame {frame_idx}")
                        break

                    entry = plan.get(frame_idx)
                    if entry is None:
                        # Missing mask/crop plan. Carry previous crop.
                        entry = last_plan_entry
                    else:
                        last_plan_entry = entry

                    box = entry["crop_bbox_original"]

                    image_crop, meta = crop_and_letterbox_image_chw(
                        frame,
                        box,
                        target_h,
                        target_w,
                    )

                    image_torch = (image_crop / 255.0).float().to(device)
                    output_prob = processor.step(image_torch)

                pred_mask = processor.output_prob_to_mask(output_prob)

                com_crop, pha_crop = mask_to_composite(image_crop, pred_mask)

                crop_fgr_writer.append_data(com_crop)
                crop_pha_writer.append_data(pha_crop)

                frame_name = f"{frame_idx:0{digits}d}.png"

                cv2.imwrite(
                    str(crop_pha_dir / frame_name),
                    pha_crop,
                )

                crop_rgb_path = None
                if save_crop_rgb:
                    crop_rgb_path = crop_rgb_dir / frame_name
                    crop_rgb = image_chw_to_rgb_u8(image_crop)
                    cv2.imwrite(
                        str(crop_rgb_path),
                        crop_rgb[..., [2, 1, 0]],
                    )

                if save_crop_images:
                    cv2.imwrite(
                        str(crop_fgr_dir / frame_name),
                        com_crop[..., [2, 1, 0]],
                    )

                full_pha_path = None

                if save_full_pha or save_full_video:
                    full_pha = unletterbox_mask_to_full(
                        pha_crop,
                        meta,
                        full_h=full_h,
                        full_w=full_w,
                    )

                    if save_full_pha:
                        full_pha_path = full_pha_dir / frame_name
                        cv2.imwrite(str(full_pha_path), full_pha)

                    if full_pha_writer is not None:
                        full_pha_writer.append_data(full_pha)

                frame_source_info = get_frame_source_info(
                    input_path,
                    frame_idx,
                    frame_paths=frame_paths,
                )

                meta_record = {
                    "frame": frame_idx,
                    "crop_box_frame_source": entry["frame"],
                    "box_original": meta["box"],
                    "crop_size": meta["crop_size"],
                    "target_size": meta["target_size"],
                    "scale": meta["scale"],
                    "resized_size": meta["resized_size"],
                    "pad": meta["pad"],
                    "full_frame_size": [full_w, full_h],
                    "input_type": frame_source_info["input_type"],
                    "input_path": frame_source_info["input_path"],
                    "input_frame_index": frame_source_info["input_frame_index"],
                    "input_frame_path": frame_source_info["input_frame_path"],
                    "input_frame_name": frame_source_info["input_frame_name"],
                    "input_video_path": frame_source_info["input_video_path"],
                    "pha_crop_path": str(crop_pha_dir / frame_name),
                    "crop_rgb_path": str(crop_rgb_path) if crop_rgb_path else None,
                    "pha_full_path": str(full_pha_path) if full_pha_path else None,
                    "crop_source": plan_source_type,
                    "bbox_expand_ratio": float(bbox_expand_ratio) if plan_source_type == "bbox_json" else None,
                    "bbox_expand_pixels": int(bbox_expand_pixels) if plan_source_type == "bbox_json" else None,
                }

                if plan_source_type == "bbox_json":
                    meta_record["bbox_xyxy_mask_space"] = entry["bbox_xyxy_mask_space"]
                    meta_record["bbox_source_size"] = entry["bbox_source_size"]
                    meta_record["mask_path"] = entry["mask_path"]

                if has_intrinsics:
                    meta_record["K_full"] = build_intrinsics_matrix(fx, fy, cx, cy)
                    meta_record["K_crop"] = compute_crop_intrinsics(fx, fy, cx, cy, meta)
                else:
                    meta_record["K_full"] = None
                    meta_record["K_crop"] = None

                meta_f.write(json.dumps(meta_record) + "\n")

    finally:
        crop_fgr_writer.close()
        crop_pha_writer.close()

        if full_pha_writer is not None:
            full_pha_writer.close()

    print(f"Done.")
    print(f"Crop alpha frames: {crop_pha_dir}")
    if save_full_pha:
        print(f"Full-frame alpha frames: {full_pha_dir}")
    print(f"Metadata: {meta_jsonl_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input_path", required=True, help="Original full-resolution video or frame folder.")
    parser.add_argument(
        "--mask_dir",
        default=None,
        help="Directory of low-res/full-res masks named 0000.png etc. Optional when --bbox_json is used.",
    )
    parser.add_argument(
        "--crop_plan",
        default=None,
        help="Legacy JSONL crop plan with per-frame crop_bbox_original entries.",
    )
    parser.add_argument(
        "--bbox_json",
        default=None,
        help="JSON output from bbox_from_mask.py computed from low-res alpha masks.",
    )
    parser.add_argument("-o", "--output_path", default="results_cropped/", help="Output folder.")
    parser.add_argument("-c", "--ckpt_path", default="pretrained_models/matanyone2.pth")

    parser.add_argument("--start_frame", type=int, default=None)
    parser.add_argument("--end_frame", type=int, default=None)

    parser.add_argument("--target_size", type=int, default=1024, help="Square crop inference size.")
    parser.add_argument("--target_width", type=int, default=None)
    parser.add_argument("--target_height", type=int, default=None)

    parser.add_argument("-w", "--warmup", type=int, default=10)
    parser.add_argument("-e", "--erode_kernel", type=int, default=10)
    parser.add_argument("-d", "--dilate_kernel", type=int, default=10)

    parser.add_argument("--mask_threshold", type=int, default=8)
    parser.add_argument("--digits", type=int, default=4)
    parser.add_argument(
        "--bbox_expand_ratio",
        type=float,
        default=0.10,
        help="Extra relative padding added around each bbox when using --bbox_json.",
    )
    parser.add_argument(
        "--bbox_expand_pixels",
        type=int,
        default=0,
        help="Extra absolute padding in full-resolution pixels when using --bbox_json.",
    )
    parser.add_argument(
        "--no_save_crop_rgb",
        action="store_true",
        help="Disable saving raw RGB crop images under rgb_crop.",
    )
    parser.add_argument("--fx", type=float, default=None, help="Full-frame camera intrinsics fx.")
    parser.add_argument("--fy", type=float, default=None, help="Full-frame camera intrinsics fy.")
    parser.add_argument("--cx", type=float, default=None, help="Full-frame camera intrinsics cx.")
    parser.add_argument("--cy", type=float, default=None, help="Full-frame camera intrinsics cy.")

    parser.add_argument("--save_crop_images", action="store_true")
    parser.add_argument("--no_full_pha", action="store_true")
    parser.add_argument("--save_full_video", action="store_true")

    args = parser.parse_args()

    if bool(args.crop_plan) == bool(args.bbox_json):
        parser.error("Specify exactly one of --crop_plan or --bbox_json.")

    if args.crop_plan and not args.mask_dir:
        parser.error("--mask_dir is required when using --crop_plan.")

    main(
        input_path=args.input_path,
        output_path=args.output_path,
        mask_dir=args.mask_dir,
        crop_plan=args.crop_plan,
        bbox_json=args.bbox_json,
        ckpt_path=args.ckpt_path,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        target_size=args.target_size,
        target_width=args.target_width,
        target_height=args.target_height,
        n_warmup=args.warmup,
        r_erode=args.erode_kernel,
        r_dilate=args.dilate_kernel,
        mask_threshold=args.mask_threshold,
        save_crop_images=args.save_crop_images,
        save_full_pha=not args.no_full_pha,
        save_full_video=args.save_full_video,
        digits=args.digits,
        bbox_expand_ratio=args.bbox_expand_ratio,
        bbox_expand_pixels=args.bbox_expand_pixels,
        save_crop_rgb=not args.no_save_crop_rgb,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
    )
