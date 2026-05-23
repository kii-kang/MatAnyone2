import os
import cv2
import tqdm
import imageio
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from matanyone2.utils.download_util import load_file_from_url
from matanyone2.utils.inference_utils import gen_dilate, gen_erosion, get_input_metadata, iter_frames_from_videos

from matanyone2.inference.inference_core import InferenceCore
from matanyone2.utils.get_default_model import get_matanyone2_model
from matanyone2.utils.device import get_default_device, safe_autocast_decorator

import warnings
warnings.filterwarnings("ignore")

device = get_default_device()

@torch.inference_mode()
@safe_autocast_decorator()
def main(input_path, mask_path, output_path, ckpt_path, n_warmup=10, r_erode=10, r_dilate=10, suffix="", save_image=False, max_size=-1, start_frame=0):

    # download ckpt for the first inference
    pretrain_model_url = "https://github.com/pq-yang/MatAnyone2/releases/download/v1.0.0/matanyone2.pth"
    ckpt_path = load_file_from_url(pretrain_model_url, 'pretrained_models')
    
    # load MatAnyone model
    matanyone2 = get_matanyone2_model(ckpt_path, device)

    # init inference processor
    processor = InferenceCore(matanyone2, cfg=matanyone2.cfg)

    # inference parameters
    r_erode = int(r_erode)
    r_dilate = int(r_dilate)
    n_warmup = int(n_warmup)
    max_size = int(max_size)
    start_frame = int(start_frame)

    # load input metadata and stream frames lazily to avoid keeping the whole video in memory
    fps, length, video_name = get_input_metadata(input_path)
    if start_frame < 0 or start_frame >= length:
        raise ValueError(f"start_frame must be in [0, {length - 1}], got {start_frame}.")

    frame_iter = iter_frames_from_videos(input_path, start_frame=start_frame)
    try:
        reference_frame = next(frame_iter).float()
    except StopIteration as exc:
        raise RuntimeError(f"No frames available from start_frame {start_frame} for {input_path}.") from exc

    length -= start_frame

    # resize if needed
    new_h, new_w = reference_frame.shape[-2:]
    resize_needed = False
    if max_size > 0:
        h, w = new_h, new_w
        min_side = min(h, w)
        if min_side > max_size:
            new_h = int(h / min_side * max_size)
            new_w = int(w / min_side * max_size)
            reference_frame = F.interpolate(
                reference_frame.unsqueeze(0),
                size=(new_h, new_w),
                mode="area",
            )[0]
            resize_needed = True
            print(f'Resize to {new_h}x{new_w} for processing...')
        
    # set output paths
    os.makedirs(output_path, exist_ok=True)
    if suffix != "":
        video_name = f'{video_name}_{suffix}'
    if save_image:
        os.makedirs(f'{output_path}/{video_name}', exist_ok=True)
        os.makedirs(f'{output_path}/{video_name}/pha', exist_ok=True)
        os.makedirs(f'{output_path}/{video_name}/fgr', exist_ok=True)

    # load the reference-frame mask
    mask = Image.open(mask_path).convert('L')
    mask = np.array(mask)

    bgr = (np.array([120, 255, 155], dtype=np.float32)/255).reshape((1, 1, 3)) # green screen to paste fgr
    objects = [1]

    # [optional] erode & dilate
    if r_dilate > 0:
        mask = gen_dilate(mask, r_dilate, r_dilate)
    if r_erode > 0:
        mask = gen_erosion(mask, r_erode, r_erode)

    mask = torch.from_numpy(mask).float().to(device)

    if resize_needed:
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(new_h, new_w), mode="nearest")
        mask = mask[0,0]

    # inference start
    fgr_writer = imageio.get_writer(f'{output_path}/{video_name}_fgr.mp4', fps=fps, quality=7)
    pha_writer = imageio.get_writer(f'{output_path}/{video_name}_pha.mp4', fps=fps, quality=7)
    total_steps = length + n_warmup

    try:
        for ti in tqdm.tqdm(range(total_steps)):
            if ti <= n_warmup:
                image = reference_frame
            else:
                image = next(frame_iter)
                if resize_needed:
                    image = F.interpolate(
                        image.unsqueeze(0).float(),
                        size=(new_h, new_w),
                        mode="area",
                    )[0]
                else:
                    image = image.float()

            image_np = np.array(image.permute(1,2,0))       # for output visualize
            image = (image / 255.).float().to(device)       # for network input

            if ti == 0:
                output_prob = processor.step(image, mask, objects=objects)      # encode given mask
                output_prob = processor.step(image, first_frame_pred=True)      # first frame for prediction
            else:
                if ti <= n_warmup:
                    output_prob = processor.step(image, first_frame_pred=True)  # reinit as the first frame for prediction
                else:
                    output_prob = processor.step(image)

            # convert output probabilities to alpha matte
            mask = processor.output_prob_to_mask(output_prob)

            # visualize prediction
            pha = mask.unsqueeze(2).cpu().numpy()
            com_np = image_np / 255. * pha + bgr * (1 - pha)

            # DONOT save the warmup frame
            if ti > (n_warmup-1):
                com_np = np.round(np.clip(com_np * 255.0, 0, 255)).astype(np.uint8)
                pha = np.round(np.clip(pha * 255.0, 0, 255)).astype(np.uint8)
                pha_frame = pha[..., 0]

                fgr_writer.append_data(com_np)
                pha_writer.append_data(pha_frame)

                if save_image:
                    frame_idx = start_frame + ti - n_warmup
                    cv2.imwrite(f'{output_path}/{video_name}/fgr/{str(frame_idx).zfill(4)}.png', com_np[...,[2,1,0]])
                    cv2.imwrite(f'{output_path}/{video_name}/pha/{str(frame_idx).zfill(4)}.png', pha_frame)
    finally:
        fgr_writer.close()
        pha_writer.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str, help='Path of the input video or frame folder.')
    parser.add_argument('-m', '--mask_path', type=str, help='Path of the segmentation mask for start_frame.')
    parser.add_argument('-o', '--output_path', type=str, default="results/", help='Output folder. Default: results')
    parser.add_argument('-c', '--ckpt_path', type=str, default="pretrained_models/matanyone2.pth", help='Path of the MatAnyone2 model.')
    parser.add_argument('-w', '--warmup', type=str, default="10", help='Number of warmup iterations for the first frame alpha prediction.')
    parser.add_argument('-e', '--erode_kernel', type=str, default="10", help='Erosion kernel on the input mask.')
    parser.add_argument('-d', '--dilate_kernel', type=str, default="10", help='Dilation kernel on the input mask.')
    parser.add_argument('--suffix', type=str, default="", help='Suffix to specify different target when saving, e.g., target1.')
    parser.add_argument('--save_image', action='store_true', default=False, help='Save output frames. Default: False')
    parser.add_argument('--max_size', type=str, default="1296", help='When positive, the video will be downsampled if min(w, h) exceeds. Default: -1 (means no limit)')
    parser.add_argument('--start_frame', type=str, default="0", help='Frame index whose mask is provided. Frames before this index are skipped.')

    
    args = parser.parse_args()

    main(input_path=args.input_path, \
         mask_path=args.mask_path, \
         output_path=args.output_path, \
         ckpt_path=args.ckpt_path, \
         n_warmup=args.warmup, \
         r_erode=args.erode_kernel, \
         r_dilate=args.dilate_kernel, \
         suffix=args.suffix, \
         save_image=args.save_image, \
         max_size=args.max_size, \
         start_frame=args.start_frame)
