import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def find_png_masks(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".png":
            raise ValueError(f"Expected a PNG mask, got: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    pattern = "**/*.png" if recursive else "*.png"
    masks = sorted(input_path.glob(pattern))
    if not masks:
        raise FileNotFoundError(f"No PNG masks found in: {input_path}")
    return masks


def mask_to_bbox(mask_path: Path, threshold: int = 0) -> dict:
    mask = Image.open(mask_path).convert("L")
    binary = mask.point(lambda value: 255 if value > threshold else 0)
    bbox = binary.getbbox()
    area = int(binary.histogram()[255])

    if bbox is None:
        return {
            "mask_path": str(mask_path),
            "bbox_xyxy": None,
            "bbox_xywh": None,
            "area_pixels": 0,
            "width": mask.width,
            "height": mask.height,
        }

    left, top, right, bottom = bbox

    return {
        "mask_path": str(mask_path),
        "bbox_xyxy": [left, top, right - 1, bottom - 1],
        "bbox_xywh": [left, top, right - left, bottom - top],
        "area_pixels": area,
        "width": mask.width,
        "height": mask.height,
    }


def build_vis_path(mask_path: Path, input_path: Path, vis_dir: Path) -> Path:
    if input_path.is_file():
        return vis_dir / mask_path.name

    return vis_dir / mask_path.relative_to(input_path)


def save_bbox_visualization(mask_path: Path, bbox_xyxy: list[int] | None, vis_path: Path) -> None:
    image = Image.open(mask_path).convert("RGB")
    if bbox_xyxy is not None:
        line_width = max(1, min(image.size) // 256)
        draw = ImageDraw.Draw(image)
        draw.rectangle(bbox_xyxy, outline=(255, 0, 0), width=line_width)

    vis_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(vis_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute tight bounding boxes from PNG masks. "
            "bbox_xyxy uses inclusive max coordinates."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="A PNG mask file or a directory containing PNG masks.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON path. Prints to stdout if omitted.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for PNG masks when input is a directory.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=0,
        help="Treat pixel values greater than this threshold as foreground.",
    )
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        help="Skip masks that contain no foreground pixels.",
    )
    parser.add_argument(
        "--vis-dir",
        type=Path,
        default=None,
        help="Optional directory to save PNG masks with the computed bbox drawn in red.",
    )
    args = parser.parse_args()

    masks = find_png_masks(args.input, args.recursive)
    entries = [(mask_path, mask_to_bbox(mask_path, args.threshold)) for mask_path in masks]

    if args.skip_empty:
        entries = [(mask_path, result) for mask_path, result in entries if result["bbox_xyxy"] is not None]

    if args.vis_dir is not None:
        for mask_path, result in entries:
            vis_path = build_vis_path(mask_path, args.input, args.vis_dir)
            save_bbox_visualization(mask_path, result["bbox_xyxy"], vis_path)

    results = [result for _, result in entries]

    output = json.dumps(results, indent=2)
    if args.output is None:
        print(output)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
