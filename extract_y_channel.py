"""Extract and save Y channels for training and validation images."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert train/val JPG images to single-channel Y PNG images."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Dataset root containing train/images and val/images",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Optional output root. Default saves to "
            "DATA_ROOT/train/y_images and DATA_ROOT/val/y_images"
        ),
    )
    return parser.parse_args()


def find_images(image_dir: Path) -> List[Path]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    images = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise RuntimeError(f"No JPG/JPEG images found in {image_dir}")
    return images


def output_dir_for_split(data_root: Path, output_root: Path, split: str) -> Path:
    if output_root is None:
        return data_root / split / "y_images"
    return output_root / split / "y_images"


def convert_split(data_root: Path, output_root: Path, split: str) -> int:
    image_dir = data_root / split / "images"
    output_dir = output_dir_for_split(data_root, output_root, split)
    images = find_images(image_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, image_path in enumerate(images, start=1):
        with Image.open(image_path) as image:
            # Pillow YCbCr channel 0: Y = 0.299R + 0.587G + 0.114B (8-bit).
            y_channel = image.convert("RGB").convert("YCbCr").getchannel("Y")
            output_path = output_dir / (image_path.stem + ".png")
            y_channel.save(output_path, format="PNG")
            width, height = y_channel.size
        print(
            "[{}] [{}/{}] {} -> {} size={}x{}".format(
                split, index, len(images), image_path.name, output_path.name, width, height
            ),
            flush=True,
        )
    print("[{}] saved {} Y images to {}".format(split, len(images), output_dir))
    return len(images)


def main() -> None:
    args = parse_args()
    train_count = convert_split(args.data_root, args.output_root, "train")
    val_count = convert_split(args.data_root, args.output_root, "val")
    print("Done: train={}, val={}".format(train_count, val_count))


if __name__ == "__main__":
    main()
