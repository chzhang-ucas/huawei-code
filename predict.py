"""Predict full-resolution sharpness maps for one image or an image directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model import SmallUNet


IMAGE_SUFFIXES = {".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="JPG file or image directory")
    parser.add_argument(
        "--mask", type=Path, required=True, help="Matching PNG file or mask directory"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def collect_pairs(input_path: Path, mask_path: Path) -> list[tuple[Path, Path]]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Input must be a JPG/JPEG file: {input_path}")
        expected_mask = mask_path / f"{input_path.stem}.png" if mask_path.is_dir() else mask_path
        if not expected_mask.is_file():
            raise FileNotFoundError(f"Mask not found: {expected_mask}")
        return [(input_path, expected_mask)]

    if not input_path.is_dir() or not mask_path.is_dir():
        raise ValueError("For batch prediction, --input and --mask must both be directories")
    images = sorted(p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise RuntimeError(f"No JPG/JPEG images found in {input_path}")
    pairs = []
    missing = []
    for image_path in images:
        current_mask = mask_path / f"{image_path.stem}.png"
        if current_mask.is_file():
            pairs.append((image_path, current_mask))
        else:
            missing.append(current_mask.name)
    if missing:
        raise FileNotFoundError("Missing masks: " + ", ".join(missing[:10]))
    return pairs


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> SmallUNet:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Invalid training checkpoint: {checkpoint_path}")
    checkpoint_args = checkpoint.get("args", {})
    base_channels = int(checkpoint_args.get("base_channels", 16))
    model = SmallUNet(base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def load_inputs(image_path: Path, mask_path: Path) -> tuple[torch.Tensor, np.ndarray]:
    with Image.open(image_path) as im:
        image_np = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    with Image.open(mask_path) as im:
        mask_np = np.asarray(im).copy()
    if mask_np.ndim != 2:
        raise ValueError(f"Mask must be a single-channel class-index PNG: {mask_path}")
    if mask_np.shape != image_np.shape[:2]:
        raise ValueError(
            f"Shape mismatch for {image_path.name}: image={image_np.shape[:2]}, "
            f"mask={mask_np.shape}"
        )
    image = torch.from_numpy(image_np.transpose(2, 0, 1)).unsqueeze(0).contiguous()
    return image, mask_np == 1


@torch.inference_mode()
def predict_one(
    model: SmallUNet, image_path: Path, mask_path: Path, device: torch.device
) -> np.ndarray:
    image, valid = load_inputs(image_path, mask_path)
    prediction = model(image.to(device))[0, 0].float().cpu().numpy()
    # Preserve the complete HxW map while defining non-class-1 pixels as zero.
    prediction[~valid] = 0.0
    return prediction.astype(np.float32, copy=False)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    pairs = collect_pairs(args.input, args.mask)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_checkpoint(args.checkpoint, device)

    for index, (image_path, mask_path) in enumerate(pairs, start=1):
        prediction = predict_one(model, image_path, mask_path, device)
        output_path = args.output_dir / f"{image_path.stem}.npy"
        np.save(output_path, prediction, allow_pickle=False)
        valid_count = int((prediction != 0).sum())
        print(f"[{index}/{len(pairs)}] saved {output_path} shape={prediction.shape} nonzero={valid_count}")


if __name__ == "__main__":
    main()
