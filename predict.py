"""Predict full-resolution sharpness maps for one image or an image directory."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

from model import SmallUNet


Y_SUFFIXES = {".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--y-input", type=Path, required=True, help="Y PNG file or directory")
    parser.add_argument(
        "--variance", type=Path, required=True, help="Variance NPY file or directory"
    )
    parser.add_argument(
        "--mask", type=Path, required=True, help="Matching PNG file or mask directory"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def collect_inputs(
    y_input: Path, variance_path: Path, mask_path: Path
) -> List[Tuple[Path, Path, Path]]:
    if y_input.is_file():
        if y_input.suffix.lower() not in Y_SUFFIXES:
            raise ValueError(f"Y input must be a PNG file: {y_input}")
        expected_variance = (
            variance_path / f"{y_input.stem}.npy" if variance_path.is_dir() else variance_path
        )
        expected_mask = mask_path / f"{y_input.stem}.png" if mask_path.is_dir() else mask_path
        if not expected_variance.is_file():
            raise FileNotFoundError(f"Variance not found: {expected_variance}")
        if not expected_mask.is_file():
            raise FileNotFoundError(f"Mask not found: {expected_mask}")
        return [(y_input, expected_variance, expected_mask)]

    if not y_input.is_dir() or not variance_path.is_dir() or not mask_path.is_dir():
        raise ValueError(
            "For batch prediction, --y-input, --variance, and --mask must be directories"
        )
    y_images = sorted(p for p in y_input.iterdir() if p.suffix.lower() in Y_SUFFIXES)
    if not y_images:
        raise RuntimeError(f"No Y PNG images found in {y_input}")
    inputs = []
    missing = []
    for y_path in y_images:
        current_variance = variance_path / f"{y_path.stem}.npy"
        current_mask = mask_path / f"{y_path.stem}.png"
        if current_variance.is_file() and current_mask.is_file():
            inputs.append((y_path, current_variance, current_mask))
        else:
            missing.append(y_path.name)
    if missing:
        raise FileNotFoundError("Missing variances or masks for: " + ", ".join(missing[:10]))
    return inputs


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Tuple[SmallUNet, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Invalid training checkpoint: {checkpoint_path}")
    checkpoint_args = checkpoint.get("args", {})
    base_channels = int(checkpoint_args.get("base_channels", 16))
    variance_scale = float(checkpoint_args.get("variance_scale", 7000.0))
    if variance_scale <= 0:
        raise ValueError(f"Invalid variance scale in checkpoint: {variance_scale}")
    model = SmallUNet(in_channels=2, base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, variance_scale


def load_inputs(
    y_path: Path, variance_path: Path, mask_path: Path, variance_scale: float
) -> Tuple[torch.Tensor, np.ndarray]:
    with Image.open(y_path) as im:
        y_np = (np.asarray(im.convert("L"), dtype=np.float32) / 255.0).copy()
    with Image.open(mask_path) as im:
        mask_np = np.asarray(im).copy()
    variance_np = np.load(variance_path, allow_pickle=False)
    if mask_np.ndim != 2 or variance_np.ndim != 2:
        raise ValueError("Y, variance, and mask must all be HxW")
    if mask_np.shape != y_np.shape or variance_np.shape != y_np.shape:
        raise ValueError(
            f"Shape mismatch for {y_path.name}: Y={y_np.shape}, "
            f"variance={variance_np.shape}, mask={mask_np.shape}"
        )
    valid = mask_np == 1
    if not valid.any():
        raise ValueError(f"Mask has no class-1 pixels: {mask_path}")
    valid_variance = variance_np[valid]
    if not np.isfinite(valid_variance).all() or (valid_variance < 0).any():
        raise ValueError(f"Variance must be finite and non-negative in class-1 area: {variance_path}")

    y_np[~valid] = 0.0
    variance_normalized = np.zeros_like(y_np, dtype=np.float32)
    variance_normalized[valid] = np.clip(
        valid_variance.astype(np.float32) / variance_scale,
        0.0,
        1.0,
    )
    # Alternative logarithmic normalization (keep training and prediction consistent):
    # variance_normalized[valid] = np.clip(
    #     np.log1p(valid_variance.astype(np.float32)) / np.log1p(variance_scale),
    #     0.0,
    #     1.0,
    # )
    network_input = np.stack((y_np, variance_normalized), axis=0)
    tensor = torch.from_numpy(network_input).unsqueeze(0).contiguous()
    return tensor, valid


@torch.no_grad()
def predict_one(
    model: SmallUNet,
    y_path: Path,
    variance_path: Path,
    mask_path: Path,
    variance_scale: float,
    device: torch.device,
) -> np.ndarray:
    network_input, valid = load_inputs(y_path, variance_path, mask_path, variance_scale)
    prediction = model(network_input.to(device))[0, 0].float().cpu().numpy()
    # Preserve the complete HxW map while defining non-class-1 pixels as zero.
    prediction[~valid] = 0.0
    return prediction.astype(np.float32, copy=False)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    inputs = collect_inputs(args.y_input, args.variance, args.mask)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, variance_scale = load_checkpoint(args.checkpoint, device)

    for index, (y_path, variance_path, mask_path) in enumerate(inputs, start=1):
        prediction = predict_one(
            model, y_path, variance_path, mask_path, variance_scale, device
        )
        output_path = args.output_dir / f"{y_path.stem}.npy"
        np.save(output_path, prediction, allow_pickle=False)
        print(
            f"[{index}/{len(inputs)}] saved {output_path} "
            f"shape={prediction.shape} variance_scale={variance_scale:g}"
        )


if __name__ == "__main__":
    main()
