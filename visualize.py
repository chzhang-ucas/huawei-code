"""Convert predicted NPY score maps into mask-aware color PNG images."""

from __future__ import annotations

import argparse
from pathlib import Path

from matplotlib import cm
import numpy as np
from PIL import Image

JET = cm.get_cmap("jet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=Path, required=True, help="NPY file or directory")
    parser.add_argument("--mask", type=Path, required=True, help="PNG file or directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def collect_pairs(prediction_path: Path, mask_path: Path) -> list[tuple[Path, Path]]:
    if prediction_path.is_file():
        if prediction_path.suffix.lower() != ".npy":
            raise ValueError(f"Prediction must be an NPY file: {prediction_path}")
        expected_mask = mask_path / f"{prediction_path.stem}.png" if mask_path.is_dir() else mask_path
        if not expected_mask.is_file():
            raise FileNotFoundError(f"Mask not found: {expected_mask}")
        return [(prediction_path, expected_mask)]

    if not prediction_path.is_dir() or not mask_path.is_dir():
        raise ValueError("For batch visualization, prediction and mask must be directories")
    predictions = sorted(prediction_path.glob("*.npy"))
    if not predictions:
        raise RuntimeError(f"No NPY predictions found in {prediction_path}")
    pairs = []
    missing = []
    for prediction in predictions:
        current_mask = mask_path / f"{prediction.stem}.png"
        if current_mask.is_file():
            pairs.append((prediction, current_mask))
        else:
            missing.append(current_mask.name)
    if missing:
        raise FileNotFoundError("Missing masks: " + ", ".join(missing[:10]))
    return pairs


def colorize(scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if scores.ndim != 2 or valid.ndim != 2 or scores.shape != valid.shape:
        raise ValueError(f"Prediction and mask must be matching HxW arrays: {scores.shape}, {valid.shape}")
    if not valid.any():
        raise ValueError("Mask has no class-1 pixels")
    if not np.isfinite(scores[valid]).all():
        raise ValueError("Prediction contains NaN or Inf in the class-1 region")

    # Equivalent color mapping to imshow(scores, cmap="jet", vmin=0, vmax=1).
    normalized = np.clip(scores, 0.0, 1.0)
    rgb = np.rint(JET(normalized, bytes=False)[..., :3] * 255).astype(np.uint8)
    rgb[~valid] = 0
    return rgb


def main() -> None:
    args = parse_args()
    pairs = collect_pairs(args.prediction, args.mask)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for index, (prediction_path, mask_path) in enumerate(pairs, start=1):
        scores = np.load(prediction_path, allow_pickle=False)
        with Image.open(mask_path) as im:
            mask = np.asarray(im).copy()
        if mask.ndim != 2:
            raise ValueError(f"Mask must be a single-channel class-index PNG: {mask_path}")
        rgb = colorize(scores, mask == 1)
        output_path = args.output_dir / f"{prediction_path.stem}_color.png"
        Image.fromarray(rgb, mode="RGB").save(output_path)
        valid_scores = scores[mask == 1]
        print(
            f"[{index}/{len(pairs)}] saved {output_path} shape={scores.shape} "
            f"range=[{valid_scores.min():.6f}, {valid_scores.max():.6f}]"
        )


if __name__ == "__main__":
    main()
