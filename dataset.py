"""Dataset and loss-mask-aware batching for sharpness map regression."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


VALID_Y_SUFFIXES = {".png"}


def discover_samples(
    y_dir: Union[str, Path],
    variance_dir: Union[str, Path],
    mask_dir: Union[str, Path],
    label_dir: Union[str, Path],
) -> List[Tuple[Path, Path, Path, Path]]:
    """Match Y PNG, variance NPY, mask PNG, and label NPY by file stem."""
    y_dir, variance_dir, mask_dir, label_dir = map(
        Path, (y_dir, variance_dir, mask_dir, label_dir)
    )
    samples: List[Tuple[Path, Path, Path, Path]] = []
    missing: List[str] = []
    for y_path in sorted(y_dir.iterdir()):
        if y_path.suffix.lower() not in VALID_Y_SUFFIXES:
            continue
        variance_path = variance_dir / f"{y_path.stem}.npy"
        mask_path = mask_dir / f"{y_path.stem}.png"
        label_path = label_dir / f"{y_path.stem}.npy"
        if variance_path.is_file() and mask_path.is_file() and label_path.is_file():
            samples.append((y_path, variance_path, mask_path, label_path))
        else:
            missing.append(y_path.name)
    if missing:
        raise FileNotFoundError(
            "Missing matching variance, mask, or label for: " + ", ".join(missing[:10])
        )
    if not samples:
        raise RuntimeError(f"No matched samples found under {y_dir}")
    return samples


class JointAugment:
    """Apply identical geometry to two-channel input, label, and valid mask."""

    def __init__(self, crop_size: Optional[Tuple[int, int]] = None) -> None:
        self.crop_size = crop_size

    def __call__(
        self, image: torch.Tensor, label: torch.Tensor, valid: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if random.random() < 0.5:
            image, label, valid = (torch.flip(x, dims=(-1,)) for x in (image, label, valid))
        if random.random() < 0.5:
            image, label, valid = (torch.flip(x, dims=(-2,)) for x in (image, label, valid))

        # A 90-degree multiple uses exact index rearrangement and no interpolation.
        k = random.randrange(4)
        if k:
            image = torch.rot90(image, k, dims=(-2, -1))
            label = torch.rot90(label, k, dims=(-2, -1))
            valid = torch.rot90(valid, k, dims=(-2, -1))

        if self.crop_size is not None:
            crop_h, crop_w = self.crop_size
            h, w = image.shape[-2:]
            if h < crop_h or w < crop_w:
                raise ValueError(
                    f"Image {(h, w)} is smaller than crop {(crop_h, crop_w)}"
                )
            # Anchor the crop on a class-1 pixel so every crop has supervision.
            valid_yx = torch.nonzero(valid[0], as_tuple=False)
            if valid_yx.numel() == 0:
                raise ValueError("Cannot crop a sample without class-1 pixels")
            anchor_y, anchor_x = valid_yx[random.randrange(len(valid_yx))].tolist()
            top_min = max(0, anchor_y - crop_h + 1)
            top_max = min(anchor_y, h - crop_h)
            left_min = max(0, anchor_x - crop_w + 1)
            left_max = min(anchor_x, w - crop_w)
            top = random.randint(top_min, top_max)
            left = random.randint(left_min, left_max)
            sl = (..., slice(top, top + crop_h), slice(left, left + crop_w))
            image, label, valid = image[sl], label[sl], valid[sl]
        return image, label, valid


class SharpnessDataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[Path, Path, Path, Path]],
        variance_scale: float = 7000.0,
        transform: Optional[Callable] = None,
    ) -> None:
        if variance_scale <= 0:
            raise ValueError("variance_scale must be positive")
        self.samples = samples
        self.variance_scale = variance_scale
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, str]]:
        y_path, variance_path, mask_path, label_path = self.samples[index]
        with Image.open(y_path) as im:
            y_np = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
        with Image.open(mask_path) as im:
            mask_np = np.asarray(im).copy()
        variance_np = np.load(variance_path, allow_pickle=False)
        label_np = np.load(label_path, allow_pickle=False)

        if mask_np.ndim != 2 or variance_np.ndim != 2 or label_np.ndim != 2:
            raise ValueError(f"Y, variance, mask, and label must be HxW: {y_path.name}")
        h, w = y_np.shape
        if mask_np.shape != (h, w) or variance_np.shape != (h, w) or label_np.shape != (h, w):
            raise ValueError(
                f"Shape mismatch for {y_path.name}: Y={(h, w)}, variance={variance_np.shape}, "
                f"mask={mask_np.shape}, label={label_np.shape}"
            )

        valid_np = mask_np == 1
        if not valid_np.any():
            raise ValueError(f"Mask has no class-1 pixels: {mask_path.name}")
        valid_variance = variance_np[valid_np]
        if not np.isfinite(valid_variance).all():
            raise ValueError(f"Variance contains NaN/Inf in class-1 area: {variance_path.name}")
        if (valid_variance < 0).any():
            raise ValueError(f"Variance contains negative values in class-1 area: {variance_path.name}")

        # Filter both network inputs with the mask before concatenation.
        y_filtered = y_np.copy()
        y_filtered[~valid_np] = 0.0
        variance_normalized = np.zeros((h, w), dtype=np.float32)
        variance_normalized[valid_np] = np.clip(
            valid_variance.astype(np.float32) / self.variance_scale,
            0.0,
            1.0,
        )
        # Alternative logarithmic normalization (keep training and prediction consistent):
        # variance_normalized[valid_np] = np.clip(
        #     np.log1p(valid_variance.astype(np.float32))
        #     / np.log1p(self.variance_scale),
        #     0.0,
        #     1.0,
        # )
        image_np = np.stack((y_filtered, variance_normalized), axis=0)
        image = torch.from_numpy(image_np).contiguous()
        label = torch.from_numpy(label_np.astype(np.float32, copy=False)).unsqueeze(0)
        valid = torch.from_numpy(valid_np).unsqueeze(0)
        if not torch.isfinite(label[valid]).all():
            raise ValueError(f"Label contains NaN/Inf in class-1 area: {label_path.name}")
        if ((label[valid] < 0) | (label[valid] > 1)).any():
            raise ValueError(f"Class-1 label values must be in [0, 1]: {label_path.name}")

        if self.transform is not None:
            image, label, valid = self.transform(image, label, valid)
        return {"image": image, "label": label, "valid": valid, "name": y_path.stem}
