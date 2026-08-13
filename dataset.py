"""Dataset and loss-mask-aware batching for sharpness map regression."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg"}


def discover_samples(
    image_dir: Union[str, Path],
    mask_dir: Union[str, Path],
    label_dir: Union[str, Path],
) -> List[Tuple[Path, Path, Path]]:
    """Match image.jpg, mask.png and label.npy by file stem."""
    image_dir, mask_dir, label_dir = map(Path, (image_dir, mask_dir, label_dir))
    samples: List[Tuple[Path, Path, Path]] = []
    missing: List[str] = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in VALID_IMAGE_SUFFIXES:
            continue
        mask_path = mask_dir / f"{image_path.stem}.png"
        label_path = label_dir / f"{image_path.stem}.npy"
        if mask_path.is_file() and label_path.is_file():
            samples.append((image_path, mask_path, label_path))
        else:
            missing.append(image_path.name)
    if missing:
        raise FileNotFoundError(
            "Missing matching PNG mask or NPY label for: " + ", ".join(missing[:10])
        )
    if not samples:
        raise RuntimeError(f"No matched samples found under {image_dir}")
    return samples


class JointAugment:
    """Apply identical geometry to image, label and valid mask."""

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
        samples: List[Tuple[Path, Path, Path]],
        transform: Optional[Callable] = None,
    ) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, str]]:
        image_path, mask_path, label_path = self.samples[index]
        with Image.open(image_path) as im:
            image_np = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
            image = torch.from_numpy(image_np.transpose(2, 0, 1)).contiguous()
        with Image.open(mask_path) as im:
            mask = torch.from_numpy(np.asarray(im).copy()).long()
        label_np = np.load(label_path, allow_pickle=False)

        if label_np.ndim != 2 or mask.ndim != 2:
            raise ValueError(f"Mask and label must be HxW: {image_path.name}")
        h, w = image.shape[-2:]
        if mask.shape != (h, w) or label_np.shape != (h, w):
            raise ValueError(
                f"Shape mismatch for {image_path.name}: image={(h, w)}, "
                f"mask={tuple(mask.shape)}, label={label_np.shape}"
            )

        label = torch.from_numpy(label_np.astype(np.float32, copy=False)).unsqueeze(0)
        valid = mask.eq(1).unsqueeze(0)
        if not valid.any():
            raise ValueError(f"Mask has no class-1 pixels: {mask_path.name}")
        if not torch.isfinite(label[valid]).all():
            raise ValueError(f"Label contains NaN/Inf in class-1 area: {label_path.name}")
        if ((label[valid] < 0) | (label[valid] > 1)).any():
            raise ValueError(f"Class-1 label values must be in [0, 1]: {label_path.name}")

        if self.transform is not None:
            image, label, valid = self.transform(image, label, valid)
        return {"image": image, "label": label, "valid": valid, "name": image_path.stem}
