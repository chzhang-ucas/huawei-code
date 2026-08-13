"""Masked regression loss and accumulated validation metrics."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def masked_smooth_l1(
    prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    valid = valid.bool()
    if not valid.any():
        # Keep the graph valid for an augmented crop containing no class-1 pixels.
        return prediction.sum() * 0.0
    return F.smooth_l1_loss(prediction[valid], target[valid])


class RegressionMeter:
    def __init__(self) -> None:
        self.absolute_error = 0.0
        self.squared_error = 0.0
        self.correct_at_005 = 0
        self.correct_at_010 = 0
        self.pixel_count = 0

    @torch.no_grad()
    def update(
        self, prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
    ) -> None:
        error = prediction[valid.bool()] - target[valid.bool()]
        self.absolute_error += error.abs().sum().item()
        self.squared_error += error.square().sum().item()
        self.correct_at_005 += (error.abs() <= 0.05).sum().item()
        self.correct_at_010 += (error.abs() <= 0.10).sum().item()
        self.pixel_count += error.numel()

    def compute(self) -> Dict[str, float]:
        if self.pixel_count == 0:
            return {
                "mae": float("nan"),
                "rmse": float("nan"),
                "acc_005": float("nan"),
                "acc_010": float("nan"),
            }
        return {
            "mae": self.absolute_error / self.pixel_count,
            "rmse": (self.squared_error / self.pixel_count) ** 0.5,
            "acc_005": self.correct_at_005 / self.pixel_count,
            "acc_010": self.correct_at_010 / self.pixel_count,
        }
