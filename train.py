"""Train and validate SmallUNet on full-resolution images."""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import JointAugment, SharpnessDataset, discover_samples
from losses import RegressionMeter, masked_smooth_l1
from model import SmallUNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/small_unet"))
    parser.add_argument(
        "--tensorboard-dir",
        type=Path,
        default=None,
        help="TensorBoard log directory (default: OUTPUT_DIR/tensorboard)",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument(
        "--variance-scale",
        type=float,
        default=7000.0,
        help="Fixed variance divisor used by linear normalization",
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crop-size", type=int, nargs=2, metavar=("H", "W"), default=None)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset: SharpnessDataset, args: argparse.Namespace, train: bool) -> DataLoader:
    # Full images have arbitrary shapes, so batch_size=1 is required unless fixed cropping is enabled.
    if args.crop_size is None and args.batch_size != 1:
        raise ValueError("Use --batch-size 1 for arbitrary full-resolution images")
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=train,
        num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.workers > 0,
    )


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class ProgressEstimator:
    """Estimate total remaining time from all completed train/validation steps."""

    def __init__(self, total_steps: int) -> None:
        self.total_steps = total_steps
        self.completed_steps = 0
        self.started_at = time.perf_counter()
        self.last_step_at = self.started_at

    def update(self) -> Tuple[float, float, float]:
        now = time.perf_counter()
        step_seconds = now - self.last_step_at
        self.last_step_at = now
        self.completed_steps += 1
        elapsed = now - self.started_at
        average_step_seconds = elapsed / self.completed_steps
        remaining_steps = max(0, self.total_steps - self.completed_steps)
        eta = average_step_seconds * remaining_steps
        return step_seconds, elapsed, eta


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: GradScaler,
    epoch: int,
    total_epochs: int,
    progress: ProgressEstimator,
) -> Dict[str, float]:
    training = optimizer is not None
    phase = "Train" if training else "Val  "
    model.train(training)
    meter = RegressionMeter()
    total_loss = 0.0
    steps = 0

    for step, batch in enumerate(loader, start=1):
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        if not valid.any():
            continue

        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with autocast(enabled=device.type == "cuda"):
                prediction = model(image)
                loss = masked_smooth_l1(prediction, label, valid)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.item()
        steps += 1
        meter.update(prediction.detach(), label, valid)
        metrics = meter.compute()
        average_loss = total_loss / steps
        learning_rate = optimizer.param_groups[0]["lr"] if training else 0.0
        step_seconds, elapsed, eta = progress.update()
        lr_text = f" LR {learning_rate:.2e}" if training else ""
        print(
            f"Epoch [{epoch:03d}/{total_epochs:03d}] {phase} "
            f"[{step:04d}/{len(loader):04d}] "
            f"Loss {loss.item():.6f} (Avg {average_loss:.6f}) "
            f"MAE {metrics['mae']:.6f} RMSE {metrics['rmse']:.6f} "
            f"Acc@0.05 {metrics['acc_005']:.2%} Acc@0.10 {metrics['acc_010']:.2%}"
            f"{lr_text} Step {step_seconds:.2f}s "
            f"Elapsed {format_duration(elapsed)} ETA {format_duration(eta)}",
            flush=True,
        )

    metrics = meter.compute()
    metrics["loss"] = total_loss / max(steps, 1)
    return metrics


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_samples = discover_samples(
        args.data_root / "train/y_images",
        args.data_root / "train/variances",
        args.data_root / "train/masks",
        args.data_root / "train/labels",
    )
    val_samples = discover_samples(
        args.data_root / "val/y_images",
        args.data_root / "val/variances",
        args.data_root / "val/masks",
        args.data_root / "val/labels",
    )
    augment = None if args.no_augment else JointAugment(tuple(args.crop_size) if args.crop_size else None)
    train_loader = make_loader(
        SharpnessDataset(train_samples, args.variance_scale, augment), args, train=True
    )
    val_loader = make_loader(
        SharpnessDataset(val_samples, args.variance_scale), args, train=False
    )

    model = SmallUNet(in_channels=2, base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    scaler = GradScaler(enabled=device.type == "cuda")
    history_path = args.output_dir / "history.csv"
    tensorboard_dir = args.tensorboard_dir or args.output_dir / "tensorboard"
    tb_writer = SummaryWriter(log_dir=str(tensorboard_dir))
    best_mae = float("inf")
    total_steps = args.epochs * (len(train_loader) + len(val_loader))
    progress = ProgressEstimator(total_steps)

    print(
        f"Training on {device}: {len(train_samples)} train samples, "
        f"{len(val_samples)} validation samples, {args.epochs} epochs, "
        f"{total_steps} total steps",
        flush=True,
    )

    fieldnames = [
        "epoch", "lr", "train_loss", "train_mae", "train_rmse",
        "train_acc_005", "train_acc_010", "val_loss", "val_mae", "val_rmse",
        "val_acc_005", "val_acc_010",
    ]
    try:
        with history_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for epoch in range(1, args.epochs + 1):
                train_metrics = run_epoch(
                    model, train_loader, device, optimizer, scaler,
                    epoch, args.epochs, progress,
                )
                with torch.no_grad():
                    val_metrics = run_epoch(
                        model, val_loader, device, None, scaler,
                        epoch, args.epochs, progress,
                    )
                scheduler.step(val_metrics["mae"])
                learning_rate = optimizer.param_groups[0]["lr"]
                row = {
                    "epoch": epoch,
                    "lr": learning_rate,
                    **{f"train_{k}": v for k, v in train_metrics.items()},
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                }
                writer.writerow(row)
                f.flush()

                for metric_name in ("loss", "mae", "rmse", "acc_005", "acc_010"):
                    # add_scalars creates separate event files for train/val.
                    # Individual tags keep every curve in this writer's one event file.
                    tb_writer.add_scalar(
                        f"train/{metric_name}", train_metrics[metric_name], epoch
                    )
                    tb_writer.add_scalar(
                        f"val/{metric_name}", val_metrics[metric_name], epoch
                    )
                tb_writer.add_scalar("optimization/learning_rate", learning_rate, epoch)
                tb_writer.flush()

                print(
                    f"Epoch [{epoch:03d}/{args.epochs:03d}] Summary | "
                    f"Train Loss {train_metrics['loss']:.6f} MAE {train_metrics['mae']:.6f} "
                    f"RMSE {train_metrics['rmse']:.6f} | "
                    f"Val Loss {val_metrics['loss']:.6f} MAE {val_metrics['mae']:.6f} "
                    f"RMSE {val_metrics['rmse']:.6f} | "
                    f"Elapsed {format_duration(time.perf_counter() - progress.started_at)} "
                    f"ETA {format_duration((time.perf_counter() - progress.started_at) / progress.completed_steps * (progress.total_steps - progress.completed_steps))}",
                    flush=True,
                )
                checkpoint = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "args": vars(args),
                }
                torch.save(checkpoint, args.output_dir / "last.pt")
                if val_metrics["mae"] < best_mae:
                    best_mae = val_metrics["mae"]
                    torch.save(checkpoint, args.output_dir / "best.pt")
    finally:
        tb_writer.close()


if __name__ == "__main__":
    main()
