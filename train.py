"""Train and validate SmallUNet on full-resolution images."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast
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


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: GradScaler,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    meter = RegressionMeter()
    total_loss = 0.0
    steps = 0

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        if not valid.any():
            continue

        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with autocast(device_type=device.type, enabled=device.type == "cuda"):
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
    metrics["loss"] = total_loss / max(steps, 1)
    return metrics


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_samples = discover_samples(
        args.data_root / "train/images",
        args.data_root / "train/masks",
        args.data_root / "train/labels",
    )
    val_samples = discover_samples(
        args.data_root / "val/images",
        args.data_root / "val/masks",
        args.data_root / "val/labels",
    )
    augment = None if args.no_augment else JointAugment(tuple(args.crop_size) if args.crop_size else None)
    train_loader = make_loader(SharpnessDataset(train_samples, augment), args, train=True)
    val_loader = make_loader(SharpnessDataset(val_samples), args, train=False)

    model = SmallUNet(base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    scaler = GradScaler(device.type, enabled=device.type == "cuda")
    history_path = args.output_dir / "history.csv"
    tensorboard_dir = args.tensorboard_dir or args.output_dir / "tensorboard"
    tb_writer = SummaryWriter(log_dir=str(tensorboard_dir))
    best_mae = float("inf")

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
                train_metrics = run_epoch(model, train_loader, device, optimizer, scaler)
                with torch.no_grad():
                    val_metrics = run_epoch(model, val_loader, device, None, scaler)
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
                    tb_writer.add_scalars(
                        f"metrics/{metric_name}",
                        {"train": train_metrics[metric_name], "val": val_metrics[metric_name]},
                        epoch,
                    )
                tb_writer.add_scalar("optimization/learning_rate", learning_rate, epoch)
                tb_writer.flush()

                print(
                    f"Epoch {epoch:03d} | train loss {train_metrics['loss']:.6f} "
                    f"MAE {train_metrics['mae']:.6f} Acc@0.10 {train_metrics['acc_010']:.2%} | "
                    f"val loss {val_metrics['loss']:.6f} MAE {val_metrics['mae']:.6f} "
                    f"RMSE {val_metrics['rmse']:.6f} Acc@0.10 {val_metrics['acc_010']:.2%}"
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
