"""
Training Script for 3D U-Net model on BraTS 2023 dataset.

Architecture: MONAI 3D U-Net with Residual Blocks + Instance Normalization.
Optimizer: AdamW with flat learning rate (no warm-up needed for CNN).

Usage:
    python scripts/train_unet.py
    python scripts/train_unet.py --epochs 100 --lr 2e-4 --batch-size 2
    python scripts/train_unet.py --data-root /path/to/brats_npz
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import torch
from src.config import get_config
from src.data import DataSplitter, create_dataloaders
from src.models import build_model
from src.training import SegmentationTrainer, get_loss_function


def parse_args():
    parser = argparse.ArgumentParser(description="Train 3D U-Net on BraTS 2023")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--steps-per-epoch", type=int, default=150, help="Steps per epoch (0=unlimited)")
    parser.add_argument("--val-interval", type=int, default=2, help="Validate every N epochs")
    parser.add_argument("--data-root", type=str, default=None, help="Dataset path")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 60)
    print("  3D U-Net — Brain Tumor Segmentation Training")
    print(f"  Device : {device}")
    if torch.cuda.is_available():
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  VRAM   : {vram:.1f} GB")
    print("=" * 60 + "\n")

    # ── Config ───────────────────────────────────────────────────────────────
    cfg = get_config(args.config)
    if args.data_root:
        data_root = args.data_root
    elif cfg.processed_npz_dir.exists():
        data_root = str(cfg.processed_npz_dir)
        print(f"[*] Auto-detected NPZ cache: {data_root}")
    else:
        data_root = str(cfg.data_root)

    # ── Data ─────────────────────────────────────────────────────────────────
    print("Initializing Data Splitter...")
    splitter = DataSplitter(data_root=data_root, val_ratio=0.15, test_ratio=0.10)
    split_config = splitter.get_split(seed=42)
    print(f"  Train: {len(split_config['train'])} | Val: {len(split_config['val'])} | Test: {len(split_config['test'])}")

    print("Creating DataLoaders...")
    dataloaders = create_dataloaders(
        data_root=data_root,
        split_config=split_config,
        batch_size=args.batch_size,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    print("Building 3D U-Net Model...")
    model = build_model(
        "unet3d",
        in_channels=cfg.get("data.in_channels", 4),
        out_channels=cfg.get("data.num_classes", 4),
    )
    print(f"  Trainable parameters: {model.num_parameters():,}")
    model_info = model.get_model_info()
    print(f"  Channels: {model_info.get('channels', 'N/A')}")

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = SegmentationTrainer(
        model=model,
        train_loader=dataloaders["train_loader"],
        val_loader=dataloaders["val_loader"],
        optimizer=optimizer,
        loss_fn=get_loss_function("dice_ce"),
        device=device,
        use_amp=not args.no_amp,
        checkpoints_dir=cfg.checkpoints_dir,
        model_name="unet3d_brats",
        steps_per_epoch=args.steps_per_epoch,
    )

    # ── Train ────────────────────────────────────────────────────────────────
    print(f"\nStarting training for {args.epochs} epochs...\n")
    history = trainer.train(max_epochs=args.epochs, val_interval=args.val_interval)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE — 3D U-Net BraTS 2023")
    print("=" * 60)
    print(f"  Best Mean Dice : {trainer.best_val_dice:.4f}")
    if history["val_dice_wt"]:
        import numpy as np
        best_idx = int(np.argmax(history["val_dice"]))
        print(f"  Best WT Dice   : {history['val_dice_wt'][best_idx]:.4f}")
        print(f"  Best TC Dice   : {history['val_dice_tc'][best_idx]:.4f}")
        print(f"  Best ET Dice   : {history['val_dice_et'][best_idx]:.4f}")
    print(f"  Checkpoint     : {cfg.checkpoints_dir}/unet3d_brats_best.pth")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
