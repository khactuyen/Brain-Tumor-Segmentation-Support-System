"""
Training Script for Swin UNETR model on BraTS 2023 dataset.

Usage:
    python scripts/train_swin_unetr.py
    python scripts/train_swin_unetr.py --epochs 300 --lr 1e-4
    python scripts/train_swin_unetr.py --pretrained /path/to/model_swinvit.pt
    python scripts/train_swin_unetr.py --use-v2
"""

import sys
import argparse
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import torch
from src.config import get_config
from src.data import DataSplitter, create_dataloaders
from src.models.swin_unetr import SwinUNETRModel, build_swin_optimizer_scheduler
from src.training import SegmentationTrainer, get_loss_function


def parse_args():
    parser = argparse.ArgumentParser(description="Train Swin UNETR on BraTS 2023")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--feature-size", type=int, default=48)
    parser.add_argument("--steps-per-epoch", type=int, default=150)
    parser.add_argument("--val-interval", type=int, default=2)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--use-v2", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--differential-lr", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 60)
    print("  Swin UNETR - Brain Tumor Segmentation Training")
    print(f"  Device : {device}")
    if torch.cuda.is_available():
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  VRAM   : {vram:.1f} GB")
    print("=" * 60 + "\n")

    cfg = get_config(args.config)
    if args.data_root:
        data_root = args.data_root
    elif cfg.processed_npz_dir.exists():
        data_root = str(cfg.processed_npz_dir)
        print(f"[*] Auto-detected NPZ cache: {data_root}")
    else:
        data_root = str(cfg.data_root)

    print("Initializing Data Splitter...")
    splitter = DataSplitter(data_root=data_root, val_ratio=0.15, test_ratio=0.10)
    split_config = splitter.get_split(seed=42)
    print(f"  Train: {len(split_config['train'])} | Val: {len(split_config['val'])} | Test: {len(split_config['test'])}")

    print("Creating DataLoaders...")
    dataloaders = create_dataloaders(data_root=data_root, split_config=split_config, batch_size=args.batch_size)

    print("Building Swin UNETR Model...")
    model = SwinUNETRModel(
        img_size=(128, 128, 128),
        in_channels=cfg.get("data.in_channels", 4),
        out_channels=cfg.get("data.num_classes", 4),
        feature_size=args.feature_size,
        use_checkpoint=True,
        use_v2=args.use_v2,
    )
    print(f"  Parameters: {model.num_parameters():,}")
    print(f"  Architecture: {'SwinUNETRv2' if args.use_v2 else 'SwinUNETR'}")

    if args.pretrained:
        model.load_pretrained_ssl_weights(args.pretrained)
    else:
        config_pretrained = cfg.get("paths.swin_pretrained_weights", "")
        if config_pretrained and Path(config_pretrained).exists():
            model.load_pretrained_ssl_weights(config_pretrained)
        else:
            print("\n  [TIP] Download pretrained weights from:")
            print("  https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt")
            print("  Then run with: --pretrained /path/to/model_swinvit.pt\n")

    print("Setting up optimizer (AdamW + cosine warm-up)...")
    optimizer, scheduler = build_swin_optimizer_scheduler(
        model=model, lr=args.lr, weight_decay=args.weight_decay,
        max_epochs=args.epochs, warmup_epochs=args.warmup_epochs,
        use_differential_lr=args.differential_lr,
    )

    trainer = SegmentationTrainer(
        model=model, train_loader=dataloaders["train_loader"], val_loader=dataloaders["val_loader"],
        optimizer=optimizer, loss_fn=get_loss_function("dice_ce"), lr_scheduler=scheduler,
        device=device, use_amp=not args.no_amp, checkpoints_dir=cfg.checkpoints_dir,
        model_name="swin_unetr_brats", steps_per_epoch=args.steps_per_epoch,
    )

    print(f"\nStarting training for {args.epochs} epochs...\n")
    history = trainer.train(max_epochs=args.epochs, val_interval=args.val_interval)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE - Swin UNETR BraTS 2023")
    print("=" * 60)
    print(f"  Best Mean Dice : {trainer.best_val_dice:.4f}")
    if history["val_dice_wt"]:
        import numpy as np
        best_idx = int(np.argmax(history["val_dice"]))
        print(f"  Best WT Dice   : {history['val_dice_wt'][best_idx]:.4f}")
        print(f"  Best TC Dice   : {history['val_dice_tc'][best_idx]:.4f}")
        print(f"  Best ET Dice   : {history['val_dice_et'][best_idx]:.4f}")
    print(f"  Checkpoint     : {cfg.checkpoints_dir}/swin_unetr_brats_best.pth")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
