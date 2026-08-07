"""
Independent Training Script for 3D U-Net model on BraTS 2023 dataset.
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


def main():
    parser = argparse.ArgumentParser(description="Train 3D U-Net Model")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--data-root", type=str, default=None, help="Path to BraTS dataset")
    args = parser.parse_args()

    # Load Config
    cfg = get_config()
    data_root = args.data_root or str(cfg.data_root)

    print("Initializing Data Splitter...")
    splitter = DataSplitter(data_root=data_root, val_ratio=0.1, test_ratio=0.1)
    split_config = splitter.get_split(seed=42)

    print("Creating DataLoaders...")
    dataloaders = create_dataloaders(
        data_root=data_root,
        split_config=split_config,
        batch_size=args.batch_size,
    )

    print("Building 3D U-Net Model...")
    model = build_model(
        "unet3d",
        in_channels=cfg.get("data.in_channels", 4),
        out_channels=cfg.get("data.num_classes", 4),
    )
    print(f"Model Parameters: {model.num_parameters():,}")

    print("Setting up Trainer...")
    trainer = SegmentationTrainer(
        model=model,
        train_loader=dataloaders["train_loader"],
        val_loader=dataloaders["val_loader"],
        optimizer=torch.optim.AdamW(model.parameters(), lr=args.lr),
        loss_fn=get_loss_function("dice_ce"),
        checkpoints_dir=cfg.checkpoints_dir,
        model_name="unet_brats",
    )

    print("Starting Training...")
    trainer.train(max_epochs=args.epochs, val_interval=2)


if __name__ == "__main__":
    main()
