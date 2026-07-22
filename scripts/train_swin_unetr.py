"""
Independent Training Script for Swin UNETR model on BraTS 2023 dataset.
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
from src.utils.logger import logger


def main():
    parser = argparse.ArgumentParser(description="Train Swin UNETR Model")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--data-root", type=str, default=None, help="Path to BraTS dataset")
    args = parser.parse_args()

    # Load Config
    cfg = get_config()
    data_root = args.data_root or str(cfg.data_root)

    logger.info("Initializing Data Splitter...")
    splitter = DataSplitter(data_root=data_root, val_ratio=0.1, test_ratio=0.1)
    split_config = splitter.get_split(seed=42)

    logger.info("Creating DataLoaders...")
    dataloaders = create_dataloaders(
        data_root=data_root,
        split_config=split_config,
        batch_size=args.batch_size,
    )

    logger.info("Building Swin UNETR Model...")
    model = build_model(
        "swin_unetr",
        img_size=(128, 128, 128),
        in_channels=cfg.get("data.in_channels", 4),
        out_channels=cfg.get("data.num_classes", 4),
        feature_size=48,
        use_checkpoint=True,
    )
    logger.info(f"Model Parameters: {model.num_parameters():,}")

    logger.info("Setting up Trainer...")
    trainer = SegmentationTrainer(
        model=model,
        train_loader=dataloaders["train_loader"],
        val_loader=dataloaders["val_loader"],
        optimizer=torch.optim.AdamW(model.parameters(), lr=args.lr),
        loss_fn=get_loss_function("dice_ce"),
        checkpoints_dir=cfg.checkpoints_dir,
        model_name="swin_unetr_brats",
    )

    logger.info("Starting Training...")
    trainer.train(max_epochs=args.epochs, val_interval=2)


if __name__ == "__main__":
    main()
