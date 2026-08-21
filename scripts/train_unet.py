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
from src.training import SegmentationTrainer, get_loss_function, get_train_transforms, get_val_transforms


def main():
    parser = argparse.ArgumentParser(description="Train 3D U-Net Model")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--data-root", type=str, default=None, help="Path to BraTS dataset or NPZ directory")
    parser.add_argument(
        "--patch-size", type=int, default=128,
        help="[MỚI] Kích thước patch 3D (cạnh lập phương) cho train + sliding-window "
             "validate. Giữ 128 để khớp với 2 epoch đã train trước đó."
    )
    parser.add_argument(
        "--num-samples", type=int, default=2,
        help="[MỚI] Số patch lấy ra từ MỖI volume cho mỗi lần load (RandCropByPosNegLabeld). "
             "Tăng số này giúp tận dụng volume đã load, giảm I/O, nhưng tốn VRAM hơn."
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="[MỚI] Đường dẫn checkpoint để train tiếp (vd: checkpoints/unet_brats_latest.pth "
             "hoặc unet_brats_best.pth). Bỏ trống nếu train từ đầu."
    )
    args = parser.parse_args()

    # Load Config
    cfg = get_config()
    if args.data_root:
        data_root = args.data_root
    elif cfg.processed_npz_dir.exists():
        data_root = str(cfg.processed_npz_dir)
        print(f"[*] Auto-detected Fast NPZ Cache Directory: {data_root}")
    else:
        data_root = str(cfg.data_root)

    print("Initializing Data Splitter...")
    splitter = DataSplitter(data_root=data_root, val_ratio=0.1, test_ratio=0.1)
    split_config = splitter.get_split(seed=42)

    print("Creating DataLoaders...")
    # [MỚI] Nối các transform MONAI đã có sẵn trong repo (nhưng trước đây
    # chưa được dùng) vào dataloader — RandCropByPosNegLabeld ưu tiên lấy
    # patch chứa vùng khối u thay vì crop ngẫu nhiên đều, giống cách đồng
    # nghiệp bạn làm trong SIC_Capstone_v3.ipynb.
    roi_size = (args.patch_size, args.patch_size, args.patch_size)
    transforms_dict = {
        "train": get_train_transforms(patch_size=roi_size, num_samples=args.num_samples),
        "val": get_val_transforms(),
    }
    dataloaders = create_dataloaders(
        data_root=data_root,
        split_config=split_config,
        batch_size=args.batch_size,
        transforms_dict=transforms_dict,
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
        roi_size=roi_size,       # [MỚI] dùng cho sliding_window_inference khi validate
        sw_batch_size=4,
        sw_overlap=0.5,
    )

    # [MỚI] Resume từ checkpoint nếu có (train tiếp, không mất tiến độ 2 epoch đã có)
    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_epoch = trainer.resume_from_checkpoint(args.resume)

    print("Starting Training...")
    trainer.train(max_epochs=args.epochs, val_interval=2, start_epoch=start_epoch)


if __name__ == "__main__":
    main()
