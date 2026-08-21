"""
Comparison Training Script: 3D U-Net vs Swin UNETR on BraTS 2023.

Trains both models on the same data pipeline using identical data splits
and produces a comprehensive comparison table + learning curve plots.

The comparison uses architecture-specific optimization:
- 3D U-Net: AdamW with flat learning rate
- Swin UNETR: AdamW with cosine annealing + linear warm-up

Usage:
    python scripts/compare_models.py
    python scripts/compare_models.py --epochs 50 --batch-size 1 --lr 2e-4
    python scripts/compare_models.py --data-root /path/to/brats_npz --epochs 100
    python scripts/compare_models.py --swin-pretrained /path/to/model_swinvit.pt

Output:
    results/comparison_results.json   — Metrics JSON
    results/comparison_plot.png       — Learning curves + bar charts
    checkpoints/unet3d_brats_best.pth — Best U-Net checkpoint
    checkpoints/swin_unetr_brats_best.pth — Best Swin UNETR checkpoint
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
from src.training import run_model_comparison


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare 3D U-Net vs Swin UNETR on BraTS 2023"
    )
    parser.add_argument(
        "--epochs", type=int, default=30,
        help="Number of training epochs per model (default: 30)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Batch size (default: 1, required for Swin UNETR on 16GB VRAM)"
    )
    parser.add_argument(
        "--lr", type=float, default=2e-4,
        help="Learning rate (default: 2e-4)"
    )
    parser.add_argument(
        "--steps-per-epoch", type=int, default=150,
        help="Max steps per epoch for fast training (0=unlimited, default: 150)"
    )
    parser.add_argument(
        "--val-interval", type=int, default=2,
        help="Validate every N epochs (default: 2)"
    )
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="Path to BraTS dataset or NPZ cache directory"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (default: configs/default.yaml)"
    )
    parser.add_argument(
        "--swin-pretrained", type=str, default=None,
        help="Path to Swin UNETR SSL pretrained weights (.pt file)"
    )
    parser.add_argument(
        "--swin-warmup", type=int, default=10,
        help="Number of warm-up epochs for Swin UNETR scheduler (default: 10)"
    )
    parser.add_argument(
        "--no-amp", action="store_true",
        help="Disable Automatic Mixed Precision"
    )
    parser.add_argument(
        "--models", nargs="+", default=["unet3d", "swin_unetr"],
        choices=["unet3d", "swin_unetr"],
        help="Models to compare (default: unet3d swin_unetr)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 65)
    print("  Brain Tumor Segmentation — Model Comparison")
    print("  Dataset: BraTS 2023 GLI Challenge")
    print(f"  Models : {' vs '.join(args.models)}")
    print(f"  Device : {device}")
    if torch.cuda.is_available():
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  VRAM   : {vram:.1f} GB")
    print("=" * 65 + "\n")

    # ── Resolve data root ────────────────────────────────────────────────────
    cfg = get_config(args.config)
    if args.data_root:
        data_root = args.data_root
    elif cfg.processed_npz_dir.exists():
        data_root = str(cfg.processed_npz_dir)
        print(f"[*] Auto-detected NPZ cache: {data_root}")
    else:
        data_root = str(cfg.data_root)
        print(f"[*] Using raw data root: {data_root}")

    # ── DataLoaders ──────────────────────────────────────────────────────────
    print("Initializing DataSplitter...")
    splitter = DataSplitter(data_root=data_root, val_ratio=0.15, test_ratio=0.10)
    split_config = splitter.get_split(seed=42)
    print(
        f"Split: Train={len(split_config['train'])} | "
        f"Val={len(split_config['val'])} | "
        f"Test={len(split_config['test'])}"
    )

    print("Creating DataLoaders...")
    dataloaders = create_dataloaders(
        data_root=data_root,
        split_config=split_config,
        batch_size=args.batch_size,
    )

    # ── Model Configs ────────────────────────────────────────────────────────
    models_cfg = []
    for arch in args.models:
        if arch == "unet3d":
            models_cfg.append({
                "name": "unet3d",
                "model_name": "unet3d_brats",
                "in_channels": 4,
                "out_channels": 4,
            })
        elif arch == "swin_unetr":
            models_cfg.append({
                "name": "swin_unetr",
                "model_name": "swin_unetr_brats",
                "in_channels": 4,
                "out_channels": 4,
                "feature_size": 48,
                "use_checkpoint": True,
            })

    # ── Run Comparison ───────────────────────────────────────────────────────
    summaries = run_model_comparison(
        train_loader=dataloaders["train_loader"],
        val_loader=dataloaders["val_loader"],
        models_cfg=models_cfg,
        max_epochs=args.epochs,
        val_interval=args.val_interval,
        lr=args.lr,
        use_amp=not args.no_amp,
        checkpoints_dir=cfg.checkpoints_dir,
        steps_per_epoch=args.steps_per_epoch,
        results_dir=cfg.results_dir,
        device=device,
        pretrained_swin_path=args.swin_pretrained,
        swin_warmup_epochs=args.swin_warmup,
    )

    print("\n" + "=" * 65)
    print("  Comparison complete!")
    print(f"  Results saved to: {cfg.results_dir}")
    print(f"  Checkpoints at:   {cfg.checkpoints_dir}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
