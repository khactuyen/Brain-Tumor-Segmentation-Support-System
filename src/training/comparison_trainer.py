"""
Comparison Trainer — Train multiple models on the same pipeline and produce
a comprehensive side-by-side results table with learning curves.

Supports:
- Sequential training of 3D U-Net and Swin UNETR on identical data splits
- BraTS-standard metrics: Dice (WT, TC, ET), HD95
- Automatic comparison plots (learning curves + bar charts)
- JSON results export for reproducibility
- Per-model training time tracking

Usage (Python):
    from src.training.comparison_trainer import run_model_comparison

    results = run_model_comparison(
        models_cfg=[
            {"name": "unet3d",     "model_name": "unet3d"},
            {"name": "swin_unetr", "model_name": "swin_unetr"},
        ],
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=30,
        ...
    )

Usage (Jupyter notebook):
    See Section 7 in SIC_Capstone_v2.ipynb.
"""

import time
import json
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from torch.utils.data import DataLoader

from ..models import build_model, BaseSegmentationModel
from ..models.swin_unetr import build_swin_optimizer_scheduler
from ..training.trainer import SegmentationTrainer
from ..training.losses import get_loss_function
from ..utils.logger import logger


# Use non-interactive backend when not in notebook/GUI environment
try:
    get_ipython()  # type: ignore
except NameError:
    matplotlib.use("Agg")


# ─── Single-model experiment runner ──────────────────────────────────────────

def _run_single_experiment(
    model: BaseSegmentationModel,
    model_label: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    max_epochs: int,
    val_interval: int,
    lr: float,
    weight_decay: float,
    use_amp: bool,
    checkpoints_dir: Path,
    steps_per_epoch: int,
    lr_scheduler_factory=None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    """Train a single model and return its training history + summary."""

    print(f"\n{'='*65}")
    print(f"  🚀 Training Model: {model_label}")
    print(f"  Parameters: {model.num_parameters():,}")
    print(f"  Device: {device}  |  AMP: {use_amp}  |  Epochs: {max_epochs}")
    print(f"{'='*65}\n")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    scheduler = None
    if lr_scheduler_factory is not None:
        scheduler = lr_scheduler_factory(optimizer, max_epochs)

    loss_fn = get_loss_function("dice_ce")

    trainer = SegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        lr_scheduler=scheduler,
        device=device,
        use_amp=use_amp,
        checkpoints_dir=checkpoints_dir,
        model_name=model_label,
        steps_per_epoch=steps_per_epoch,
    )

    t_start = time.time()
    history = trainer.train(max_epochs=max_epochs, val_interval=val_interval)
    total_time = time.time() - t_start

    # Compute final best metrics from history
    best_idx = int(np.argmax(history["val_dice"])) if history["val_dice"] else 0

    summary = {
        "model": model_label,
        "parameters": model.num_parameters(),
        "best_val_dice": trainer.best_val_dice,
        "best_val_dice_wt": history["val_dice_wt"][best_idx] if history["val_dice_wt"] else 0.0,
        "best_val_dice_tc": history["val_dice_tc"][best_idx] if history["val_dice_tc"] else 0.0,
        "best_val_dice_et": history["val_dice_et"][best_idx] if history["val_dice_et"] else 0.0,
        "total_train_time_min": total_time / 60.0,
        "epochs_trained": max_epochs,
        "history": history,
    }

    print(f"\n✅ Finished training {model_label}!")
    print(f"   Best Mean Dice : {summary['best_val_dice']:.4f}")
    print(f"   WT / TC / ET   : {summary['best_val_dice_wt']:.4f} / "
          f"{summary['best_val_dice_tc']:.4f} / "
          f"{summary['best_val_dice_et']:.4f}")
    print(f"   Training time  : {summary['total_train_time_min']:.1f} min\n")

    return summary


# ─── Swin UNETR experiment runner (with cosine warm-up scheduler) ─────────────

def _run_swin_experiment(
    model: BaseSegmentationModel,
    model_label: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    max_epochs: int,
    val_interval: int,
    lr: float,
    weight_decay: float,
    use_amp: bool,
    checkpoints_dir: Path,
    steps_per_epoch: int,
    warmup_epochs: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    """
    Train Swin UNETR with its recommended optimizer/scheduler setup.

    Uses AdamW + cosine annealing with linear warm-up, as recommended
    in the original Swin UNETR paper for optimal transformer training.
    """
    print(f"\n{'='*65}")
    print(f"  🚀 Training Model: {model_label} (Transformer Schedule)")
    print(f"  Parameters: {model.num_parameters():,}")
    print(f"  Device: {device}  |  AMP: {use_amp}  |  Epochs: {max_epochs}")
    print(f"  Optimizer: AdamW  |  Scheduler: Cosine + {warmup_epochs} warmup")
    print(f"{'='*65}\n")

    optimizer, scheduler = build_swin_optimizer_scheduler(
        model=model,
        lr=lr,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        warmup_epochs=warmup_epochs,
    )

    loss_fn = get_loss_function("dice_ce")

    trainer = SegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        lr_scheduler=scheduler,
        device=device,
        use_amp=use_amp,
        checkpoints_dir=checkpoints_dir,
        model_name=model_label,
        steps_per_epoch=steps_per_epoch,
    )

    t_start = time.time()
    history = trainer.train(max_epochs=max_epochs, val_interval=val_interval)
    total_time = time.time() - t_start

    best_idx = int(np.argmax(history["val_dice"])) if history["val_dice"] else 0

    summary = {
        "model": model_label,
        "parameters": model.num_parameters(),
        "best_val_dice": trainer.best_val_dice,
        "best_val_dice_wt": history["val_dice_wt"][best_idx] if history["val_dice_wt"] else 0.0,
        "best_val_dice_tc": history["val_dice_tc"][best_idx] if history["val_dice_tc"] else 0.0,
        "best_val_dice_et": history["val_dice_et"][best_idx] if history["val_dice_et"] else 0.0,
        "total_train_time_min": total_time / 60.0,
        "epochs_trained": max_epochs,
        "history": history,
    }

    print(f"\n✅ Finished training {model_label}!")
    print(f"   Best Mean Dice : {summary['best_val_dice']:.4f}")
    print(f"   WT / TC / ET   : {summary['best_val_dice_wt']:.4f} / "
          f"{summary['best_val_dice_tc']:.4f} / "
          f"{summary['best_val_dice_et']:.4f}")
    print(f"   Training time  : {summary['total_train_time_min']:.1f} min\n")

    return summary


# ─── Main comparison runner ───────────────────────────────────────────────────

def run_model_comparison(
    train_loader: DataLoader,
    val_loader: DataLoader,
    models_cfg: Optional[List[Dict[str, Any]]] = None,
    max_epochs: int = 30,
    val_interval: int = 2,
    lr: float = 2e-4,
    weight_decay: float = 1e-4,
    use_amp: bool = True,
    checkpoints_dir: Union[str, Path] = "checkpoints",
    steps_per_epoch: int = 150,
    results_dir: Union[str, Path] = "results",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    pretrained_swin_path: Optional[str] = None,
    swin_warmup_epochs: int = 10,
) -> List[Dict[str, Any]]:
    """
    Train multiple models on the same data pipeline and produce a
    comprehensive comparison report.

    Args:
        train_loader: Shared training DataLoader
        val_loader: Shared validation DataLoader
        models_cfg: List of model config dicts, e.g.
            [{"name": "unet3d", "model_name": "unet3d_exp"},
             {"name": "swin_unetr", "model_name": "swin_unetr_exp"}]
        max_epochs: Total training epochs per model
        val_interval: Validation frequency (epochs)
        lr: Learning rate
        weight_decay: Weight decay
        use_amp: Automatic Mixed Precision
        checkpoints_dir: Directory to save checkpoints
        steps_per_epoch: Step limit per epoch (0 = unlimited)
        results_dir: Directory to save comparison results JSON / plots
        device: "cuda" or "cpu"
        pretrained_swin_path: Optional path to Swin UNETR SSL pretrained weights
        swin_warmup_epochs: Number of warm-up epochs for Swin UNETR scheduler

    Returns:
        List of summary dicts for each model
    """
    if models_cfg is None:
        models_cfg = [
            {"name": "unet3d",     "model_name": "unet3d_brats"},
            {"name": "swin_unetr", "model_name": "swin_unetr_brats"},
        ]

    checkpoints_dir = Path(checkpoints_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []

    for cfg in models_cfg:
        arch_name = cfg["name"]
        model_label = cfg.get("model_name", arch_name)

        # Build model
        build_kwargs = {k: v for k, v in cfg.items() if k not in ("name", "model_name")}
        model = build_model(arch_name, **build_kwargs)

        # Load pretrained SSL weights for Swin UNETR if provided
        if arch_name in ("swin_unetr", "swin") and pretrained_swin_path:
            model.load_pretrained_ssl_weights(pretrained_swin_path)

        # Use specialized training loop for Swin UNETR (cosine warm-up scheduler)
        if arch_name in ("swin_unetr", "swin"):
            summary = _run_swin_experiment(
                model=model,
                model_label=model_label,
                train_loader=train_loader,
                val_loader=val_loader,
                max_epochs=max_epochs,
                val_interval=val_interval,
                lr=lr,
                weight_decay=weight_decay,
                use_amp=use_amp,
                checkpoints_dir=checkpoints_dir,
                steps_per_epoch=steps_per_epoch,
                warmup_epochs=swin_warmup_epochs,
                device=device,
            )
        else:
            summary = _run_single_experiment(
                model=model,
                model_label=model_label,
                train_loader=train_loader,
                val_loader=val_loader,
                max_epochs=max_epochs,
                val_interval=val_interval,
                lr=lr,
                weight_decay=weight_decay,
                use_amp=use_amp,
                checkpoints_dir=checkpoints_dir,
                steps_per_epoch=steps_per_epoch,
                device=device,
            )
        all_summaries.append(summary)

    # Save JSON results
    json_path = results_dir / "comparison_results.json"
    serializable = []
    for s in all_summaries:
        item = {k: v for k, v in s.items() if k != "history"}
        serializable.append(item)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\n📊 Results saved to: {json_path}")

    # Plot & save comparison figure
    fig_path = results_dir / "comparison_plot.png"
    _plot_comparison(all_summaries, val_interval=val_interval, save_path=fig_path)
    print(f"📈 Comparison plot saved to: {fig_path}")

    # Print summary table
    _print_summary_table(all_summaries)

    return all_summaries


# ─── Plotting & Table helpers ─────────────────────────────────────────────────

def _plot_comparison(
    summaries: List[Dict[str, Any]],
    val_interval: int = 2,
    save_path: Optional[Path] = None,
) -> None:
    """Plot training loss and validation Dice curves for all models."""

    # Professional color palette
    palette = ["#2563EB", "#DC2626", "#059669", "#D97706", "#7C3AED"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    for idx, s in enumerate(summaries):
        color = palette[idx % len(palette)]
        history = s["history"]
        label = s["model"]

        # Training Loss
        epochs = range(1, len(history["train_loss"]) + 1)
        axes[0].plot(epochs, history["train_loss"], label=label, color=color, lw=2)

        # Validation Dice (Mean)
        val_epochs = list(range(val_interval, len(history["train_loss"]) + 1, val_interval))
        if len(history["val_dice"]) == len(val_epochs):
            axes[1].plot(
                val_epochs, history["val_dice"],
                label=f"{label} Mean", color=color, marker="o", lw=2
            )

    axes[0].set_title("Training Loss (DiceCE)", fontweight="bold", fontsize=12)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].set_title("Validation Mean Dice (WT+TC+ET)/3", fontweight="bold", fontsize=12)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice Score")
    axes[1].set_ylim(0, 1.0)
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Bar chart: Best WT/TC/ET per model
    model_names = [s["model"] for s in summaries]
    x = np.arange(len(model_names))
    width = 0.25
    bars_wt = [s["best_val_dice_wt"] for s in summaries]
    bars_tc = [s["best_val_dice_tc"] for s in summaries]
    bars_et = [s["best_val_dice_et"] for s in summaries]

    axes[2].bar(x - width, bars_wt, width, label="WT", color="#10B981")
    axes[2].bar(x,         bars_tc, width, label="TC", color="#F59E0B")
    axes[2].bar(x + width, bars_et, width, label="ET", color="#EF4444")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(model_names, rotation=15)
    axes[2].set_title("Best Dice per Sub-Region", fontweight="bold", fontsize=12)
    axes[2].set_ylabel("Dice Score")
    axes[2].set_ylim(0, 1.0)
    axes[2].legend()
    axes[2].grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.suptitle(
        "🧠 Brain Tumor Segmentation — Model Comparison (BraTS 2023)",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[ComparisonTrainer] Plot saved to {save_path}")
    plt.show()


def _print_summary_table(summaries: List[Dict[str, Any]]) -> None:
    """Print a rich comparison table to stdout."""
    header = (
        f"\n{'='*80}\n"
        f"{'MODEL COMPARISON TABLE':^80}\n"
        f"{'BraTS 2023 — Validation Dice Scores':^80}\n"
        f"{'='*80}"
    )
    print(header)
    print(
        f"{'Model':<20} {'Params':>10} {'Mean Dice':>12} "
        f"{'WT':>8} {'TC':>8} {'ET':>8} {'Time(min)':>10}"
    )
    print("-" * 80)
    for s in summaries:
        print(
            f"{s['model']:<20} {s['parameters']:>10,} "
            f"{s['best_val_dice']:>12.4f} "
            f"{s['best_val_dice_wt']:>8.4f} "
            f"{s['best_val_dice_tc']:>8.4f} "
            f"{s['best_val_dice_et']:>8.4f} "
            f"{s['total_train_time_min']:>10.1f}"
        )
    print("=" * 80)

    # Winner
    if summaries:
        best = max(summaries, key=lambda x: x["best_val_dice"])
        print(f"\n🏆 Best model: {best['model']} (Mean Dice = {best['best_val_dice']:.4f})\n")
