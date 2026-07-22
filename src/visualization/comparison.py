"""
Model comparison visualization tools (Bar charts, Side-by-side overlays, Radar plots).
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Tuple
from matplotlib.colors import ListedColormap
from .viewer import SliceViewer


def plot_model_comparison_bar(
    results_dict: Dict[str, Dict[str, float]],
    metrics_to_plot: Optional[List[str]] = None,
    title: str = "Model Performance Comparison",
    figsize: Tuple[int, int] = (8, 5),
) -> plt.Figure:
    """
    Create a bar chart comparing performance metrics between models.

    Args:
        results_dict: Dict mapping model_name -> dict of metrics (e.g. {"unet": {"dice": 0.85, ...}})
        metrics_to_plot: List of metric keys to plot (default: dice, iou, precision, recall)
        title: Title of chart
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    if metrics_to_plot is None:
        metrics_to_plot = ["dice", "iou", "precision", "recall"]

    models = list(results_dict.keys())
    x = np.arange(len(metrics_to_plot))
    width = 0.35 / max(1, len(models) / 2)

    fig, ax = plt.subplots(figsize=figsize)

    for i, model in enumerate(models):
        metrics_values = [results_dict[model].get(m, 0.0) for m in metrics_to_plot]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, metrics_values, width, label=model.upper())

        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(
                    f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_ylabel("Score (0.0 - 1.0)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics_to_plot], fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    return fig


def plot_side_by_side_prediction(
    image_slice: np.ndarray,
    ground_truth_slice: Optional[np.ndarray],
    unet_mask_slice: Optional[np.ndarray],
    swin_mask_slice: Optional[np.ndarray],
    title: str = "Segmentation Model Side-by-Side Comparison",
    figsize: Tuple[int, int] = (16, 4),
) -> plt.Figure:
    """
    Create side-by-side comparison of MRI, GT, U-Net prediction, and Swin UNETR prediction.

    Args:
        image_slice: 2D MRI slice
        ground_truth_slice: Optional 2D GT mask
        unet_mask_slice: Optional 2D U-Net prediction
        swin_mask_slice: Optional 2D Swin UNETR prediction
        title: Figure title
        figsize: Figure dimensions

    Returns:
        Matplotlib Figure object
    """
    panels = [("MRI Image", image_slice, None)]

    if ground_truth_slice is not None:
        panels.append(("Ground Truth", image_slice, ground_truth_slice))
    if unet_mask_slice is not None:
        panels.append(("3D U-Net Prediction", image_slice, unet_mask_slice))
    if swin_mask_slice is not None:
        panels.append(("Swin UNETR Prediction", image_slice, swin_mask_slice))

    fig, axes = plt.subplots(1, len(panels), figsize=figsize)
    if len(panels) == 1:
        axes = [axes]

    cmap = ListedColormap([
        (0, 0, 0, 0),
        (1.0, 0.2, 0.2, 0.6),
        (0.2, 0.8, 0.2, 0.5),
        (1.0, 0.9, 0.1, 0.7),
    ])

    for ax, (p_title, p_img, p_mask) in zip(axes, panels):
        ax.imshow(p_img, cmap="gray")
        if p_mask is not None and np.any(p_mask):
            ax.imshow(p_mask, cmap=cmap, vmin=0, vmax=3)
        ax.set_title(p_title, fontsize=11, fontweight="bold")
        ax.axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig
