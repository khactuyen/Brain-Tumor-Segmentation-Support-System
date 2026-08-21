"""
Training package for Brain Tumor Segmentation Support System.
"""

from .augmentation import get_train_transforms, get_val_transforms
from .losses import get_loss_function
from .trainer import SegmentationTrainer
from .comparison_trainer import run_model_comparison

__all__ = [
    "get_train_transforms",
    "get_val_transforms",
    "get_loss_function",
    "SegmentationTrainer",
    "run_model_comparison",
]
