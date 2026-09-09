"""
Data Augmentation pipeline using MONAI transforms for 3D MRI.
"""

from typing import Tuple, List, Optional
import monai.transforms as mt


def get_train_transforms(
    patch_size: Tuple[int, int, int] = (128, 128, 128),
    num_samples: int = 4,
) -> mt.Compose:
    """
    Get MONAI data augmentation pipeline for training.

    Args:
        patch_size: Spatial patch dimensions (H, W, D)
        num_samples: Number of patches per volume

    Returns:
        MONAI Compose transform
    """
    return mt.Compose(
        [
            # Random cropping of spatial patches centered around foreground/background
            mt.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=patch_size,
                pos=1.0,
                neg=1.0,
                num_samples=num_samples,
                image_key="image",
                image_threshold=0,
            ),
            # Spatial flip augmentations
            mt.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            mt.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            mt.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            # 90-degree rotations
            mt.RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            # Intensity augmentations
            mt.RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
            mt.RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            # Ensure correct data types
            mt.EnsureTyped(keys=["image", "label"]),
        ]
    )


def get_val_transforms() -> mt.Compose:
    """
    Get MONAI transform pipeline for validation (eval mode).

    Returns:
        MONAI Compose transform
    """
    return mt.Compose(
        [
            mt.EnsureTyped(keys=["image", "label"]),
        ]
    )
