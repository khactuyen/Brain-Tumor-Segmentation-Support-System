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
            # [FIX - Bước 2] Tăng pos=3.0 để 75% patch luôn cắt trúng vùng u.
            # Mặc định pos=1.0, neg=1.0 nghĩa là 50% patch là nền não bình
            # thường — lãng phí bước lặp và giảm tốc độ học đặc trưng u nhỏ.
            # pos=3.0, neg=1.0 => xác suất chọn patch foreground = 3/(3+1) = 75%.
            mt.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=patch_size,
                pos=3.0,
                neg=1.0,
                num_samples=num_samples,
                image_key="image",
                image_threshold=0,
            ),
            # Spatial augmentations
            mt.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            mt.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            mt.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            mt.RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            # Intensity augmentations cơ bản
            mt.RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
            mt.RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            # [NEW - Bước 2] Augmentation y tế chuyên sâu (mimí pipeline nnU-Net):
            # Gaussian Noise: mô phỏng nhiễu thuật số của máy MRI.
            mt.RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.1),
            # Gaussian Smooth: mô phỏng mức độ nét mờ khác nhau giữa các ca chụp.
            mt.RandGaussianSmoothd(
                keys=["image"],
                prob=0.15,
                sigma_x=(0.5, 1.0),
                sigma_y=(0.5, 1.0),
                sigma_z=(0.5, 1.0),
            ),
            # Contrast Adjust: mô phỏng sự khác biệt về độ tương phản giữa các
            # máy chụp khác nhau (field strength, coil type).
            mt.RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.5)),
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
