"""
Preprocessing module for MRI data normalization and spatial transforms.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional


class MRIPreprocessor:
    """Preprocess MRI data: normalize intensity, resize, crop."""

    def __init__(
        self,
        target_shape: Tuple[int, int, int] = (240, 240, 160),
        normalize_method: str = "z_score",
    ):
        """
        Initialize preprocessor.

        Args:
            target_shape: Target spatial dimensions (H, W, D) for model input
            normalize_method: "z_score", "min_max", or "none"
        """
        self.target_shape = target_shape
        self.normalize_method = normalize_method

    def normalize_intensity(
        self, image: np.ndarray, method: str = None
    ) -> np.ndarray:
        """
        Normalize intensity values.

        Args:
            image: Input image (can be multi-channel)
            method: Normalization method ("z_score", "min_max", "none")

        Returns:
            Normalized image
        """
        if method is None:
            method = self.normalize_method

        if method == "none":
            return image

        # Handle multi-channel images
        if image.ndim == 4:  # (C, H, W, D)
            normalized = np.zeros_like(image)
            for c in range(image.shape[0]):
                normalized[c] = self._normalize_single_channel(
                    image[c], method
                )
            return normalized
        else:
            return self._normalize_single_channel(image, method)

    def _normalize_single_channel(
        self, image: np.ndarray, method: str
    ) -> np.ndarray:
        """Normalize a single channel."""
        if method == "z_score":
            mean = np.mean(image)
            std = np.std(image)
            if std > 0:
                return (image - mean) / std
            else:
                return image - mean

        elif method == "min_max":
            img_min = np.min(image)
            img_max = np.max(image)
            if img_max > img_min:
                return (image - img_min) / (img_max - img_min)
            else:
                return np.zeros_like(image)

        else:
            return image

    def crop_or_pad(
        self, image: np.ndarray, target_shape: Tuple[int, int, int] = None
    ) -> np.ndarray:
        """
        Crop or pad image to target shape.

        Args:
            image: Input image (H, W, D) or (C, H, W, D)
            target_shape: Target (H, W, D)

        Returns:
            Image cropped/padded to target shape
        """
        if target_shape is None:
            target_shape = self.target_shape

        if image.ndim == 4:  # Multi-channel
            result = np.zeros(
                (image.shape[0],) + target_shape, dtype=image.dtype
            )
            for c in range(image.shape[0]):
                result[c] = self._crop_or_pad_single(image[c], target_shape)
            return result
        else:
            return self._crop_or_pad_single(image, target_shape)

    def _crop_or_pad_single(
        self, image: np.ndarray, target_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """Crop or pad single channel image."""
        current_shape = image.shape
        target_h, target_w, target_d = target_shape

        result = np.zeros(target_shape, dtype=image.dtype)

        # Calculate crop/pad ranges
        h_start = max(0, (current_shape[0] - target_h) // 2)
        w_start = max(0, (current_shape[1] - target_w) // 2)
        d_start = max(0, (current_shape[2] - target_d) // 2)

        h_end = min(current_shape[0], h_start + target_h)
        w_end = min(current_shape[1], w_start + target_w)
        d_end = min(current_shape[2], d_start + target_d)

        h_crop = h_end - h_start
        w_crop = w_end - w_start
        d_crop = d_end - d_start

        result[:h_crop, :w_crop, :d_crop] = image[h_start:h_end, w_start:w_end, d_start:d_end]

        return result

    def preprocess(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        normalize: bool = True,
        crop_pad: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Full preprocessing pipeline.

        Args:
            image: Input image (C, H, W, D) or (H, W, D)
            mask: Optional segmentation mask
            normalize: Whether to normalize intensity
            crop_pad: Whether to crop/pad to target shape

        Returns:
            (processed_image, processed_mask)
        """
        # Normalize intensity
        if normalize:
            image = self.normalize_intensity(image)

        # Crop/pad
        if crop_pad:
            image = self.crop_or_pad(image)
            if mask is not None:
                mask = self.crop_or_pad(mask)

        return image, mask

    def to_tensor(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Convert numpy arrays to PyTorch tensors.

        Args:
            image: Numpy array
            mask: Optional mask array
            device: "cpu" or "cuda"

        Returns:
            (image_tensor, mask_tensor)
        """
        image_tensor = torch.from_numpy(image).float().to(device)

        mask_tensor = None
        if mask is not None:
            mask_tensor = torch.from_numpy(mask).float().to(device)

        return image_tensor, mask_tensor
