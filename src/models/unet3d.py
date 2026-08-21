"""
3D U-Net Model implementation using MONAI.

Architecture:
    Classic encoder-decoder architecture with skip connections.
    Uses residual blocks with Instance Normalization, designed for
    efficient local feature extraction in volumetric medical images.

Reference:
    Çiçek et al. (2016). "3D U-Net: Learning Dense Volumetric
    Segmentation from Sparse Annotation." MICCAI 2016.
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Dict, Any
from monai.networks.nets import UNet

from .base_model import BaseSegmentationModel


class UNet3DModel(BaseSegmentationModel):
    """
    3D U-Net Architecture for Brain Tumor Segmentation.

    Key features:
    - Residual convolution blocks with Instance Normalization
    - Efficient local feature extraction via hierarchical downsampling
    - Skip connections preserve fine-grained spatial information
    - Lightweight compared to Transformer-based models (~2.5M params)
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        channels: Tuple[int, ...] = (16, 32, 64, 128, 256),
        strides: Tuple[int, ...] = (2, 2, 2, 2),
        num_res_units: int = 2,
        dropout: float = 0.2,
    ):
        """
        Initialize 3D U-Net model.

        Args:
            in_channels: Input channels (default: 4 for T1n, T1c, T2w, T2f)
            out_channels: Output classes (default: 4)
            channels: Feature channels per layer
            strides: Downsampling strides per layer
            num_res_units: Residual units per layer
            dropout: Dropout probability
        """
        super().__init__(in_channels=in_channels, out_channels=out_channels)

        self.channels = channels
        self.strides = strides

        self.net = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            dropout=dropout,
            norm="instance",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (B, in_channels, H, W, D)

        Returns:
            Output logits of shape (B, out_channels, H, W, D)
        """
        return self.net(x)

    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata including architecture specifics."""
        info = super().get_model_info()
        info.update(
            {
                "architecture": "3D U-Net",
                "channels": list(self.channels),
                "strides": list(self.strides),
            }
        )
        return info
