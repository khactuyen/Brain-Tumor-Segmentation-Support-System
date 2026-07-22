"""
3D U-Net Model implementation using MONAI.
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional
from monai.networks.nets import UNet

from .base_model import BaseSegmentationModel


class UNet3DModel(BaseSegmentationModel):
    """3D U-Net Architecture for Brain Tumor Segmentation."""

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
