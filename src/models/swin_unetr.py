"""
Swin UNETR 3D Transformer Model implementation using MONAI.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from monai.networks.nets import SwinUNETR

from .base_model import BaseSegmentationModel


class SwinUNETRModel(BaseSegmentationModel):
    """Swin UNETR Architecture for 3D Brain Tumor Segmentation."""

    def __init__(
        self,
        img_size: Tuple[int, int, int] = (128, 128, 128),
        in_channels: int = 4,
        out_channels: int = 4,
        feature_size: int = 48,
        use_checkpoint: bool = True,
        spatial_dims: int = 3,
        dropout_path_rate: float = 0.0,
    ):
        """
        Initialize Swin UNETR model.

        Args:
            img_size: Spatial dimensions of input patch (H, W, D)
            in_channels: Input channels (4 modalities)
            out_channels: Output classes (4 classes)
            feature_size: Embedding feature dimension
            use_checkpoint: Use activation checkpointing to save GPU memory
            spatial_dims: Spatial dimension (3 for 3D volumes)
            dropout_path_rate: Drop path rate for transformer blocks
        """
        super().__init__(in_channels=in_channels, out_channels=out_channels)

        self.img_size = img_size
        self.feature_size = feature_size

        self.net = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=dropout_path_rate,
            use_v2=False,
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
