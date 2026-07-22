"""
Models package for Brain Tumor Segmentation Support System.
"""

from typing import Union, Dict, Any
from .base_model import BaseSegmentationModel
from .unet3d import UNet3DModel
from .swin_unetr import SwinUNETRModel

MODEL_REGISTRY = {
    "unet": UNet3DModel,
    "unet3d": UNet3DModel,
    "swin_unetr": SwinUNETRModel,
    "swin": SwinUNETRModel,
}


def build_model(
    model_name: str,
    in_channels: int = 4,
    out_channels: int = 4,
    **kwargs: Any
) -> BaseSegmentationModel:
    """
    Factory function to build a segmentation model by name.

    Args:
        model_name: "unet3d" or "swin_unetr"
        in_channels: Number of input channels
        out_channels: Number of output channels
        **kwargs: Model specific arguments

    Returns:
        Instance of BaseSegmentationModel
    """
    key = model_name.lower().strip()
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model name '{model_name}'. Available options: {list(MODEL_REGISTRY.keys())}"
        )

    model_cls = MODEL_REGISTRY[key]
    return model_cls(in_channels=in_channels, out_channels=out_channels, **kwargs)


__all__ = [
    "BaseSegmentationModel",
    "UNet3DModel",
    "SwinUNETRModel",
    "build_model",
    "MODEL_REGISTRY",
]
