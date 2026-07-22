"""
Abstract base model interface for 3D Segmentation models.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Union, Optional


class BaseSegmentationModel(nn.Module, ABC):
    """Abstract Base Class for 3D Medical Image Segmentation Models."""

    def __init__(self, in_channels: int = 4, out_channels: int = 4):
        """
        Initialize base model.

        Args:
            in_channels: Number of input MRI modalities (default: 4 - T1n, T1c, T2w, T2f)
            out_channels: Number of output segmentation classes/channels
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        pass

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform inference/prediction on input tensor.

        Args:
            x: Input tensor (B, C, H, W, D)

        Returns:
            Logits or Softmax probabilities tensor
        """
        self.eval()
        return self.forward(x)

    def load_checkpoint(self, checkpoint_path: Union[str, Path], device: str = "cpu") -> Dict[str, Any]:
        """
        Load model weights from a checkpoint file.

        Args:
            checkpoint_path: Path to .pth/.pt file
            device: Map location ("cpu" or "cuda")

        Returns:
            Checkpoint dict containing metadata
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        checkpoint = torch.load(path, map_location=device)

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            self.load_state_dict(checkpoint["state_dict"])
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.load_state_dict(checkpoint)

        return checkpoint if isinstance(checkpoint, dict) else {}

    def save_checkpoint(
        self,
        output_path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Path:
        """
        Save model weights and training state.

        Args:
            output_path: Destination path
            optimizer: Optional optimizer state
            epoch: Optional current epoch
            metrics: Optional evaluation metrics

        Returns:
            Saved file path
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "state_dict": self.state_dict(),
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "model_name": self.__class__.__name__,
        }

        if optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()
        if epoch is not None:
            state["epoch"] = epoch
        if metrics is not None:
            state["metrics"] = metrics

        torch.save(state, path)
        return path

    def num_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata."""
        return {
            "model_name": self.__class__.__name__,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "num_parameters": self.num_parameters(),
        }
