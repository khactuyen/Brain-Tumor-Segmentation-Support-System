"""
Inference Engine for 3D Brain Tumor Segmentation models.
Provides Sliding Window Inference and post-processing.
"""

import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
from monai.inferers import sliding_window_inference

from ..models.base_model import BaseSegmentationModel
from .postprocessor import MaskPostprocessor
from .prediction_result import PredictionResult
from ..utils.metrics import compute_metrics, calculate_tumor_volume
from ..utils.logger import logger


class InferenceEngine:
    """Engine for performing inference with 3D Segmentation models."""

    def __init__(
        self,
        model: BaseSegmentationModel,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        roi_size: Tuple[int, int, int] = (128, 128, 128),
        sw_batch_size: int = 4,
        overlap: float = 0.5,
        postprocessor: Optional[MaskPostprocessor] = None,
    ):
        """
        Initialize InferenceEngine.

        Args:
            model: Instance of BaseSegmentationModel
            device: Execution device ("cuda" or "cpu")
            roi_size: Sliding window spatial patch size
            sw_batch_size: Batch size during sliding window
            overlap: Spatial overlap ratio between adjacent windows (0.0 to 1.0)
            postprocessor: Optional MaskPostprocessor instance
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.postprocessor = postprocessor or MaskPostprocessor()

    @torch.no_grad()
    def predict_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Perform sliding window inference on a 3D MRI tensor.

        Args:
            image_tensor: Shape (1, 4, H, W, D) or (4, H, W, D)

        Returns:
            Logits tensor of shape (1, num_classes, H, W, D)
        """
        if image_tensor.ndim == 4:
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dim -> (1, C, H, W, D)

        image_tensor = image_tensor.to(self.device)

        logits = sliding_window_inference(
            inputs=image_tensor,
            roi_size=self.roi_size,
            sw_batch_size=self.sw_batch_size,
            predictor=self.model,
            overlap=self.overlap,
            mode="gaussian",
        )

        return logits

    def run_case(
        self,
        case_id: str,
        image_tensor: torch.Tensor,
        ground_truth: Optional[torch.Tensor] = None,
        affine: Optional[np.ndarray] = None,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        apply_postprocess: bool = True,
    ) -> PredictionResult:
        """
        Run full inference pipeline on a case.

        Args:
            case_id: Identifier for case
            image_tensor: Tensor of shape (4, H, W, D) or (1, 4, H, W, D)
            ground_truth: Optional ground truth mask tensor (1, H, W, D)
            affine: 4x4 affine matrix
            spacing: Voxel spacing (mm)
            apply_postprocess: Whether to apply mask postprocessing

        Returns:
            PredictionResult object
        """
        logger.info(f"Running inference for case {case_id} using {self.model.__class__.__name__}...")
        start_time = time.time()

        # Sliding Window Inference
        logits = self.predict_tensor(image_tensor)

        # Convert logits to class labels (Argmax over channel dim)
        probs = torch.softmax(logits, dim=1)
        pred_mask_tensor = torch.argmax(probs, dim=1).squeeze(0)  # (H, W, D)
        pred_mask = pred_mask_tensor.cpu().numpy().astype(np.uint8)

        # Postprocessing
        if apply_postprocess:
            pred_mask = self.postprocessor.process(pred_mask)

        inference_time = time.time() - start_time
        logger.info(f"Inference completed in {inference_time:.2f}s")

        # Calculate Volume
        tumor_volume = calculate_tumor_volume(pred_mask, spacing=spacing)

        # Calculate Metrics if Ground Truth is provided
        metrics = {}
        if ground_truth is not None:
            if isinstance(ground_truth, torch.Tensor):
                gt_np = ground_truth.squeeze().cpu().numpy().astype(np.uint8)
            else:
                gt_np = ground_truth.astype(np.uint8)

            metrics = compute_metrics(pred_mask, gt_np, spacing=spacing)

        model_name = self.model.__class__.__name__.replace("Model", "")

        return PredictionResult(
            case_id=case_id,
            model_name=model_name,
            mask=pred_mask,
            inference_time=inference_time,
            metrics=metrics,
            tumor_volume_cm3=tumor_volume,
            affine=affine,
            metadata={"roi_size": self.roi_size, "overlap": self.overlap},
        )
