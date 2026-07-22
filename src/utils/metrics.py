"""
Utility functions for metrics, evaluation, and visualization.
Supports BraTS tumor regions: Whole Tumor (WT), Tumor Core (TC), Enhancing Tumor (ET).
"""

import numpy as np
import torch
from typing import Tuple, Union, Dict
from scipy.ndimage import distance_transform_edt


def dice_score(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    smooth: float = 1e-5,
) -> float:
    """Calculate Dice coefficient."""
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()

    pred = pred.astype(bool).flatten()
    target = target.astype(bool).flatten()

    intersection = np.sum(pred * target)
    dice = (2 * intersection + smooth) / (np.sum(pred) + np.sum(target) + smooth)

    return float(dice)


def iou_score(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    smooth: float = 1e-5,
) -> float:
    """Calculate Intersection over Union (IoU)."""
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()

    pred = pred.astype(bool).flatten()
    target = target.astype(bool).flatten()

    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target) - intersection
    iou = (intersection + smooth) / (union + smooth)

    return float(iou)


def precision_score(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    smooth: float = 1e-5,
) -> float:
    """Calculate Precision."""
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()

    pred = pred.astype(bool).flatten()
    target = target.astype(bool).flatten()

    true_positives = np.sum(pred * target)
    false_positives = np.sum(pred * ~target)
    precision = (true_positives + smooth) / (true_positives + false_positives + smooth)

    return float(precision)


def recall_score(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    smooth: float = 1e-5,
) -> float:
    """Calculate Recall (Sensitivity)."""
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()

    pred = pred.astype(bool).flatten()
    target = target.astype(bool).flatten()

    true_positives = np.sum(pred * target)
    false_negatives = np.sum(~pred * target)
    recall = (true_positives + smooth) / (true_positives + false_negatives + smooth)

    return float(recall)


def hausdorff_distance_95(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """Calculate 95th percentile Hausdorff Distance in mm."""
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()

    pred = pred.astype(bool)
    target = target.astype(bool)

    if not np.any(pred) and not np.any(target):
        return 0.0
    if not np.any(pred) or not np.any(target):
        return 373.0  # Standard max penalty distance in BraTS challenge

    dist_pred = distance_transform_edt(~pred, sampling=spacing)
    dist_target = distance_transform_edt(~target, sampling=spacing)

    surface_dist = np.concatenate([dist_pred[target], dist_target[pred]])
    hd95 = np.percentile(surface_dist, 95)

    return float(hd95)


def compute_metrics(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Dict[str, float]:
    """Compute global segmentation metrics."""
    return {
        "dice": dice_score(pred, target),
        "iou": iou_score(pred, target),
        "precision": precision_score(pred, target),
        "recall": recall_score(pred, target),
        "hd95": hausdorff_distance_95(pred, target, spacing),
    }


def compute_brats_metrics(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Dict[str, float]:
    """
    Compute BraTS sub-region metrics (Whole Tumor, Tumor Core, Enhancing Tumor).

    BraTS Regions:
    - WT (Whole Tumor): labels > 0
    - TC (Tumor Core): labels 1 or 3
    - ET (Enhancing Tumor): label 3
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()

    # Whole Tumor (WT)
    pred_wt = pred > 0
    target_wt = target > 0

    # Tumor Core (TC): NCR/NET (1) + ET (3)
    pred_tc = np.isin(pred, [1, 3])
    target_tc = np.isin(target, [1, 3])

    # Enhancing Tumor (ET): ET (3)
    pred_et = pred == 3
    target_et = target == 3

    metrics = {
        "dice_wt": dice_score(pred_wt, target_wt),
        "dice_tc": dice_score(pred_tc, target_tc),
        "dice_et": dice_score(pred_et, target_et),
        "hd95_wt": hausdorff_distance_95(pred_wt, target_wt, spacing),
        "hd95_tc": hausdorff_distance_95(pred_tc, target_tc, spacing),
        "hd95_et": hausdorff_distance_95(pred_et, target_et, spacing),
        "dice_mean": (
            dice_score(pred_wt, target_wt)
            + dice_score(pred_tc, target_tc)
            + dice_score(pred_et, target_et)
        ) / 3.0,
    }

    return metrics


def calculate_tumor_volume(
    seg_mask: Union[np.ndarray, torch.Tensor],
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """Calculate tumor volume in cm³."""
    if isinstance(seg_mask, torch.Tensor):
        seg_mask = seg_mask.cpu().numpy()

    num_voxels = np.sum(seg_mask > 0)
    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
    volume_mm3 = num_voxels * voxel_volume_mm3
    return float(volume_mm3 / 1000.0)
