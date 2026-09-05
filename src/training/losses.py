"""
Loss functions for 3D Brain Tumor Segmentation.
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from monai.losses import DiceCELoss, DiceLoss, FocalLoss


def get_loss_function(
    loss_name: str = "dice_ce",
    include_background: bool = False,  # [FIX - Bước 1] Mặc định tắt background
    to_onehot_y: bool = True,
    softmax: bool = True,
    squared_pred: bool = True,
    smooth_nr: float = 1e-5,
    smooth_dr: float = 1e-5,
) -> nn.Module:
    """
    Factory function to get loss function instance.

    Args:
        loss_name: "dice_ce", "dice", or "focal"
        include_background: Whether to include background channel in loss
        to_onehot_y: Convert targets to one-hot encoding automatically
        softmax: Apply softmax to predicted logits
        squared_pred: Square predictions in denominator for Dice
        smooth_nr: Numerator smoothing factor
        smooth_dr: Denominator smoothing factor

    Returns:
        PyTorch Loss Module
    """
    name = loss_name.lower().strip()

    if name in ["dice_ce", "diceceloss"]:
        return DiceCELoss(
            # [FIX - Bước 1] include_background=False: background chiếm ~90%
            # voxels, nếu đưa vào loss nó áp đảo gradient, mô hình thiên về
            # dự đoán nền thay vì tập trung học biên giới u nhỏ (đặc biệt ET).
            # Chuẩn nnU-Net và các đội top BraTS đều tắt background loss.
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            softmax=softmax,
            squared_pred=squared_pred,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
        )
    elif name in ["dice", "diceloss"]:
        return DiceLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            softmax=softmax,
            squared_pred=squared_pred,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
        )
    elif name in ["focal", "focalloss"]:
        return FocalLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
        )
    else:
        raise ValueError(f"Unsupported loss function '{loss_name}'. Use 'dice_ce', 'dice', or 'focal'.")
