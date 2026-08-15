"""
Trainer class for 3D Brain Tumor Segmentation models.

Supports:
 - AMP (Automatic Mixed Precision) for GPU memory savings
 - BraTS-standard metrics: Dice (WT, TC, ET), HD95
 - Sliding Window Inference for full-volume validation
 - Learning rate scheduler
 - Best checkpoint saving based on mean Val Dice
"""

import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from torch.utils.data import DataLoader

from monai.inferers import sliding_window_inference

from ..models.base_model import BaseSegmentationModel
from ..utils.logger import logger
from ..utils.metrics import compute_brats_metrics
from .losses import get_loss_function


class SegmentationTrainer:
    """Trainer pipeline for training 3D Medical Segmentation Models."""

    def __init__(
        self,
        model: BaseSegmentationModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_fn: Optional[nn.Module] = None,
        lr_scheduler: Optional[Any] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_amp: bool = True,
        checkpoints_dir: Union[str, Path] = "checkpoints",
        model_name: str = "model",
        # Sliding Window Inference config
        sw_roi_size: tuple = (128, 128, 128),
        sw_batch_size: int = 4,
        sw_overlap: float = 0.25,
        # Steps per epoch limit (0 = no limit)
        steps_per_epoch: int = 0,
    ):
        """
        Initialize trainer.

        Args:
            model: Instance of BaseSegmentationModel
            train_loader: DataLoader for training set
            val_loader: Optional DataLoader for validation set
            optimizer: PyTorch optimizer (default: AdamW lr=2e-4)
            loss_fn: PyTorch loss module (default: DiceCELoss)
            lr_scheduler: Learning rate scheduler
            device: Training device ("cuda" or "cpu")
            use_amp: Whether to use Automatic Mixed Precision
            checkpoints_dir: Directory to save model checkpoints
            model_name: Base name for saved checkpoints
            sw_roi_size: Patch size for sliding window inference
            sw_batch_size: Batch size for sliding window inference
            sw_overlap: Patch overlap ratio for sliding window inference
            steps_per_epoch: Max steps per epoch (0 = unlimited)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_amp = use_amp and (device == "cuda")
        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.sw_roi_size = sw_roi_size
        self.sw_batch_size = sw_batch_size
        self.sw_overlap = sw_overlap
        self.steps_per_epoch = steps_per_epoch

        # Default Optimizer
        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(), lr=2e-4, weight_decay=1e-4
        )

        # Default Loss Function
        self.loss_fn = loss_fn or get_loss_function("dice_ce")

        # Scheduler
        self.lr_scheduler = lr_scheduler

        # AMP GradScaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Tracking metrics
        self.best_val_dice = -1.0
        self.history: Dict[str, List] = {
            "train_loss": [],
            "val_loss": [],
            "val_dice": [],
            "val_dice_wt": [],
            "val_dice_tc": [],
            "val_dice_et": [],
        }

    # ─── Training Epoch ──────────────────────────────────────────────────────
    def train_epoch(self, epoch: int) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        step_count = 0
        start_time = time.time()

        for batch_idx, batch_data in enumerate(self.train_loader):
            # Step limit (for fast experimentation on Colab)
            if self.steps_per_epoch > 0 and step_count >= self.steps_per_epoch:
                break

            # Handle both dict batches and list of patches (RandCropByPosNegLabeld)
            if isinstance(batch_data, list):
                images = torch.cat([b["image"] for b in batch_data], dim=0).to(self.device)
                labels = torch.cat([b["label"] for b in batch_data], dim=0).to(self.device)
            elif isinstance(batch_data, dict):
                images = batch_data["image"].to(self.device)
                labels = batch_data["label"].to(self.device)
                # Flatten num_samples dimension if present (B, N, C, H, W, D)
                if images.ndim == 6:
                    B, N, C, H, W, D = images.shape
                    images = images.view(B * N, C, H, W, D)
                    labels = labels.view(B * N, 1, H, W, D)
            elif isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                images, labels = batch_data[0].to(self.device), batch_data[1].to(self.device)
            else:
                raise ValueError(f"Unsupported batch format: {type(batch_data)}")

            self.optimizer.zero_grad()

            # Forward pass with AMP
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)

            # Backward pass & step
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            step_count += 1

        avg_loss = total_loss / max(1, step_count)
        elapsed = time.time() - start_time
        logger.info(
            f"Epoch [{epoch:03d}] Train Loss: {avg_loss:.4f} "
            f"({step_count} steps, {elapsed:.1f}s)"
        )
        return avg_loss

    # ─── Validation ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Run validation loop with BraTS-standard Dice metrics.

        Uses Sliding Window Inference for full-volume prediction, avoiding
        out-of-memory errors on 240×240×155 MRI volumes.
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        all_dice_wt, all_dice_tc, all_dice_et = [], [], []

        for batch_data in self.val_loader:
            # Parse batch
            if isinstance(batch_data, dict):
                images = batch_data["image"].to(self.device)
                labels = batch_data["label"].to(self.device)
            elif isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                images, labels = batch_data[0].to(self.device), batch_data[1].to(self.device)
            else:
                continue

            # Sliding Window Inference on full-volume
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = sliding_window_inference(
                    inputs=images,
                    roi_size=self.sw_roi_size,
                    sw_batch_size=self.sw_batch_size,
                    predictor=self.model,
                    overlap=self.sw_overlap,
                    mode="gaussian",
                )
                loss = self.loss_fn(outputs, labels)

            total_loss += loss.item()

            # Compute BraTS Dice metrics
            pred_label = torch.argmax(outputs, dim=1, keepdim=True).cpu().numpy()
            true_label = labels.cpu().numpy()

            for b in range(pred_label.shape[0]):
                m = compute_brats_metrics(pred_label[b, 0], true_label[b, 0])
                all_dice_wt.append(m["dice_wt"])
                all_dice_tc.append(m["dice_tc"])
                all_dice_et.append(m["dice_et"])

        avg_loss = total_loss / max(1, len(self.val_loader))
        mean_wt = float(np.mean(all_dice_wt)) if all_dice_wt else 0.0
        mean_tc = float(np.mean(all_dice_tc)) if all_dice_tc else 0.0
        mean_et = float(np.mean(all_dice_et)) if all_dice_et else 0.0
        mean_dice = (mean_wt + mean_tc + mean_et) / 3.0

        logger.info(
            f"Epoch [{epoch:03d}] Val Loss: {avg_loss:.4f} | "
            f"Mean Dice: {mean_dice:.4f} "
            f"[WT: {mean_wt:.4f}, TC: {mean_tc:.4f}, ET: {mean_et:.4f}]"
        )
        return {
            "val_loss": avg_loss,
            "val_dice": mean_dice,
            "val_dice_wt": mean_wt,
            "val_dice_tc": mean_tc,
            "val_dice_et": mean_et,
        }

    # ─── Full Training Loop ───────────────────────────────────────────────────
    def train(self, max_epochs: int = 100, val_interval: int = 2) -> Dict[str, Any]:
        """
        Run full training loop.

        Args:
            max_epochs: Total epochs to train
            val_interval: Perform validation every N epochs

        Returns:
            Dictionary containing training history
        """
        logger.info(
            f"Starting training [{self.model_name}] "
            f"on {self.device} | AMP={self.use_amp} | "
            f"Epochs={max_epochs} | Val every {val_interval} epochs"
        )

        for epoch in range(1, max_epochs + 1):
            train_loss = self.train_epoch(epoch)
            self.history["train_loss"].append(train_loss)

            if epoch % val_interval == 0 and self.val_loader is not None:
                val_metrics = self.validate(epoch)

                self.history["val_loss"].append(val_metrics.get("val_loss", 0.0))
                self.history["val_dice"].append(val_metrics.get("val_dice", 0.0))
                self.history["val_dice_wt"].append(val_metrics.get("val_dice_wt", 0.0))
                self.history["val_dice_tc"].append(val_metrics.get("val_dice_tc", 0.0))
                self.history["val_dice_et"].append(val_metrics.get("val_dice_et", 0.0))

                # Save best checkpoint
                val_dice = val_metrics.get("val_dice", 0.0)
                if val_dice > self.best_val_dice:
                    self.best_val_dice = val_dice
                    best_path = self.checkpoints_dir / f"{self.model_name}_best.pth"
                    self.model.save_checkpoint(
                        best_path,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        metrics=val_metrics,
                    )
                    logger.info(
                        f"⭐ New Best Model Saved! "
                        f"Val Dice: {val_dice:.4f} → {best_path}"
                    )

            # Update learning rate scheduler
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # Save final checkpoint
        final_path = self.checkpoints_dir / f"{self.model_name}_final.pth"
        self.model.save_checkpoint(final_path, optimizer=self.optimizer, epoch=max_epochs)
        logger.info(f"Training complete! Final model saved → {final_path}")

        return self.history
