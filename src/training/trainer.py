"""
Trainer class for 3D Brain Tumor Segmentation models.
"""

import time
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, Union
from torch.utils.data import DataLoader
from monai.inferers import sliding_window_inference

from ..models.base_model import BaseSegmentationModel
from ..utils.logger import logger
from ..utils.metrics import dice_score
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
        roi_size: tuple = (128, 128, 128),
        sw_batch_size: int = 4,
        sw_overlap: float = 0.5,
    ):
        """
        Initialize trainer.

        Args:
            model: Instance of BaseSegmentationModel
            train_loader: DataLoader for training set
            val_loader: Optional DataLoader for validation set
            optimizer: PyTorch optimizer (default: AdamW with lr=2e-4)
            loss_fn: PyTorch loss module (default: DiceCELoss)
            lr_scheduler: Learning rate scheduler
            device: Training device ("cuda" or "cpu")
            use_amp: Whether to use Automatic Mixed Precision
            checkpoints_dir: Directory to save model checkpoints
            model_name: Base name for saved checkpoints
            roi_size: [MỚI] Kích thước cửa sổ trượt khi validate (mặc định
                128³, khớp với patch_size dùng lúc train để mô hình luôn
                thấy input cùng kích thước đã học).
            sw_batch_size: [MỚI] Số cửa sổ trượt xử lý cùng lúc trên GPU.
            sw_overlap: [MỚI] Độ chồng lấn giữa các cửa sổ trượt (0-1).
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_amp = use_amp and (device == "cuda")
        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.sw_overlap = sw_overlap

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
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_dice": [],
        }

    def train_epoch(self, epoch: int) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            # Format inputs
            if isinstance(batch, (list, tuple)):
                images, labels = batch[0], batch[1]
            elif isinstance(batch, dict):
                images, labels = batch["image"], batch["label"]
            else:
                raise ValueError("Unsupported batch format")

            images = images.to(self.device)
            labels = labels.to(self.device)

            # [FIX] Bỏ đoạn crop tay 128³ ở đây — giờ việc lấy patch đã được
            # RandCropByPosNegLabeld (MONAI) làm sẵn ngay trong DataLoader
            # (xem get_train_transforms trong training/augmentation.py),
            # ưu tiên lấy patch chứa vùng khối u thay vì crop ngẫu nhiên đều
            # như trước — giống cách đồng nghiệp bạn làm trong notebook.
            # images/labels ở đây đã đúng kích thước patch, forward thẳng.

            self.optimizer.zero_grad()

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

        avg_loss = total_loss / max(1, len(self.train_loader))
        elapsed = time.time() - start_time
        logger.info(f"Epoch [{epoch:03d}] Train Loss: {avg_loss:.4f} ({elapsed:.1f}s)")
        return avg_loss

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Run validation loop."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        dice_scores = []

        for batch in self.val_loader:
            if isinstance(batch, (list, tuple)):
                images, labels = batch[0], batch[1]
            elif isinstance(batch, dict):
                images, labels = batch["image"], batch["label"]
            else:
                continue

            images = images.to(self.device)
            labels = labels.to(self.device)

            # [FIX] Trước đây center-crop 128³ khi validate — nghĩa là chỉ
            # đánh giá đúng 1 vùng nhỏ ở giữa ảnh, bỏ sót phần còn lại của
            # não. Giờ dùng sliding_window_inference (MONAI) giống đồng
            # nghiệp: quét toàn bộ ảnh gốc bằng cửa sổ 128³ trượt có chồng
            # lấn, ghép kết quả lại đủ full volume — vừa không tràn VRAM
            # (mỗi lần chỉ xử lý 1 cửa sổ), vừa đánh giá đúng trên ảnh đầy
            # đủ như lúc inference thực tế.
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = sliding_window_inference(
                    inputs=images,
                    roi_size=self.roi_size,
                    sw_batch_size=self.sw_batch_size,
                    predictor=self.model,
                    overlap=self.sw_overlap,
                )
                loss = self.loss_fn(outputs, labels)

            total_loss += loss.item()

            # [FIX] Trước đây threshold softmax >0.5 theo từng kênh riêng lẻ
            # rồi mới đưa vào dice_score để argmax lại — sai logic cho bài
            # toán multi-class loại trừ lẫn nhau. Lấy argmax trực tiếp trên
            # logits để ra đúng lớp dự đoán cho từng voxel.
            preds = torch.argmax(outputs, dim=1)
            val_dice = dice_score(preds, labels)
            dice_scores.append(val_dice)

        avg_loss = total_loss / max(1, len(self.val_loader))
        avg_dice = float(sum(dice_scores) / max(1, len(dice_scores)))

        logger.info(f"Epoch [{epoch:03d}] Val Loss: {avg_loss:.4f} | Val Dice: {avg_dice:.4f}")
        return {"val_loss": avg_loss, "val_dice": avg_dice}

    def resume_from_checkpoint(self, checkpoint_path: Union[str, Path]) -> int:
        """
        [MỚI] Nạp lại model + optimizer + best_val_dice từ checkpoint đã lưu
        để train tiếp mà không mất tiến độ (dùng khi Colab hết GPU/session).

        Args:
            checkpoint_path: đường dẫn tới file .pth (vd: unet_brats_best.pth)

        Returns:
            epoch đã lưu trong checkpoint — vòng lặp sẽ train tiếp từ epoch+1
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint_path}")

        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["state_dict"])

        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "metrics" in checkpoint and "val_dice" in checkpoint["metrics"]:
            self.best_val_dice = checkpoint["metrics"]["val_dice"]

        start_epoch = checkpoint.get("epoch", 0)
        logger.info(
            f"✅ Resumed from checkpoint '{path.name}' — epoch {start_epoch}, "
            f"best_val_dice={self.best_val_dice:.4f}. Sẽ train tiếp từ epoch {start_epoch + 1}."
        )
        return start_epoch

    def train(
        self,
        max_epochs: int = 100,
        val_interval: int = 1,
        start_epoch: int = 0,
    ) -> Dict[str, Any]:
        """
        Run full training loop.

        Args:
            max_epochs: Total epochs to train (mục tiêu cuối cùng, vd 50-100)
            val_interval: Perform validation every N epochs
            start_epoch: [MỚI] epoch bắt đầu train tiếp (0 nếu train từ đầu,
                hoặc = epoch đã lưu trong checkpoint nếu resume)

        Returns:
            Dictionary containing training history
        """
        logger.info(
            f"Starting training for {self.model_name} on {self.device} (AMP={self.use_amp}) "
            f"| epoch {start_epoch + 1} -> {max_epochs}"
        )

        for epoch in range(start_epoch + 1, max_epochs + 1):
            train_loss = self.train_epoch(epoch)
            self.history["train_loss"].append(train_loss)

            if epoch % val_interval == 0 and self.val_loader is not None:
                val_metrics = self.validate(epoch)
                val_loss = val_metrics.get("val_loss", 0.0)
                val_dice = val_metrics.get("val_dice", 0.0)

                self.history["val_loss"].append(val_loss)
                self.history["val_dice"].append(val_dice)

                # Save best checkpoint
                if val_dice > self.best_val_dice:
                    self.best_val_dice = val_dice
                    best_path = self.checkpoints_dir / f"{self.model_name}_best.pth"
                    self.model.save_checkpoint(
                        best_path,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        metrics={"val_dice": val_dice, "val_loss": val_loss},
                    )
                    logger.info(f"⭐ New Best Model Saved! Val Dice: {val_dice:.4f} -> {best_path}")

            # [MỚI] Luôn lưu checkpoint "latest" sau MỖI epoch (khác với "best"),
            # để nếu Colab ngắt kết nối giữa chừng, bạn chỉ mất tối đa 1 epoch
            # thay vì mất hết đến lần validate cuối. Dùng file này để resume.
            latest_path = self.checkpoints_dir / f"{self.model_name}_latest.pth"
            self.model.save_checkpoint(
                latest_path,
                optimizer=self.optimizer,
                epoch=epoch,
                metrics={"val_dice": self.best_val_dice},
            )

            # Update scheduler
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # Save final checkpoint
        final_path = self.checkpoints_dir / f"{self.model_name}_final.pth"
        self.model.save_checkpoint(final_path, optimizer=self.optimizer, epoch=max_epochs)
        logger.info(f"Training completed! Final model saved to {final_path}")

        return self.history
