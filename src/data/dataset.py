"""
PyTorch Dataset class for BraTS data with train/val/test split.
Supports both fast preprocessed .npz cache loading and raw NIfTI loading with fallbacks.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
import json

from .data_loader import BraTSDataLoader
from .preprocessor import MRIPreprocessor


class PreprocessedBraTSDataset(Dataset):
    """Ultra-fast PyTorch Dataset loading directly from preprocessed .npz files."""

    def __init__(
        self,
        npz_dir: Union[str, Path],
        case_list: List[str],
        transforms=None,
        device: str = "cpu",
    ):
        """
        Initialize preprocessed dataset.

        Args:
            npz_dir: Path to directory containing .npz files
            case_list: List of case IDs
            transforms: Optional MONAI transforms
            device: "cpu" or "cuda"
        """
        self.npz_dir = Path(npz_dir)
        self.case_list = case_list
        self.transforms = transforms
        self.device = device

    def __len__(self) -> int:
        return len(self.case_list)

    def __getitem__(self, idx: int) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        case_id = self.case_list[idx]
        npz_path = self.npz_dir / f"{case_id}.npz"

        if not npz_path.exists():
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")

        with np.load(npz_path) as data:
            image_np = data["image"].astype(np.float32)  # (4, H, W, D)
            if "label" in data:
                label_np = data["label"].astype(np.int64)
            elif "seg" in data:
                label_np = data["seg"].astype(np.int64)
            else:
                label_np = np.zeros((1,) + image_np.shape[1:], dtype=np.int64)

            if label_np.ndim == 3:
                label_np = np.expand_dims(label_np, axis=0)  # (1, H, W, D)

        if self.transforms:
            sample = {
                "image": image_np,
                "label": label_np,
                "case_id": case_id,
            }
            return self.transforms(sample)

        image_tensor = torch.from_numpy(image_np).float().to(self.device)
        label_tensor = torch.from_numpy(label_np).float().to(self.device)
        return image_tensor, label_tensor


class BraTSDataset(Dataset):
    """PyTorch Dataset for BraTS 2023 data with automated NPZ fast path & fallback."""

    def __init__(
        self,
        data_root: str,
        case_list: List[str],
        preprocessor: Optional[MRIPreprocessor] = None,
        device: str = "cpu",
        transforms=None,
    ):
        """
        Initialize dataset.

        Args:
            data_root: Path to BraTS dataset root or processed .npz directory
            case_list: List of case IDs to include
            preprocessor: MRIPreprocessor instance (if None, use default)
            device: "cpu" or "cuda"
            transforms: Optional MONAI transforms
        """
        self.data_loader = BraTSDataLoader(data_root)
        self.case_list = case_list
        self.device = device
        self.preprocessor = preprocessor or MRIPreprocessor()
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.case_list)

    def __getitem__(self, idx: int) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Get a single sample.

        Returns:
            (image_tensor, seg_tensor) or dict if transforms specified
        """
        case_id = self.case_list[idx]

        # Fast NPZ path if available
        if self.data_loader.cache_dir and (self.data_loader.cache_dir / f"{case_id}.npz").exists():
            npz_path = self.data_loader.cache_dir / f"{case_id}.npz"
            with np.load(npz_path) as data:
                image_np = data["image"].astype(np.float32)
                label_np = data["label"].astype(np.int64) if "label" in data else data.get("seg", None)
                if label_np is not None and label_np.ndim == 3:
                    label_np = np.expand_dims(label_np, axis=0)

            if self.transforms:
                sample = {"image": image_np, "label": label_np, "case_id": case_id}
                return self.transforms(sample)

            image_tensor = torch.from_numpy(image_np).float().to(self.device)
            seg_tensor = torch.from_numpy(label_np).float().to(self.device) if label_np is not None else None
            return image_tensor, seg_tensor

        # Standard NIfTI fallback path with on-the-fly preprocessing
        image_tensor, seg_tensor = self.data_loader.load_case_as_tensor(
            case_id, device=self.device
        )

        image_np = image_tensor.cpu().numpy()
        seg_np = seg_tensor.cpu().numpy() if seg_tensor is not None else None

        image_np, seg_np = self.preprocessor.preprocess(
            image_np, seg_np, normalize=True, crop_pad=True
        )

        if self.transforms:
            sample = {"image": image_np, "label": seg_np, "case_id": case_id}
            return self.transforms(sample)

        image_tensor = torch.from_numpy(image_np).float().to(self.device)
        seg_tensor = torch.from_numpy(seg_np).float().to(self.device) if seg_np is not None else None
        return image_tensor, seg_tensor


class DataSplitter:
    """Helper to split dataset into train/val/test."""

    def __init__(self, data_root: str, val_ratio: float = 0.1, test_ratio: float = 0.1):
        """
        Initialize splitter.

        Args:
            data_root: Path to BraTS dataset root
            val_ratio: Fraction for validation (0.0-1.0)
            test_ratio: Fraction for test (0.0-1.0)
        """
        self.data_root = data_root
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.loader = BraTSDataLoader(data_root)

    def get_split(self, seed: int = 42) -> dict:
        """
        Get train/val/test split.

        Args:
            seed: Random seed for reproducibility

        Returns:
            dict with keys "train", "val", "test", each containing list of case IDs
        """
        np.random.seed(seed)

        all_cases = self.loader.list_cases()
        case_ids = [c.name for c in all_cases]

        # Shuffle
        np.random.shuffle(case_ids)

        # Split
        n_total = len(case_ids)
        n_test = int(n_total * self.test_ratio)
        n_val = int(n_total * self.val_ratio)
        n_train = n_total - n_test - n_val

        split = {
            "train": case_ids[:n_train],
            "val": case_ids[n_train : n_train + n_val],
            "test": case_ids[n_train + n_val :],
        }

        return split

    def save_split(self, split: dict, output_file: str = "data_split.json"):
        """Save split configuration to JSON."""
        output_path = Path(output_file)
        with open(output_path, "w") as f:
            json.dump(split, f, indent=2)
        print(f"Split saved to {output_path}")

    def load_split(self, input_file: str = "data_split.json") -> dict:
        """Load split configuration from JSON."""
        input_path = Path(input_file)
        with open(input_path, "r") as f:
            split = json.load(f)
        return split


def create_dataloaders(
    data_root: str,
    split_config: dict,
    batch_size: int = 2,
    num_workers: int = 0,
    device: str = "cpu",
    preprocessor: Optional[MRIPreprocessor] = None,
    transforms_dict: Optional[dict] = None,
) -> dict:
    """
    Create PyTorch DataLoaders for train/val/test with fast NPZ cache support.

    Args:
        data_root: Path to BraTS dataset root or processed NPZ directory
        split_config: Dict with "train", "val", "test" keys
        batch_size: Batch size for loading
        num_workers: Number of worker processes
        device: "cpu" or "cuda"
        preprocessor: Optional custom preprocessor
        transforms_dict: Optional dict mapping split names to MONAI transforms

    Returns:
        dict with keys "train_loader", "val_loader", "test_loader"
    """
    if preprocessor is None:
        preprocessor = MRIPreprocessor()

    dataloaders = {}
    transforms_dict = transforms_dict or {}

    for split_name in ["train", "val", "test"]:
        if split_name not in split_config:
            continue

        trans = transforms_dict.get(split_name, None)

        dataset = BraTSDataset(
            data_root=data_root,
            case_list=split_config[split_name],
            preprocessor=preprocessor,
            device=device,
            transforms=trans,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        dataloaders[f"{split_name}_loader"] = dataloader

    return dataloaders
