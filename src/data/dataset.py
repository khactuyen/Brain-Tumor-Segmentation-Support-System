"""
PyTorch Dataset class for BraTS data with train/val/test split.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple, Optional
import json

from .data_loader import BraTSDataLoader
from .preprocessor import MRIPreprocessor


class BraTSDataset(Dataset):
    """PyTorch Dataset for BraTS 2023 data."""

    def __init__(
        self,
        data_root: str,
        case_list: List[str],
        preprocessor: Optional[MRIPreprocessor] = None,
        device: str = "cpu",
    ):
        """
        Initialize dataset.

        Args:
            data_root: Path to BraTS dataset root
            case_list: List of case IDs to include
            preprocessor: MRIPreprocessor instance (if None, use default)
            device: "cpu" or "cuda"
        """
        self.data_loader = BraTSDataLoader(data_root)
        self.case_list = case_list
        self.device = device
        self.preprocessor = preprocessor or MRIPreprocessor()

    def __len__(self) -> int:
        return len(self.case_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Get a single sample.

        Returns:
            (image_tensor, seg_tensor) where
            image_tensor: (4, H, W, D)
            seg_tensor: (1, H, W, D) or None
        """
        case_id = self.case_list[idx]
        image_tensor, seg_tensor = self.data_loader.load_case_as_tensor(
            case_id, device=self.device
        )

        # Apply preprocessing
        image_np = image_tensor.cpu().numpy()
        seg_np = seg_tensor.cpu().numpy() if seg_tensor is not None else None

        image_np, seg_np = self.preprocessor.preprocess(
            image_np, seg_np, normalize=True, crop_pad=True
        )

        # Convert back to tensors
        image_tensor = torch.from_numpy(image_np).float().to(self.device)
        if seg_np is not None:
            seg_tensor = torch.from_numpy(seg_np).float().to(self.device)
        else:
            seg_tensor = None

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
) -> dict:
    """
    Create PyTorch DataLoaders for train/val/test.

    Args:
        data_root: Path to BraTS dataset root
        split_config: Dict with "train", "val", "test" keys
        batch_size: Batch size for loading
        num_workers: Number of worker processes
        device: "cpu" or "cuda"
        preprocessor: Optional custom preprocessor

    Returns:
        dict with keys "train_loader", "val_loader", "test_loader"
    """
    if preprocessor is None:
        preprocessor = MRIPreprocessor()

    dataloaders = {}
    for split_name in ["train", "val", "test"]:
        if split_name not in split_config:
            continue

        dataset = BraTSDataset(
            data_root=data_root,
            case_list=split_config[split_name],
            preprocessor=preprocessor,
            device=device,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
        )

        dataloaders[f"{split_name}_loader"] = dataloader

    return dataloaders
