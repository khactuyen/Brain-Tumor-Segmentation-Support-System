"""
Data loader module for BraTS 2023 dataset.
Reads NIfTI files and converts to PyTorch tensors.
"""

import os
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, Tuple, Optional
import torch


class BraTSDataLoader:
    """Load BraTS 2023 data from NIfTI files."""

    MODALITIES = ["t1n", "t1c", "t2w", "t2f"]  # T1, T1-contrast, T2, FLAIR
    SEGMENTATION_SUFFIX = "seg"

    def __init__(self, data_root: str):
        """
        Initialize data loader.

        Args:
            data_root: Path to BraTS dataset root directory
                      (e.g., .../ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData)
        """
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root not found: {data_root}")

    def list_cases(self) -> list:
        """List all case directories in dataset."""
        cases = sorted([d for d in self.data_root.iterdir() if d.is_dir()])
        return cases

    def load_case(self, case_id: str) -> Dict[str, np.ndarray]:
        """
        Load all MRI sequences and segmentation for a case.

        Args:
            case_id: Case ID (e.g., "BraTS-GLI-00000-000")

        Returns:
            dict with keys: "t1n", "t1c", "t2w", "t2f", "seg" (if exists)
            Each value is a numpy array (H, W, D)
        """
        case_dir = self.data_root / case_id
        if not case_dir.exists():
            raise FileNotFoundError(f"Case directory not found: {case_id}")

        data = {}

        # Load MRI modalities
        for modality in self.MODALITIES:
            filename = f"{case_id}-{modality}.nii.gz"
            filepath = case_dir / filename
            if not filepath.exists():
                raise FileNotFoundError(f"Missing modality file: {filename}")

            nii = nib.load(filepath)
            data[modality] = nii.get_fdata()

        # Load segmentation if exists
        seg_filename = f"{case_id}-{self.SEGMENTATION_SUFFIX}.nii.gz"
        seg_filepath = case_dir / seg_filename
        if seg_filepath.exists():
            nii = nib.load(seg_filepath)
            data[self.SEGMENTATION_SUFFIX] = nii.get_fdata()

        return data

    def load_case_as_tensor(
        self, case_id: str, device: str = "cpu"
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Load case and convert to PyTorch tensors.

        Args:
            case_id: Case ID
            device: "cpu" or "cuda"

        Returns:
            (image_tensor, seg_tensor) where
            image_tensor: shape (4, H, W, D) - 4 modalities stacked
            seg_tensor: shape (1, H, W, D) or None if no segmentation
        """
        data = self.load_case(case_id)

        # Stack modalities into single tensor
        modality_arrays = [data[mod] for mod in self.MODALITIES]
        image_array = np.stack(modality_arrays, axis=0)  # (4, H, W, D)
        image_tensor = torch.from_numpy(image_array).float().to(device)

        # Load segmentation if available
        seg_tensor = None
        if self.SEGMENTATION_SUFFIX in data:
            seg_array = data[self.SEGMENTATION_SUFFIX]
            seg_tensor = torch.from_numpy(seg_array).unsqueeze(0).float().to(device)

        return image_tensor, seg_tensor

    def get_case_info(self, case_id: str) -> Dict:
        """Get metadata info for a case."""
        case_dir = self.data_root / case_id
        files = sorted(case_dir.glob("*.nii.gz"))

        info = {
            "case_id": case_id,
            "num_files": len(files),
            "files": [f.name for f in files],
        }

        # Get shape from first modality
        first_mod_file = case_dir / f"{case_id}-{self.MODALITIES[0]}.nii.gz"
        if first_mod_file.exists():
            nii = nib.load(first_mod_file)
            info["shape"] = nii.shape
            info["spacing"] = nii.header.get_zooms()[:3]

        return info
