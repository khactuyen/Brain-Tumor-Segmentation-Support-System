import os
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from concurrent.futures import ThreadPoolExecutor
import torch


class BraTSDataLoader:
    """Fast & Optimized Data Loader for BraTS 2023 dataset."""

    MODALITIES = ["t1n", "t1c", "t2w", "t2f"]  # T1, T1-contrast, T2, FLAIR
    SEGMENTATION_SUFFIX = "seg"

    def __init__(self, data_root: str, cache_dir: Optional[str] = None):
        """
        Initialize optimized data loader.

        Args:
            data_root: Path to BraTS dataset root directory
            cache_dir: Optional path to directory containing preprocessed .npz files
        """
        self.data_root = Path(data_root)
        
        # Check default cache directory locations
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is None:
            candidate_caches = [
                self.data_root.parent / "processed_npz",
                self.data_root / "processed_npz",
                Path("Datasets/processed_npz"),
                Path("/content/drive/MyDrive/BraTS2023/processed_npz"),
            ]
            for c in candidate_caches:
                if c.exists():
                    self.cache_dir = c
                    break

    def list_cases(self) -> List[Path]:
        """List all case directories or cached case IDs in dataset."""
        if self.cache_dir and self.cache_dir.exists():
            npz_files = sorted(self.cache_dir.glob("*.npz"))
            if len(npz_files) > 0:
                # Return dummy path objects matching case directory naming
                return [self.data_root / f.stem for f in npz_files]

        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root not found: {self.data_root}")

        cases = sorted([d for d in self.data_root.iterdir() if d.is_dir() and d.name.startswith("BraTS")])
        if not cases:
            cases = sorted([d for d in self.data_root.iterdir() if d.is_dir() and not d.name.startswith(".")])
        return cases

    def _load_single_nii(self, filepath: Path) -> np.ndarray:
        """Load single NIfTI file directly into float32 numpy array without float64 casting."""
        nii = nib.load(filepath)
        return np.asarray(nii.dataobj, dtype=np.float32)

    def load_case(self, case_id: str) -> Dict[str, np.ndarray]:
        """
        Load all MRI sequences and segmentation for a case.
        Uses fast NPZ cache if available, or parallel multi-threaded NIfTI uncompression.

        Args:
            case_id: Case ID (e.g., "BraTS-GLI-00000-000")

        Returns:
            dict with keys: "t1n", "t1c", "t2w", "t2f", "seg" (if exists)
        """
        # 1. Try loading from fast NPZ cache
        if self.cache_dir and self.cache_dir.exists():
            npz_path = self.cache_dir / f"{case_id}.npz"
            if npz_path.exists():
                with np.load(npz_path) as npz_data:
                    data = {}
                    image_stack = npz_data["image"]  # shape (4, H, W, D)
                    for idx, modality in enumerate(self.MODALITIES):
                        data[modality] = image_stack[idx]
                    if "seg" in npz_data:
                        data[self.SEGMENTATION_SUFFIX] = npz_data["seg"].astype(np.float32)
                    return data

        # 2. Direct NIfTI load with parallel thread pool for 4x speedup
        case_dir = self.data_root / case_id
        if not case_dir.exists():
            raise FileNotFoundError(f"Case directory not found: {case_id}")

        data = {}
        files_to_load = {}

        for modality in self.MODALITIES:
            filepath = case_dir / f"{case_id}-{modality}.nii.gz"
            if not filepath.exists():
                raise FileNotFoundError(f"Missing modality file: {filepath.name}")
            files_to_load[modality] = filepath

        seg_filepath = case_dir / f"{case_id}-{self.SEGMENTATION_SUFFIX}.nii.gz"
        if seg_filepath.exists():
            files_to_load[self.SEGMENTATION_SUFFIX] = seg_filepath

        # Parallel multi-thread reading
        with ThreadPoolExecutor(max_workers=len(files_to_load)) as executor:
            future_to_mod = {
                executor.submit(self._load_single_nii, path): mod
                for mod, path in files_to_load.items()
            }
            for future in future_to_mod:
                mod = future_to_mod[future]
                data[mod] = future.result()

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
            (image_tensor, seg_tensor)
        """
        # Fast NPZ path
        if self.cache_dir and self.cache_dir.exists():
            npz_path = self.cache_dir / f"{case_id}.npz"
            if npz_path.exists():
                with np.load(npz_path) as npz_data:
                    image_array = npz_data["image"].astype(np.float32)
                    image_tensor = torch.from_numpy(image_array).to(device)

                    seg_tensor = None
                    if "seg" in npz_data:
                        seg_array = npz_data["seg"].astype(np.float32)
                        if seg_array.ndim == 3:
                            seg_array = np.expand_dims(seg_array, axis=0)
                        seg_tensor = torch.from_numpy(seg_array).to(device)

                    return image_tensor, seg_tensor

        # Standard path with fast multi-threading
        data = self.load_case(case_id)
        modality_arrays = [data[mod] for mod in self.MODALITIES]
        image_array = np.stack(modality_arrays, axis=0)  # (4, H, W, D)
        image_tensor = torch.from_numpy(image_array).float().to(device)

        seg_tensor = None
        if self.SEGMENTATION_SUFFIX in data:
            seg_array = data[self.SEGMENTATION_SUFFIX]
            if seg_array.ndim == 3:
                seg_array = np.expand_dims(seg_array, axis=0)
            seg_tensor = torch.from_numpy(seg_array).float().to(device)

        return image_tensor, seg_tensor

    def save_case_to_npz(self, case_id: str, output_dir: Path) -> Path:
        """
        Convert a case to compressed float32/uint8 NPZ for fast future loading.

        Args:
            case_id: Case ID
            output_dir: Destination directory

        Returns:
            Path to saved .npz file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        target_npz = output_dir / f"{case_id}.npz"

        data = self.load_case(case_id)
        modality_arrays = [data[mod].astype(np.float32) for mod in self.MODALITIES]
        image_stack = np.stack(modality_arrays, axis=0)  # (4, H, W, D)

        save_dict = {"image": image_stack}
        if self.SEGMENTATION_SUFFIX in data:
            save_dict["seg"] = data[self.SEGMENTATION_SUFFIX].astype(np.uint8)

        np.savez_compressed(target_npz, **save_dict)
        return target_npz

    def get_case_info(self, case_id: str) -> Dict:
        """Get metadata info for a case."""
        case_dir = self.data_root / case_id
        files = sorted(case_dir.glob("*.nii.gz")) if case_dir.exists() else []

        info = {
            "case_id": case_id,
            "num_files": len(files),
            "files": [f.name for f in files],
        }

        first_mod_file = case_dir / f"{case_id}-{self.MODALITIES[0]}.nii.gz"
        if first_mod_file.exists():
            nii = nib.load(first_mod_file)
            info["shape"] = nii.shape
            info["spacing"] = nii.header.get_zooms()[:3]

        return info

