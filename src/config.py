"""
Configuration Manager for Brain Tumor Segmentation Support System.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from .utils.io import load_yaml, ensure_dir


class Config:
    """Class to manage system configurations."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize Config manager.

        Args:
            config_path: Path to YAML config file. Defaults to configs/default.yaml.
        """
        if config_path is None:
            # Default to root configs/default.yaml
            root_dir = Path(__file__).resolve().parent.parent
            config_path = root_dir / "configs" / "default.yaml"

        self.config_path = Path(config_path)
        self._cfg = self._load()

    def _load(self) -> Dict[str, Any]:
        """Load YAML config file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
        return load_yaml(self.config_path)

    @property
    def raw_config(self) -> Dict[str, Any]:
        """Get raw config dictionary."""
        return self._cfg

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get config value using dot-notation path (e.g. 'data.data_root').

        Args:
            key_path: Dot-separated string key path
            default: Default value if key is not found

        Returns:
            Config value
        """
        keys = key_path.split(".")
        val = self._cfg
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    @property
    def project_name(self) -> str:
        return self.get("project.name", "Brain Tumor Segmentation Support System")

    @property
    def root_dir(self) -> Path:
        return Path(__file__).resolve().parent.parent

    @property
    def data_root(self) -> Path:
        """
        Resolve BraTS dataset path.

        Priority:
        1. BRATS_DATA_ROOT environment variable
        2. DATA_ROOT environment variable
        3. paths.data_root from configs/default.yaml
        4. common Google Drive / Colab locations
        """
        raw_candidates = [
            os.getenv("BRATS_DATA_ROOT"),
            os.getenv("DATA_ROOT"),
            self.get(
                "paths.data_root",
                "Datasets/brats2023-gli-dataset/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
            ),
            # Windows Google Drive for Desktop locations
            "G:/My Drive/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
            "G:/MyDrive/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
            "G:/My Drive/SIC_Capstone_2026/data/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
            "D:/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
            "C:/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
            # Google Colab (Linux) locations
            "/content/drive/MyDrive/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
            "/content/drive/MyDrive/SIC_Capstone_2026/data/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
            "/content/drive/MyDrive/SIC_Capstone_2026/data/BraTS2023/brats2023-gli-dataset/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
        ]

        for cand in raw_candidates:
            if not cand:
                continue
            path = Path(cand).expanduser()
            if path.is_absolute():
                if path.exists():
                    return path
            else:
                if path.exists():
                    return path.resolve()
                proj_path = self.root_dir / path
                if proj_path.exists():
                    return proj_path.resolve()

        # Dynamic Windows Drive Letter Scanner (scans G:, H:, I:, D:, C:, etc. for Google Drive for Desktop)
        import string
        sub_paths = [
            "My Drive/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
            "MyDrive/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
            "BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
            "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
        ]
        for letter in "GHIJKLMNOPQRSTUVWXYZCDEFAB":
            root_drive = Path(f"{letter}:/")
            if root_drive.exists():
                for sub in sub_paths:
                    candidate = root_drive / sub
                    if candidate.exists():
                        return candidate.resolve()

        default_rel = Path(raw_candidates[2])
        return (self.root_dir / default_rel).resolve() if not default_rel.is_absolute() else default_rel

    @property
    def checkpoints_dir(self) -> Path:
        p = Path(self.get("paths.checkpoints_dir", "checkpoints"))
        if not p.is_absolute():
            p = self.root_dir / p
        ensure_dir(p)
        return p.resolve()

    @property
    def results_dir(self) -> Path:
        p = Path(self.get("paths.results_dir", "results"))
        if not p.is_absolute():
            p = self.root_dir / p
        ensure_dir(p)
        return p.resolve()

    @property
    def logs_dir(self) -> Path:
        p = Path(self.get("paths.logs_dir", "logs"))
        if not p.is_absolute():
            p = self.root_dir / p
        ensure_dir(p)
        return p.resolve()

    @property
    def processed_npz_dir(self) -> Path:
        """
        Resolve processed .npz dataset path (local or Google Drive).
        """
        raw_candidates = [
            os.getenv("BRATS_CACHE_DIR"),
            os.getenv("BRATS_PROCESSED_DIR"),
            self.get("paths.processed_npz_dir", "Datasets/processed_npz"),
            "G:/My Drive/BraTS2023/processed_npz",
            "G:/MyDrive/BraTS2023/processed_npz",
            "D:/BraTS2023/processed_npz",
            "/content/drive/MyDrive/BraTS2023/processed_npz",
            "/content/drive/MyDrive/SIC_Capstone_2026/data/BraTS2023/processed_npz",
        ]
        for cand in raw_candidates:
            if not cand:
                continue
            path = Path(cand).expanduser()
            if path.is_absolute():
                if path.exists():
                    return path
            else:
                if path.exists():
                    return path.resolve()
                proj_path = self.root_dir / path
                if proj_path.exists():
                    return proj_path.resolve()

        sub_npz_paths = [
            "My Drive/BraTS2023/processed_npz",
            "MyDrive/BraTS2023/processed_npz",
            "BraTS2023/processed_npz",
            "processed_npz",
        ]
        for letter in "GHIJKLMNOPQRSTUVWXYZCDEFAB":
            root_drive = Path(f"{letter}:/")
            if root_drive.exists():
                for sub in sub_npz_paths:
                    candidate = root_drive / sub
                    if candidate.exists():
                        return candidate.resolve()

        default_rel = Path(raw_candidates[2])
        return (self.root_dir / default_rel).resolve() if not default_rel.is_absolute() else default_rel

    @property
    def num_classes(self) -> int:
        return self.get("data.num_classes", 4)

    @property
    def target_shape(self) -> tuple:
        return tuple(self.get("data.target_shape", [240, 240, 160]))


# Global config instance
def get_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    return Config(config_path)
