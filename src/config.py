"""
Configuration Manager for Brain Tumor Segmentation Support System.
"""

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

    # Convenience properties
    @property
    def project_name(self) -> str:
        return self.get("project.name", "Brain Tumor Segmentation Support System")

    @property
    def data_root(self) -> Path:
        return Path(self.get("paths.data_root", "Datasets/brats2023-gli-dataset/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"))

    @property
    def checkpoints_dir(self) -> Path:
        p = Path(self.get("paths.checkpoints_dir", "checkpoints"))
        ensure_dir(p)
        return p

    @property
    def results_dir(self) -> Path:
        p = Path(self.get("paths.results_dir", "results"))
        ensure_dir(p)
        return p

    @property
    def logs_dir(self) -> Path:
        p = Path(self.get("paths.logs_dir", "logs"))
        ensure_dir(p)
        return p

    @property
    def num_classes(self) -> int:
        return self.get("data.num_classes", 4)

    @property
    def target_shape(self) -> tuple:
        return tuple(self.get("data.target_shape", [240, 240, 160]))


# Global config instance
def get_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    return Config(config_path)
