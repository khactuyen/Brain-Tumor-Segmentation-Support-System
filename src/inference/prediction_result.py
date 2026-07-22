"""
Dataclass structure for storing model inference predictions and metrics.
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union
from ..utils.io import save_nifti, save_json, ensure_dir


@dataclass
class PredictionResult:
    """Class representing the result of a 3D segmentation inference."""

    case_id: str
    model_name: str
    mask: np.ndarray  # Predicted mask 3D numpy array (H, W, D)
    inference_time: float  # In seconds
    metrics: Dict[str, float] = field(default_factory=dict)
    tumor_volume_cm3: float = 0.0
    affine: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save(self, output_dir: Union[str, Path]) -> Dict[str, Path]:
        """
        Save prediction mask as NIfTI file and metadata/metrics as JSON.

        Args:
            output_dir: Root directory for output results

        Returns:
            Dict containing saved file paths ("mask_path", "json_path")
        """
        out_dir = Path(output_dir) / self.case_id
        ensure_dir(out_dir)

        # Save NIfTI Mask
        mask_filename = f"{self.case_id}_{self.model_name}_mask.nii.gz"
        mask_path = out_dir / mask_filename
        save_nifti(self.mask, mask_path, affine=self.affine)

        # Save JSON Info
        json_filename = f"{self.case_id}_{self.model_name}_info.json"
        json_path = out_dir / json_filename

        info = {
            "case_id": self.case_id,
            "model_name": self.model_name,
            "inference_time_seconds": round(self.inference_time, 4),
            "tumor_volume_cm3": round(self.tumor_volume_cm3, 4),
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
            "mask_shape": list(self.mask.shape),
            "metadata": self.metadata,
        }
        save_json(info, json_path)

        return {"mask_path": mask_path, "json_path": json_path}

    def to_dict(self) -> Dict[str, Any]:
        """Convert result metadata to dictionary."""
        return {
            "case_id": self.case_id,
            "model_name": self.model_name,
            "inference_time": self.inference_time,
            "tumor_volume_cm3": self.tumor_volume_cm3,
            "metrics": self.metrics,
        }
