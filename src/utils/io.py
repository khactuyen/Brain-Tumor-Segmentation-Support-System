"""
I/O utilities for file handling, NIfTI reading/writing, and configuration loading.
"""

import os
import json
import yaml
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, Any, Union, Tuple, Optional


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if not."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {file_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """Save dictionary to a YAML file."""
    path = Path(file_path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False)


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load data from a JSON file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], file_path: Union[str, Path], indent: int = 2) -> None:
    """Save dictionary to a JSON file."""
    path = Path(file_path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def save_nifti(
    data: np.ndarray,
    output_path: Union[str, Path],
    affine: Optional[np.ndarray] = None,
    header: Optional[nib.Nifti1Header] = None,
) -> Path:
    """
    Save numpy array as NIfTI file (.nii.gz).

    Args:
        data: 3D or 4D numpy array
        output_path: Output file path
        affine: 4x4 affine matrix (default: identity if None)
        header: NIfTI header (optional)

    Returns:
        Path to saved file
    """
    path = Path(output_path)
    ensure_dir(path.parent)

    if affine is None:
        affine = np.eye(4)

    nii = nib.Nifti1Image(data.astype(np.float32), affine=affine, header=header)
    nib.save(nii, str(path))
    return path


def load_nifti(file_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load NIfTI file.

    Args:
        file_path: Path to .nii or .nii.gz file

    Returns:
        (data_array, affine_matrix)
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {file_path}")

    nii = nib.load(str(path))
    return nii.get_fdata(), nii.affine
