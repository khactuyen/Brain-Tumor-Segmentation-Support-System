#!/usr/bin/env python3
"""
Brain Tumor Segmentation BraTS 2023 - 3D U-Net & Swin UNETR Pipeline (Standalone Script)
Loads preprocessed NPZ directly from disk/Drive if available, skipping re-conversion.
"""

import inspect
import os
import random
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from zipfile import BadZipFile

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm

import monai.transforms as mt
from monai.data import DataLoader as MonaiDataLoader
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.networks.nets import SwinUNETR, UNet
from monai.utils import set_determinism
from fpdf import FPDF

# 1. Environment & Device Setup
IN_COLAB = False
try:
    from google.colab import drive
    IN_COLAB = True
    if not Path("/content/drive/MyDrive").exists():
        drive.mount("/content/drive", force_remount=False)
    print("Google Drive is ready")
except ImportError:
    print("Running outside Google Colab")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
set_determinism(seed=SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {props.total_memory / (1024 ** 3):.2f} GB")

# 2. Paths Configuration
raw_candidates = [
    Path("/content/drive/MyDrive/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"),
    Path("/content/drive/MyDrive/SIC_Capstone_2026/data/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"),
    Path("Datasets/brats2023-gli-dataset/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"),
    Path("Datasets/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"),
    Path("D:/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"),
    Path("D:/SIC_Capstone 2026/Datasets/brats2023-gli-dataset/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"),
]
drive_npz_candidates = [
    Path("/content/drive/MyDrive/BraTS2023/processed_npz"),
    Path("/content/drive/MyDrive/SIC_Capstone_2026/data/BraTS2023/processed_npz"),
    Path("/content/drive/MyDrive/SIC_Capstone_2026/processed_npz"),
    Path("/content/drive/MyDrive/processed_npz"),
    Path("/content/processed_npz"),
    Path("Datasets/processed_npz"),
    Path("processed_npz"),
    Path("D:/BraTS2023/processed_npz"),
    Path("D:/SIC_Capstone 2026/Datasets/processed_npz"),
    Path("D:/SIC_Capstone 2026/processed_npz"),
]

default_raw_dir = raw_candidates[0] if IN_COLAB else raw_candidates[2]
default_npz_dir = drive_npz_candidates[0] if IN_COLAB else drive_npz_candidates[5]
data_root = next((p for p in raw_candidates if p.exists()), default_raw_dir)

# Auto-detect existing NPZ directory containing cached files
drive_npz_dir = next((p for p in drive_npz_candidates if p.exists() and len(list(p.glob("*.npz"))) > 0), None)
if drive_npz_dir is None:
    drive_npz_dir = next((p for p in drive_npz_candidates if p.exists()), default_npz_dir)

local_npz_dir = Path("/content/processed_npz") if IN_COLAB else drive_npz_dir
checkpoints_dir = Path("/content/drive/MyDrive/BraTS2023/checkpoints") if IN_COLAB else Path("checkpoints")
reports_dir = Path("/content/drive/MyDrive/BraTS2023/reports") if IN_COLAB else Path("reports")
predictions_dir = Path("/content/drive/MyDrive/BraTS2023/predictions") if IN_COLAB else Path("predictions")

for output_dir in (checkpoints_dir, reports_dir, predictions_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

PREPROCESS_VERSION = "brats_npz_v3_affine_spacing_strict_labels"
FORCE_REBUILD_CACHE = False
MODALITIES = ("t1n", "t1c", "t2w", "t2f")
LABEL_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3}
ALLOWED_RAW_LABELS = set(LABEL_MAP)

VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
VAL_SUBSET_SIZE = 20
PATCH_SIZE = (96, 96, 96)
BATCH_SIZE = 1
NUM_SAMPLES = 1
NUM_WORKERS = 2
STEPS_PER_EPOCH = 150
MAX_EPOCHS = 30
FAST_VAL_INTERVAL = 2
FULL_VAL_INTERVAL = 10
SW_BATCH_SIZE = 1
SW_OVERLAP = 0.5
LR = 2e-4
WEIGHT_DECAY = 1e-4

RUN_SWIN_UNETR = True
MODELS_TO_TRAIN = ["unet3d"] + (["swin_unetr"] if RUN_SWIN_UNETR else [])

# 3. Flexible NPZ Loaders & Preprocessors
CACHE_ERRORS = (OSError, ValueError, TypeError, KeyError, EOFError, BadZipFile)

def load_validated_cache(npz_path: Path) -> Dict[str, np.ndarray]:
    """Load one NPZ cache file flexibly, supporting all preprocessed and custom formats."""
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=True) as data:
        # 1. Image
        img_key = next((k for k in ("image", "img", "data", "mri") if k in data), None)
        if img_key is None:
            if len(data.files) > 0:
                img_key = data.files[0]
            else:
                raise ValueError(f"No array found in {npz_path}")
        image = np.asarray(data[img_key])

        if image.ndim == 4 and image.shape[-1] in (1, 4) and image.shape[0] not in (1, 4):
            image = np.moveaxis(image, -1, 0)
        elif image.ndim == 3:
            image = np.expand_dims(image, axis=0)

        # 2. Label
        lbl_key = next((k for k in ("label", "seg", "mask", "target", "labels") if k in data), None)
        if lbl_key is not None:
            label = np.asarray(data[lbl_key])
            if label.ndim == 4 and label.shape[-1] == 1:
                label = np.squeeze(label, axis=-1)
            if label.ndim == 3:
                label = np.expand_dims(label, axis=0)
            if np.any(label == 4):
                label = label.copy()
                label[label == 4] = 3
        else:
            label = np.zeros((1,) + image.shape[1:], dtype=np.uint8)

        # 3. Affine
        if "affine" in data and hasattr(data["affine"], "shape") and data["affine"].shape == (4, 4):
            affine = np.asarray(data["affine"], dtype=np.float64)
        else:
            affine = np.eye(4, dtype=np.float64)

        # 4. Spacing
        if "spacing" in data and hasattr(data["spacing"], "__len__") and len(data["spacing"]) == 3:
            spacing = np.asarray(data["spacing"], dtype=np.float32)
        else:
            spacing = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        return {
            "image": image.astype(np.float32, copy=False),
            "label": label.astype(np.uint8, copy=False),
            "affine": affine,
            "spacing": spacing,
        }

def cache_is_current(npz_path: Path) -> bool:
    npz_path = Path(npz_path)
    if not npz_path.is_file() or npz_path.stat().st_size == 0:
        return False
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            return len(data.files) > 0
    except CACHE_ERRORS:
        return False

def preprocess_case(case_id: str, src_dir: Path, dst_dir: Path, force: bool = False) -> Path:
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    out_path = dst_dir / f"{case_id}.npz"
    if not force and cache_is_current(out_path):
        return out_path
    
    case_folder = Path(src_dir) / case_id
    if not case_folder.is_dir():
        raise FileNotFoundError(f"Case directory not found: {case_folder}")
    
    channels = []
    reference_shape = None
    reference_affine = None
    spacing = None
    for modality in MODALITIES:
        file_path = case_folder / f"{case_id}-{modality}.nii.gz"
        if not file_path.exists():
            raise FileNotFoundError(f"Missing modality for {case_id}: {file_path.name}")

        nii = nib.load(file_path)
        array = np.asarray(nii.dataobj, dtype=np.float32)
        if array.ndim != 3:
            raise ValueError(f"Expected a 3D volume for {file_path}, got shape {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite MRI intensities in {file_path}")

        if reference_shape is None:
            reference_shape = array.shape
            reference_affine = np.asarray(nii.affine, dtype=np.float64)
            spacing = np.asarray(nib.affines.voxel_sizes(nii.affine), dtype=np.float32)
            if not np.isfinite(reference_affine).all() or abs(np.linalg.det(reference_affine[:3, :3])) <= 1e-8:
                raise ValueError(f"Invalid affine in {file_path}")
            if not np.isfinite(spacing).all() or np.any(spacing <= 0):
                raise ValueError(f"Invalid voxel spacing in {file_path}")
        else:
            if array.shape != reference_shape:
                raise ValueError(f"Modality shape mismatch in {case_id}: {array.shape} != {reference_shape}")
            if not np.allclose(nii.affine, reference_affine, rtol=0.0, atol=1e-4):
                raise ValueError(f"Modality affine mismatch in {case_id}: {file_path.name}")

        brain_mask = array > 0
        if not brain_mask.any():
            raise ValueError(f"Empty MRI signal in {file_path}")
        values = array[brain_mask]
        low, high = np.percentile(values, [0.5, 99.5])
        array = np.clip(array, low, high)
        mean_value = float(array[brain_mask].mean())
        std_value = float(array[brain_mask].std())
        if std_value <= 1e-6:
            raise ValueError(f"Near-zero intensity variance in {file_path}")
        array = (array - mean_value) / std_value
        array[~brain_mask] = 0.0
        channels.append(array.astype(np.float32, copy=False))

    image = np.stack(channels, axis=0)
    seg_path = case_folder / f"{case_id}-seg.nii.gz"
    if not seg_path.exists():
        raise FileNotFoundError(f"Missing training segmentation for {case_id}: {seg_path.name}")

    seg_nii = nib.load(seg_path)
    raw_seg = np.asarray(seg_nii.dataobj)
    if raw_seg.shape != reference_shape:
        raise ValueError(f"Segmentation shape mismatch in {case_id}: {raw_seg.shape} != {reference_shape}")
    if not np.isfinite(raw_seg).all():
        raise ValueError(f"Non-finite segmentation values in {case_id}")
    if not np.allclose(seg_nii.affine, reference_affine, rtol=0.0, atol=1e-4):
        raise ValueError(f"Segmentation affine mismatch in {case_id}")

    raw_labels = set(np.unique(raw_seg).tolist())
    unexpected = raw_labels - ALLOWED_RAW_LABELS
    if unexpected:
        raise ValueError(f"Unexpected labels in {case_id}: {sorted(unexpected)}")
    if not np.any(raw_seg > 0):
        raise ValueError(f"Empty tumor segmentation in {case_id}")

    label = np.zeros_like(raw_seg, dtype=np.uint8)
    for source_label, target_label in LABEL_MAP.items():
        label[raw_seg == source_label] = target_label
    label = label[None, ...]

    temp_path = out_path.with_suffix(".tmp")
    with temp_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            image=image,
            label=label,
            seg=label[0],
            affine=reference_affine,
            spacing=spacing,
            preprocess_version=np.asarray(PREPROCESS_VERSION),
            case_id=np.asarray(case_id),
        )
    temp_path.replace(out_path)
    return out_path

def prepare_dataset(drive_dir: Path, local_dir: Path, raw_dir: Path) -> List[str]:
    drive_dir = Path(drive_dir)
    local_dir = Path(local_dir)
    raw_dir = Path(raw_dir)
    drive_dir.mkdir(parents=True, exist_ok=True)
    local_dir.mkdir(parents=True, exist_ok=True)

    existing_drive_npz = {f.stem: f for f in drive_dir.glob("*.npz") if not f.name.startswith(".") and f.stat().st_size > 0}
    existing_local_npz = {f.stem: f for f in local_dir.glob("*.npz") if not f.name.startswith(".") and f.stat().st_size > 0}

    raw_cases = []
    if raw_dir.exists():
        raw_cases = sorted(
            directory.name
            for directory in raw_dir.iterdir()
            if directory.is_dir() and (directory.name.startswith("BraTS") or not directory.name.startswith("."))
        )

    reusable_cases = [
        cid for cid in raw_cases
        if cid in existing_drive_npz or cid in existing_local_npz
    ] if raw_cases else list(existing_drive_npz.keys() or existing_local_npz.keys())

    if FORCE_REBUILD_CACHE:
        cases_to_build = raw_cases
    elif raw_cases:
        cases_to_build = [cid for cid in raw_cases if cid not in reusable_cases]
    else:
        cases_to_build = []

    if cases_to_build:
        print(f"Found {len(reusable_cases)} existing NPZ files ({len(reusable_cases)} conversion skipped). Preprocessing {len(cases_to_build)} missing cases from raw NIfTI -> NPZ...")
        for case_id in tqdm(cases_to_build, desc="Preprocess missing cases"):
            preprocess_case(case_id, raw_dir, drive_dir, force=FORCE_REBUILD_CACHE)
        existing_drive_npz = {f.stem: f for f in drive_dir.glob("*.npz") if not f.name.startswith(".") and f.stat().st_size > 0}
    elif raw_cases:
        print(f"All {len(raw_cases)} raw cases already have preprocessed NPZ files ({len(raw_cases)} conversion skipped). Loading directly from cache.")
    elif reusable_cases:
        print(f"Found {len(reusable_cases)} existing preprocessed NPZ files (conversion skipped). Loading directly from cache.")

    drive_files = sorted(drive_dir.glob("*.npz"))
    drive_files = [f for f in drive_files if not f.name.startswith(".") and f.stat().st_size > 0]

    if not drive_files and existing_local_npz:
        drive_files = sorted(local_dir.glob("*.npz"))
        drive_files = [f for f in drive_files if not f.name.startswith(".") and f.stat().st_size > 0]

    if not drive_files:
        raise RuntimeError(f"No NPZ files found in {drive_dir} or {local_dir}, and no raw data available in {raw_dir}.")

    if local_dir.resolve() != drive_dir.resolve() and len(existing_drive_npz) > 0:
        files_to_sync = []
        for source in drive_files:
            destination = local_dir / source.name
            if not destination.exists() or destination.stat().st_size != source.stat().st_size:
                files_to_sync.append((source, destination))

        if files_to_sync:
            for source, destination in tqdm(files_to_sync, desc="Sync NPZ to local SSD"):
                shutil.copy2(source, destination)
        else:
            print(f"Local SSD cache is already up to date ({len(drive_files)} files).")
        target_dir = local_dir
    else:
        target_dir = drive_dir if drive_files else local_dir

    case_ids = sorted([f.stem for f in target_dir.glob("*.npz") if not f.name.startswith(".") and f.stat().st_size > 0])
    if not case_ids and drive_files:
        case_ids = sorted([f.stem for f in drive_files])
    print(f"Loaded {len(case_ids)} preprocessed cases from {target_dir}")
    return case_ids

# 4. Dataset & DataLoaders
class BraTSDataset3D(Dataset):
    def __init__(self, npz_dir: Union[str, Path], case_list: List[str], transforms=None):
        self.npz_dir = Path(npz_dir)
        self.case_list = list(case_list)
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.case_list)

    def __getitem__(self, index: int) -> Dict:
        case_id = self.case_list[index]
        npz_path = self.npz_dir / f"{case_id}.npz"
        if not npz_path.exists() and drive_npz_dir.exists():
            fallback_path = drive_npz_dir / f"{case_id}.npz"
            if fallback_path.exists():
                npz_path = fallback_path

        try:
            cached = load_validated_cache(npz_path)
        except Exception as exc:
            raise RuntimeError(f"Invalid or unreadable cache: {npz_path}") from exc

        sample = {
            "image": cached["image"],
            "label": cached["label"].astype(np.int64),
            "case_id": case_id,
            "affine": cached["affine"],
            "spacing": cached["spacing"],
        }
        return self.transforms(sample) if self.transforms else sample

def main():
    all_cases = prepare_dataset(drive_npz_dir, local_npz_dir, data_root)
    print(f"Valid cases: {len(all_cases)}")

    train_val_cases, test_cases = train_test_split(all_cases, test_size=TEST_SPLIT, random_state=SEED, shuffle=True)
    relative_val_size = VAL_SPLIT / (1.0 - TEST_SPLIT)
    train_cases, val_cases = train_test_split(train_val_cases, test_size=relative_val_size, random_state=SEED, shuffle=True)

    print(f"Train cases: {len(train_cases)}, Val cases: {len(val_cases)}, Test cases: {len(test_cases)}")

if __name__ == "__main__":
    main()
