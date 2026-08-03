"""
inference_utils.py — Tiện ích suy diễn (inference) cho Web App.

Hỗ trợ 2 loại model:
  - ONNX Runtime (khuyến nghị cho production, không cần PyTorch)
  - PyTorch TorchScript (.pt) — fallback nếu cần

Quy trình:
  nii_paths (4 file .nii.gz) -> preprocess -> tensor -> model -> mask (numpy)
"""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
MODALITIES = ["t1n", "t1c", "t2w", "t2f"]
NUM_CLASSES = 4          # 0=Background, 1=NCR, 2=ED, 3=ET
ROI_SIZE    = (128, 128, 128)
OVERLAP     = 0.5

CLASS_NAMES = {
    0: "Background",
    1: "NCR (Necrotic Core)",
    2: "ED (Edema)",
    3: "ET (Enhancing Tumor)",
}

# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing (same as BraTSDataset3D in notebook)
# ──────────────────────────────────────────────────────────────────────────────

def _clip_and_normalize(volume: np.ndarray) -> np.ndarray:
    """Percentile clip [0.5, 99.5] + z-score normalization trên vùng não."""
    brain_mask = volume > 0
    if brain_mask.sum() > 0:
        vals = volume[brain_mask]
        lo, hi = np.percentile(vals, [0.5, 99.5])
        volume = np.clip(volume, lo, hi)
        mean = volume[brain_mask].mean()
        std  = volume[brain_mask].std()
        if std > 0:
            volume = (volume - mean) / std
        volume[~brain_mask] = 0.0
    return volume.astype(np.float32)


def preprocess_case(
    nii_paths: Dict[str, Union[str, Path]],
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, ...]]:
    """
    Đọc 4 file NIfTI và trả về image tensor + affine + spacing.

    Args:
        nii_paths: dict với keys = MODALITIES, values = đường dẫn file .nii.gz
                   Ví dụ: {"t1n": "path/t1n.nii.gz", "t1c": ..., "t2w": ..., "t2f": ...}

    Returns:
        image   : np.float32 array shape (4, H, W, D)
        affine  : np.ndarray (4, 4)
        spacing : tuple of 3 floats (voxel size mm)
    """
    channels = []
    ref_nii = None
    for mod in MODALITIES:
        path = nii_paths.get(mod)
        if path is None:
            raise ValueError(f"Thiếu modality '{mod}' trong nii_paths")
        nii = nib.load(str(path))
        if ref_nii is None:
            ref_nii = nii
        arr = nii.get_fdata().astype(np.float32)
        arr = _clip_and_normalize(arr)
        channels.append(arr)

    image = np.stack(channels, axis=0)  # (4, H, W, D)
    affine = ref_nii.affine
    spacing = tuple(float(z) for z in ref_nii.header.get_zooms()[:3])
    return image, affine, spacing


def preprocess_from_bytes(
    nii_bytes: Dict[str, bytes],
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, ...]]:
    """
    Giống preprocess_case nhưng nhận raw bytes (dùng cho FastAPI UploadFile).

    Args:
        nii_bytes: dict keys = MODALITIES, values = bytes của file .nii.gz
    """
    channels = []
    ref_nii = None
    for mod in MODALITIES:
        raw = nii_bytes.get(mod)
        if raw is None:
            raise ValueError(f"Thiếu modality '{mod}' trong nii_bytes")
        fh = io.BytesIO(raw)
        nii = nib.load(fh)
        if ref_nii is None:
            ref_nii = nii
        arr = nii.get_fdata().astype(np.float32)
        arr = _clip_and_normalize(arr)
        channels.append(arr)

    image = np.stack(channels, axis=0)
    affine = ref_nii.affine
    spacing = tuple(float(z) for z in ref_nii.header.get_zooms()[:3])
    return image, affine, spacing


# ──────────────────────────────────────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────────────────────────────────────

class _ONNXModel:
    """Wrapper nhỏ gọn cho ONNX Runtime session."""

    def __init__(self, onnx_path: str):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime chưa được cài. Chạy: pip install onnxruntime-gpu  "
                "(hoặc onnxruntime nếu không có GPU)"
            )
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if _cuda_available()
            else ["CPUExecutionProvider"]
        )
        logger.info(f"Loading ONNX model from {onnx_path} (providers={providers})")
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """x: (1, 4, H, W, D) float32 numpy -> (1, C, H, W, D) float32 numpy"""
        return self.session.run([self.output_name], {self.input_name: x})[0]


class _TorchScriptModel:
    """Wrapper cho TorchScript (.pt) model."""

    def __init__(self, pt_path: str, device: str = "auto"):
        import torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        logger.info(f"Loading TorchScript model from {pt_path} on {device}")
        self.model = torch.jit.load(pt_path, map_location=device)
        self.model.eval()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        import torch
        t = torch.from_numpy(x).to(self.device)
        with torch.no_grad():
            out = self.model(t)
        return out.cpu().numpy()


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def load_model(model_path: Union[str, Path], model_format: str = "auto"):
    """
    Load model từ file .onnx hoặc .pt (TorchScript).

    Args:
        model_path  : Đường dẫn tới file model
        model_format: "onnx", "torchscript", hoặc "auto" (detect từ extension)

    Returns:
        Callable model (nhận numpy array, trả numpy array)
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file không tồn tại: {path}")

    if model_format == "auto":
        ext = path.suffix.lower()
        if ext == ".onnx":
            model_format = "onnx"
        elif ext in (".pt", ".pth"):
            model_format = "torchscript"
        else:
            raise ValueError(f"Không nhận ra định dạng model từ extension '{ext}'")

    if model_format == "onnx":
        return _ONNXModel(str(path))
    elif model_format == "torchscript":
        return _TorchScriptModel(str(path))
    else:
        raise ValueError(f"model_format phải là 'onnx' hoặc 'torchscript', nhận '{model_format}'")


# ──────────────────────────────────────────────────────────────────────────────
# Sliding Window Inference (pure numpy — không cần PyTorch)
# ──────────────────────────────────────────────────────────────────────────────

def sliding_window_inference(
    model,
    image: np.ndarray,
    roi_size: Tuple[int, int, int] = ROI_SIZE,
    overlap: float = OVERLAP,
) -> np.ndarray:
    """
    Sliding window inference thuần numpy.

    Args:
        model    : Callable (1,4,rH,rW,rD) -> (1,C,rH,rW,rD)
        image    : (4, H, W, D) float32
        roi_size : patch size
        overlap  : tỷ lệ overlap [0, 1)

    Returns:
        logits: (C, H, W, D) float32 — trung bình có trọng số Gaussian
    """
    C_in, H, W, D = image.shape
    rH, rW, rD = roi_size
    C_out = NUM_CLASSES

    stride = tuple(max(1, int(r * (1 - overlap))) for r in roi_size)

    # Output accumulator + weight map
    output  = np.zeros((C_out, H, W, D), dtype=np.float32)
    weights = np.zeros((H, W, D), dtype=np.float32)

    # Gaussian weight kernel
    gw = _gaussian_kernel(roi_size)

    # Generate start positions
    starts_h = _sliding_starts(H, rH, stride[0])
    starts_w = _sliding_starts(W, rW, stride[1])
    starts_d = _sliding_starts(D, rD, stride[2])

    total = len(starts_h) * len(starts_w) * len(starts_d)
    logger.info(f"Sliding window: {total} patches, roi={roi_size}, overlap={overlap}")

    for h0 in starts_h:
        for w0 in starts_w:
            for d0 in starts_d:
                h1, w1, d1 = h0 + rH, w0 + rW, d0 + rD
                patch = image[:, h0:h1, w0:w1, d0:d1][np.newaxis]  # (1,4,rH,rW,rD)
                pred  = model(patch)[0]                              # (C,rH,rW,rD)
                output[:, h0:h1, w0:w1, d0:d1] += pred * gw
                weights[h0:h1, w0:w1, d0:d1]   += gw

    weights = np.maximum(weights, 1e-8)
    output /= weights[np.newaxis]
    return output


def _sliding_starts(size: int, roi: int, stride: int) -> List[int]:
    starts = list(range(0, max(1, size - roi), stride))
    if not starts or starts[-1] + roi < size:
        starts.append(max(0, size - roi))
    return starts


def _gaussian_kernel(roi_size: Tuple[int, int, int]) -> np.ndarray:
    """Gaussian importance map cho sliding window."""
    def _g1d(n):
        x = np.linspace(-1, 1, n)
        return np.exp(-0.5 * (x / 0.5) ** 2).astype(np.float32)

    gz = _g1d(roi_size[2])
    gy = _g1d(roi_size[1])
    gx = _g1d(roi_size[0])
    return gx[:, None, None] * gy[None, :, None] * gz[None, None, :]


# ──────────────────────────────────────────────────────────────────────────────
# Postprocessing
# ──────────────────────────────────────────────────────────────────────────────

def postprocess(logits: np.ndarray) -> np.ndarray:
    """
    Chuyển logits (C, H, W, D) thành mask (H, W, D) uint8.
    Áp dụng argmax theo channel dimension.
    """
    mask = np.argmax(logits, axis=0).astype(np.uint8)
    return mask


# ──────────────────────────────────────────────────────────────────────────────
# High-level predict API
# ──────────────────────────────────────────────────────────────────────────────

def predict(
    model,
    image: np.ndarray,
    roi_size: Tuple[int, int, int] = ROI_SIZE,
    overlap: float = OVERLAP,
) -> Tuple[np.ndarray, Dict]:
    """
    Suy diễn đầy đủ: image tensor -> mask + metadata.

    Args:
        model  : model được load bởi load_model()
        image  : (4, H, W, D) float32
        roi_size, overlap: tham số sliding window

    Returns:
        mask    : (H, W, D) uint8 — 0=BG, 1=NCR, 2=ED, 3=ET
        metadata: dict chứa thống kê kết quả
    """
    t0 = time.time()
    logits = sliding_window_inference(model, image, roi_size, overlap)
    mask   = postprocess(logits)
    elapsed = time.time() - t0

    # Tumor statistics
    total_vox = mask.size
    stats = {}
    for cls_id, cls_name in CLASS_NAMES.items():
        vox = int((mask == cls_id).sum())
        stats[cls_name] = {
            "voxels": vox,
            "percent": round(100.0 * vox / total_vox, 4),
        }

    metadata = {
        "inference_time_s": round(elapsed, 2),
        "mask_shape": list(mask.shape),
        "class_stats": stats,
        "has_tumor": bool((mask > 0).any()),
    }
    logger.info(f"Predict done in {elapsed:.2f}s | tumor={'yes' if metadata['has_tumor'] else 'no'}")
    return mask, metadata


def mask_to_nifti(mask: np.ndarray, affine: np.ndarray) -> nib.Nifti1Image:
    """Chuyển mask numpy thành NIfTI image để download."""
    return nib.Nifti1Image(mask.astype(np.int16), affine)
