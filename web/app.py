"""
app.py — FastAPI Web App cho Brain Tumor Segmentation.

Endpoints:
  GET  /              — Trang chào mừng
  GET  /health        — Healthcheck (JSON)
  POST /predict       — Upload 4 file NIfTI, nhận mask JSON + NIfTI
  GET  /models        — Danh sách model có sẵn

Khởi chạy:
  uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Yêu cầu biến môi trường (tuỳ chọn):
  MODEL_PATH   — Đường dẫn model (.onnx hoặc .pt). Mặc định: ../models/unet.onnx
  MODEL_FORMAT — "onnx" hoặc "torchscript". Mặc định: auto
  ROI_SIZE     — Patch size VD "128,128,128". Mặc định: 128,128,128
  OVERLAP      — Sliding window overlap. Mặc định: 0.5
"""

from __future__ import annotations

import io
import json
import logging
import os
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, HTMLResponse

from inference_utils import (
    load_model,
    predict,
    preprocess_from_bytes,
    mask_to_nifti,
    MODALITIES,
    CLASS_NAMES,
)

# ──────────────────────────────────────────────────────────────────────────────
# Config từ môi trường
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH   = os.getenv("MODEL_PATH",   str(Path(__file__).parent.parent / "models" / "unet.onnx"))
MODEL_FORMAT = os.getenv("MODEL_FORMAT", "auto")
_roi_str     = os.getenv("ROI_SIZE", "128,128,128")
ROI_SIZE     = tuple(int(x) for x in _roi_str.split(","))
OVERLAP      = float(os.getenv("OVERLAP", "0.5"))

# ──────────────────────────────────────────────────────────────────────────────
# App khởi tạo
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Brain Tumor Segmentation API",
    description=(
        "API phân đoạn khối u não 3D từ MRI BraTS 2023. "
        "Upload 4 modalities (T1n, T1c, T2w, T2f) để nhận kết quả segmentation."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model được load lazy khi server khởi động
_model = None


def get_model():
    global _model
    if _model is None:
        if not Path(MODEL_PATH).exists():
            raise HTTPException(
                status_code=503,
                detail=f"Model chưa được train/export. Không tìm thấy: {MODEL_PATH}",
            )
        logger.info(f"Loading model: {MODEL_PATH}")
        _model = load_model(MODEL_PATH, MODEL_FORMAT)
    return _model


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    return """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Brain Tumor Segmentation API</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', system-ui, sans-serif;
                background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                color: #fff;
            }
            .card {
                background: rgba(255,255,255,0.05);
                backdrop-filter: blur(20px);
                border: 1px solid rgba(255,255,255,0.1);
                border-radius: 24px;
                padding: 3rem 4rem;
                max-width: 600px;
                text-align: center;
                box-shadow: 0 25px 50px rgba(0,0,0,0.4);
            }
            .icon { font-size: 4rem; margin-bottom: 1rem; }
            h1 { font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem; }
            .subtitle {
                color: rgba(255,255,255,0.6);
                font-size: 1rem;
                margin-bottom: 2rem;
                line-height: 1.6;
            }
            .badge {
                display: inline-block;
                background: rgba(99,102,241,0.3);
                border: 1px solid rgba(99,102,241,0.5);
                border-radius: 999px;
                padding: 0.3rem 1rem;
                font-size: 0.85rem;
                margin: 0.3rem;
                color: #a5b4fc;
            }
            .links { margin-top: 2rem; display: flex; gap: 1rem; justify-content: center; }
            .btn {
                padding: 0.75rem 2rem;
                border-radius: 12px;
                font-weight: 600;
                text-decoration: none;
                transition: all 0.2s;
                font-size: 0.95rem;
            }
            .btn-primary {
                background: linear-gradient(135deg, #6366f1, #8b5cf6);
                color: #fff;
                box-shadow: 0 4px 15px rgba(99,102,241,0.4);
            }
            .btn-primary:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(99,102,241,0.5); }
            .btn-outline {
                border: 1px solid rgba(255,255,255,0.3);
                color: rgba(255,255,255,0.8);
            }
            .btn-outline:hover { background: rgba(255,255,255,0.1); transform: translateY(-2px); }
        </style>
    </head>
    <body>
        <div class="card">
            <div class="icon">🧠</div>
            <h1>Brain Tumor Segmentation</h1>
            <p class="subtitle">
                API phân đoạn khối u não 3D từ MRI BraTS 2023<br>
                Sử dụng 3D U-Net & Swin UNETR với MONAI
            </p>
            <div>
                <span class="badge">4 Modalities: T1n · T1c · T2w · T2f</span>
                <span class="badge">3 Tumor Regions</span>
                <span class="badge">ONNX / TorchScript</span>
            </div>
            <div class="links">
                <a href="/docs" class="btn btn-primary">📋 API Docs</a>
                <a href="/health" class="btn btn-outline">💚 Health Check</a>
                <a href="/redoc" class="btn btn-outline">📄 ReDoc</a>
            </div>
        </div>
    </body>
    </html>
    """


@app.get("/health", tags=["System"])
async def health():
    """Kiểm tra trạng thái server và model."""
    model_path = Path(MODEL_PATH)
    model_ready = model_path.exists()
    return {
        "status": "ok" if model_ready else "model_not_found",
        "model_path": str(model_path),
        "model_exists": model_ready,
        "roi_size": list(ROI_SIZE),
        "overlap": OVERLAP,
    }


@app.get("/models", tags=["System"])
async def list_models():
    """Liệt kê các file model (.onnx, .pt) trong thư mục models/."""
    models_dir = Path(MODEL_PATH).parent
    models = []
    if models_dir.exists():
        for ext in ("*.onnx", "*.pt"):
            for f in models_dir.glob(ext):
                models.append({
                    "name": f.name,
                    "path": str(f),
                    "size_mb": round(f.stat().st_size / 1024 / 1024, 2),
                    "format": "onnx" if f.suffix == ".onnx" else "torchscript",
                })
    return {"models": models, "models_dir": str(models_dir)}


@app.post("/predict", tags=["Inference"])
async def predict_endpoint(
    t1n: UploadFile = File(..., description="T1 native MRI (.nii.gz)"),
    t1c: UploadFile = File(..., description="T1 post-contrast MRI (.nii.gz)"),
    t2w: UploadFile = File(..., description="T2 weighted MRI (.nii.gz)"),
    t2f: UploadFile = File(..., description="T2 FLAIR MRI (.nii.gz)"),
    return_nifti: bool = False,
):
    """
    Phân đoạn khối u não từ 4 modalities MRI.

    **Upload 4 file NIfTI** (.nii.gz):
    - **t1n**: T1 native
    - **t1c**: T1 post-contrast
    - **t2w**: T2 weighted
    - **t2f**: T2 FLAIR

    **Trả về**:
    - JSON: mask flattened + metadata (mặc định)
    - NIfTI binary nếu `return_nifti=true`

    **Nhãn mask**: 0=Background, 1=NCR, 2=ED, 3=ET
    """
    model = get_model()

    # Đọc bytes từ các file upload
    try:
        nii_bytes = {
            "t1n": await t1n.read(),
            "t1c": await t1c.read(),
            "t2w": await t2w.read(),
            "t2f": await t2f.read(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi đọc file upload: {e}")

    # Kiểm tra file không rỗng
    for mod, data in nii_bytes.items():
        if len(data) == 0:
            raise HTTPException(status_code=400, detail=f"File '{mod}' rỗng")

    # Tiền xử lý
    try:
        image, affine, spacing = preprocess_from_bytes(nii_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Lỗi tiền xử lý NIfTI: {e}")

    # Dự đoán
    try:
        mask, metadata = predict(model, image, roi_size=ROI_SIZE, overlap=OVERLAP)
    except Exception as e:
        logger.exception("Lỗi khi chạy inference")
        raise HTTPException(status_code=500, detail=f"Lỗi inference: {e}")

    metadata["spacing_mm"] = list(spacing)
    metadata["modalities_received"] = list(nii_bytes.keys())

    if return_nifti:
        nifti_img = mask_to_nifti(mask, affine)
        buf = io.BytesIO()
        nib.save(nifti_img, buf)
        buf.seek(0)
        return Response(
            content=buf.read(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": "attachment; filename=segmentation.nii.gz"},
        )

    # Trả JSON (mask flatten để tránh quá nặng; client tự reshape)
    return JSONResponse({
        "mask": mask.flatten().tolist(),
        "mask_shape": list(mask.shape),
        "metadata": metadata,
        "class_legend": {str(k): v for k, v in CLASS_NAMES.items()},
    })


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
