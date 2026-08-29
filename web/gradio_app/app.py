"""
app.py — Gradio Web Interface cho Brain Tumor Segmentation.

Giao diện 3 tab:
  Tab 1 — 🧠 Phân đoạn khối u: Upload 4 file NIfTI → dự đoán + visualization + xuất PDF
  Tab 2 — 🔬 Xem MRI:          Xem từng modality MRI đã upload (3-view)
  Tab 3 — 📖 Hướng dẫn:        Giải thích quy trình + các nhãn phân đoạn

Chạy local:
    pip install -r requirements.txt
    python app.py

Deploy Hugging Face Spaces:
    Xem README.md hoặc README_HF.md ở root project
"""

from __future__ import annotations

import io
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import nibabel as nib
import numpy as np

# Thêm thư mục cha vào path để import inference_utils từ web/
_THIS_DIR   = Path(__file__).parent
_WEB_DIR    = _THIS_DIR.parent
_PROJ_DIR   = _WEB_DIR.parent
sys.path.insert(0, str(_WEB_DIR))     # để import inference_utils
sys.path.insert(0, str(_THIS_DIR))    # để import visualization, report_generator

from visualization import (
    plot_segmentation_overlay,
    plot_segmentation_to_file,
    plot_mri_multiview,
    build_stats_html,
    MODALITY_DISPLAY,
)
from report_generator import generate_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
MODEL_PATH   = os.getenv("MODEL_PATH", str(_PROJ_DIR / "models" / "unet.onnx"))
MODEL_FORMAT = os.getenv("MODEL_FORMAT", "auto")
_roi_str     = os.getenv("ROI_SIZE", "128,128,128")
ROI_SIZE     = tuple(int(x) for x in _roi_str.split(","))
OVERLAP      = float(os.getenv("OVERLAP", "0.5"))
DEMO_MODE    = os.getenv("DEMO_MODE", "1") == "1"   # True nếu không có model

MODALITIES = ["t1n", "t1c", "t2w", "t2f"]

# ──────────────────────────────────────────────────────────────────────────────
# Lazy model loading
# ──────────────────────────────────────────────────────────────────────────────
_model = None

def _get_model():
    global _model
    if _model is not None:
        return _model
    model_file = Path(MODEL_PATH)
    if not model_file.exists():
        return None
    try:
        from inference_utils import load_model
        _model = load_model(MODEL_PATH, MODEL_FORMAT)
        logger.info(f"Model loaded: {MODEL_PATH}")
        return _model
    except Exception as e:
        logger.error(f"Lỗi load model: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# NIfTI preprocessing helpers
# ──────────────────────────────────────────────────────────────────────────────

def _clip_normalize(volume: np.ndarray) -> np.ndarray:
    brain = volume > 0
    if brain.sum() > 0:
        lo, hi = np.percentile(volume[brain], [0.5, 99.5])
        volume = np.clip(volume, lo, hi)
        mean, std = volume[brain].mean(), volume[brain].std()
        if std > 0:
            volume = (volume - mean) / std
        volume[~brain] = 0.0
    return volume.astype(np.float32)


def _load_nifti_file(file_obj) -> Tuple[np.ndarray, np.ndarray, Tuple]:
    """Load NIfTI từ Gradio file object (path string)."""
    nii = nib.load(str(file_obj.name) if hasattr(file_obj, "name") else str(file_obj))
    arr = nii.get_fdata().astype(np.float32)
    affine  = nii.affine
    spacing = tuple(float(z) for z in nii.header.get_zooms()[:3])
    return arr, affine, spacing


def _preprocess_uploads(files: dict) -> Tuple[np.ndarray, np.ndarray, Tuple]:
    """
    Đọc 4 NIfTI files và trả về:
        image (4, H, W, D), affine (4,4), spacing (3,)
    """
    channels = []
    ref_affine  = None
    ref_spacing = None

    for mod in MODALITIES:
        f = files[mod]
        if f is None:
            raise ValueError(f"Vui lòng upload file {mod.upper()}")
        arr, affine, spacing = _load_nifti_file(f)
        arr = _clip_normalize(arr)
        if ref_affine is None:
            ref_affine  = affine
            ref_spacing = spacing
        channels.append(arr)

    image = np.stack(channels, axis=0)   # (4, H, W, D)
    return image, ref_affine, ref_spacing


# ──────────────────────────────────────────────────────────────────────────────
# Inference (thật hoặc demo)
# ──────────────────────────────────────────────────────────────────────────────

def _run_inference(image: np.ndarray) -> Tuple[np.ndarray, float]:
    """Chạy inference và trả về (mask, elapsed_s)."""
    model = _get_model()

    # ── Real inference ────────────────────────────────────────────────────────
    if model is not None:
        from inference_utils import sliding_window_inference, postprocess
        t0 = time.time()

        def _model_call(x):
            return model(x)

        logits = sliding_window_inference(model, image, ROI_SIZE, OVERLAP)
        mask   = postprocess(logits)
        return mask, time.time() - t0

    # ── Demo mode: tạo mask giả để minh họa giao diện ────────────────────────
    logger.warning("DEMO MODE: Model chưa có — tạo dummy mask")
    H, W, D = image.shape[1:]
    t0 = time.time()
    time.sleep(0.3)   # giả lập thời gian inference

    rng  = np.random.default_rng(42)
    mask = np.zeros((H, W, D), dtype=np.uint8)

    # Tạo một khối cầu giả ở giữa để demo
    cx, cy, cz = H // 2, W // 2, D // 2
    r_outer, r_inner = min(H, W, D) // 6, min(H, W, D) // 12

    zz, yy, xx = np.ogrid[:H, :W, :D]
    dist = np.sqrt((zz - cx)**2 + (yy - cy)**2 + (xx - cz)**2)

    mask[dist < r_outer] = 2   # Edema (xanh)
    mask[dist < r_inner * 1.5] = 3  # ET (cam)
    mask[dist < r_inner] = 1   # NCR (đỏ)

    return mask, time.time() - t0


# ──────────────────────────────────────────────────────────────────────────────
# Main prediction handler
# ──────────────────────────────────────────────────────────────────────────────

def predict_and_visualize(
    t1n_file, t1c_file, t2w_file, t2f_file,
    case_id: str,
    modality_for_bg: str,
    export_nifti: bool,
):
    """
    Handler chính cho Tab 1 — nhận 4 NIfTI files, trả về:
    (overlay_image, stats_html, nifti_path_or_none, status_text)
    """
    files = {
        "t1n": t1n_file,
        "t1c": t1c_file,
        "t2w": t2w_file,
        "t2f": t2f_file,
    }

    # Kiểm tra đủ 4 file
    missing = [k.upper() for k, v in files.items() if v is None]
    if missing:
        return (
            None, None, None, None,
            f"❌ Thiếu file: {', '.join(missing)}. Vui lòng upload đủ 4 modalities.",
        )

    case_id = case_id.strip() or "BraTS-Unknown"
    mod_idx = MODALITIES.index(modality_for_bg) if modality_for_bg in MODALITIES else 3

    try:
        # 1. Tiền xử lý
        image, affine, spacing = _preprocess_uploads(files)

        # 2. Inference
        mask, elapsed = _run_inference(image)

        # 3. Visualization overlay (Gradio Image)
        overlay_arr = plot_segmentation_overlay(
            image_4d=image,
            mask=mask,
            case_id=case_id,
            modality_idx=mod_idx,
        )

        # 4. Stats HTML
        stats_html = build_stats_html(mask, spacing, elapsed)

        # 5. NIfTI output (optional)
        nifti_path = None
        if export_nifti:
            nii_img = nib.Nifti1Image(mask.astype(np.int16), affine)
            tmp = tempfile.NamedTemporaryFile(
                suffix=".nii.gz", prefix=f"seg_{case_id}_", delete=False
            )
            nib.save(nii_img, tmp.name)
            nifti_path = tmp.name

        demo_warn = "\n\n⚠️ **DEMO MODE**: Model chưa được load — mask hiển thị là dummy để minh họa giao diện." if _get_model() is None else ""

        status = f"✅ Phân đoạn hoàn thành trong **{elapsed:.2f}s**{demo_warn}"
        return overlay_arr, stats_html, nifti_path, None, status

    except Exception as e:
        logger.exception("Lỗi prediction")
        return None, None, None, None, f"❌ Lỗi: {str(e)}"


def export_pdf_report(
    t1n_file, t1c_file, t2w_file, t2f_file,
    case_id: str,
    modality_for_bg: str,
):
    """
    Handler xuất PDF report — chạy lại inference để có mask, rồi tạo PDF.
    """
    files = {
        "t1n": t1n_file,
        "t1c": t1c_file,
        "t2w": t2w_file,
        "t2f": t2f_file,
    }
    missing = [k.upper() for k, v in files.items() if v is None]
    if missing:
        return None, f"❌ Thiếu file: {', '.join(missing)}"

    case_id  = case_id.strip() or "BraTS-Unknown"
    mod_idx  = MODALITIES.index(modality_for_bg) if modality_for_bg in MODALITIES else 3
    model_name = "unet3d" if _get_model() is not None else "demo_model"

    try:
        image, affine, spacing = _preprocess_uploads(files)
        mask, elapsed = _run_inference(image)

        # Tạo ảnh overlay để nhúng vào PDF
        slice_img_path = plot_segmentation_to_file(
            image_4d=image, mask=mask, case_id=case_id,
            modality_idx=mod_idx, dpi=150,
        )

        # Tạo PDF
        pdf_path = generate_report(
            case_id=case_id,
            model_name=model_name,
            mask=mask,
            spacing=spacing,
            inference_time_s=elapsed,
            metrics=None,     # Không có ground truth → bỏ metrics
            slice_image_path=slice_img_path,
        )

        return pdf_path, f"✅ PDF đã tạo: {Path(pdf_path).name}"

    except Exception as e:
        logger.exception("Lỗi tạo PDF")
        return None, f"❌ Lỗi tạo PDF: {str(e)}"


# ──────────────────────────────────────────────────────────────────────────────
# MRI Viewer handler (Tab 2)
# ──────────────────────────────────────────────────────────────────────────────

def view_mri(t1n_file, t1c_file, t2w_file, t2f_file, selected_modality: str):
    """Hiển thị 3-view của một modality MRI."""
    files = {
        "t1n": t1n_file,
        "t1c": t1c_file,
        "t2w": t2w_file,
        "t2f": t2f_file,
    }
    mod_key = selected_modality.split("—")[0].strip().lower()
    file_obj = files.get(mod_key)
    if file_obj is None:
        return None, f"❌ Vui lòng upload file {mod_key.upper()} ở Tab 1 trước."
    try:
        arr, _, _ = _load_nifti_file(file_obj)
        display_name = MODALITY_DISPLAY.get(mod_key, mod_key.upper())
        img_arr = plot_mri_multiview(arr, modality_name=display_name)
        return img_arr, "✅ Hiển thị thành công"
    except Exception as e:
        return None, f"❌ Lỗi: {str(e)}"


# ──────────────────────────────────────────────────────────────────────────────
# CSS Theme
# ──────────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', system-ui, sans-serif !important; }

body, .gradio-container {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1527 50%, #0a0e1a 100%) !important;
    min-height: 100vh;
}

.gradio-container {
    max-width: 1200px !important;
}

/* Header */
.app-header {
    text-align: center;
    padding: 28px 20px 10px;
    background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(139,92,246,0.10));
    border-bottom: 1px solid rgba(99,102,241,0.3);
    border-radius: 16px;
    margin-bottom: 20px;
}
.app-header h1 {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #818cf8, #c084fc, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
}
.app-header p {
    color: rgba(200,210,240,0.7);
    font-size: 0.95rem;
    margin: 0;
}

/* Tabs */
.tab-nav button {
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    border-radius: 10px 10px 0 0 !important;
    color: rgba(160,180,220,0.8) !important;
    transition: all 0.2s !important;
}
.tab-nav button.selected {
    color: #a5b4fc !important;
    border-bottom: 2px solid #6366f1 !important;
    background: rgba(99,102,241,0.12) !important;
}

/* Upload areas */
.upload-box .wrap {
    border: 2px dashed rgba(99,102,241,0.4) !important;
    border-radius: 12px !important;
    background: rgba(99,102,241,0.05) !important;
    transition: border-color 0.2s !important;
}
.upload-box .wrap:hover {
    border-color: rgba(99,102,241,0.8) !important;
}

/* Buttons */
button.primary-btn, .btn-primary {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    color: white !important;
    padding: 10px 24px !important;
    cursor: pointer !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.35) !important;
}
button.primary-btn:hover, .btn-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99,102,241,0.5) !important;
}

/* Inputs */
input[type="text"], .gr-text-input {
    background: rgba(15,20,40,0.8) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    border-radius: 8px !important;
    color: #e0e8ff !important;
}

/* Status text */
.status-box {
    padding: 10px 14px;
    border-radius: 8px;
    border-left: 3px solid #6366f1;
    background: rgba(99,102,241,0.08);
    color: #c7d2fe;
    font-size: 0.9rem;
    margin-top: 8px;
}

/* Info cards */
.info-card {
    background: rgba(30,42,70,0.7);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
}
"""

# ──────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    model_status = "🟢 Model sẵn sàng" if Path(MODEL_PATH).exists() else "🟡 Demo Mode (Model chưa load)"

    with gr.Blocks(
        css=CUSTOM_CSS,
        title="Brain Tumor Segmentation",
        theme=gr.themes.Base(
            primary_hue="violet",
            secondary_hue="indigo",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "system-ui"],
        ).set(
            body_background_fill="transparent",
            block_background_fill="rgba(15,22,40,0.7)",
            block_border_color="rgba(99,102,241,0.25)",
            block_border_width="1px",
            block_radius="12px",
            input_background_fill="rgba(10,15,35,0.8)",
            button_primary_background_fill="linear-gradient(135deg, #6366f1, #8b5cf6)",
            button_primary_text_color="white",
        ),
    ) as demo:
        # ── Header ────────────────────────────────────────────────────────────
        gr.HTML(f"""
        <div class="app-header">
          <h1>🧠 Brain Tumor Segmentation</h1>
          <p>Phân đoạn khối u não 3D từ MRI BraTS 2023 · 3D U-Net &amp; Swin UNETR · MONAI</p>
          <p style="margin-top:8px;font-size:0.8rem;color:#64748b;">{model_status} &nbsp;|&nbsp; 4 Modalities: T1n · T1c · T2w · T2f &nbsp;|&nbsp; 3 Tumor Regions: NCR · ED · ET</p>
        </div>
        """)

        # Shared state: lưu file objects giữa các tab
        with gr.Row():
            with gr.Column(scale=1, min_width=200):
                pass  # spacer

        # ── Tabs ──────────────────────────────────────────────────────────────
        with gr.Tabs(elem_classes="tab-nav") as tabs:

            # ================================================================
            # TAB 1 — Phân đoạn khối u
            # ================================================================
            with gr.TabItem("🧠  Phân đoạn khối u", id="tab_predict"):
                with gr.Row():
                    # Left panel: Uploads
                    with gr.Column(scale=1):
                        gr.Markdown("### 📂 Upload 4 file MRI (.nii hoặc .nii.gz)")
                        gr.Markdown(
                            "_BraTS format: T1n · T1c · T2w · T2f — tất cả phải cùng case_",
                            elem_classes="muted",
                        )

                        t1n_input = gr.File(
                            label="T1n — T1 Native",
                            file_types=[".nii", ".nii.gz", ".gz"],
                            elem_classes="upload-box",
                        )
                        t1c_input = gr.File(
                            label="T1c — T1 Post-Contrast",
                            file_types=[".nii", ".nii.gz", ".gz"],
                            elem_classes="upload-box",
                        )
                        t2w_input = gr.File(
                            label="T2w — T2 Weighted",
                            file_types=[".nii", ".nii.gz", ".gz"],
                            elem_classes="upload-box",
                        )
                        t2f_input = gr.File(
                            label="T2f — T2 FLAIR",
                            file_types=[".nii", ".nii.gz", ".gz"],
                            elem_classes="upload-box",
                        )

                        gr.Markdown("---")
                        case_id_input = gr.Textbox(
                            label="Case ID (tùy chọn)",
                            placeholder="VD: BraTS-GLI-00000-000",
                            value="BraTS-Unknown",
                        )
                        bg_modality = gr.Dropdown(
                            label="Modality nền cho overlay",
                            choices=["t2f", "t1n", "t1c", "t2w"],
                            value="t2f",
                        )
                        export_nifti_cb = gr.Checkbox(
                            label="Xuất mask NIfTI (.nii.gz)",
                            value=False,
                        )

                        predict_btn = gr.Button(
                            "🚀 Bắt đầu phân đoạn",
                            variant="primary",
                            size="lg",
                        )

                    # Right panel: Results
                    with gr.Column(scale=2):
                        gr.Markdown("### 📊 Kết quả phân đoạn")

                        status_out = gr.Markdown(
                            "_Kết quả sẽ hiển thị sau khi bạn bấm **Bắt đầu phân đoạn**_",
                            elem_id="status_text",
                        )

                        overlay_out = gr.Image(
                            label="Segmentation Overlay (Axial · Coronal · Sagittal)",
                            show_label=True,
                            type="numpy",
                            elem_id="overlay_img",
                        )

                        stats_out = gr.HTML(
                            label="Thống kê khối u",
                            elem_id="stats_html",
                        )

                        with gr.Row():
                            nifti_out = gr.File(
                                label="Mask NIfTI (download)",
                                visible=True,
                                interactive=False,
                            )

                # ── PDF Export ───────────────────────────────────────────────
                gr.Markdown("---")
                gr.Markdown("### 📄 Xuất báo cáo PDF")
                gr.Markdown(
                    "Báo cáo PDF bao gồm: thông tin case, bảng thể tích khối u, "
                    "ảnh visualization 3-view và disclaimer nghiên cứu."
                )

                with gr.Row():
                    export_pdf_btn = gr.Button(
                        "📋 Xuất báo cáo PDF",
                        variant="secondary",
                        size="lg",
                    )

                with gr.Row():
                    pdf_status_out = gr.Markdown("")
                    pdf_out = gr.File(
                        label="📥 Tải xuống PDF",
                        interactive=False,
                    )

            # ================================================================
            # TAB 2 — MRI Viewer
            # ================================================================
            with gr.TabItem("🔬  Xem MRI", id="tab_viewer"):
                gr.Markdown("### 🔬 MRI Modality Viewer")
                gr.Markdown(
                    "Xem 3-view (Axial · Coronal · Sagittal) của từng modality MRI đã upload.\n"
                    "Vui lòng upload file ở Tab **Phân đoạn khối u** trước."
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        modality_select = gr.Dropdown(
                            label="Chọn Modality",
                            choices=[
                                "t1n — T1 Native",
                                "t1c — T1 Post-Contrast",
                                "t2w — T2 Weighted",
                                "t2f — T2 FLAIR",
                            ],
                            value="t2f — T2 FLAIR",
                        )
                        view_btn = gr.Button("🔍 Hiển thị", variant="primary")
                        viewer_status = gr.Markdown("")

                    with gr.Column(scale=3):
                        mri_viewer_out = gr.Image(
                            label="MRI 3-View",
                            type="numpy",
                            show_label=True,
                        )

            # ================================================================
            # TAB 3 — Hướng dẫn
            # ================================================================
            with gr.TabItem("📖  Hướng dẫn", id="tab_guide"):
                gr.HTML("""
                <div style="max-width:800px;margin:0 auto;color:#c7d2fe;line-height:1.7;">

                  <h2 style="color:#a5b4fc;border-bottom:1px solid rgba(99,102,241,0.3);padding-bottom:8px;">
                    📖 Hướng dẫn sử dụng
                  </h2>

                  <h3 style="color:#818cf8;margin-top:20px;">🔢 Bước 1: Chuẩn bị dữ liệu</h3>
                  <p>Chuẩn bị <strong>4 file MRI NIfTI</strong> (.nii hoặc .nii.gz) của cùng một bệnh nhân, theo chuẩn BraTS 2023:</p>
                  <ul>
                    <li><span style="color:#e41a1c">●</span> <strong>T1n</strong> — T1 Native (trước tiêm thuốc cản quang)</li>
                    <li><span style="color:#4daf4a">●</span> <strong>T1c</strong> — T1 Post-Contrast (sau tiêm)</li>
                    <li><span style="color:#ff7f00">●</span> <strong>T2w</strong> — T2 Weighted</li>
                    <li><span style="color:#9b59b6">●</span> <strong>T2f</strong> — T2 FLAIR (Fluid Attenuated Inversion Recovery)</li>
                  </ul>

                  <h3 style="color:#818cf8;margin-top:20px;">🚀 Bước 2: Upload và phân đoạn</h3>
                  <ol>
                    <li>Vào tab <strong>🧠 Phân đoạn khối u</strong></li>
                    <li>Upload 4 file MRI vào đúng ô tương ứng</li>
                    <li>Nhập Case ID (tùy chọn) để nhận dạng báo cáo</li>
                    <li>Chọn modality nền cho overlay (khuyến nghị T2f/FLAIR)</li>
                    <li>Bấm <strong>🚀 Bắt đầu phân đoạn</strong></li>
                  </ol>

                  <h3 style="color:#818cf8;margin-top:20px;">📊 Kết quả & Các nhãn phân đoạn</h3>
                  <table style="width:100%;border-collapse:collapse;margin-top:10px;">
                    <thead>
                      <tr style="background:rgba(99,102,241,0.2);">
                        <th style="padding:8px;text-align:left;border:1px solid rgba(99,102,241,0.3);">Nhãn</th>
                        <th style="padding:8px;text-align:left;border:1px solid rgba(99,102,241,0.3);">Màu</th>
                        <th style="padding:8px;text-align:left;border:1px solid rgba(99,102,241,0.3);">Ý nghĩa</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td style="padding:8px;border:1px solid rgba(99,102,241,0.2);">0 — Background</td>
                        <td style="padding:8px;border:1px solid rgba(99,102,241,0.2);">Trong suốt</td>
                        <td style="padding:8px;border:1px solid rgba(99,102,241,0.2);">Mô não bình thường / nền</td>
                      </tr>
                      <tr style="background:rgba(228,26,28,0.08);">
                        <td style="padding:8px;border:1px solid rgba(99,102,241,0.2);">1 — NCR</td>
                        <td style="padding:8px;border:1px solid rgba(99,102,241,0.2);"><span style="color:#e41a1c;font-weight:bold;">■ Đỏ</span></td>
                        <td style="padding:8px;border:1px solid rgba(99,102,241,0.2);">Necrotic Core — Lõi hoại tử, trung tâm khối u chết</td>
                      </tr>
                      <tr style="background:rgba(77,175,74,0.08);">
                        <td style="padding:8px;border:1px solid rgba(99,102,241,0.2);">2 — ED</td>
                        <td style="padding:8px;border:1px solid rgba(99,102,241,0.2);"><span style="color:#4daf4a;font-weight:bold;">■ Xanh lá</span></td>
                        <td style="padding:8px;border:1px solid rgba(99,102,241,0.2);">Edema — Vùng phù não xung quanh khối u</td>
                      </tr>
                      <tr style="background:rgba(255,127,0,0.08);">
                        <td style="padding:8px;border:1px solid rgba(99,102,241,0.2);">3 — ET</td>
                        <td style="padding:8px;border:1px solid rgba(99,102,241,0.2);"><span style="color:#ff7f00;font-weight:bold;">■ Cam</span></td>
                        <td style="padding:8px;border:1px solid rgba(99,102,241,0.2);">Enhancing Tumor — Vùng khối u hoạt động, bắt cản quang</td>
                      </tr>
                    </tbody>
                  </table>

                  <h3 style="color:#818cf8;margin-top:24px;">📄 Xuất báo cáo PDF</h3>
                  <p>Sau khi phân đoạn, bấm <strong>📋 Xuất báo cáo PDF</strong> để tải về file PDF bao gồm:</p>
                  <ul>
                    <li>Thông tin case và model</li>
                    <li>Bảng thể tích (cm³) cho WT, TC, NCR, ED, ET</li>
                    <li>Ảnh visualization 3-view overlay</li>
                    <li>Disclaimer nghiên cứu</li>
                  </ul>

                  <div style="background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);border-radius:10px;padding:14px;margin-top:20px;">
                    <strong style="color:#f87171;">⚠️ Disclaimer quan trọng</strong><br>
                    Hệ thống này được phát triển cho mục đích <strong>nghiên cứu</strong> trong khuôn khổ SIC Capstone 2026.
                    Kết quả phân đoạn <strong>KHÔNG được dùng</strong> cho chẩn đoán lâm sàng hoặc ra quyết định điều trị.
                    Mọi kết quả cần được xem xét bởi bác sĩ chuyên khoa thần kinh có kinh nghiệm.
                  </div>

                </div>
                """)

        # ── Event Handlers ────────────────────────────────────────────────────

        # Predict
        predict_btn.click(
            fn=predict_and_visualize,
            inputs=[
                t1n_input, t1c_input, t2w_input, t2f_input,
                case_id_input, bg_modality, export_nifti_cb,
            ],
            outputs=[overlay_out, stats_out, nifti_out, pdf_out, status_out],
        )

        # Export PDF
        export_pdf_btn.click(
            fn=export_pdf_report,
            inputs=[
                t1n_input, t1c_input, t2w_input, t2f_input,
                case_id_input, bg_modality,
            ],
            outputs=[pdf_out, pdf_status_out],
        )

        # MRI Viewer
        view_btn.click(
            fn=view_mri,
            inputs=[
                t1n_input, t1c_input, t2w_input, t2f_input,
                modality_select,
            ],
            outputs=[mri_viewer_out, viewer_status],
        )

    return demo


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        share=False,
        show_error=True,
    )
