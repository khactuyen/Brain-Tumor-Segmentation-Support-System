"""
report_generator.py — Tạo PDF research report cho kết quả phân đoạn khối u não.

Port từ notebook SIC_Capstone_v2.ipynb (Cell 21-22).
Thích nghi để dùng trong Gradio web app: nhận numpy arrays thay vì
đọc từ file, trả về bytes/path thay vì lưu trực tiếp.
"""

from __future__ import annotations

import io
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = {
    0: "Background",
    1: "NCR (Necrotic Core)",
    2: "ED (Edema)",
    3: "ET (Enhancing Tumor)",
}

REGION_LABELS = {
    "Whole Tumor (WT)": "wt",
    "Tumor Core (TC)":  "tc",
    "Enhancing Tumor (ET)": "et",
}

REGION_COLORS = {
    "NCR (Necrotic Core)": (228, 26, 28),    # #e41a1c — đỏ
    "ED (Edema)":           (77, 175, 74),    # #4daf4a — xanh lá
    "ET (Enhancing Tumor)": (255, 127, 0),    # #ff7f00 — cam
}


# ──────────────────────────────────────────────────────────────────────────────
# Volume calculation helpers
# ──────────────────────────────────────────────────────────────────────────────

def compute_volumes(
    mask: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Dict[str, float]:
    """
    Tính thể tích (cm³) cho từng vùng khối u.

    Args:
        mask   : (H, W, D) uint8 — 0=BG, 1=NCR, 2=ED, 3=ET
        spacing: kích thước voxel theo mm (vx, vy, vz)

    Returns:
        dict với keys: 'ncr_cm3', 'ed_cm3', 'et_cm3',
                       'wt_cm3', 'tc_cm3', voxel_volume_cm3
    """
    voxel_vol_mm3 = float(np.prod(spacing))
    voxel_vol_cm3 = voxel_vol_mm3 / 1000.0

    ncr_vox = int((mask == 1).sum())
    ed_vox  = int((mask == 2).sum())
    et_vox  = int((mask == 3).sum())
    wt_vox  = int((mask > 0).sum())                   # WT = NCR + ED + ET
    tc_vox  = int(np.isin(mask, [1, 3]).sum())        # TC = NCR + ET

    return {
        "ncr_cm3": round(ncr_vox * voxel_vol_cm3, 3),
        "ed_cm3":  round(ed_vox  * voxel_vol_cm3, 3),
        "et_cm3":  round(et_vox  * voxel_vol_cm3, 3),
        "wt_cm3":  round(wt_vox  * voxel_vol_cm3, 3),
        "tc_cm3":  round(tc_vox  * voxel_vol_cm3, 3),
        "voxel_volume_cm3": voxel_vol_cm3,
        "ncr_voxels": ncr_vox,
        "ed_voxels":  ed_vox,
        "et_voxels":  et_vox,
        "wt_voxels":  wt_vox,
        "tc_voxels":  tc_vox,
    }


# ──────────────────────────────────────────────────────────────────────────────
# PDF Report Class (port từ ResearchSegmentationReport trong notebook)
# ──────────────────────────────────────────────────────────────────────────────

def _make_pdf_report(
    case_id: str,
    model_name: str,
    mask: np.ndarray,
    spacing: Tuple[float, float, float],
    inference_time_s: float,
    metrics: Optional[Dict] = None,
    slice_image_path: Optional[str] = None,
) -> bytes:
    """
    Tạo PDF report và trả về bytes.

    Args:
        case_id          : ID case (VD: "BraTS-GLI-00000-000")
        model_name       : Tên model ("unet3d" hoặc "swin_unetr")
        mask             : (H, W, D) uint8 mask
        spacing          : (sx, sy, sz) voxel size mm
        inference_time_s : Thời gian inference (giây)
        metrics          : Optional dict chứa Dice/IoU/HD95 (có thể None)
        slice_image_path : Optional path tới ảnh PNG của overlay

    Returns:
        bytes của file PDF
    """
    try:
        from fpdf import FPDF
    except ImportError:
        raise ImportError(
            "fpdf2 chưa được cài. Chạy: pip install fpdf2"
        )

    volumes = compute_volumes(mask, spacing)
    has_tumor = bool((mask > 0).any())
    report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Khởi tạo PDF ──────────────────────────────────────────────────────────
    class _BrainTumorReport(FPDF):
        def header(self):
            self.set_font("Helvetica", "B", 14)
            self.cell(
                0, 8,
                "BRAIN TUMOR SEGMENTATION RESEARCH REPORT",
                new_x="LMARGIN", new_y="NEXT", align="C",
            )
            self.set_font("Helvetica", "I", 8)
            self.cell(
                0, 5,
                "Research use only — not for clinical diagnosis or treatment",
                new_x="LMARGIN", new_y="NEXT", align="C",
            )
            self.set_draw_color(180, 180, 180)
            self.line(self.l_margin, self.get_y() + 1,
                      self.w - self.r_margin, self.get_y() + 1)
            self.ln(4)

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(
                0, 10,
                f"Page {self.page_no()} | SIC Capstone 2026 — Brain Tumor Segmentation",
                align="C",
            )
            self.set_text_color(0, 0, 0)

    pdf = _BrainTumorReport()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── Thông tin Case ────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_fill_color(230, 235, 245)
    pdf.cell(0, 7, "Case Information", new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf.ln(2)

    pdf.set_font("Helvetica", "", 10)
    info_rows = [
        ("Case ID",          case_id),
        ("Model",            model_name),
        ("Report Generated", report_time),
        ("Inference Time",   f"{inference_time_s:.2f} s"),
        ("Voxel Spacing",    f"{spacing[0]:.2f} × {spacing[1]:.2f} × {spacing[2]:.2f} mm"),
        ("Volume Shape",     f"{mask.shape[0]} × {mask.shape[1]} × {mask.shape[2]} voxels"),
        ("Tumor Detected",   "YES ✓" if has_tumor else "NO — No tumor voxels found"),
    ]
    for label, value in info_rows:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(55, 6, f"{label}:", border="B")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, str(value), border="B", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # ── Bảng Thể Tích Khối U ─────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_fill_color(230, 235, 245)
    pdf.cell(0, 7, "Tumor Volume Summary", new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf.ln(2)

    # Header
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(200, 210, 230)
    col_w = [65, 35, 35, 35]
    headers = ["Region", "Voxels", "Volume (cm³)", "% of Brain"]
    total_vox = mask.size
    for i, h in enumerate(headers):
        pdf.cell(col_w[i], 7, h, border=1, fill=True, align="C")
    pdf.ln()

    # Rows
    pdf.set_font("Helvetica", "", 9)
    vol_rows = [
        ("Whole Tumor (WT)",     volumes["wt_voxels"],  volumes["wt_cm3"]),
        ("  Tumor Core (TC)",    volumes["tc_voxels"],  volumes["tc_cm3"]),
        ("    NCR — Necrotic",   volumes["ncr_voxels"], volumes["ncr_cm3"]),
        ("    ET — Enhancing",   volumes["et_voxels"],  volumes["et_cm3"]),
        ("ED — Edema",           volumes["ed_voxels"],  volumes["ed_cm3"]),
    ]
    for i, (region, vox, vol_cm3) in enumerate(vol_rows):
        fill = i % 2 == 0
        pdf.set_fill_color(245, 248, 255) if fill else pdf.set_fill_color(255, 255, 255)
        pct = 100.0 * vox / total_vox if total_vox > 0 else 0.0
        pdf.cell(col_w[0], 6, region, border=1, fill=fill)
        pdf.cell(col_w[1], 6, f"{vox:,}", border=1, fill=fill, align="R")
        pdf.cell(col_w[2], 6, f"{vol_cm3:.3f}", border=1, fill=fill, align="R")
        pdf.cell(col_w[3], 6, f"{pct:.4f}%", border=1, fill=fill, align="R")
        pdf.ln()
    pdf.ln(5)

    # ── Metrics (nếu có) ──────────────────────────────────────────────────────
    if metrics:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_fill_color(230, 235, 245)
        pdf.cell(0, 7, "Segmentation Metrics", new_x="LMARGIN", new_y="NEXT", fill=True)
        pdf.ln(2)

        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(200, 210, 230)
        metric_cols = [45, 22, 22, 22, 22, 22, 22]
        metric_headers = ["Region", "Dice", "IoU", "Precision", "Recall", "HD95 (mm)", "Vol cm³"]
        for w, h in zip(metric_cols, metric_headers):
            pdf.cell(w, 7, h, border=1, fill=True, align="C")
        pdf.ln()

        pdf.set_font("Helvetica", "", 9)
        for i, region_label in enumerate(["WT", "TC", "ET"]):
            key = region_label.lower()
            row = [
                f"  {region_label}",
                f"{metrics.get(f'dice_{key}', float('nan')):.4f}",
                f"{metrics.get(f'iou_{key}', float('nan')):.4f}",
                f"{metrics.get(f'precision_{key}', float('nan')):.4f}",
                f"{metrics.get(f'recall_{key}', float('nan')):.4f}",
                f"{metrics.get(f'hd95_{key}', float('nan')):.2f}" if not np.isnan(metrics.get(f'hd95_{key}', float('nan'))) else "N/A",
                f"{volumes.get(f'{key}_cm3', 0):.3f}",
            ]
            fill = i % 2 == 0
            pdf.set_fill_color(245, 248, 255) if fill else pdf.set_fill_color(255, 255, 255)
            for w, val in zip(metric_cols, row):
                pdf.cell(w, 6, val, border=1, fill=fill, align="C")
            pdf.ln()
        pdf.ln(5)

    # ── Ảnh Overlay (nếu có) ─────────────────────────────────────────────────
    if slice_image_path and Path(slice_image_path).exists():
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_fill_color(230, 235, 245)
        pdf.cell(0, 7, "Segmentation Visualization", new_x="LMARGIN", new_y="NEXT", fill=True)
        pdf.ln(3)

        # Căn giữa ảnh
        img_w = 170
        x_offset = (pdf.w - img_w) / 2
        pdf.image(str(slice_image_path), x=x_offset, w=img_w)
        pdf.ln(3)

        # Caption
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 5,
                 "Colors: Red = NCR (Necrotic Core) | Green = ED (Edema) | Orange = ET (Enhancing Tumor)",
                 align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        pdf.ln(4)

    # ── Disclaimer / Limitations ──────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_fill_color(255, 230, 230)
    pdf.cell(0, 7, "⚠ Limitations & Disclaimer", new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf.ln(2)

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 0, 0)
    disclaimers = [
        "• This report is for RESEARCH PURPOSES ONLY and must not be used for clinical decision-making.",
        "• Results are generated by an AI model trained on BraTS 2023 dataset and may contain errors.",
        "• Segmentation accuracy depends on MRI quality, preprocessing, and scanner parameters.",
        "• All findings must be verified by a qualified neuroradiologist before any clinical use.",
        "• The model has not been validated for clinical deployment (CE/FDA clearance not obtained).",
    ]
    for line in disclaimers:
        pdf.multi_cell(0, 5, line)
    pdf.set_text_color(0, 0, 0)

    # ── Output bytes ──────────────────────────────────────────────────────────
    return bytes(pdf.output())


def generate_report(
    case_id: str,
    model_name: str,
    mask: np.ndarray,
    spacing: Tuple[float, float, float],
    inference_time_s: float,
    metrics: Optional[Dict] = None,
    slice_image_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> str:
    """
    Tạo PDF report và lưu ra file.

    Args:
        case_id, model_name, mask, spacing, inference_time_s, metrics: như _make_pdf_report
        slice_image_path : Optional path PNG ảnh visualization
        output_path      : Nếu None → lưu vào tempfile và trả về path

    Returns:
        str — đường dẫn tới file PDF đã lưu
    """
    pdf_bytes = _make_pdf_report(
        case_id=case_id,
        model_name=model_name,
        mask=mask,
        spacing=spacing,
        inference_time_s=inference_time_s,
        metrics=metrics,
        slice_image_path=slice_image_path,
    )

    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(
            suffix=".pdf",
            prefix=f"brats_report_{case_id}_",
            delete=False,
        )
        output_path = tmp.name
        tmp.close()

    Path(output_path).write_bytes(pdf_bytes)
    return output_path
