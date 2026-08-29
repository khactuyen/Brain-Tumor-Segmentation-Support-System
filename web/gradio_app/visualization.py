"""
visualization.py — Trực quan hóa kết quả phân đoạn MRI.

Port từ hàm visualize_slices trong notebook SIC_Capstone_v2.ipynb (Cell 19).
Trả về matplotlib Figure hoặc PIL Image để dùng trong Gradio.
"""

from __future__ import annotations

import io
import tempfile
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import ListedColormap
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
# Constants — màu mask (giống hệt notebook)
# ──────────────────────────────────────────────────────────────────────────────
# Transparent cho background, sau đó NCR, ED, ET
SEG_CMAP = ListedColormap(["none", "#e41a1c", "#4daf4a", "#ff7f00"])
SEG_ALPHA = 0.55

LABEL_COLORS = {
    1: ("#e41a1c", "NCR — Necrotic Core"),
    2: ("#4daf4a", "ED — Edema"),
    3: ("#ff7f00", "ET — Enhancing Tumor"),
}

MODALITY_DISPLAY = {
    "t1n": "T1 Native",
    "t1c": "T1 Post-Contrast",
    "t2w": "T2 Weighted",
    "t2f": "T2 FLAIR",
}

# ──────────────────────────────────────────────────────────────────────────────
# Slice selection helper
# ──────────────────────────────────────────────────────────────────────────────

def _find_best_slices(mask: np.ndarray) -> Tuple[int, int, int]:
    """
    Tìm slice có nhiều voxel tumor nhất cho 3 mặt phẳng.
    Trả về (axial_z, coronal_y, sagittal_x).
    Fallback về slice giữa nếu mask rỗng.
    """
    H, W, D = mask.shape

    if mask.max() == 0:
        return D // 2, H // 2, W // 2

    z_sum = np.sum(mask > 0, axis=(0, 1))   # axial   (D,)
    y_sum = np.sum(mask > 0, axis=(0, 2))   # coronal (H,)
    x_sum = np.sum(mask > 0, axis=(1, 2))   # sagittal(W,)

    z_idx = int(np.argmax(z_sum)) if z_sum.max() > 0 else D // 2
    y_idx = int(np.argmax(y_sum)) if y_sum.max() > 0 else H // 2
    x_idx = int(np.argmax(x_sum)) if x_sum.max() > 0 else W // 2

    return z_idx, y_idx, x_idx


def _normalize_display(volume: np.ndarray) -> np.ndarray:
    """Normalize về [0, 1] để hiển thị."""
    lo, hi = volume.min(), volume.max()
    if hi == lo:
        return np.zeros_like(volume, dtype=np.float32)
    return ((volume - lo) / (hi - lo)).astype(np.float32)

# ──────────────────────────────────────────────────────────────────────────────
# Main visualization: 3-view segmentation overlay
# ──────────────────────────────────────────────────────────────────────────────

def plot_segmentation_overlay(
    image_4d: np.ndarray,
    mask: np.ndarray,
    case_id: str = "Unknown",
    modality_idx: int = 3,       # mặc định FLAIR (index 3)
    figsize: Tuple = (15, 5),
    dpi: int = 120,
) -> np.ndarray:
    """
    Vẽ 3-view overlay (axial / coronal / sagittal) — giống notebook Cell 19.

    Args:
        image_4d    : (4, H, W, D) float32 — 4 modalities
        mask        : (H, W, D) uint8 — segmentation mask
        case_id     : Tên case để hiển thị trên title
        modality_idx: Index modality dùng làm background (0-3)
        figsize     : Kích thước figure
        dpi         : DPI output

    Returns:
        np.ndarray (H, W, 3) uint8 — ảnh RGB để dùng trong gr.Image
    """
    bg = _normalize_display(image_4d[modality_idx])   # (H, W, D)
    H, W, D = bg.shape
    z_idx, y_idx, x_idx = _find_best_slices(mask)

    views = [
        ("Axial",    bg[:, :, z_idx],   mask[:, :, z_idx]),
        ("Coronal",  bg[:, y_idx, :],   mask[:, y_idx, :]),
        ("Sagittal", bg[x_idx, :, :],   mask[x_idx, :, :]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=figsize, facecolor="#0d0d1a")
    fig.suptitle(
        f"Brain Tumor Segmentation — {case_id}",
        fontsize=13, color="white", fontweight="bold", y=1.01,
    )

    for ax, (title, bg_slice, mask_slice) in zip(axes, views):
        ax.set_facecolor("#0d0d1a")
        # Background MRI (gray)
        ax.imshow(
            np.rot90(bg_slice), cmap="gray",
            aspect="equal", interpolation="bilinear",
        )
        # Mask overlay
        mask_display = np.rot90(mask_slice.astype(np.float32))
        mask_display[mask_display == 0] = np.nan   # background trong suốt
        ax.imshow(
            mask_display, cmap=SEG_CMAP,
            alpha=SEG_ALPHA, vmin=0, vmax=3,
            aspect="equal", interpolation="nearest",
        )
        ax.set_title(title, color="white", fontsize=10, pad=4)
        ax.axis("off")

    # Legend
    legend_patches = [
        mpatches.Patch(color=color, label=label)
        for _, (color, label) in LABEL_COLORS.items()
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=3,
        fontsize=9,
        framealpha=0.2,
        labelcolor="white",
        facecolor="#1a1a2e",
        edgecolor="none",
        bbox_to_anchor=(0.5, -0.08),
    )

    plt.tight_layout(pad=0.5)

    # Render ra numpy array
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    return np.array(img)


def plot_segmentation_to_file(
    image_4d: np.ndarray,
    mask: np.ndarray,
    case_id: str = "Unknown",
    modality_idx: int = 3,
    dpi: int = 150,
) -> str:
    """
    Giống plot_segmentation_overlay nhưng lưu ra file PNG tạm.
    Trả về path để nhúng vào PDF.
    """
    arr = plot_segmentation_overlay(
        image_4d, mask, case_id, modality_idx, dpi=dpi,
    )
    img = Image.fromarray(arr)
    tmp = tempfile.NamedTemporaryFile(
        suffix=".png", prefix="brats_overlay_", delete=False
    )
    img.save(tmp.name, dpi=(dpi, dpi))
    tmp.close()
    return tmp.name


# ──────────────────────────────────────────────────────────────────────────────
# Single-modality MRI viewer
# ──────────────────────────────────────────────────────────────────────────────

def plot_mri_multiview(
    volume: np.ndarray,
    modality_name: str = "MRI",
    figsize: Tuple = (15, 5),
    dpi: int = 110,
) -> np.ndarray:
    """
    Hiển thị 3-view của một modality MRI (không có mask).

    Args:
        volume       : (H, W, D) float32 — một modality
        modality_name: Tên hiển thị
        figsize, dpi : Layout params

    Returns:
        np.ndarray (H, W, 3) uint8
    """
    vol = _normalize_display(volume)
    H, W, D = vol.shape

    z_idx, y_idx, x_idx = D // 2, H // 2, W // 2

    views = [
        ("Axial",    vol[:, :, z_idx]),
        ("Coronal",  vol[:, y_idx, :]),
        ("Sagittal", vol[x_idx, :, :]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=figsize, facecolor="#0d0d1a")
    fig.suptitle(modality_name, fontsize=13, color="white", fontweight="bold")

    for ax, (title, sl) in zip(axes, views):
        ax.set_facecolor("#0d0d1a")
        ax.imshow(np.rot90(sl), cmap="gray", aspect="equal")
        ax.set_title(title, color="white", fontsize=10, pad=4)
        ax.axis("off")

    plt.tight_layout(pad=0.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    return np.array(img)


# ──────────────────────────────────────────────────────────────────────────────
# Stats helpers
# ──────────────────────────────────────────────────────────────────────────────

def build_stats_html(
    mask: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    inference_time_s: float = 0.0,
) -> str:
    """
    Tạo HTML table thống kê tumor để hiển thị trong Gradio gr.HTML.
    """
    from report_generator import compute_volumes

    vol = compute_volumes(mask, spacing)
    has_tumor = bool((mask > 0).any())

    rows_data = [
        ("Whole Tumor (WT)",       vol["wt_voxels"],  vol["wt_cm3"],  "#4a90d9"),
        ("Tumor Core (TC)",        vol["tc_voxels"],  vol["tc_cm3"],  "#9b59b6"),
        ("NCR — Necrotic Core",    vol["ncr_voxels"], vol["ncr_cm3"], "#e41a1c"),
        ("ED — Edema",             vol["ed_voxels"],  vol["ed_cm3"],  "#4daf4a"),
        ("ET — Enhancing Tumor",   vol["et_voxels"],  vol["et_cm3"],  "#ff7f00"),
    ]

    total_vox = mask.size

    rows_html = ""
    for name, vox, cm3, color in rows_data:
        pct = 100.0 * vox / total_vox if total_vox > 0 else 0.0
        dot = f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{color};margin-right:6px;"></span>'
        rows_html += f"""
        <tr>
          <td style="padding:6px 10px;">{dot}{name}</td>
          <td style="padding:6px 10px;text-align:right;">{vox:,}</td>
          <td style="padding:6px 10px;text-align:right;">{cm3:.3f}</td>
          <td style="padding:6px 10px;text-align:right;">{pct:.4f}%</td>
        </tr>"""

    status_color = "#27ae60" if has_tumor else "#7f8c8d"
    status_text  = "✅ Tumor Detected" if has_tumor else "⬜ No Tumor Detected"

    return f"""
    <div style="font-family:'Segoe UI',system-ui,sans-serif;color:#e0e0e0;">
      <div style="display:flex;gap:20px;margin-bottom:14px;flex-wrap:wrap;">
        <div style="background:#1e2a3a;border-radius:10px;padding:10px 18px;border-left:4px solid {status_color};">
          <div style="font-size:11px;color:#888;margin-bottom:2px;">STATUS</div>
          <div style="font-size:15px;font-weight:700;color:{status_color};">{status_text}</div>
        </div>
        <div style="background:#1e2a3a;border-radius:10px;padding:10px 18px;border-left:4px solid #6366f1;">
          <div style="font-size:11px;color:#888;margin-bottom:2px;">INFERENCE TIME</div>
          <div style="font-size:15px;font-weight:700;color:#a5b4fc;">{inference_time_s:.2f}s</div>
        </div>
        <div style="background:#1e2a3a;border-radius:10px;padding:10px 18px;border-left:4px solid #06b6d4;">
          <div style="font-size:11px;color:#888;margin-bottom:2px;">VOXEL SPACING</div>
          <div style="font-size:15px;font-weight:700;color:#67e8f9;">{spacing[0]:.1f}×{spacing[1]:.1f}×{spacing[2]:.1f} mm</div>
        </div>
      </div>
      <table style="width:100%;border-collapse:collapse;background:#141b2d;border-radius:10px;overflow:hidden;">
        <thead>
          <tr style="background:#1e2a3a;">
            <th style="padding:8px 10px;text-align:left;color:#94a3b8;font-size:11px;">REGION</th>
            <th style="padding:8px 10px;text-align:right;color:#94a3b8;font-size:11px;">VOXELS</th>
            <th style="padding:8px 10px;text-align:right;color:#94a3b8;font-size:11px;">VOLUME (cm³)</th>
            <th style="padding:8px 10px;text-align:right;color:#94a3b8;font-size:11px;">% OF SCAN</th>
          </tr>
        </thead>
        <tbody>{rows_html}
        </tbody>
      </table>
      <p style="font-size:10px;color:#666;margin-top:8px;font-style:italic;">
        ⚠ For research use only. Not for clinical diagnosis.
      </p>
    </div>
    """
