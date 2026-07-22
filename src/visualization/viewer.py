"""
Slice visualizer for 3D MRI volumes and segmentation mask overlays.
Supports 3 orthogonal views: Axial, Coronal, Sagittal.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union, Dict
from matplotlib.colors import ListedColormap


class SliceViewer:
    """Viewer for rendering 3D MRI slices and tumor mask overlays."""

    # Standard BraTS Color Palette
    # 0: Background (Transparent)
    # 1: NCR/NET (Red) - Necrotic & Non-Enhancing Tumor Core
    # 2: ED (Green) - Peritumoral Edema
    # 3: ET (Yellow) - Enhancing Tumor
    TUMOR_COLORS = {
        0: [0, 0, 0, 0],          # Transparent background
        1: [1.0, 0.0, 0.0, 0.6],  # Red
        2: [0.0, 0.8, 0.2, 0.5],  # Green
        3: [1.0, 1.0, 0.0, 0.7],  # Yellow
    }

    @staticmethod
    def extract_slice(volume: np.ndarray, plane: str = "axial", slice_idx: Optional[int] = None) -> np.ndarray:
        """
        Extract 2D slice from 3D volume.

        Args:
            volume: 3D numpy array (H, W, D) or 4D (C, H, W, D)
            plane: "axial" (D), "coronal" (H), or "sagittal" (W)
            slice_idx: Index of slice (if None, use middle slice)

        Returns:
            2D numpy array
        """
        plane = plane.lower().strip()
        if volume.ndim == 4:
            # Multi-channel image, select first channel if not specified
            volume = volume[0]

        h, w, d = volume.shape

        if plane == "axial":
            idx = slice_idx if slice_idx is not None else d // 2
            idx = max(0, min(d - 1, idx))
            return volume[:, :, idx]
        elif plane == "coronal":
            idx = slice_idx if slice_idx is not None else h // 2
            idx = max(0, min(h - 1, idx))
            return volume[idx, :, :]
        elif plane == "sagittal":
            idx = slice_idx if slice_idx is not None else w // 2
            idx = max(0, min(w - 1, idx))
            return volume[:, idx, :]
        else:
            raise ValueError(f"Invalid plane '{plane}'. Choose 'axial', 'coronal', or 'sagittal'.")

    @classmethod
    def plot_slice_with_overlay(
        cls,
        image_slice: np.ndarray,
        mask_slice: Optional[np.ndarray] = None,
        title: str = "MRI Slice",
        alpha: float = 0.5,
        figsize: Tuple[int, int] = (6, 6),
    ) -> plt.Figure:
        """
        Create a Matplotlib figure of an MRI slice with tumor mask overlay.

        Args:
            image_slice: 2D numpy array
            mask_slice: Optional 2D mask array
            title: Title for plot
            alpha: Transparency alpha for overlay
            figsize: Figure dimensions

        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(image_slice, cmap="gray")

        if mask_slice is not None and np.any(mask_slice):
            # Overlay mask using colormap
            cmap = ListedColormap([
                (0, 0, 0, 0),         # 0: transparent
                (1.0, 0.2, 0.2, alpha), # 1: Red
                (0.2, 0.8, 0.2, alpha), # 2: Green
                (1.0, 0.9, 0.1, alpha), # 3: Yellow
            ])
            ax.imshow(mask_slice, cmap=cmap, vmin=0, vmax=3)

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()
        return fig

    @classmethod
    def plot_ortho_views(
        cls,
        volume: np.ndarray,
        mask: Optional[np.ndarray] = None,
        title: str = "3D Orthogonal Slice Views",
        figsize: Tuple[int, int] = (15, 5),
    ) -> plt.Figure:
        """
        Plot 3 orthogonal views (Axial, Coronal, Sagittal) side-by-side.

        Args:
            volume: 3D MRI volume (H, W, D) or 4D (C, H, W, D)
            mask: Optional 3D mask volume
            title: Overall figure title
            figsize: Figure size

        Returns:
            Matplotlib Figure object
        """
        if volume.ndim == 4:
            volume = volume[0]

        h, w, d = volume.shape
        axial = cls.extract_slice(volume, "axial", d // 2)
        coronal = cls.extract_slice(volume, "coronal", h // 2)
        sagittal = cls.extract_slice(volume, "sagittal", w // 2)

        m_axial = cls.extract_slice(mask, "axial", d // 2) if mask is not None else None
        m_coronal = cls.extract_slice(mask, "coronal", h // 2) if mask is not None else None
        m_sagittal = cls.extract_slice(mask, "sagittal", w // 2) if mask is not None else None

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        views = [
            ("Axial (Transverse)", axial, m_axial),
            ("Coronal (Frontal)", coronal, m_coronal),
            ("Sagittal (Side)", sagittal, m_sagittal),
        ]

        cmap = ListedColormap([
            (0, 0, 0, 0),
            (1.0, 0.2, 0.2, 0.6),
            (0.2, 0.8, 0.2, 0.5),
            (1.0, 0.9, 0.1, 0.7),
        ])

        for ax, (v_title, v_img, v_mask) in zip(axes, views):
            ax.imshow(v_img, cmap="gray")
            if v_mask is not None and np.any(v_mask):
                ax.imshow(v_mask, cmap=cmap, vmin=0, vmax=3)
            ax.set_title(v_title, fontsize=11, fontweight="bold")
            ax.axis("off")

        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        return fig
