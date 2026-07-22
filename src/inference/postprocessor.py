"""
Post-processing operations for 3D segmentation masks.
Khử nhiễu, nối liền vùng và loại bỏ các đảo nhỏ (connected component analysis).
"""

import numpy as np
from typing import Tuple, Optional
from scipy.ndimage import label, binary_closing, binary_opening, binary_fill_holes


class MaskPostprocessor:
    """Postprocessor for 3D segmentation masks."""

    def __init__(
        self,
        min_component_size: int = 64,
        apply_closing: bool = True,
        apply_fill_holes: bool = True,
    ):
        """
        Initialize postprocessor.

        Args:
            min_component_size: Minimum voxel size for connected components (smaller removed)
            apply_closing: Apply morphological closing to smooth boundaries
            apply_fill_holes: Fill internal holes in masks
        """
        self.min_component_size = min_component_size
        self.apply_closing = apply_closing
        self.apply_fill_holes = apply_fill_holes

    def remove_small_components(self, mask: np.ndarray) -> np.ndarray:
        """
        Remove small connected components from binary or multi-class mask.

        Args:
            mask: 3D numpy array

        Returns:
            Filtered mask
        """
        if not np.any(mask):
            return mask

        cleaned_mask = np.zeros_like(mask)
        classes = np.unique(mask)

        for c in classes:
            if c == 0:
                continue

            binary_c = (mask == c)
            labeled_array, num_features = label(binary_c)

            for i in range(1, num_features + 1):
                component_size = np.sum(labeled_array == i)
                if component_size >= self.min_component_size:
                    cleaned_mask[labeled_array == i] = c

        return cleaned_mask

    def fill_holes(self, mask: np.ndarray) -> np.ndarray:
        """Fill internal holes in 3D masks."""
        if not np.any(mask):
            return mask

        filled = np.zeros_like(mask)
        classes = np.unique(mask)

        for c in classes:
            if c == 0:
                continue

            binary_c = (mask == c)
            filled_c = binary_fill_holes(binary_c)
            filled[filled_c] = c

        return filled

    def apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        """Apply morphological closing and opening."""
        if not np.any(mask):
            return mask

        processed = np.zeros_like(mask)
        classes = np.unique(mask)

        for c in classes:
            if c == 0:
                continue

            binary_c = (mask == c)
            if self.apply_closing:
                binary_c = binary_closing(binary_c, structure=np.ones((3, 3, 3)))
            processed[binary_c] = c

        return processed

    def process(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply full post-processing pipeline.

        Args:
            mask: 3D numpy array

        Returns:
            Processed 3D numpy array
        """
        output = mask.copy()

        if self.min_component_size > 0:
            output = self.remove_small_components(output)

        if self.apply_fill_holes:
            output = self.fill_holes(output)

        if self.apply_closing:
            output = self.apply_morphology(output)

        return output
