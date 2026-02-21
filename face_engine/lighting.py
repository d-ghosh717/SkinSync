"""
face_engine/lighting.py
-----------------------
Lighting normalization using:
  - BGR → LAB color space conversion
  - CLAHE (Contrast Limited Adaptive Histogram Equalization) on L-channel
  - LAB → BGR conversion back
"""

import cv2
import numpy as np


class LightingNormalizer:
    """
    Applies CLAHE-based lighting normalization to a BGR frame.

    Parameters
    ----------
    clip_limit : float
        CLAHE clip limit (contrast limit). Default 2.0.
    tile_grid_size : tuple[int, int]
        Grid size for CLAHE. Default (8, 8).
    """

    def __init__(self, clip_limit: float = 2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size,
        )

    def normalize(self, frame: np.ndarray) -> np.ndarray:
        """
        Normalize lighting of a BGR frame.

        Returns
        -------
        np.ndarray
            Lighting-normalized BGR frame (same dtype/shape as input).
        """
        # BGR → LAB
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        # Split channels
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE to the L (lightness) channel only
        l_normalized = self.clahe.apply(l_channel)

        # Merge back and convert to BGR
        lab_normalized = cv2.merge([l_normalized, a_channel, b_channel])
        return cv2.cvtColor(lab_normalized, cv2.COLOR_LAB2BGR)
