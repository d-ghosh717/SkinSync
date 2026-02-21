"""
face_engine/tryon.py
---------------------
Phase 3 – Virtual Foundation Try-On

Pipeline:
  1. Build skin-only polygon mask from FaceMesh landmarks
       - Jaw + cheek outline
       - SUBTRACT eyes, eyebrows, lips
  2. Map detected tone → foundation BGR colour
  3. Alpha-blend foundation colour over the skin region

Press V in main.py to toggle try-on on/off.
"""

import cv2
import numpy as np


# ── Foundation colour table (BGR) ─────────────────────────────────────────
FOUNDATION_COLORS: dict[str, tuple[int, int, int]] = {
    "Fair":   (180, 195, 205),   # warm ivory
    "Medium": (130, 160, 180),   # beige-nude
    "Tan":    (90,  120, 140),   # caramel
    "Deep":   (50,  70,  90),    # rich mahogany
}

# Default alpha blend strength (0 = invisible, 1 = fully opaque)
ALPHA_DEFAULT = 0.38


# ── FaceMesh landmark index groups ────────────────────────────────────────
# Face oval / jaw outline  (MediaPipe FACEMESH_FACE_OVAL connection set → indices)
FACE_OVAL_IDS = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109, 10,
]

# Left eye (to subtract)
LEFT_EYE_IDS  = [33, 7, 163, 144, 145, 153, 154, 155, 133,
                  246, 161, 160, 159, 158, 157, 173, 33]
RIGHT_EYE_IDS = [362, 382, 381, 380, 374, 373, 390, 249, 263,
                  466, 388, 387, 386, 385, 384, 398, 362]

# Eyebrows (to subtract)
LEFT_BROW_IDS  = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 70]
RIGHT_BROW_IDS = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276, 300]

# Lips (to subtract)
LIPS_IDS = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    375, 321, 405, 314, 17, 84, 181, 91, 146, 61,
]

# Nose bridge / tip (to subtract — avoids weird nose blending)
NOSE_IDS = [168, 6, 197, 195, 5, 4, 45, 220, 115, 48, 64,
            98,  97, 2,  326, 327, 294, 278, 344, 440, 168]


class VirtualTryOn:
    """
    Renders a virtual foundation try-on over the skin region.

    Usage
    -----
    tryon = VirtualTryOn(alpha=0.38)
    frame = tryon.apply(frame, face_landmarks, tone_label)
    """

    def __init__(self, alpha: float = ALPHA_DEFAULT):
        self.alpha = alpha

    # ── Internal: build a polygon from landmark IDs ────────────────────────
    @staticmethod
    def _poly(landmarks, ids: list[int], h: int, w: int) -> np.ndarray:
        """Return (N, 1, 2) int32 array for cv2.fillPoly."""
        pts = []
        for idx in ids:
            lm = landmarks.landmark[idx]
            px = int(lm.x * w)
            py = int(lm.y * h)
            pts.append([px, py])
        return np.array(pts, dtype=np.int32).reshape(-1, 1, 2)

    # ── Build skin mask ────────────────────────────────────────────────────
    def _build_mask(
        self,
        landmarks,
        h: int,
        w: int,
    ) -> np.ndarray:
        """
        Create a single-channel uint8 mask (255 = skin, 0 = exclude).
        """
        mask = np.zeros((h, w), dtype=np.uint8)

        # Fill face oval
        face_poly  = self._poly(landmarks, FACE_OVAL_IDS,  h, w)
        cv2.fillPoly(mask, [face_poly], 255)

        # Punch out exclusions
        for ids in [
            LEFT_EYE_IDS, RIGHT_EYE_IDS,
            LEFT_BROW_IDS, RIGHT_BROW_IDS,
            LIPS_IDS, NOSE_IDS,
        ]:
            excl_poly = self._poly(landmarks, ids, h, w)
            cv2.fillPoly(mask, [excl_poly], 0)

        # Smooth the mask edges for natural blending
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        return mask

    # ── Public API ─────────────────────────────────────────────────────────
    def apply(
        self,
        frame: np.ndarray,
        landmarks,          # NormalizedLandmarkList
        tone_label: str,    # "Fair" / "Medium" / "Tan" / "Deep"
    ) -> np.ndarray:
        """
        Apply foundation try-on to frame (modifies in-place) and return it.

        Returns the original frame unchanged if landmarks is None
        or tone is not recognised.
        """
        if landmarks is None or tone_label not in FOUNDATION_COLORS:
            return frame

        h, w = frame.shape[:2]
        bgr = FOUNDATION_COLORS[tone_label]

        # Build soft mask
        mask = self._build_mask(landmarks, h, w)   # uint8 0–255

        # Create a flat foundation layer
        foundation = np.full((h, w, 3), bgr, dtype=np.uint8)

        # Per-pixel alpha based on mask value (soft edges from GaussianBlur)
        alpha_map = (mask.astype(np.float32) / 255.0) * self.alpha
        alpha_map = alpha_map[:, :, np.newaxis]      # broadcast to 3 channels

        # Blend: out = foundation * alpha + frame * (1 - alpha)
        blended = (foundation.astype(np.float32) * alpha_map
                   + frame.astype(np.float32) * (1.0 - alpha_map))
        np.clip(blended, 0, 255, out=blended)
        np.copyto(frame, blended.astype(np.uint8))

        return frame

    def set_alpha(self, alpha: float):
        """Clamp and update blend strength."""
        self.alpha = max(0.0, min(1.0, alpha))
