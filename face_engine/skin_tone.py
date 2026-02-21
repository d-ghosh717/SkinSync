"""
face_engine/skin_tone.py
-------------------------
Phase 2 – Skin tone detection pipeline.

Steps:
  1. Extract cheek ROI pixels using FaceMesh landmarks
  2. Convert to LAB colour space
  3. Average L (lightness) value maps to one of 4 tone classes
  4. Temporal smoothing  – rolling window of last 10 predictions
  5. Confidence score    – % agreement within the window
  6. "Hold still" hint   – shown when confidence < 65%

Tone mapping (L channel, 0-255 scale):
  L > 170  → Fair
  140-170  → Medium
  110-140  → Tan
  <  110   → Deep
"""

from collections import deque
from dataclasses import dataclass, field
import cv2
import numpy as np


# ── Tone definitions ───────────────────────────────────────────────────────
TONE_CLASSES = ["Fair", "Medium", "Tan", "Deep"]

# BGR colours used for the HUD colour chip
TONE_CHIP_COLOR = {
    "Fair":   (200, 190, 210),   # pinkish-light
    "Medium": (120, 155, 185),   # warm mid
    "Tan":    (80,  120, 160),   # caramel-ish
    "Deep":   (40,  65,  100),   # dark brown
}

CONFIDENCE_THRESHOLD = 0.65   # below this → show "Hold still"
WINDOW_SIZE          = 10     # rolling smoothing window


# ── Cheek landmark indices (MediaPipe FaceMesh 468-point model) ────────────
# Left cheek cluster (mirrored in face space → right side on screen due to flip)
LEFT_CHEEK_IDS  = [205, 206, 207, 187, 147, 123, 116, 111, 117, 118, 119, 100,
                   126, 209, 49,  36,  31,  228, 229, 230]
# Right cheek cluster
RIGHT_CHEEK_IDS = [425, 426, 427, 411, 376, 352, 345, 340, 346, 347, 348, 329,
                   355, 449, 279, 266, 261, 448, 449, 450]


@dataclass
class SkinToneResult:
    tone:       str             # "Fair" / "Medium" / "Tan" / "Deep"
    confidence: float           # 0.0 – 1.0
    l_value:    float           # raw avg LAB L channel value
    stable:     bool            # True when confidence >= threshold
    hint:       str             # user-facing message


class SkinToneAnalyzer:
    """
    Analyses skin tone from face mesh landmarks.

    Parameters
    ----------
    window_size : int
        Number of past frames kept for majority-vote smoothing.
    """

    def __init__(self, window_size: int = WINDOW_SIZE):
        self._history: deque[str] = deque(maxlen=window_size)

    # ── Internal: landmark pixel extraction ───────────────────────────────
    @staticmethod
    def _roi_pixels(
        frame: np.ndarray,
        landmarks,
        ids: list[int],
    ) -> np.ndarray:
        """
        Return BGR pixel array for a set of landmark IDs.
        Each pixel is sampled from a 3×3 patch to reduce noise.
        """
        h, w = frame.shape[:2]
        pixels = []
        for idx in ids:
            lm = landmarks.landmark[idx]
            px = int(lm.x * w)
            py = int(lm.y * h)
            # 3×3 patch clipped to frame bounds
            patch = frame[
                max(0, py - 1): min(h, py + 2),
                max(0, px - 1): min(w, px + 2)
            ]
            if patch.size > 0:
                pixels.append(patch.reshape(-1, 3))
        return np.vstack(pixels) if pixels else np.empty((0, 3), dtype=np.uint8)

    # ── Internal: L-value → tone class ────────────────────────────────────
    @staticmethod
    def _classify_l(l_mean: float) -> str:
        if l_mean > 170:
            return "Fair"
        elif l_mean > 140:
            return "Medium"
        elif l_mean > 110:
            return "Tan"
        else:
            return "Deep"

    # ── Internal: smoothed result from history ─────────────────────────────
    def _smoothed(self) -> tuple[str, float]:
        """Majority vote + confidence from rolling history window."""
        if not self._history:
            return "Unknown", 0.0
        counts = {t: 0 for t in TONE_CLASSES}
        for t in self._history:
            counts[t] += 1
        winner = max(counts, key=counts.__getitem__)
        confidence = counts[winner] / len(self._history)
        return winner, confidence

    # ── Public API ─────────────────────────────────────────────────────────
    def analyze(
        self,
        frame: np.ndarray,
        landmarks,       # NormalizedLandmarkList or None
    ) -> SkinToneResult | None:
        """
        Run skin tone analysis for one frame.

        Returns None if no landmarks available.
        """
        if landmarks is None:
            return None

        # Collect cheek pixels from both sides
        left_px  = self._roi_pixels(frame, landmarks, LEFT_CHEEK_IDS)
        right_px = self._roi_pixels(frame, landmarks, RIGHT_CHEEK_IDS)
        pixels   = np.vstack([left_px, right_px]) if (
            left_px.size > 0 and right_px.size > 0
        ) else (left_px if left_px.size > 0 else right_px)

        if pixels.shape[0] == 0:
            return None

        # BGR → LAB, extract L channel mean
        lab     = cv2.cvtColor(pixels.reshape(1, -1, 3), cv2.COLOR_BGR2LAB)
        l_mean  = float(lab[:, :, 0].mean())

        # Classify and push to history
        raw_tone = self._classify_l(l_mean)
        self._history.append(raw_tone)

        # Smooth + confidence
        tone, confidence = self._smoothed()
        stable = confidence >= CONFIDENCE_THRESHOLD

        hint = (
            "Hold still for better accuracy" if not stable
            else f"Skin Tone: {tone}  ({confidence*100:.0f}%)"
        )
        return SkinToneResult(
            tone=tone,
            confidence=confidence,
            l_value=l_mean,
            stable=stable,
            hint=hint,
        )

    def reset(self):
        """Clear history (e.g. when face leaves frame)."""
        self._history.clear()
