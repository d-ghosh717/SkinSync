"""
face_engine/quality.py
----------------------
Frame and face quality checks:
  - Blur detection via Laplacian variance
  - Face size ratio check (face bbox vs frame area)
"""

from dataclasses import dataclass
import cv2
import numpy as np


# ── Thresholds ─────────────────────────────────────────────────────────────
BLUR_THRESHOLD     = 80.0   # Laplacian variance; below = blurry
MIN_FACE_RATIO     = 0.12   # Face bbox must cover ≥12% of frame area


@dataclass
class QualityResult:
    passed:     bool
    blur_score: float
    size_ratio: float   # 0.0–1.0
    message:    str


class QualityChecker:
    """Runs blur and face-size quality checks on each frame."""

    # ── Blur check ─────────────────────────────────────────────────────────
    def check_blur(self, frame: np.ndarray) -> float:
        """Return Laplacian variance of the grayscale frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # ── Face size check ────────────────────────────────────────────────────
    def check_face_size(
        self,
        landmarks,           # mediapipe NormalizedLandmarkList or None
        frame: np.ndarray,
    ) -> float:
        """
        Estimate face bounding-box area as a fraction of frame area.
        Returns 0.0 if no landmarks given.
        """
        if landmarks is None:
            return 0.0

        h, w = frame.shape[:2]
        xs = [lm.x for lm in landmarks.landmark]
        ys = [lm.y for lm in landmarks.landmark]
        fx_min, fx_max = min(xs), max(xs)
        fy_min, fy_max = min(ys), max(ys)

        face_w = (fx_max - fx_min) * w
        face_h = (fy_max - fy_min) * h
        face_area  = face_w * face_h
        frame_area = w * h
        return face_area / frame_area if frame_area > 0 else 0.0

    # ── Combined quality check ─────────────────────────────────────────────
    def evaluate(
        self,
        frame: np.ndarray,
        landmarks=None,
    ) -> QualityResult:
        blur_score = self.check_blur(frame)
        size_ratio = self.check_face_size(landmarks, frame)

        if blur_score < BLUR_THRESHOLD:
            return QualityResult(
                passed=False,
                blur_score=blur_score,
                size_ratio=size_ratio,
                message="! Blurry frame",
            )
        if landmarks is not None and size_ratio < MIN_FACE_RATIO:
            return QualityResult(
                passed=False,
                blur_score=blur_score,
                size_ratio=size_ratio,
                message="! Face too small",
            )
        if landmarks is None:
            return QualityResult(
                passed=False,
                blur_score=blur_score,
                size_ratio=0.0,
                message="Searching for face...",
            )

        return QualityResult(
            passed=True,
            blur_score=blur_score,
            size_ratio=size_ratio,
            message="Quality OK",
        )
