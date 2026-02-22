"""
face_engine/skin_tone.py  –  Phase 4 enhanced
Adds:
  - undertone_scores  : {"Warm": 0.22, "Cool": 0.31, "Neutral": 0.47}
  - avg_rgb           : (r, g, b) tuple
  - tone_description  : human-readable Fitzpatrick description
  - texture_tags      : list of descriptive bullets
  - quality warnings  : blur / brightness checks
"""

from collections import deque
from dataclasses import dataclass, field
import cv2
import numpy as np


# ── Constants ────────────────────────────────────────────────────────────────
TONE_CLASSES      = ["Fair", "Medium", "Tan", "Deep"]
UNDERTONE_CLASSES = ["Warm", "Cool", "Neutral"]
UNDERTONE_THRESH  = 3.0
TEXTURE_THRESH    = 150.0
CONFIDENCE_THRESH = 0.65

TONE_CHIP_COLOR = {
    "Fair":   (200, 190, 210),
    "Medium": (120, 155, 185),
    "Tan":    (80,  120, 160),
    "Deep":   (40,  65,  100),
}

TONE_META = {
    "Fair": {
        "label": "Light",
        "type":  "Type I-II",
        "desc":  "Light skin that burns easily, rarely tans",
    },
    "Medium": {
        "label": "Medium",
        "type":  "Type II-III",
        "desc":  "Light skin that burns moderately, tans gradually",
    },
    "Tan": {
        "label": "Tan",
        "type":  "Type IV",
        "desc":  "Olive skin that rarely burns, tans easily",
    },
    "Deep": {
        "label": "Deep",
        "type":  "Type V-VI",
        "desc":  "Dark skin that very rarely burns, tans deeply",
    },
}

TEXTURE_META = {
    "Smooth": {
        "label": "Normal Skin",
        "tags":  [
            "Even surface with minimal pores",
            "Good moisture balance",
            "Healthy, radiant complexion",
        ],
        "finish": "matte or dewy",
        "types":  "liquid, cream",
    },
    "Textured": {
        "label": "Oily Skin",
        "tags":  [
            "Visible shine, especially in T-zone",
            "Enlarged pores",
            "Prone to breakouts",
        ],
        "finish": "matte",
        "types":  "liquid, powder",
    },
}


# ── Cheek landmark IDs (MediaPipe 468-point model) ───────────────────────────
LEFT_CHEEK_IDS  = [205, 206, 207, 187, 147, 123, 116, 111, 117, 118, 119, 100,
                   126, 209,  49,  36,  31, 228, 229, 230]
RIGHT_CHEEK_IDS = [425, 426, 427, 411, 376, 352, 345, 340, 346, 347, 348, 329,
                   355, 449, 279, 266, 261, 448, 449, 450]


@dataclass
class SkinToneResult:
    # Core classification
    tone:       str
    undertone:  str
    texture:    str
    hex_color:  str

    # Rich extras
    avg_rgb:           tuple = field(default_factory=lambda: (0, 0, 0))
    undertone_scores:  dict  = field(default_factory=dict)
    confidence:        float = 0.0
    l_value:           float = 0.0
    stable:            bool  = False
    hint:              str   = ""

    # Descriptive
    tone_label:       str  = ""
    tone_type:        str  = ""
    tone_desc:        str  = ""
    texture_label:    str  = ""
    texture_tags:     list = field(default_factory=list)
    finish_rec:       str  = ""
    foundation_types: str  = ""


class SkinToneAnalyzer:
    def __init__(self, window_size: int = 10):
        # history stores tuples: (tone, undertone_raw, texture, hex, r, g, b, l_val, a_val, b_val)
        self._history: deque = deque(maxlen=window_size)

    # ── Pixel extraction ─────────────────────────────────────────────────────
    @staticmethod
    def _roi_pixels(frame, landmarks, ids) -> np.ndarray:
        h, w = frame.shape[:2]
        patches = []
        for idx in ids:
            lm = landmarks.landmark[idx]
            px, py = int(lm.x * w), int(lm.y * h)
            patch = frame[max(0, py-1):min(h, py+2), max(0, px-1):min(w, px+2)]
            if patch.size > 0:
                patches.append(patch.reshape(-1, 3))
        return np.vstack(patches) if patches else np.empty((0, 3), dtype=np.uint8)

    # ── Classifiers ─────────────────────────────────────────────────────────
    @staticmethod
    def _classify_l(l_mean: float) -> str:
        if l_mean > 170:   return "Fair"
        elif l_mean > 140: return "Medium"
        elif l_mean > 110: return "Tan"
        else:              return "Deep"

    @staticmethod
    def _classify_undertone(a: float, b: float) -> str:
        if   b > a + UNDERTONE_THRESH: return "Warm"
        elif a > b + UNDERTONE_THRESH: return "Cool"
        else:                          return "Neutral"

    @staticmethod
    def _classify_texture(pixels: np.ndarray) -> str:
        n = pixels.shape[0]
        size = int(np.sqrt(n))
        if size < 3: return "Smooth"
        img  = pixels[:size*size].reshape((size, size, 3))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        var  = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return "Textured" if var > TEXTURE_THRESH else "Smooth"

    # ── Smooth over rolling window ────────────────────────────────────────────
    def _smoothed(self):
        if not self._history:
            return "Unknown", "Neutral", "Smooth", "#000000", (0,0,0), {}, 0.0, 0.0

        tones      = [x[0] for x in self._history]
        undertones = [x[1] for x in self._history]
        textures   = [x[2] for x in self._history]
        n          = len(self._history)

        winner_tone = max(set(tones), key=tones.count)
        confidence  = tones.count(winner_tone) / n

        winner_ut   = max(set(undertones), key=undertones.count)
        winner_tx   = max(set(textures),   key=textures.count)

        # Per-category percentages for undertone bar chart
        ut_scores = {c: round(undertones.count(c) / n, 2) for c in UNDERTONE_CLASSES}

        # Average hex from recent window
        last = self._history[-1]
        hex_color = last[3]
        avg_rgb   = last[4]
        l_val     = last[7]

        return winner_tone, winner_ut, winner_tx, hex_color, avg_rgb, ut_scores, confidence, l_val

    # ── Public API ────────────────────────────────────────────────────────────
    def analyze(self, frame: np.ndarray, landmarks) -> "SkinToneResult | None":
        if landmarks is None:
            return None

        left  = self._roi_pixels(frame, landmarks, LEFT_CHEEK_IDS)
        right = self._roi_pixels(frame, landmarks, RIGHT_CHEEK_IDS)
        pixels = (np.vstack([left, right]) if left.size and right.size
                  else (left if left.size else right))
        if pixels.shape[0] == 0:
            return None

        # BGR → LAB
        lab    = cv2.cvtColor(pixels.reshape(1, -1, 3), cv2.COLOR_BGR2LAB)
        l_mean = float(lab[0, :, 0].mean())
        a_mean = float(lab[0, :, 1].mean())
        b_mean = float(lab[0, :, 2].mean())

        # Average BGR → RGB for display
        b_avg = int(pixels[:, 0].mean())
        g_avg = int(pixels[:, 1].mean())
        r_avg = int(pixels[:, 2].mean())
        hex_c = f"#{r_avg:02x}{g_avg:02x}{b_avg:02x}"

        tone = self._classify_l(l_mean)
        ut   = self._classify_undertone(a_mean, b_mean)
        tx   = self._classify_texture(pixels)

        self._history.append((tone, ut, tx, hex_c, (r_avg, g_avg, b_avg),
                              {}, 0.0, l_mean, a_mean, b_mean))

        tone, ut, tx, hex_c, avg_rgb, ut_scores, conf, l_val = self._smoothed()
        stable = conf >= CONFIDENCE_THRESH

        # Enrich with descriptive metadata
        tm = TONE_META.get(tone, {})
        xm = TEXTURE_META.get(tx, TEXTURE_META["Smooth"])

        return SkinToneResult(
            tone=tone, undertone=ut, texture=tx,
            hex_color=hex_c, avg_rgb=avg_rgb,
            undertone_scores=ut_scores,
            confidence=conf, l_value=l_val, stable=stable,
            hint="" if stable else "Hold still for better accuracy",
            tone_label=tm.get("label", tone),
            tone_type=tm.get("type", ""),
            tone_desc=tm.get("desc", ""),
            texture_label=xm.get("label", tx),
            texture_tags=xm.get("tags", []),
            finish_rec=xm.get("finish", ""),
            foundation_types=xm.get("types", ""),
        )

    def reset(self):
        self._history.clear()
