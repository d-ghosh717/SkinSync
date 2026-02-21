"""
face_engine/hud.py  (Phase 3 – with Try-On status)
----------------------------------------------------
Adds:
  - Try-On badge (top-right) when active
  - Alpha % display next to badge
  - [V] toggle row in bottom panel
  - Foundation colour preview chip when try-on is on
"""

import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np


FONT         = cv2.FONT_HERSHEY_SIMPLEX
COLOR_GREEN  = (0,  220, 80)
COLOR_YELLOW = (0,  210, 230)
COLOR_RED    = (60,  80, 255)
COLOR_WHITE  = (220, 220, 220)
COLOR_GRAY   = (130, 130, 130)
COLOR_CYAN   = (220, 200, 0)
COLOR_ORANGE = (0,  160, 255)
COLOR_PURPLE = (200, 80,  180)


@dataclass
class ToggleState:
    show_landmarks: bool = True   # T
    lighting_view:  bool = False  # L
    mesh_mode:      bool = True   # R


class FPSCounter:
    def __init__(self, window: int = 20):
        self._times: deque = deque(maxlen=window)

    def tick(self) -> float:
        self._times.append(time.perf_counter())
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0


class HUDRenderer:
    def __init__(self):
        self.fps_counter  = FPSCounter()
        self._panel_cache = None

    # ── Panel BG cache ─────────────────────────────────────────────────────
    def _get_panel_bg(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if self._panel_cache and self._panel_cache[0] == (h, w):
            return self._panel_cache[1]

        bg = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.rectangle(bg, (10,     10), (258,   104), (25, 25, 25), -1)  # top-left
        cv2.rectangle(bg, (w-214,  10), (w-10,  140), (25, 25, 25), -1)  # right: skin
        py = h - 56
        cv2.rectangle(bg, (10, py), (w-10,  h-8),  (25, 25, 25), -1)    # bottom toggles

        self._panel_cache = ((h, w), bg)
        return bg

    @staticmethod
    def _txt(frame, text, pos, color, scale=0.50, thick=1):
        x, y = pos
        cv2.putText(frame, text, (x+1, y+1), FONT, scale, (0,0,0),  thick+1, cv2.LINE_AA)
        cv2.putText(frame, text, (x,   y  ), FONT, scale, color,    thick,   cv2.LINE_AA)

    @staticmethod
    def _conf_bar(frame, x, y, w, confidence, stable):
        filled    = int(w * confidence)
        bar_color = COLOR_GREEN if stable else COLOR_ORANGE
        cv2.rectangle(frame, (x, y), (x+w, y+10), (60, 60, 60), -1)
        if filled > 0:
            cv2.rectangle(frame, (x, y), (x+filled, y+10), bar_color, -1)
        cv2.rectangle(frame, (x, y), (x+w, y+10), (80, 80, 80), 1)

    @staticmethod
    def _chip(frame, x, y, size, color_bgr):
        cv2.rectangle(frame, (x, y), (x+size, y+size), color_bgr, -1)
        cv2.rectangle(frame, (x, y), (x+size, y+size), (180,180,180), 1)

    # ── Main render ────────────────────────────────────────────────────────
    def render(
        self,
        frame: np.ndarray,
        quality,
        toggles: ToggleState,
        skin_tone=None,
        tryon_on: bool = False,
        tryon_alpha: float = 0.38,
    ):
        fps = self.fps_counter.tick()
        h, w = frame.shape[:2]

        panel_bg = self._get_panel_bg(frame)
        cv2.addWeighted(panel_bg, 0.50, frame, 0.50, 0, frame)

        # ── Top-left: FPS + quality ────────────────────────────────────────
        fps_color = COLOR_GREEN if fps >= 20 else COLOR_YELLOW if fps >= 12 else COLOR_RED
        self._txt(frame, f"FPS: {fps:5.1f}", (18, 32), fps_color, scale=0.60)
        self._txt(frame,
                  f"Blur:{quality.blur_score:5.0f}  Face:{quality.size_ratio*100:4.1f}%",
                  (18, 54), COLOR_CYAN, scale=0.44)
        q_color = COLOR_GREEN if quality.passed else COLOR_RED
        self._txt(frame, quality.message, (18, 76), q_color, scale=0.48)
        self._txt(frame, "480p detect | 640p display", (18, 96), COLOR_GRAY, scale=0.36)

        # ── Right: Skin tone + try-on ──────────────────────────────────────
        px = w - 204
        if skin_tone is not None:
            from face_engine.skin_tone  import TONE_CHIP_COLOR
            from face_engine.tryon      import FOUNDATION_COLORS

            chip_col = TONE_CHIP_COLOR.get(skin_tone.tone, (128,128,128))
            self._chip(frame, px+4,  16, 34, chip_col)
            self._txt(frame, skin_tone.tone,   (px+46, 36), COLOR_WHITE, scale=0.60)
            self._txt(frame, f"L:{skin_tone.l_value:5.1f}", (px+46, 56), COLOR_GRAY, scale=0.42)

            bar_x = px + 4
            self._conf_bar(frame, bar_x, 64, 190, skin_tone.confidence, skin_tone.stable)
            conf_color = COLOR_GREEN if skin_tone.stable else COLOR_ORANGE
            self._txt(frame, f"{skin_tone.confidence*100:.0f}% conf",
                      (bar_x, 86), conf_color, scale=0.44)

            if not skin_tone.stable:
                self._txt(frame, "Hold still...", (bar_x, 106), COLOR_ORANGE, scale=0.43)

            # Foundation chip when try-on is on
            if tryon_on:
                found_col = FOUNDATION_COLORS.get(skin_tone.tone, (128,128,128))
                self._chip(frame, px+4, 112, 16, found_col)
                self._txt(frame, f"Foundation  α:{tryon_alpha:.0%}",
                          (px+24, 124), COLOR_PURPLE, scale=0.42)
        else:
            self._txt(frame, "Skin Tone", (px+4, 32), COLOR_GRAY, scale=0.46)
            self._txt(frame, "No face",   (px+4, 54), COLOR_GRAY, scale=0.46)

        # ── Bottom: toggles ────────────────────────────────────────────────
        py = h - 50
        lm  = "ON " if toggles.show_landmarks else "OFF"
        lit = "NORM" if toggles.lighting_view  else "RAW "
        rn  = "MESH" if toggles.mesh_mode      else "DOTS"
        vo  = "ON " if tryon_on               else "OFF"
        self._txt(frame,
                  f"[T]{lm} [L]{lit} [R]{rn} [V]TryOn:{vo} [+/-]α  [Q]Quit",
                  (14, py+16), COLOR_GRAY, scale=0.41)

        # ── Try-on badge (top-right corner) ───────────────────────────────
        if tryon_on:
            self._txt(frame, f"TRY-ON  α:{tryon_alpha:.0%}",
                      (w - 168, h - 28), COLOR_PURPLE, scale=0.50)
        else:
            self._txt(frame, f"{fps:.0f}fps",
                      (w - 68,  h - 28), fps_color, scale=0.48)

        return frame, fps
