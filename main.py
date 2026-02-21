"""
main.py – Skin1 Phase 3: Virtual Foundation Try-On
====================================================
New in Phase 3:
  - VirtualTryOn: skin-region mask + foundation alpha blend
  - V key → toggle try-on on/off
  - +/- keys → adjust blend strength in real time

Controls
--------
  T  – toggle landmark drawing
  L  – toggle lighting view (CLAHE)
  R  – toggle render mode (mesh vs dots)
  V  – toggle virtual try-on
  +  – increase blend alpha (+0.05)
  -  – decrease blend alpha (-0.05)
  Q  – quit
"""

import sys
import cv2
import numpy

from face_engine.detector   import FaceDetector
from face_engine.lighting   import LightingNormalizer
from face_engine.quality    import QualityChecker, QualityResult
from face_engine.hud        import HUDRenderer, ToggleState
from face_engine.skin_tone  import SkinToneAnalyzer
from face_engine.tryon      import VirtualTryOn


# ── Configuration ─────────────────────────────────────────────────────────
CAMERA_INDEX         = 0
WINDOW_NAME          = "Skin1 | Phase 3 – Virtual Try-On"
DISPLAY_W, DISPLAY_H = 960, 540
CAP_W,  CAP_H        = 640, 480
CAP_FPS              = 30
QUALITY_EVERY_N      = 3
TONE_EVERY_N         = 2


def open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {index}.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
    cap.set(cv2.CAP_PROP_FPS, CAP_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def main():
    cap          = open_camera(CAMERA_INDEX)
    detector     = FaceDetector(max_num_faces=1)
    normalizer   = LightingNormalizer(clip_limit=2.0, tile_grid_size=(8, 8))
    checker      = QualityChecker()
    tone_analyzer = SkinToneAnalyzer(window_size=10)
    tryon        = VirtualTryOn(alpha=0.38)
    hud          = HUDRenderer()
    toggles      = ToggleState()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_W, DISPLAY_H)

    print("━" * 62)
    print("  Skin1 – Phase 3 | Virtual Foundation Try-On")
    print("━" * 62)
    print("  T → landmarks  L → lighting  R → render  V → try-on  Q → quit")
    print("  + / -  →  adjust blend strength")
    print("━" * 62)

    frame_count    = 0
    landmarks_list = []
    last_quality   = checker.evaluate(
        numpy.zeros((CAP_H, CAP_W, 3), dtype=numpy.uint8)
    )
    last_tone  = None
    tryon_on   = False

    while True:
        ret, raw = cap.read()
        if not ret or raw is None:
            continue

        raw = cv2.flip(raw, 1)
        frame_count += 1

        # ── Lighting normalization ─────────────────────────────────────────
        display = normalizer.normalize(raw) if toggles.lighting_view else raw

        # ── Face detection ─────────────────────────────────────────────────
        landmarks_list = detector.process(display)
        first_face     = landmarks_list[0] if landmarks_list else None

        if first_face is None and last_tone is not None:
            tone_analyzer.reset()
            last_tone = None

        # ── Quality check ──────────────────────────────────────────────────
        if frame_count % QUALITY_EVERY_N == 0:
            last_quality = checker.evaluate(display, first_face)

        # ── Skin tone ─────────────────────────────────────────────────────
        if first_face is not None and frame_count % TONE_EVERY_N == 0:
            result = tone_analyzer.analyze(display, first_face)
            if result is not None:
                last_tone = result

        # ── Virtual try-on (BEFORE landmark drawing so mesh sits on top) ───
        if tryon_on and first_face is not None and last_tone is not None:
            tryon.apply(display, first_face, last_tone.tone)

        # ── Landmark drawing ───────────────────────────────────────────────
        detector.draw(
            display,
            landmarks_list,
            show_landmarks=toggles.show_landmarks,
            mesh_mode=toggles.mesh_mode,
        )

        # ── HUD ───────────────────────────────────────────────────────────
        display, _fps = hud.render(
            display, last_quality, toggles, last_tone,
            tryon_on=tryon_on, tryon_alpha=tryon.alpha,
        )

        cv2.imshow(WINDOW_NAME, display)

        # ── Keys ──────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), ord('Q')):
            print("\n[INFO] Quit. Bye!")
            break
        elif key in (ord('t'), ord('T')):
            toggles.show_landmarks = not toggles.show_landmarks
            print(f"[T] Landmarks → {'ON' if toggles.show_landmarks else 'OFF'}")
        elif key in (ord('l'), ord('L')):
            toggles.lighting_view = not toggles.lighting_view
            print(f"[L] Lighting  → {'NORMALIZED' if toggles.lighting_view else 'RAW'}")
        elif key in (ord('r'), ord('R')):
            toggles.mesh_mode = not toggles.mesh_mode
            print(f"[R] Render    → {'MESH' if toggles.mesh_mode else 'DOTS'}")
        elif key in (ord('v'), ord('V')):
            tryon_on = not tryon_on
            print(f"[V] Try-On    → {'ON' if tryon_on else 'OFF'}")
        elif key in (ord('+'), ord('=')):
            tryon.set_alpha(tryon.alpha + 0.05)
            print(f"[+] Alpha     → {tryon.alpha:.2f}")
        elif key in (ord('-'), ord('_')):
            tryon.set_alpha(tryon.alpha - 0.05)
            print(f"[-] Alpha     → {tryon.alpha:.2f}")

        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
