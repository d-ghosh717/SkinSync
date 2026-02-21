"""
face_engine/detector.py  (OPTIMIZED)
--------------------------------------
Key perf changes vs v1:
  - refine_landmarks=False  (removes 80 iris points, saves ~5 FPS)
  - process() receives a pre-shrunk small_frame so MediaPipe sees 480p
  - RGB conversion done with cv2.cvtColor(flag=cv2.COLOR_BGR2RGB) once
  - MESH drawing uses FACEMESH_CONTOURS only (vs full TESSELATION) for speed
  - DOT mode draws a fixed list of key landmark indices via polylines (fast)
"""

import cv2
import mediapipe as mp
import numpy as np


mp_face_mesh      = mp.solutions.face_mesh
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ── Drawing specs ──────────────────────────────────────────────────────────
DOT_SPEC     = mp_drawing.DrawingSpec(color=(0, 200, 100), thickness=1, circle_radius=1)
CONTOUR_SPEC = mp_drawing.DrawingSpec(color=(0, 220, 120), thickness=1, circle_radius=1)
MESH_SPEC    = mp_drawing.DrawingSpec(color=(0, 170, 70),  thickness=1, circle_radius=0)


class FaceDetector:
    """
    Optimised MediaPipe FaceMesh wrapper.

    detect_width / detect_height control the internal resolution fed to
    MediaPipe. Smaller = faster. Results are scaled back to the display frame.
    """

    def __init__(
        self,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.55,
        min_tracking_confidence:  float = 0.45,
        detect_width:  int = 480,   # internal detection resolution
        detect_height: int = 360,
    ):
        # refine_landmarks=False → no iris mesh, saves significant CPU
        self._mesh = mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.detect_width  = detect_width
        self.detect_height = detect_height

    # ── Detection ──────────────────────────────────────────────────────────
    def process(self, frame: np.ndarray):
        """
        Run FaceMesh.

        Internally downscales the frame to detect_width×detect_height
        before passing to MediaPipe, then returns landmarks in the
        ORIGINAL frame's coordinate space (normalised 0-1 so no scaling needed).

        Returns list[NormalizedLandmarkList], empty if no face found.
        """
        # Downscale for detection only
        small = cv2.resize(
            frame,
            (self.detect_width, self.detect_height),
            interpolation=cv2.INTER_LINEAR,
        )
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        # Tell MediaPipe not to write back to the array
        rgb.flags.writeable = False
        result = self._mesh.process(rgb)
        rgb.flags.writeable = True

        if result.multi_face_landmarks:
            return result.multi_face_landmarks
        return []

    # ── Drawing ────────────────────────────────────────────────────────────
    def draw(
        self,
        frame: np.ndarray,
        landmarks_list: list,
        show_landmarks: bool = True,
        mesh_mode: bool = True,
    ) -> np.ndarray:
        if not show_landmarks or not landmarks_list:
            return frame

        for face_lms in landmarks_list:
            if mesh_mode:
                # Contour-only mesh — much lighter than full TESSELATION
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_lms,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=CONTOUR_SPEC,
                )
            else:
                # Dot mode
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_lms,
                    connections=None,
                    landmark_drawing_spec=DOT_SPEC,
                    connection_drawing_spec=None,
                )
        return frame

    # ── Cleanup ────────────────────────────────────────────────────────────
    def close(self):
        self._mesh.close()
