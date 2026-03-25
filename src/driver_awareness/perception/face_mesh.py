"""
face_mesh.py
------------
Perception layer — face landmark detection only.

Responsibilities:
  - Accept a BGR frame
  - Detect facial landmarks using MediaPipe FaceMesh
  - Return normalized (N,3) landmarks, pixel coordinates, and iris subsets

Does NOT:
  - Draw on frames
  - Compute EAR, blink counts, or any derived feature
  - Hold state across frames (stateless per-call detection)
  - Make decisions
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np


# ---------------------------------------------------------------------------
# MediaPipe iris landmark indices within the 478-point mesh
# ---------------------------------------------------------------------------
# Right iris: 468, 469, 470, 471, 472 (5 points)
# Left iris:  473, 474, 475, 476, 477 (5 points)
_RIGHT_IRIS_IDX = list(range(468, 473))
_LEFT_IRIS_IDX = list(range(473, 478))


@dataclass(frozen=True)
class FaceMeshResult:
    """
    Immutable result object returned by FaceMesh.detect().

    Attributes
    ----------
    landmarks_norm : np.ndarray, shape (478, 3)
        Normalized landmark coordinates (x, y, z) in [0, 1] range.
        x and y are relative to frame width/height; z is relative depth.
    landmarks_px : np.ndarray, shape (478, 2)
        Pixel-space landmark positions (x_px, y_px) as integers.
    left_iris_px : np.ndarray, shape (5, 2)
        Pixel positions of the 5 left-iris landmarks.
    right_iris_px : np.ndarray, shape (5, 2)
        Pixel positions of the 5 right-iris landmarks.
    frame_width : int
    frame_height : int
    """

    landmarks_norm: np.ndarray   # (478, 3)
    landmarks_px: np.ndarray     # (478, 2)
    left_iris_px: np.ndarray     # (5, 2)
    right_iris_px: np.ndarray    # (5, 2)
    frame_width: int
    frame_height: int


class FaceMesh:
    """
    Wraps MediaPipe FaceMesh for single-face landmark detection.

    Usage
    -----
        detector = FaceMesh()
        detector.start()
        result = detector.detect(bgr_frame)  # FaceMeshResult | None
        detector.release()

    Or as a context manager:
        with FaceMesh() as detector:
            result = detector.detect(frame)
    """

    def __init__(
        self,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """
        Parameters
        ----------
        refine_landmarks :
            Enable iris and attention mesh refinement (required for iris tracking).
        min_detection_confidence :
            Confidence threshold for face detection.
        min_tracking_confidence :
            Confidence threshold for landmark tracking between frames.
        """
        self._refine = refine_landmarks
        self._det_conf = min_detection_confidence
        self._trk_conf = min_tracking_confidence
        self._mesh: mp.solutions.face_mesh.FaceMesh | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Initialize the MediaPipe FaceMesh model."""
        if self._mesh is not None:
            raise RuntimeError(
                "FaceMesh is already running. Call release() before calling start() again."
            )
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,           # video mode → tracking between frames
            max_num_faces=1,
            refine_landmarks=self._refine,
            min_detection_confidence=self._det_conf,
            min_tracking_confidence=self._trk_conf,
        )

    def release(self) -> None:
        """Close the MediaPipe model and free resources."""
        if self._mesh is not None:
            self._mesh.close()
            self._mesh = None

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(self, bgr_frame: np.ndarray) -> FaceMeshResult | None:
        """
        Run face mesh detection on a single BGR frame.

        Parameters
        ----------
        bgr_frame : np.ndarray
            A (H, W, 3) BGR image as returned by OpenCV.

        Returns
        -------
        FaceMeshResult if a face is found, otherwise None.

        Raises
        ------
        RuntimeError if start() has not been called.
        """
        if self._mesh is None:
            raise RuntimeError("FaceMesh is not started. Call start() first.")

        h, w = bgr_frame.shape[:2]

        # MediaPipe expects RGB
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_result = self._mesh.process(rgb)

        if not mp_result.multi_face_landmarks:
            return None

        # Take the first (and only) detected face
        raw_landmarks = mp_result.multi_face_landmarks[0].landmark

        # --- Normalized coordinates (x, y, z) in [0,1] ---
        landmarks_norm = np.array(
            [[lm.x, lm.y, lm.z] for lm in raw_landmarks],
            dtype=np.float32,
        )  # shape: (478, 3)

        # --- Pixel coordinates (x_px, y_px) ---
        landmarks_px = np.array(
            [[int(lm.x * w), int(lm.y * h)] for lm in raw_landmarks],
            dtype=np.int32,
        )  # shape: (478, 2)

        # --- Iris subsets ---
        left_iris_px = landmarks_px[_LEFT_IRIS_IDX]    # (5, 2)
        right_iris_px = landmarks_px[_RIGHT_IRIS_IDX]  # (5, 2)

        return FaceMeshResult(
            landmarks_norm=landmarks_norm,
            landmarks_px=landmarks_px,
            left_iris_px=left_iris_px,
            right_iris_px=right_iris_px,
            frame_width=w,
            frame_height=h,
        )

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False