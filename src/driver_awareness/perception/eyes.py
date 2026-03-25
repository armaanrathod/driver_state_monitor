"""
eyes.py
-------
Feature extraction layer — eye metrics only.

Responsibilities:
  - Compute Eye Aspect Ratio (EAR) from facial landmarks
  - Detect blink events using a closed → open state transition

Does NOT:
  - Access the camera
  - Use MediaPipe directly
  - Draw on frames
  - Track time or produce awareness scores
  - Hold mutable global state (BlinkDetector is the explicit stateful unit)

Reference
---------
Soukupova & Cech (2016): "Real-Time Eye Blink Detection using Facial Landmarks"
EAR = (‖p2−p6‖ + ‖p3−p5‖) / (2 · ‖p1−p4‖)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# MediaPipe FaceMesh landmark indices for each eye
# Source: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
# ---------------------------------------------------------------------------

# Each eye needs 6 landmarks that form the EAR hexagon:
#   p1 (left corner), p2 (upper-left), p3 (upper-right),
#   p4 (right corner), p5 (lower-right), p6 (lower-left)

LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

def compute_ear(landmarks_norm: np.ndarray, eye_indices: list[int]) -> float:
    """
    Compute the Eye Aspect Ratio (EAR) for one eye.

    Parameters
    ----------
    landmarks_norm : np.ndarray, shape (N, 3)
        Normalized landmark array from FaceMeshResult.landmarks_norm.
    eye_indices : list[int]
        Six landmark indices in the order [p1, p2, p3, p4, p5, p6].

    Returns
    -------
    float
        EAR value. Typical open-eye range is 0.25–0.40.
        Values below ~0.20 typically indicate a closed eye.
    """
    p = landmarks_norm[eye_indices, :2]  # only x, y — shape (6, 2)

    vertical_a = np.linalg.norm(p[1] - p[5])   # ‖p2 - p6‖
    vertical_b = np.linalg.norm(p[2] - p[4])   # ‖p3 - p5‖
    horizontal = np.linalg.norm(p[0] - p[3])   # ‖p1 - p4‖

    if horizontal < 1e-6:
        return 0.0

    return float((vertical_a + vertical_b) / (2.0 * horizontal))


def compute_mean_ear(landmarks_norm: np.ndarray) -> float:
    """
    Compute the average EAR across both eyes.

    Parameters
    ----------
    landmarks_norm : np.ndarray, shape (478, 3)

    Returns
    -------
    float — mean EAR of left and right eyes.
    """
    left = compute_ear(landmarks_norm, LEFT_EYE_IDX)
    right = compute_ear(landmarks_norm, RIGHT_EYE_IDX)
    return (left + right) / 2.0


# ---------------------------------------------------------------------------
# Stateful blink detector
# ---------------------------------------------------------------------------

@dataclass
class BlinkDetector:
    """
    Detects blink events using a closed → open state transition.

    A blink is counted only when the eye transitions from CLOSED back to OPEN,
    preventing a single prolonged closure from registering multiple blinks.

    Parameters
    ----------
    ear_threshold : float
        EAR value at or below which the eye is considered closed (default 0.20).
    consec_frames_closed : int
        Minimum number of consecutive frames the eye must be below the threshold
        before a closure is registered (debounce, default 2).

    Usage
    -----
        detector = BlinkDetector()
        blinked = detector.update(mean_ear)  # True on the frame a blink completes
    """

    ear_threshold: float = 0.20
    consec_frames_closed: int = 2

    # Internal state — not part of the public API
    _closed_frame_count: int = field(default=0, init=False, repr=False)
    _eye_was_closed: bool = field(default=False, init=False, repr=False)
    _total_blinks: int = field(default=0, init=False, repr=False)

    def update(self, ear: float) -> bool:
        """
        Feed a new EAR value and return whether a blink just completed.

        Parameters
        ----------
        ear : float
            Mean EAR for the current frame.

        Returns
        -------
        bool
            True exactly on the frame when the eye reopens after a confirmed closure.
        """
        blink_detected = False

        if ear <= self.ear_threshold:
            self._closed_frame_count += 1
            if self._closed_frame_count >= self.consec_frames_closed:
                self._eye_was_closed = True
        else:
            # Eye is open — check if it was previously confirmed-closed
            if self._eye_was_closed:
                self._total_blinks += 1
                blink_detected = True
            self._closed_frame_count = 0
            self._eye_was_closed = False

        return blink_detected

    def reset(self) -> None:
        """Reset all state (e.g., after a long gap with no face detected)."""
        self._closed_frame_count = 0
        self._eye_was_closed = False

    @property
    def total_blinks(self) -> int:
        """Total blink count since construction or last explicit reset."""
        return self._total_blinks

    @property
    def is_closed(self) -> bool:
        """True if the eye is currently in a confirmed-closed state."""
        return self._eye_was_closed


# ---------------------------------------------------------------------------
# Convenience result bundle
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EyeMetrics:
    """
    Snapshot of eye-related metrics for a single frame.

    Attributes
    ----------
    left_ear : float
    right_ear : float
    mean_ear : float
    is_closed : bool       — True if eyes are in confirmed-closed state
    blink_detected : bool  — True only on the frame a blink completes
    total_blinks : int     — Running count from the BlinkDetector
    """

    left_ear: float
    right_ear: float
    mean_ear: float
    is_closed: bool
    blink_detected: bool
    total_blinks: int


def process_eyes(
    landmarks_norm: np.ndarray,
    blink_detector: BlinkDetector,
) -> EyeMetrics:
    """
    Compute all eye metrics for one frame and update the blink detector.

    Parameters
    ----------
    landmarks_norm : np.ndarray, shape (478, 3)
        From FaceMeshResult.landmarks_norm.
    blink_detector : BlinkDetector
        Stateful detector — caller owns and persists this across frames.

    Returns
    -------
    EyeMetrics
    """
    left_ear = compute_ear(landmarks_norm, LEFT_EYE_IDX)
    right_ear = compute_ear(landmarks_norm, RIGHT_EYE_IDX)
    mean_ear = (left_ear + right_ear) / 2.0

    blinked = blink_detector.update(mean_ear)

    return EyeMetrics(
        left_ear=left_ear,
        right_ear=right_ear,
        mean_ear=mean_ear,
        is_closed=blink_detector.is_closed,
        blink_detected=blinked,
        total_blinks=blink_detector.total_blinks,
    )