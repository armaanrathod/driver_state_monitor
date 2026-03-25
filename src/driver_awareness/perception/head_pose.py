"""
head_pose.py
------------
Perception layer — head orientation estimation from facial landmarks.

Responsibilities:
  - Accept normalized MediaPipe landmarks (478, 3) and frame dimensions
  - Project 6 canonical 3D face points onto 2D image space (solvePnP)
  - Decompose the resulting rotation matrix into Yaw / Pitch / Roll (degrees)
  - Return a structured, immutable HeadPoseResult with a Direction label

Does NOT:
  - Draw lines, axes, or annotations on frames
  - Hold state across frames — fully stateless, one call per frame
  - Access the camera, MediaPipe, or any I/O device
  - Perform temporal smoothing (that belongs in the temporal layer)

Architecture position
---------------------
    FaceMesh (landmarks_norm) → HeadPoseEstimator.estimate() → HeadPoseResult
                                        ↓
                           (future) TemporalAggregator

Algorithm
---------
Uses OpenCV's solvePnP (ITERATIVE method) to solve the Perspective-n-Point
problem between a known 3D canonical face model and the 2D projected landmark
positions derived from MediaPipe's normalized coordinates.

Rotation vector → rotation matrix (Rodrigues) → Euler angles (RQDecomp3x3).

Camera intrinsics
-----------------
Real intrinsic parameters are rarely available at runtime. We use the
standard approximation:
  focal_length ≈ frame_width  (assumes ~60° horizontal FOV)
  principal_point = (frame_width / 2, frame_height / 2)
This is well-established for face-based head-pose estimation and produces
accurate relative angle measurements suitable for driver monitoring.

References
----------
- Gee & Cipolla (1994): Determining the gaze of faces in images.
- Kazemi & Sullivan (2014): One millisecond face alignment with an ensemble
  of regression trees (6-point model convention).
- OpenCV solvePnP documentation:
  https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Gaze direction label
# ---------------------------------------------------------------------------

class Direction(str, Enum):
    """
    Coarse head-orientation label derived from Yaw and Pitch.

    The string mixin lets callers use the value directly as a display
    string without calling ``.value``.
    """
    FORWARD    = "FORWARD"
    LEFT       = "LEFT"
    RIGHT      = "RIGHT"
    UP         = "UP"
    DOWN       = "DOWN"
    DOWN_LEFT  = "DOWN_LEFT"
    DOWN_RIGHT = "DOWN_RIGHT"
    UP_LEFT    = "UP_LEFT"
    UP_RIGHT   = "UP_RIGHT"


# ---------------------------------------------------------------------------
# Canonical 3D face model
# ---------------------------------------------------------------------------
# Six anatomically stable landmarks expressed in a face-centric metric
# coordinate system (millimetres), with the nose tip at the origin.
#
# Coordinate axes (right-hand, OpenCV convention):
#   +X  → right side of the face (from the subject's perspective)
#   +Y  → downward
#   +Z  → out of the face toward the camera
#
# These values follow the widely-used Kazemi & Sullivan (2014) convention
# and are tuned for the MediaPipe 478-point landmark topology.
#
# Landmark index → anatomical location:
#   1   → Nose tip              (origin)
#   152 → Chin                  (mentolabial sulcus)
#   33  → Right eye outer corner
#   263 → Left eye outer corner
#   61  → Right mouth corner
#   291 → Left mouth corner
# ---------------------------------------------------------------------------

# Index ordering must match _IMAGE_POINT_INDICES below — one-to-one.
_MODEL_POINTS_3D = np.array(
    [
        (  0.0,    0.0,    0.0),   # 1   Nose tip
        (  0.0,  -63.6,  -12.5),   # 152 Chin
        ( 43.3,   32.7,  -26.0),   # 33  Right eye outer corner (+X = right)
        (-43.3,   32.7,  -26.0),   # 263 Left eye outer corner  (−X = left)
        ( 28.9,  -28.9,  -24.1),   # 61  Right mouth corner
        (-28.9,  -28.9,  -24.1),   # 291 Left mouth corner
    ],
    dtype=np.float64,
)

# Corresponding MediaPipe FaceMesh landmark indices (0-indexed, range 0–477).
# Order must be identical to _MODEL_POINTS_3D rows above.
_IMAGE_POINT_INDICES: list[int] = [1, 152, 33, 263, 61, 291]


# ---------------------------------------------------------------------------
# Direction classification thresholds (degrees)
# ---------------------------------------------------------------------------

_YAW_THRESHOLD_DEG: float   = 15.0   # |yaw|  > this → LEFT or RIGHT
_PITCH_THRESHOLD_DEG: float = 12.0   # |pitch| > this → UP or DOWN


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HeadPoseResult:
    """
    Immutable head-orientation result for a single frame.

    Attributes
    ----------
    yaw : float
        Rotation around the vertical (Y) axis, in degrees.
        Negative = driver looking LEFT; Positive = looking RIGHT.
    pitch : float
        Rotation around the lateral (X) axis, in degrees.
        Negative = looking DOWN; Positive = looking UP.
    roll : float
        Rotation around the depth (Z) axis, in degrees.
        Describes head tilt (ear toward shoulder).
    direction : Direction
        Coarse gaze label derived from yaw and pitch thresholds.
    is_distracted : bool
        True if direction is anything other than FORWARD.
        Convenience flag for the temporal aggregation layer.
    """

    yaw: float
    pitch: float
    roll: float
    direction: Direction
    is_distracted: bool


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------

class HeadPoseEstimator:
    """
    Stateless head-pose estimator using the solvePnP algorithm.

    The estimator is instantiated once and can process any number of frames.
    It holds no mutable state — every call to ``estimate()`` is independent.

    Usage
    -----
        estimator = HeadPoseEstimator()

        # Inside your frame loop (after FaceMesh.detect()):
        result: HeadPoseResult | None = estimator.estimate(
            landmarks_norm=face_mesh_result.landmarks_norm,
            frame_width=face_mesh_result.frame_width,
            frame_height=face_mesh_result.frame_height,
        )
        if result:
            print(result.direction, result.yaw, result.pitch)

    Notes
    -----
    - ``estimate()`` returns None when solvePnP fails (degenerate geometry,
      very extreme angles, or corrupted landmarks). Callers should handle None
      the same way they handle a missing face detection.
    - All angles are in **degrees** for human readability and threshold
      comparison. Convert to radians only if you need trig downstream.
    """

    # ------------------------------------------------------------------
    # estimate()
    # ------------------------------------------------------------------

    def estimate(
        self,
        landmarks_norm: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> HeadPoseResult | None:
        """
        Compute head pose angles from normalized MediaPipe landmarks.

        Parameters
        ----------
        landmarks_norm : np.ndarray, shape (478, 3)
            Normalized landmark array from ``FaceMeshResult.landmarks_norm``.
            Coordinates are in [0, 1] range; z is relative depth only.
        frame_width : int
            Width of the source frame in pixels.
        frame_height : int
            Height of the source frame in pixels.

        Returns
        -------
        HeadPoseResult if solvePnP converges, otherwise None.

        Raises
        ------
        ValueError
            If ``landmarks_norm`` does not have shape (N, 3) with N >= 292
            (minimum index required is 291).
        """
        if landmarks_norm.ndim != 2 or landmarks_norm.shape[1] < 3:
            raise ValueError(
                f"landmarks_norm must have shape (N, 3), got {landmarks_norm.shape}."
            )
        if landmarks_norm.shape[0] < 292:
            raise ValueError(
                f"landmarks_norm must contain at least 292 rows "
                f"(highest required index is 291), got {landmarks_norm.shape[0]}."
            )

        # 1. Build camera intrinsic matrix
        camera_matrix = _build_camera_matrix(frame_width, frame_height)

        # 2. Extract 2D image points from normalized landmarks
        image_points_2d = _extract_image_points(
            landmarks_norm, frame_width, frame_height
        )

        # 3. Solve PnP
        success, rvec, tvec = cv2.solvePnP(
            _MODEL_POINTS_3D,
            image_points_2d,
            camera_matrix,
            _DIST_COEFFS,          # assume no lens distortion
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return None

        # 4. Rotation vector → rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rvec)

        # 5. Decompose rotation matrix → Euler angles (degrees)
        yaw, pitch, roll = _rotation_matrix_to_euler_degrees(rotation_matrix)

        # 6. Classify direction
        direction = _classify_direction(yaw, pitch)

        return HeadPoseResult(
            yaw=round(yaw, 2),
            pitch=round(pitch, 2),
            roll=round(roll, 2),
            direction=direction,
            is_distracted=(direction is not Direction.FORWARD),
        )


# ---------------------------------------------------------------------------
# Module-level constants (computed once at import time)
# ---------------------------------------------------------------------------

# No lens distortion assumed — reasonable for built-in webcams and phone
# cameras at the distances relevant for driver monitoring.
_DIST_COEFFS = np.zeros((4, 1), dtype=np.float64)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_camera_matrix(width: int, height: int) -> np.ndarray:
    """
    Build an approximate 3×3 camera intrinsic matrix.

    focal_length ≈ frame_width
        Derived from the assumption of ~60° horizontal field of view,
        which holds for typical laptop webcams and phone front cameras.

    principal_point = (width / 2, height / 2)
        Assumes the optical centre is at the image centre.

    Returns
    -------
    np.ndarray, shape (3, 3), dtype float64.
    """
    focal_length = float(width)
    cx = width  / 2.0
    cy = height / 2.0

    return np.array(
        [
            [focal_length, 0.0,          cx],
            [0.0,          focal_length, cy],
            [0.0,          0.0,          1.0],
        ],
        dtype=np.float64,
    )


def _extract_image_points(
    landmarks_norm: np.ndarray,
    frame_width: int,
    frame_height: int,
) -> np.ndarray:
    """
    Convert the 6 canonical landmarks from normalized to pixel coordinates.

    Parameters
    ----------
    landmarks_norm : np.ndarray, shape (478, 3)
    frame_width, frame_height : int

    Returns
    -------
    np.ndarray, shape (6, 1, 2), dtype float64
        Required shape for cv2.solvePnP.
    """
    pts = landmarks_norm[_IMAGE_POINT_INDICES, :2].copy()  # (6, 2) — x, y only
    pts[:, 0] *= frame_width
    pts[:, 1] *= frame_height
    # solvePnP expects shape (N, 1, 2) or (N, 2) — use (N, 1, 2) to be explicit
    return pts.reshape(6, 1, 2).astype(np.float64)


def _rotation_matrix_to_euler_degrees(R: np.ndarray) -> tuple[float, float, float]:
    """
    Decompose a 3×3 rotation matrix into Yaw, Pitch, Roll in degrees.

    Uses ``cv2.RQDecomp3x3`` which performs an RQ decomposition and returns
    angles in degrees directly, avoiding gimbal-lock edge cases common in
    manual atan2 decompositions.

    Convention (right-hand, camera-facing the subject):
      Yaw   (Y-axis): negative = subject turns LEFT, positive = RIGHT
      Pitch (X-axis): negative = subject looks DOWN, positive = UP
      Roll  (Z-axis): negative = head tilts right, positive = tilts left

    Parameters
    ----------
    R : np.ndarray, shape (3, 3)

    Returns
    -------
    (yaw, pitch, roll) : tuple[float, float, float] — all in degrees
    """
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(R)

    # RQDecomp3x3 returns (pitch, yaw, roll) in OpenCV convention
    # angles[0] = rotation around X (pitch)
    # angles[1] = rotation around Y (yaw)
    # angles[2] = rotation around Z (roll)
    pitch_deg = angles[0]
    yaw_deg   = angles[1]
    roll_deg  = angles[2]

    return float(yaw_deg), float(pitch_deg), float(roll_deg)


def _classify_direction(yaw: float, pitch: float) -> Direction:
    """
    Map yaw and pitch angles to a coarse Direction label.

    Diagonal labels are assigned when both yaw and pitch exceed their
    respective thresholds simultaneously, capturing oblique head turns.

    Parameters
    ----------
    yaw   : float — degrees, negative = LEFT, positive = RIGHT
    pitch : float — degrees, negative = DOWN, positive = UP

    Returns
    -------
    Direction
    """
    looking_left  = yaw   < -_YAW_THRESHOLD_DEG
    looking_right = yaw   >  _YAW_THRESHOLD_DEG
    looking_down  = pitch < -_PITCH_THRESHOLD_DEG
    looking_up    = pitch >  _PITCH_THRESHOLD_DEG

    # Diagonal cases first (both axes exceed threshold)
    if looking_down and looking_left:
        return Direction.DOWN_LEFT
    if looking_down and looking_right:
        return Direction.DOWN_RIGHT
    if looking_up and looking_left:
        return Direction.UP_LEFT
    if looking_up and looking_right:
        return Direction.UP_RIGHT

    # Single-axis cases
    if looking_left:
        return Direction.LEFT
    if looking_right:
        return Direction.RIGHT
    if looking_down:
        return Direction.DOWN
    if looking_up:
        return Direction.UP

    return Direction.FORWARD