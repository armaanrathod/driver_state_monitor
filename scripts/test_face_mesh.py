"""
test_face_mesh.py
-----------------
Integration layer — orchestrates Camera → FaceMesh → Eyes → Temporal pipeline.

Runs the full perception + feature extraction loop and renders an optional
debug overlay. Visualization is a development aid only; system correctness
does not depend on it.

Usage
-----
    export PYTHONPATH=$PYTHONPATH:$(pwd)/src
    python scripts/test_face_mesh.py           # with debug overlay
    python scripts/test_face_mesh.py --no-viz  # headless / without drawing
    python scripts/test_face_mesh.py --frames 300   # stop after N frames
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2

# Hardware / root-level
from camera import Camera

# Perception / Feature extraction from src/
from driver_awareness.perception.eyes import (
    BlinkDetector,
    EyeMetrics,
    LEFT_EYE_IDX,
    RIGHT_EYE_IDX,
    process_eyes,
)
from driver_awareness.perception.face_mesh import FaceMesh, FaceMeshResult
from driver_awareness.perception.head_pose import HeadPoseEstimator, HeadPoseResult
from driver_awareness.logic.temporal import TemporalAggregator, TemporalMetrics


# ---------------------------------------------------------------------------
# Debug visualization helpers  (behind the --viz flag)
# ---------------------------------------------------------------------------

def _draw_eye_landmarks(
    frame,
    result: FaceMeshResult,
    eye_indices: list[int],
    color: tuple[int, int, int],
) -> None:
    """Draw the 6 EAR landmark points for one eye."""
    for idx in eye_indices:
        x, y = result.landmarks_px[idx]
        cv2.circle(frame, (x, y), 2, color, -1)


def _draw_iris(frame, iris_px, color: tuple[int, int, int]) -> None:
    """Draw the 5 iris landmark points."""
    for x, y in iris_px:
        cv2.circle(frame, (x, y), 2, color, -1)


def _draw_hud(
    frame,
    metrics: EyeMetrics,
    pose: HeadPoseResult | None,
    temporal: TemporalMetrics,
    fps: float,
) -> None:
    """Render a minimal HUD overlay onto the frame."""
    h, w = frame.shape[:2]

    # --- Header Metrics ---
    lines = [
        f"FPS: {fps:.1f}",
        f"EAR: {metrics.mean_ear:.3f} ({'CLOSED' if metrics.is_closed else 'OPEN'})",
        f"Blinks: {metrics.total_blinks}",
    ]
    
    # --- Pose Metrics ---
    if pose:
        lines.append(f"Pose: {pose.direction} (Y:{pose.yaw:.1f} P:{pose.pitch:.1f})")
    else:
        lines.append("Pose: UNKNOWN")

    # --- Temporal Metrics ---
    lines.append(f"PERCLOS: {temporal.perclos:.2f}")
    lines.append(f"Drowsiness: {temporal.drowsiness_score:.2f}")

    y_start = 24
    for i, text in enumerate(lines):
        # Color logic: Red if distracted or drowsy
        color = (0, 220, 0) # Green
        if "CLOSED" in text or temporal.drowsiness_score > 0.5 or (pose and pose.is_distracted and "Pose:" in text):
            color = (0, 0, 220) # Red
            
        cv2.putText(
            frame, text,
            (10, y_start + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA,
        )

    # --- Events ---
    if metrics.blink_detected:
        cv2.putText(
            frame, "BLINK",
            (w // 2 - 30, h - 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA,
        )
    
    if temporal.microsleep_detected:
        cv2.putText(
            frame, "MICROSLEEP DETECTED!",
            (w // 2 - 100, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA,
        )


def render_debug(
    frame,
    result: FaceMeshResult,
    metrics: EyeMetrics,
    pose: HeadPoseResult | None,
    temporal: TemporalMetrics,
    fps: float,
) -> None:
    """Compose all debug visuals onto the frame (in-place)."""
    _draw_eye_landmarks(frame, result, LEFT_EYE_IDX, color=(0, 255, 0))
    _draw_eye_landmarks(frame, result, RIGHT_EYE_IDX, color=(0, 255, 0))
    _draw_iris(frame, result.left_iris_px, color=(255, 100, 0))
    _draw_iris(frame, result.right_iris_px, color=(255, 100, 0))
    _draw_hud(frame, metrics, pose, temporal, fps)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run(
    device_index: int = 0,
    visualize: bool = True,
    max_frames: int | None = None,
    ear_threshold: float = 0.20,
) -> None:
    """
    Main perception loop.

    Parameters
    ----------
    device_index  : camera device index (usually 0 for built-in webcam)
    visualize     : if True, render debug overlay and open a display window
    max_frames    : stop after this many frames (None = run until 'q' pressed)
    ear_threshold : EAR value below which eyes are considered closed
    """

    camera = Camera(device_index=device_index)
    detector = FaceMesh(refine_landmarks=True)
    blink_detector = BlinkDetector(ear_threshold=ear_threshold)
    pose_estimator = HeadPoseEstimator()
    temporal_aggregator = TemporalAggregator(ear_closed_threshold=ear_threshold)

    frame_count = 0
    no_face_count = 0
    fps = 0.0
    t_prev = time.perf_counter()

    print("[INFO] Starting pipeline. Press 'q' to quit.")

    with camera, detector:
        while True:
            # ---- Termination conditions --------------------------------
            if max_frames is not None and frame_count >= max_frames:
                print(f"[INFO] Reached max_frames={max_frames}. Stopping.")
                break

            if visualize and cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] 'q' pressed. Stopping.")
                break

            # ---- Capture -----------------------------------------------
            frame = camera.read()
            if frame is None:
                print("[WARN] Empty frame received. Skipping.")
                continue

            frame_count += 1

            # ---- FPS estimate ------------------------------------------
            t_now = time.perf_counter()
            elapsed = t_now - t_prev
            if elapsed > 0:
                fps = 1.0 / elapsed
            t_prev = t_now

            # ---- Perception --------------------------------------------
            result: FaceMeshResult | None = detector.detect(frame)

            if result is None:
                no_face_count += 1
                # Reset blink state when face is lost to avoid false events
                blink_detector.reset()
                temporal_aggregator.reset()

                if visualize:
                    cv2.putText(
                        frame, "No face detected",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 0, 200), 1, cv2.LINE_AA,
                    )
                    cv2.imshow("Driver Awareness — Debug", frame)
                continue

            # ---- Feature extraction ------------------------------------
            # 1. Eyes
            metrics: EyeMetrics = process_eyes(result.landmarks_norm, blink_detector)
            
            # 2. Head Pose
            pose: HeadPoseResult | None = pose_estimator.estimate(
                result.landmarks_norm, 
                result.frame_width, 
                result.frame_height
            )
            
            # 3. Temporal Aggregation
            temporal: TemporalMetrics = temporal_aggregator.update(metrics)

            # ---- Console logging ---------------------------
            if metrics.blink_detected:
                print(
                    f"[BLINK #{metrics.total_blinks:03d}]  "
                    f"frame={frame_count}  EAR={metrics.mean_ear:.3f}"
                )
            
            if temporal.microsleep_detected:
                print(f"[ALERT] Microsleep detected at frame {frame_count}!")

            # ---- Debug visualization (optional) ------------------------
            if visualize:
                render_debug(frame, result, metrics, pose, temporal, fps)
                cv2.imshow("Driver Awareness — Debug", frame)

    if visualize:
        cv2.destroyAllWindows()

    # ---- Summary -------------------------------------------------------
    print("\n[SUMMARY]")
    print(f"  Total frames processed : {frame_count}")
    print(f"  Frames without face    : {no_face_count}")
    print(f"  Total blinks detected  : {blink_detector.total_blinks}")
    if frame_count > 0:
        print(f"  Final Drowsiness Score : {temporal.drowsiness_score:.2f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Driver awareness pipeline — integration test"
    )
    parser.add_argument(
        "--device", type=int, default=0,
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--no-viz", dest="visualize", action="store_false", default=True,
        help="Disable debug visualization window",
    )
    parser.add_argument(
        "--frames", type=int, default=None, dest="max_frames",
        help="Stop after N frames (default: run until 'q')",
    )
    parser.add_argument(
        "--ear-threshold", type=float, default=0.20,
        help="EAR below which eyes are considered closed (default: 0.20)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        run(
            device_index=args.device,
            visualize=args.visualize,
            max_frames=args.max_frames,
            ear_threshold=args.ear_threshold,
        )
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
        sys.exit(0)
