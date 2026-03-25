# Code Understanding: Driver Awareness System

This document provides a deep dive into the architecture and data flow of the Driver Awareness System.

## System Architecture

The system is designed as a **unidirectional perception pipeline** with a strict separation between hardware access, machine learning inference, and feature logic.

### 1. The Pipeline Flow
`Camera` (Hardware) → `FaceMesh` (Perception) → `Eyes` (Logic) → `Script` (Orchestration/UI)

---

## Module Breakdown

### A. Camera Module (`camera.py`)
- **Responsibility:** Pure hardware abstraction.
- **Key Pattern:** Lifecycle management using context managers (`__enter__`/`__exit__`).
- **Input:** None (Physical hardware).
- **Output:** Raw `np.ndarray` (BGR format).
- **Design Philosophy:** Avoids "leaking" CV2 state into the rest of the application.

### B. Face Mesh Module (`src/driver_awareness/perception/face_mesh.py`)
- **Responsibility:** Landmark localization.
- **Key Pattern:** Stateless detection calls returning immutable dataclasses (`FaceMeshResult`).
- **Input:** BGR Frame.
- **Output:** Normalized coordinates (0.0 to 1.0) and pixel coordinates.
- **Design Philosophy:** Acts as a "Pure Perception" layer. It doesn't know what an "eye" is; it only knows where point #362 is.

### C. Eyes Module (`src/driver_awareness/perception/eyes.py`)
- **Responsibility:** Semantic feature extraction.
- **Key Patterns:** 
    - Pure functions for EAR (`compute_ear`).
    - Stateful state-machine for blinks (`BlinkDetector`).
- **Input:** Normalized landmark coordinates.
- **Output:** `EyeMetrics` (EAR values, blink flags, blink counts).
- **Design Philosophy:** Decoupled from MediaPipe and OpenCV. It operates on coordinate data, making it highly unit-testable.

---

## Data Models

### `FaceMeshResult`
Encapsulates the raw output of the ML model:
- `landmarks_norm`: (478, 3) - Global face structure.
- `left_iris_px` / `right_iris_px`: High-precision subsets for future gaze tracking.

### `EyeMetrics`
Encapsulates the semantic state of the driver:
- `mean_ear`: Average openness.
- `blink_detected`: Single-frame trigger for event logging.
- `is_closed`: Boolean state for drowsiness detection.

---

## Design Analysis

### Strengths
1. **Modularity:** You can replace MediaPipe with another model without changing the `Eyes` logic.
2. **Testability:** The `Eyes` module can be tested using static JSON landmark data without needing a camera.
3. **Safety:** Immutable dataclasses prevent accidental modification of landmarks during the processing loop.

### Identified Weaknesses
1. **Script Redundancy:** `scripts/test_face_mesh.py` is currently a duplicate of `face_mesh.py`. It needs to be rewritten to import and use the modules correctly.
2. **Redundant Color Conversion:** If multiple detectors (Face, Hands, Pose) were used, they would all convert BGR to RGB independently.
3. **Threshold Hardcoding:** EAR thresholds (0.20) are currently hardcoded in the class definition.

---

## Future Improvements
- **Temporal Aggregation:** Adding a layer to track EAR over minutes (PERCLOS) rather than just per-frame blinks.
- **Configuration Layer:** Using a YAML or JSON file to manage thresholds and camera indices.
- **Visualization Module:** Moving drawing logic out of scripts and into a dedicated `driver_awareness.ui` module.
