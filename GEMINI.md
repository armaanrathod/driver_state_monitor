# GEMINI: Project Context & Instructions

This file provides the foundational context and instructional mandates for the Driver Awareness System.

## Project Overview
A modular, real-time driver behavior analysis system focused on edge computer vision. The project serves as a learning platform for designing clean perception pipelines with strict separation of concerns.

### Main Technologies
- **Python:** Core logic.
- **OpenCV:** Hardware access and visualization.
- **MediaPipe:** FaceMesh for facial landmark detection.
- **NumPy:** Efficient numerical operations (EAR calculation).

### Architecture: The Perception Pipeline
The system follows a strict unidirectional flow:
**Camera (Hardware) → Face Mesh (Perception) → Feature Modules (Logic) → Orchestrator (Integration)**

1. **Camera (`camera.py`):** Low-level hardware abstraction with explicit lifecycle management.
2. **Face Mesh (`src/driver_awareness/perception/face_mesh.py`):** Pure perception layer; detects landmarks but performs no semantic analysis.
3. **Eyes (`src/driver_awareness/perception/eyes.py`):** Pure logic layer; computes EAR and detects blinks from raw landmarks.
4. **Test Script (`scripts/test_face_mesh.py`):** The integration layer that "glues" the pipeline together and provides optional debug visualization.

---

## Building and Running

### Installation
The project is structured as a portable Python package. Install it in editable mode:
```bash
pip install -e .
```

### Running the Integration Pipeline
Run the main test script from the root directory:
```bash
# Standard run with visualization
python scripts/test_face_mesh.py

# Headless run (no window)
python scripts/test_face_mesh.py --no-viz

# Limit to N frames for testing
python scripts/test_face_mesh.py --frames 100
```

---

## Development Conventions

### 1. Separation of Concerns (Mandatory)
- **Hardware Layer:** Only `camera.py` should interact with the physical device.
- **Perception Layer:** Models should return raw data (landmarks/coordinates) in immutable dataclasses. No drawing logic here.
- **Logic Layer:** Modules must be "pure" (side-effect free) where possible. They should take coordinates as input and return semantic metrics.
- **UI/Visualization:** Drawing and display logic should be confined to orchestration scripts or dedicated UI modules.

### 2. Lifecycle Management
All hardware or ML resources (Camera, MediaPipe models) **must** implement a start/release lifecycle and support the context manager (`with` statement) pattern to prevent resource leaks.

### 3. Data Structures
Use `FaceMeshResult` and `EyeMetrics` dataclasses to pass data between modules. Avoid passing raw lists or dictionaries where possible to maintain type safety.

### 4. Import Patterns
Always use absolute imports starting from the `driver_awareness` package (e.g., `from driver_awareness.perception.eyes import ...`).

---

## Instructional Mandates for AI
- **Strictly Adhere to Modular Design:** Do not add drawing logic to perception modules.
- **Maintain Statelessness:** Keep perception and logic modules stateless or explicitly stateful (like `BlinkDetector`) to allow for easy unit testing.
- **Verify with Test Script:** Always verify changes by running `scripts/test_face_mesh.py` in a headless environment if visualization is not possible.
