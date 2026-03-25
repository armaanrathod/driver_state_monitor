# DA: Driver Awareness System

A modular, real-time perception pipeline designed for driver behavior analysis. This system uses computer vision to monitor driver attention, fatigue, and head orientation in real-time.

---

## 🚀 System Architecture: The Pipeline
The project follows a strict **unidirectional pipeline** with a clean separation of concerns:

**Camera (Hardware) → Face Mesh (Perception) → Feature Modules (Logic) → Temporal Aggregation (Memory)**

1.  **Camera:** Captures raw BGR frames from the device.
2.  **Face Mesh:** Maps 478 3D landmarks and iris positions.
3.  **Eyes:** Computes Eye Aspect Ratio (EAR) and detects blinks.
4.  **Head Pose:** Calculates Yaw, Pitch, and Roll to track gaze direction.
5.  **Temporal:** Aggregates frame-level data into long-term trends (PERCLOS, Drowsiness Score).

---

## 🛠️ Module Breakdown

### 1. Camera Module (`camera.py`)
- **Hardware Abstraction:** Manages the lifecycle of the camera device.
- **Lifecycle:** Start, read, and release pattern using context managers.
- **Output:** Raw image frames (`np.ndarray`).

### 2. Face Mesh Module (`face_mesh.py`)
- **Perception Layer:** Wraps MediaPipe FaceMesh in video mode.
- **Precision:** Detects 478 landmarks + 10 iris-specific points.
- **Design:** Pure perception; returns coordinates without drawing or analysis.

### 3. Eyes Module (`eyes.py`)
- **Feature Extraction:** Computes the **Eye Aspect Ratio (EAR)**.
- **Blink Detection:** A state-transition machine that detects blinks (Closed → Open) with noise debouncing.
- **Pure Logic:** Decoupled from hardware and ML; operates solely on landmark data.

### 4. Head Pose Module (`head_pose.py`)
- **Spatial Focus:** Uses the **SolvePnP** algorithm to estimate 3D orientation.
- **Angles:** Outputs Yaw (left/right), Pitch (up/down), and Roll (tilt) in degrees.
- **Classification:** Automatically labels the direction (e.g., "FORWARD", "LEFT", "DOWN").

### 5. Temporal Module (`temporal.py`)
- **Memory Layer:** Maintains a sliding window (60s) of all driver observations.
- **PERCLOS:** Calculates the industry-standard "Percentage of Eye Closure" for drowsiness.
- **Drowsiness Score:** A composite index (0.0 to 1.0) based on PERCLOS, micro-sleep events, and blink rate deviations.

---

## 💻 Installation

The project is structured as a portable Python package.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/DA.git
   cd DA
   ```

2. **Install in editable mode:**
   ```bash
   pip install -e .
   ```
   *This automatically installs dependencies: OpenCV, MediaPipe, and NumPy.*

---

## 🏃 Usage

Run the main integration script to see the system in action with a real-time HUD:

```bash
# Set your path (if not using the pip installation)
$env:PYTHONPATH = ".;src"  # Windows
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/src  # Linux/Mac

# Run with Visualization
python scripts/test_face_mesh.py

# Run Headless (CLI summary only)
python scripts/test_face_mesh.py --no-viz

# Limit frames for testing
python scripts/test_face_mesh.py --frames 300
```

---

## 📈 Roadmap
- [ ] **Phase 3:** Awareness Scoring & Alerts (Orchestration logic).
- [ ] **Phase 4:** Dedicated Visualization UI module.
- [ ] **Phase 5:** Persistent Event Logging (JSON/CSV).

---

## ⚖️ Design Philosophy
- **Separation of Concerns:** Hardware, Perception, and Logic never mix.
- **Stateless Perception:** Modules process frames independently where possible.
- **Deterministic Logic:** Core algorithms are side-effect free and testable.
