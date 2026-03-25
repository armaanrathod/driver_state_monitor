# Driver Awareness System: Development Plan

A modular, real-time perception pipeline for driver behavior analysis.

## Phase 1: Core Perception (COMPLETED)
- [x] **Camera Module:** Hardware abstraction with lifecycle management.
- [x] **Face Mesh Module:** MediaPipe landmark detection (478 points).
- [x] **Eyes Module:** EAR computation and state-based blink detection.
- [x] **Integration Script:** Orchestrator with real-time debug HUD.
- [x] **Package Structure:** Portable `setup.py` and `src/` layout.

## Phase 2: Memory & Focus (ACTIVE)
### 2.1 Focus (Head Pose Estimation)
- [ ] **Module:** `src/driver_awareness/perception/head_pose.py`
- [ ] **Task:** Implement 3D-to-2D projection for Yaw, Pitch, and Roll.
- [ ] **Logic:** Detect distraction (looking away) and drowsiness (head nodding).
- [ ] **Integration:** Add pose metrics to the `test_face_mesh.py` HUD.

### 2.2 Memory (Temporal Aggregation)
- [ ] **Module:** `src/driver_awareness/logic/temporal.py`
- [ ] **Task:** Implement PERCLOS (Percentage of Eye Closure) over a sliding window.
- [ ] **Logic:** Detect micro-sleep events (>1.5s closure).
- [ ] **Trend:** Calculate a real-time Drowsiness Score (0.0 to 1.0).

## Phase 3: Awareness Scoring & Alerts (PLANNED)
- [ ] **Orchestration:** Combine EAR, PERCLOS, and Head Pose into a single Risk Index.
- [ ] **Visualization:** Create a dedicated `driver_awareness.ui` module for clean drawing logic.
- [ ] **Logging:** Implement persistent event logging (JSON/CSV) for behavior analysis.

## Phase 4: Refinement & Robustness (PLANNED)
- [ ] **Testing:** Add a `tests/` suite with mock landmark data for logic validation.
- [ ] **Config:** Centralize thresholds (EAR, Pose angles) into a `config.yaml` file.
- [ ] **Hardware:** Optimize camera capture for higher FPS and lower latency.
