\# Project Overview



This is a modular, real-time driver behavior analysis system, developed as a learning project focused on edge computer vision systems design. The primary goal is to understand how to design a clean, real-time perception pipeline with proper separation of concerns (perception, feature extraction, temporal logic, decision-making).



The system is built incrementally, validating each module before expanding.



\## Architecture



The system follows a pipeline architecture:



\*\*Camera → Face Mesh → Feature Modules (e.g., Eyes) → (planned) Temporal Aggregation → (planned) Awareness Score\*\*



\### Core Modules:



\*   \*\*`camera.py` (Camera Module):\*\*

&#x20;   \*   OpenCV-based for video capture.

&#x20;   \*   Explicit lifecycle management (start once, release once).

&#x20;   \*   Provides continuous BGR frames.

&#x20;   \*   Designed for cross-platform compatibility and to avoid double initialization/tracking resets.

\*   \*\*`driver\_awareness/perception/face\_mesh.py` (Face Mesh Module):\*\*

&#x20;   \*   Utilizes MediaPipe FaceMesh in video mode.

&#x20;   \*   Returns normalized landmark coordinates (N, 3), pixel coordinates, and iris landmark subsets.

&#x20;   \*   Acts as a pure perception layer, without drawing or scoring logic.

\*   \*\*`driver\_awareness/perception/eyes.py` (Eyes Module):\*\*

&#x20;   \*   A pure logic module that takes face landmarks as input.

&#x20;   \*   Computes the Eye Aspect Ratio (EAR).

&#x20;   \*   Implements blink detection using state-transition logic (closed → open).

&#x20;   \*   Contains no camera access, visualization, or scoring logic, adhering to the separation of concerns principle.

\*   \*\*`test\_face\_mesh.py` (Integration Layer):\*\*

&#x20;   \*   Orchestrates the `Camera`, `FaceMeshDetector`, and `EyeMetrics` modules.

&#x20;   \*   Includes optional debug visualization (eye landmark points, EAR values, blink events) for development aid.



\## Main Technologies



\*   \*\*Python:\*\* Primary programming language.

\*   \*\*OpenCV:\*\* For camera interaction, frame processing, and drawing visualizations.

\*   \*\*MediaPipe:\*\* Specifically MediaPipe FaceMesh for robust facial landmark detection.

\*   \*\*NumPy:\*\* For efficient numerical operations, particularly in EAR calculation.



\## Building and Running



This project uses Python. To set up and run the main integration script:



1\.  \*\*Install Dependencies:\*\*

&#x20;   You will need `opencv-python`, `mediapipe`, and `numpy`.

&#x20;   ```bash

&#x20;   pip install opencv-python mediapipe numpy

&#x20;   ```

&#x20;   \*(Note: `opencv-python` might be `opencv-python-headless` for server environments without GUI dependencies).\*



2\.  \*\*Run the Integration Script:\*\*

&#x20;   The `test\_face\_mesh.py` script demonstrates the integration of the camera, face mesh, and eye metrics modules with real-time visualization.

&#x20;   ```bash

&#x20;   python test\_face\_mesh.py

&#x20;   ```

&#x20;   Press `q` to quit the visualization window.



\## Development Conventions



\*   \*\*Real-time execution:\*\* Designed for on-device processing only, with no cloud dependency.

\*   \*\*Separation of Concerns:\*\* Clear boundaries between perception, feature computation, and future temporal/decision layers.

\*   \*\*Module Reusability:\*\* Modules are intended to be side-effect free and easily reusable.

\*   \*\*Visualization as Development Aid:\*\* Debug visualizations are optional and decoupled from core logic, ensuring system correctness does not rely on them.

\*   \*\*Incremental Development:\*\* The project is built module by module, with validation at each stage.



