I am building a modular real-time driver behavior analysis system as a learning project focused on edge computer vision systems design.



The goal is not to create a startup or production automotive system. The goal is to understand how to design a clean, real-time perception pipeline with proper separation of concerns (perception, feature extraction, temporal logic, decision-making).



The system currently follows this architecture:



Camera → Face Mesh → Feature Modules → (planned) Temporal Aggregation → (planned) Awareness Score



Current State

Camera Module

OpenCV-based

Explicit lifecycle management (start once, release once)

Provides continuous BGR frames

Designed to avoid double initialization and tracking resets

Face Mesh Module

Uses MediaPipe FaceMesh in video mode

Returns:

Normalized landmark coordinates (N, 3)

Pixel coordinates

Iris landmark subsets

Does not contain drawing logic or scoring logic

Pure perception layer

Eyes Module

Pure logic module

Takes face landmarks as input

Computes Eye Aspect Ratio (EAR)

Implements blink detection using state-transition logic (closed → open)

No camera access

No visualization inside the module

No scoring logic

Deterministic and import-safe

Integration Layer

Orchestrates camera + face mesh + eyes

Contains optional debug visualization (behind a flag)

Visualization is limited to:

Eye landmark points

EAR values

Blink events

Visualization is considered a development aid, not a core feature

Design Principles Being Followed

Real-time execution

On-device processing only

No cloud

No ML training yet (only MediaPipe inference)

Clear separation between:

Perception (landmarks)

Feature computation (EAR, blink)

Future temporal logic

Future decision layer

Modules must be reusable and side-effect free

Visualization must not be required for system correctness

What Is Not Implemented Yet

Head pose estimation (yaw/pitch trend tracking)

Distraction trend modeling

Hand-to-face interaction detection

Temporal aggregation layer (sustained event tracking, decay logic)

Risk normalization

Continuous awareness score

Intent



This project is meant to help me understand:



Real-time computer vision pipeline design

State-based event detection

Feature normalization

Edge system architecture

How to evolve a perception system without overcoupling logic and UI



It is intentionally being built incrementally, validating each module before expanding.





