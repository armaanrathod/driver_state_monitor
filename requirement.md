# Driver Awareness Requirements and Setup

## Purpose

This file documents the exact environment and run setup needed for this project so it can be reproduced on another machine.

## Python Version

- Recommended: Python 3.11.x
- Virtual environment location used in this project: ../.venv (workspace root)

## Python Packages (Pinned)

Install these in the virtual environment:

- numpy
- opencv-python
- mediapipe==0.10.14

Why pin MediaPipe:

- Newer MediaPipe versions may not expose mp.solutions, while this codebase currently uses that API.

## VS Code Runtime Path Setup

This project is inside a parent workspace folder. To make imports work from VS Code Run/Debug:

1. Workspace root .env should include:

PYTHONPATH=driver-awareness;driver-awareness/src

2. Workspace root .vscode/launch.json should set:

- cwd to ${workspaceFolder}/driver-awareness
- envFile to ${workspaceFolder}/.env
- interpreter to ${workspaceFolder}/.venv/Scripts/python.exe

3. Workspace root .vscode/settings.json should set:

- python.defaultInterpreterPath to ${workspaceFolder}/.venv/Scripts/python.exe
- python.envFile to ${workspaceFolder}/.env

## Run Command (Windows)

From workspace root:

PowerShell:

$env:PYTHONPATH='driver-awareness;driver-awareness/src'
.\.venv\Scripts\python.exe .\driver-awareness\scripts\test_face_mesh.py --no-viz --frames 100

From project root:

PowerShell:

$env:PYTHONPATH='src'
..\.venv\Scripts\python.exe .\scripts\test_face_mesh.py --no-viz --frames 100

## Common Issues on Another System and Fixes

1. ModuleNotFoundError: No module named camera

Cause:

- Script launched from a folder that is not the project root.

Fix:

- Ensure cwd is driver-awareness when running.
- Ensure PYTHONPATH includes both project root and src, or run from project root with PYTHONPATH=src.

2. ModuleNotFoundError for driver_awareness package

Cause:

- src is missing from PYTHONPATH.

Fix:

- Set PYTHONPATH to include src.

3. PowerShell cannot run Activate.ps1

Cause:

- Execution policy blocks script activation.

Fix options:

- Use direct interpreter path without activation:
  .\.venv\Scripts\python.exe ...
- Or temporarily allow scripts for current process:
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

4. AttributeError: module mediapipe has no attribute solutions

Cause:

- Incompatible MediaPipe version.

Fix:

- Install mediapipe==0.10.14 in the active virtual environment.

5. Webcam or camera open errors

Cause:

- Camera in use by another app, missing permissions, or wrong device index.

Fix:

- Close other camera apps.
- Try --device 1 or --device 2.
- Check OS camera privacy permissions.

## Reproducible Onboarding Checklist for New Machine

1. Clone repository.
2. Create virtual environment with Python 3.11.
3. Install packages listed above.
4. Open the parent workspace folder (the folder that contains driver-awareness and .venv).
5. Ensure .env and .vscode configs exist at workspace root.
6. Run a quick smoke test with --no-viz and --frames 50.

## Notes

- Current runtime import issue (camera) was solved by workspace-root VS Code configuration and PYTHONPATH setup.
- Any future move of folder structure should update .env and launch.json paths accordingly.
