"""
camera.py
---------
Perception layer — hardware access only.

Responsibilities:
  - Open a camera device once
  - Yield BGR frames on demand
  - Release the device cleanly on shutdown

Does NOT:
  - Process or analyze frames
  - Draw on frames
  - Make decisions
"""

import cv2


class Camera:
    """
    Manages the lifecycle of a single camera device.

    Usage
    -----
        cam = Camera(device_index=0)
        cam.start()
        frame = cam.read()   # returns BGR ndarray or None
        cam.release()

    Or as a context manager:
        with Camera() as cam:
            frame = cam.read()
    """

    def __init__(self, device_index: int = 0, width: int = 640, height: int = 480):
        self._index = device_index
        self._width = width
        self._height = height
        self._cap: cv2.VideoCapture | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the camera. Raises RuntimeError if already started or device fails."""
        if self._cap is not None and self._cap.isOpened():
            raise RuntimeError(
                "Camera is already running. Call release() before calling start() again."
            )

        self._cap = cv2.VideoCapture(self._index)

        if not self._cap.isOpened():
            self._cap = None
            raise RuntimeError(
                f"Failed to open camera device {self._index}. "
                "Check that the device is connected and not in use by another application."
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, 30)

    def release(self) -> None:
        """Release the camera device so other applications can use it."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def read(self):
        """
        Capture one frame from the camera.

        Returns
        -------
        numpy.ndarray (H, W, 3) BGR image, or None if capture failed.

        Raises
        ------
        RuntimeError if the camera has not been started.
        """
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("Camera is not started. Call start() first.")

        ok, frame = self._cap.read()
        return frame if ok else None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def frame_width(self) -> int:
        if self._cap is None:
            return self._width
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def frame_height(self) -> int:
        if self._cap is None:
            return self._height
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False  # do not suppress exceptions