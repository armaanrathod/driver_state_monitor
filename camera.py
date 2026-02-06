import cv2
import time


class CrossPlatformCamera:
    """
    Simple, reliable OpenCV-based camera interface.
    Owns the camera lifecycle explicitly.
    """

    def __init__(self, camera_index=0, width=640, height=480, fps=30):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None

    def start(self) -> bool:
        """Initialize and start the camera once."""
        if self.cap is not None:
            return True  # already started

        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            self.cap = None
            return False

        # Set capture properties (best-effort)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        return True

    def read(self):
        """Read a single frame."""
        if self.cap is None:
            return False, None
        return self.cap.read()

    def frames(self):
        """
        Generator yielding frames.
        Camera MUST be started before calling this.
        """
        if self.cap is None:
            raise RuntimeError("Camera not started. Call start() first.")

        while True:
            success, frame = self.read()
            if not success or frame is None:
                break
            yield frame

    def release(self):
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def has_face(self, face_data):
        """Check if face is detected."""
        return face_data is not None
    
    def draw_status(self, frame, face_data, fps=None):
        """Draw face detection status and FPS on frame."""
        if self.has_face(face_data):
            status_text = "Face: DETECTED"
            status_color = (0, 255, 0)
        else:
            status_text = "Face: NOT DETECTED"
            status_color = (0, 0, 255)
        
        if fps is not None:
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, status_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        return frame
