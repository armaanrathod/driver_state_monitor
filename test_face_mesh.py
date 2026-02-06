import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'driver_awareness'))

from camera import CrossPlatformCamera
from perception.face_mesh import FaceMeshDetector
import cv2
import time

def main():
    camera = CrossPlatformCamera(camera_index=0)
    detector = FaceMeshDetector()

    if not camera.start():
        print("Failed to start camera")
        return

    fps_start_time = time.time()
    fps_frame_count = 0
    fps_display = 0.0

    try:
        for frame in camera.frames():
            face_data = detector.process(frame)
            
            frame = detector.draw(frame, face_data)
            frame = camera.draw_status(frame, face_data, fps_display)

            fps_frame_count += 1
            if time.time() - fps_start_time >= 1.0:
                fps_display = fps_frame_count / (time.time() - fps_start_time)
                fps_frame_count = 0
                fps_start_time = time.time()

            cv2.imshow("Driver Awareness", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        detector.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
