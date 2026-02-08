import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'driver_awareness'))

from camera import CrossPlatformCamera
from perception.face_mesh import FaceMeshDetector
from perception.eyes import EyeMetrics
import cv2
import time

def main():
    camera = CrossPlatformCamera(camera_index=0)
    detector = FaceMeshDetector()
    eye_metrics = EyeMetrics()

    if not camera.start():
        print("Failed to start camera")
        return

    fps_start_time = time.time()
    fps_frame_count = 0
    fps_display = 0.0

    try:
        for frame in camera.frames():
            face_data = detector.process(frame)
            
            # Process eye metrics if face is detected
            if face_data is not None:
                landmarks = face_data['landmarks']
                eye_data = eye_metrics.process(landmarks)
                
                # Temporary visualization for eye metrics validation
                if eye_data['avg_ear'] is not None:
                    pixel_coords = face_data['pixel_coords']
                    
                    # Draw eye landmark points
                    for idx in eye_metrics.LEFT_EYE_INDICES + eye_metrics.RIGHT_EYE_INDICES:
                        x, y = pixel_coords[idx]
                        cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
                    
                    # Display EAR value
                    ear_text = f"EAR: {eye_data['avg_ear']:.3f}"
                    cv2.putText(frame, ear_text, (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # Display BLINK when event detected
                    if eye_data['blink']:
                        cv2.putText(frame, "BLINK", (10, 130), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
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
