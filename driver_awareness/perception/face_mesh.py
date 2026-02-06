import cv2
import numpy as np
import mediapipe as mp


class FaceMeshDetector:
    
    def __init__(
        self,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ):
        self.mp_face_mesh = mp.solutions.face_mesh

        # IMPORTANT: static_image_mode MUST be False for video streams
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
        self.RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]
        
    def process(self, frame):
        if frame is None:
            return None

        # MediaPipe requires RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks is None:
            return None
        
        face_landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        
        landmarks = np.empty((len(face_landmarks.landmark), 3), dtype=np.float32)
        pixel_coords = np.empty((len(face_landmarks.landmark), 2), dtype=np.int32)
        
        for i, lm in enumerate(face_landmarks.landmark):
            landmarks[i] = (lm.x, lm.y, lm.z)
            pixel_coords[i] = (int(lm.x * w), int(lm.y * h))
        
        iris_left = None
        iris_right = None
        
        if landmarks.shape[0] > max(self.RIGHT_IRIS_INDICES):
            iris_left = landmarks[self.LEFT_IRIS_INDICES]
            iris_right = landmarks[self.RIGHT_IRIS_INDICES]
        
        return {
            'landmarks': landmarks,
            'pixel_coords': pixel_coords,
            'iris_left': iris_left,
            'iris_right': iris_right
        }
    
    def draw(self, frame, face_data):
        if face_data is None:
            return frame
        
        pixel_coords = face_data['pixel_coords']
        
        for (x, y) in pixel_coords:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        
        if face_data['iris_left'] is not None:
            iris_left_pixels = pixel_coords[self.LEFT_IRIS_INDICES]
            iris_right_pixels = pixel_coords[self.RIGHT_IRIS_INDICES]
            
            for (x, y) in iris_left_pixels:
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
            for (x, y) in iris_right_pixels:
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
        
        return frame
    
    def close(self):
        self.face_mesh.close()
