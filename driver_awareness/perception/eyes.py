import numpy as np


class EyeMetrics:
    """
    Extracts raw eye metrics from face mesh landmarks.
    Computes Eye Aspect Ratio (EAR) and detects blinks.
    """
    
    # Eye landmark indices for MediaPipe face mesh (478 landmarks)
    # Ordered as: [p1, p2, p3, p4, p5, p6] for EAR formula
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    
    # Simple blink detection threshold
    BLINK_THRESHOLD = 0.21
    
    def __init__(self):
        self.eye_closed = False
    
    def compute_ear(self, eye_landmarks):
        """
        Compute Eye Aspect Ratio for a single eye.
        
        EAR formula: (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
        
        Args:
            eye_landmarks: NumPy array of shape (6, 3) containing 6 eye landmarks
        
        Returns:
            float: Eye Aspect Ratio value
        """
        p1, p2, p3, p4, p5, p6 = eye_landmarks
        
        # Vertical distances
        vertical1 = np.linalg.norm(p2 - p6)
        vertical2 = np.linalg.norm(p3 - p5)
        
        # Horizontal distance
        horizontal = np.linalg.norm(p1 - p4)
        
        # Avoid division by zero
        if horizontal < 1e-6:
            return 0.0
        
        ear = (vertical1 + vertical2) / (2.0 * horizontal)
        return ear
    
    def process(self, landmarks):
        """
        Process face landmarks to extract eye metrics.
        
        Args:
            landmarks: NumPy array of shape (N, 3) with normalized face landmarks
                      (as returned by face_mesh.py)
        
        Returns:
            dict with keys:
                - 'left_ear': float or None
                - 'right_ear': float or None
                - 'avg_ear': float or None (average of left and right)
                - 'blink': bool (True only on closed → open transition)
        """
        # Check if we have enough landmarks based on max index used
        max_index = max(max(self.LEFT_EYE_INDICES), max(self.RIGHT_EYE_INDICES))
        if landmarks is None or landmarks.shape[0] <= max_index:
            return {
                'left_ear': None,
                'right_ear': None,
                'avg_ear': None,
                'blink': False
            }
        
        # Extract eye landmarks using fixed indices
        left_eye = landmarks[self.LEFT_EYE_INDICES]
        right_eye = landmarks[self.RIGHT_EYE_INDICES]
        
        # Compute EAR for each eye
        left_ear = self.compute_ear(left_eye)
        right_ear = self.compute_ear(right_eye)
        
        # Average EAR
        avg_ear = (left_ear + right_ear) / 2.0
        
        # State-transition-based blink detection
        blink = False
        if avg_ear < self.BLINK_THRESHOLD:
            # Eyes are closed
            self.eye_closed = True
        else:
            # Eyes are open
            if self.eye_closed:
                # Transition from closed to open = blink event
                blink = True
            self.eye_closed = False
        
        return {
            'left_ear': left_ear,
            'right_ear': right_ear,
            'avg_ear': avg_ear,
            'blink': blink
        }
