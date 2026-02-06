# Camera Module

## What It Does

`camera.py` connects to your device's camera and captures video. It works on Windows, Mac, Linux, and iPhone/iPad.

## Key Features

- Works across different devices and operating systems
- Continuously captures video frames
- Shows how fast frames are being captured (frames per second)
- Automatically cleans up when done

## How to Use

### Simple Example

```python
from camera import CrossPlatformCamera

# Connect to your camera (0 = main camera)
camera = CrossPlatformCamera(camera_index=0)

# Get video frames one by one
for frame in camera.capture_frames():
    # Do something with each frame
    # (frame is just an image)
    pass
```

### Test It

Run this command to see if it works:

```bash
python camera.py
```

Press 'q' to quit.

## What You Need

**For computers (Windows/Mac/Linux):**
- Python (version 3.7 or newer)
- OpenCV library
- NumPy library

**For iPhone/iPad:**
- Kivy framework
- Additional iOS setup

## Main Functions

**`start()`** - Turn on the camera

**`read_frame()`** - Grab one image from the camera

**`capture_frames()`** - Keep getting images continuously

**`release()`** - Turn off the camera and free up resources

## Settings

- Default resolution: 640×480 pixels
- Default speed: 30 frames per second
- Camera index 0 = your main camera
- Camera index 1, 2, etc. = other cameras (if available)

## Notes

- Each frame is a regular image you can process or save
- The camera turns off automatically when your program ends
- If something goes wrong, error messages will help you troubleshoot

---

# Face Mesh Module

## What It Does

`face_mesh.py` finds faces in images and marks key points on them (like eyes, nose, mouth). It uses Google's MediaPipe technology to detect 478 facial landmarks.

## Purpose

This module is a "perception" tool - it only detects and returns face landmarks. It does NOT:
- Calculate if eyes are closed
- Figure out where someone is looking
- Track changes over time
- Make any decisions or judgments
- Draw on frames
- Store frame history
- Compute derived features like Eye Aspect Ratio or head pose

## How to Use

```python
from driver_awareness.perception.face_mesh import FaceMeshDetector
import cv2

# Set up the detector once
detector = FaceMeshDetector()

# Get a frame from somewhere (like camera.py)
frame = cv2.imread('photo.jpg')

# Process the frame
result = detector.process(frame)

if result:
    # Face was found!
    landmarks = result['landmarks']  # All face points (normalized 0-1)
    pixel_coords = result['pixel_coords']  # Face points in pixels
    iris_left = result['iris_left']  # Left iris points
    iris_right = result['iris_right']  # Right iris points
else:
    # No face found
    pass

# Clean up when done
detector.close()
```

## What You Get Back

When a face is detected, you get a dictionary with:

- **landmarks**: All face mesh points with normalized coordinates (values 0-1)
- **pixel_coords**: Same points but in actual pixel positions
- **iris_left**: 5 points marking the left iris (or None if not available)
- **iris_right**: 5 points marking the right iris (or None if not available)

If no face is found, you get `None`.

## Settings

You can customize when creating the detector:

- **max_num_faces** (default: 1): How many faces to look for
- **refine_landmarks** (default: True): Include detailed iris points (landmarks 468-477)
- **min_detection_confidence** (default: 0.5): How sure it must be to detect a face (0.0 to 1.0)
- **min_tracking_confidence** (default: 0.5): How sure it must be when tracking (0.0 to 1.0)

## Input Requirements

- **Frame format**: BGR image (numpy array from OpenCV)
- **Any resolution**: The detector handles images of any size
- **Single frame**: Process one frame at a time

## Iris Landmarks

- Total facial landmarks: 478 points
- Iris points are indices 468-477 in the full landmark set
- Left iris: indices 468-472 (5 points)
- Right iris: indices 473-477 (5 points)
- Only available when `refine_landmarks=True`

## Coordinate Systems

The detector returns landmarks in two formats:

1. **Normalized coordinates** (0-1 range):
   - `x`: 0 = left edge, 1 = right edge
   - `y`: 0 = top edge, 1 = bottom edge
   - `z`: depth information (relative to face center)

2. **Pixel coordinates**: Actual positions in the image
   - Calculated by multiplying normalized coords by image dimensions
   - Ready to use for drawing or further processing

## Performance Notes

- The detector is set up once and reused for speed
- Optimized for real-time video processing
- Automatically converts image colors (BGR to RGB) for MediaPipe
- Returns data as NumPy arrays for easy processing

## What You Need

- Python 3.7+
- OpenCV (cv2)
- MediaPipe
- NumPy
