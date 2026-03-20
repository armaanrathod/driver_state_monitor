# Driver State Monitor


A real-time computer vision system for detecting facial landmarks as a foundation for driver drowsiness detection.

---

## 🚀 Overview

This project captures live webcam input and processes it using MediaPipe Face Mesh to extract detailed facial landmarks (478 points). These landmarks are used as a base for building drowsiness detection features such as eye closure and fatigue estimation.

---

## ⚙️ System Architecture

The system is modular and consists of:

### 1. Camera Module (`camera.py`)
- Captures real-time video stream from webcam
- Handles frame acquisition for processing pipeline

### 2. Face Mesh Module (`face_mesh.py`)
- Detects face using MediaPipe
- Extracts 478 facial landmarks
- Provides precise coordinates for facial feature analysis

---

## 🧠 Key Features

- Real-time face mesh detection
- High-resolution facial landmark tracking (478 points)
- Modular pipeline for extending into drowsiness detection
- Supports real-time processing (~30 FPS)

---

## 📊 Current Progress

- ✅ Camera pipeline implemented  
- ✅ Face mesh detection working  
- 🔄 Working on eye-aspect ratio and drowsiness logic  

---

## 🛠️ Tech Stack

- Python  
- OpenCV  
- MediaPipe  
- NumPy  

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python main.py
