"Smart Assistive Technology: Real-Time Object Detection and Speech Output for the Visually Impaired"
Objective: Enable visually impaired individuals to perceive their environment through real-time object detection and audio feedback.

Project Description:
This project is a real-time object detection system utilizing the YOLOv10 model, integrated with a Streamlit web interface. It enables users to upload images, process videos, or use a webcam to detect objects dynamically. The system provides both visual and auditory feedback through text-to-speech, making it accessible for various applications, including accessibility tools and security systems.

Features:
User Interface: Designed using Streamlit for an interactive and user-friendly experience.
Multiple Detection Modes: Supports image, video, and live webcam object detection.
YOLOv10 Integration: Uses the YOLOv10 model for high-accuracy object detection.
Real-time Processing: Processes and displays object detection results with bounding boxes and confidence levels.
Text-to-Speech (TTS) Support: Converts detected object names into speech using pyttsx3.
Custom Styling: Enhances the UI with CSS for a modern look and feel.

Technology Stack:
Programming Language: Python
Libraries Used:
cv2 (OpenCV) for image processing
cv zone for enhanced display
pyttsx3 for text-to-speech
numpy for array operations
streamlit for web-based UI
ultralytics (YOLOv10) for object detection
PIL (Pillow) for image processing
tempfile for handling temporary video files
threading for asynchronous speech processing

How It Works:
The user selects a detection mode (Image, Video, or Webcam) from the Streamlite sidebar.
If Image mode is selected, the user uploads an image, which is then processed using the YOLOv10 model.
If Video mode is selected, a video file is uploaded, and each frame is analyzed to detect objects.
If Webcam mode is selected, the application accesses the webcam feed and detects objects in real time.
Detected objects are highlighted with bounding boxes and labels showing object names and confidence percentages.
The names of detected objects are spoken aloud using the TTS engine.
In video and webcam mode, the system speaks detected objects every 2 seconds to prevent excessive speech output.

Setup & Installation:
Clone the repository.
Install required dependencies.
Run the Streamlit application.
