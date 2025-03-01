import cv2
import cvzone
import pyttsx3
import numpy as np
import streamlit as st
import threading
import time
from ultralytics import YOLO
from PIL import Image
import tempfile

# ========== UI CUSTOMIZATION ==========
st.set_page_config(page_title="YOLOv10 Object Detector", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        .big-title { font-size: 36px; font-weight: bold; color: #FF4B4B; text-align: center; }
        .sub-header { font-size: 20px; font-weight: bold; color: #333333; }
        .stButton>button { background-color: #ff5733; color: white; font-size: 18px; border-radius: 10px; }
        .stFileUploader { font-size: 16px; }
    </style>
""", unsafe_allow_html=True)


# ========== TEXT-TO-SPEECH ==========
def speak(text):
    def tts():
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)  # Adjust speed
        engine.setProperty('volume', 1.0)  # Full volume
        engine.say(text)
        engine.runAndWait()

    threading.Thread(target=tts, daemon=True).start()


# Load YOLO model
objectModel = YOLO("yolov10n.pt")

# Sidebar for options
st.sidebar.title("🔧 Settings")
option = st.sidebar.radio("Choose Detection Mode:", ["Image", "Video", "Webcam"])
st.sidebar.write("👆 Select how you want to perform object detection.")

# ========== IMAGE DETECTION ==========
if option == "Image":
    st.markdown('<p class="big-title">🖼️ Image Object Detection</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_cv = np.array(image)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

        objects = objectModel(image_cv)
        detected_objects = []

        for object in objects:
            if object.boxes is None or len(object.boxes) == 0:
                continue

            for box in object.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                classNum = int(box.cls[0])
                className = objectModel.names[classNum]
                confidence = box.conf[0] * 100

                cv2.rectangle(image_cv, (x1, y1), (x2, y2), (50, 50, 255), 2)
                cvzone.putTextRect(image_cv, f"{className} | {confidence:.1f}%",
                                   [x1 + 8, y1 - 12], scale=1, thickness=1)

                detected_objects.append(f"{className} ({confidence:.1f}%)")

        image_result = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        st.image(image_result, caption="📌 Detection Results", use_column_width=True)

        if detected_objects:
            detection_text = ", ".join(detected_objects)
            st.success(f"✅ Detected Objects: {detection_text}")
            speak(detection_text)

# ========== VIDEO & WEBCAM DETECTION ==========
elif option in ["Video", "Webcam"]:
    st.markdown(f'<p class="big-title">🎥 {option} Object Detection</p>', unsafe_allow_html=True)

    if option == "Video":
        uploaded_video = st.file_uploader("📤 Upload a Video", type=["mp4", "avi", "mov"])
        if uploaded_video is None:
            st.warning("Please upload a video file.")
            st.stop()
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
    else:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_placeholder = st.empty()
    status_text = st.empty()
    last_spoken_time = 0
    last_detected_objects = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        objects = objectModel(frame)
        detected_objects = set()

        for object in objects:
            if object.boxes is None or len(object.boxes) == 0:
                continue

            for box in object.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                classNum = int(box.cls[0])
                className = objectModel.names[classNum]
                confidence = box.conf[0] * 100

                cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 50, 255), 2)
                cvzone.putTextRect(frame, f"{className} | {confidence:.1f}%",
                                   [x1 + 8, y1 - 12], scale=1, thickness=1)

                detected_objects.add(className)

        frame_placeholder.image(frame, channels="BGR")

        if detected_objects:
            status_text.markdown(f"✅ **Detected Objects:** {', '.join(detected_objects)}")

        # Speak every 2 seconds if new objects are found
        current_time = time.time()
        if (current_time - last_spoken_time > 2) and (detected_objects != last_detected_objects):
            speak(", ".join(detected_objects))
            last_spoken_time = current_time
            last_detected_objects = detected_objects.copy()

    cap.release()
    st.success("🎬 Processing Complete!")
