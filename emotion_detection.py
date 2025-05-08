import cv2
import os
import numpy as np
import time
from tensorflow.keras.models import load_model

print("Script started")

# Load Model & Haar Cascade
model_path = os.path.abspath("models/emotion_recognition_model.h5")
print("Loading model from:", model_path)

model = load_model(model_path, compile=False)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

emotion_labels = ["Angry", "Disgust", "Fear",
                  "Happy", "Sad", "Surprise", "Neutral"]

print("Initializing camera...")
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

# FPS Calculation Variables
fps_start_time = time.time()
frame_count = 0

while True:
    frame_count += 1
    ret, frame = cap.read()
    if not ret:
        print("Failed to access camera. Exiting...")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
        face_roi = face_roi.reshape(1, 48, 48, 1) / 255.0
        emotion_label = emotion_labels[np.argmax(model.predict(face_roi))]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Calculate FPS
    if frame_count >= 30:  # Measure FPS every 30 frames
        fps = frame_count / (time.time() - fps_start_time)
        print(f"FPS: {fps:.2f}")
        fps_start_time = time.time()
        frame_count = 0

    cv2.imshow("Emotion Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Script ended")
