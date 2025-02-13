import cv2
import mediapipe as mp
import time
import numpy as np
import pandas as pd
import pygame  # For sound alerts
import os  # To check file existence
from plyer import notification
import streamlit as st
import joblib
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import winsound  # For system beep sound (Windows only)
import threading  # For running webcam + Streamlit together

# 🔹 Initialize MediaPipe FaceMesh for eye tracking
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 🔹 Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

if not cap.isOpened():
    st.error("Error: Could not open webcam.")
    st.stop()

# 🔹 Load alert sound
ALERT_SOUND_PATH = "alert.wav"

pygame.mixer.init()
if os.path.exists(ALERT_SOUND_PATH):
    alert_sound = pygame.mixer.Sound(ALERT_SOUND_PATH)
else:
    alert_sound = None  # Fallback option if file is missing

# 🔹 Eye tracking variables
focus_counter = 0
distraction_threshold = 10  # Frames before triggering an alert
last_alert_time = 0
focus_data = []
stop_tracking = False  # To properly exit tracking

# 🔹 Alert Function
def send_alert(message):
    """Sends a notification and plays a sound alert."""
    global last_alert_time
    if time.time() - last_alert_time > 10:  # Prevent spam alerts
        notification.notify(title="👀 Focus Alert!", message=message, timeout=5)
        if alert_sound:
            alert_sound.play()
        else:
            winsound.Beep(1000, 500)  # Fallback: System beep sound
        last_alert_time = time.time()

# 🔹 Save Focus Data
def save_focus_data(state):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    focus_data.append([timestamp, state])
    df = pd.DataFrame(focus_data, columns=["Timestamp", "Focus State"])
    df.to_csv("focus_log.csv", index=False)

# 🔹 Webcam Tracking Loop (Runs in a Thread)
def webcam_tracking():
    global focus_counter, stop_tracking

    while cap.isOpened():
        if stop_tracking:
            break  # Stop tracking when user exits

        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot capture frame. Exiting.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        focus_state = 1  # Assume focused initially

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_eye_top = face_landmarks.landmark[159].y
                left_eye_bottom = face_landmarks.landmark[145].y
                right_eye_top = face_landmarks.landmark[386].y
                right_eye_bottom = face_landmarks.landmark[374].y

                left_eye_open = abs(left_eye_bottom - left_eye_top)
                right_eye_open = abs(right_eye_bottom - right_eye_top)
                avg_eye_open = (left_eye_open + right_eye_open) / 2
                adjusted_threshold = max(0.015, avg_eye_open * 0.5)

                # Head Movement Detection
                nose_x = face_landmarks.landmark[1].x
                if nose_x < 0.3 or nose_x > 0.7:  # If looking too left or right
                    send_alert("You're looking away! Stay focused.")
                    focus_state = 0  # Mark as distracted

                # Determine Focus State
                if avg_eye_open < adjusted_threshold:
                    focus_state = 0  # Mark as distracted

        # Save Focus Data
        save_focus_data(focus_state)

        if focus_state == 0:
            focus_counter += 1
            if focus_counter >= distraction_threshold:
                send_alert("You are distracted! Get back to work.")
                focus_counter = 0
        else:
            focus_counter = 0

        # Show Webcam Feed
        cv2.imshow("Live Focus Tracker", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or 'ESC' to exit
            stop_tracking = True
            break

    cap.release()
    cv2.destroyAllWindows()

# 🔹 Streamlit Dashboard (Fixed)
def dashboard():
    st.set_page_config(page_title="Focus Tracker", layout="wide")
    st.title("🎯 Live Focus Tracker Dashboard")
    st.write("🔵 **Webcam is running in the background. Press `q` or `ESC` to stop.**")

    focus_chart = st.empty()
    focus_table = st.empty()
    
    refresh_button = st.button("🔄 Refresh Data")

    # Ensure there's enough data before computing rolling mean
    if len(focus_data) > 0:
        df = pd.DataFrame(focus_data, columns=["Timestamp", "Focus State"])
        if len(df) > 5:  # Avoid errors when dataset is too small
            df["Focus State"] = df["Focus State"].rolling(5, min_periods=1).mean()
        focus_chart.line_chart(df["Focus State"])
        focus_table.dataframe(df.tail(10))

    # Refreshing mechanism
    if refresh_button:
        st.rerun()  # Corrected from `st.experimental_rerun()`

# 🔹 Run Webcam & Streamlit Together
threading.Thread(target=webcam_tracking, daemon=True).start()
dashboard()
