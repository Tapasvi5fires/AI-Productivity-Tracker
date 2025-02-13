import cv2
import mediapipe as mp
import time
import numpy as np
import pandas as pd
import pygame
import os
import joblib
import streamlit as st
from plyer import notification
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import threading

# 🔹 Initialize MediaPipe FaceMesh for eye tracking
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 🔹 Load alert sound
ALERT_SOUND_PATH = "alert.wav"
pygame.mixer.init()
alert_sound = pygame.mixer.Sound(ALERT_SOUND_PATH) if os.path.exists(ALERT_SOUND_PATH) else None

# 🔹 Eye tracking variables
focus_counter = 0
distraction_threshold = 10
last_alert_time = 0
focus_data = []
stop_tracking = False
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# 🔹 Alert Function
def send_alert(message):
    """Sends a notification and plays a sound alert."""
    global last_alert_time
    if time.time() - last_alert_time > 10:
        notification.notify(title="👀 Focus Alert!", message=message, timeout=5)
        if alert_sound:
            alert_sound.play()
        last_alert_time = time.time()

# 🔹 Save Focus Data
def save_focus_data(state):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    focus_data.append([timestamp, state])
    df = pd.DataFrame(focus_data, columns=["Timestamp", "Focus State"])
    df.to_csv("focus_log.csv", index=False)

# 🔹 Load or Train ML Model for Predictions
MODEL_PATH = "focus_predictor.pkl"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = RandomForestClassifier(n_estimators=100)
    model.fit(np.random.rand(100, 1), np.random.randint(2, size=100))
    joblib.dump(model, MODEL_PATH)

# 🔹 Webcam Tracking Loop
def webcam_tracking():
    global focus_counter, stop_tracking

    while cap.isOpened():
        if stop_tracking:
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        focus_state = 1  # Assume focused

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

                nose_x = face_landmarks.landmark[1].x
                if nose_x < 0.3 or nose_x > 0.7:
                    send_alert("You're looking away! Stay focused.")
                    focus_state = 0

                if avg_eye_open < adjusted_threshold:
                    focus_state = 0

        save_focus_data(focus_state)

        if focus_state == 0:
            focus_counter += 1
            if focus_counter >= distraction_threshold:
                send_alert("You are distracted! Get back to work.")
                focus_counter = 0
        else:
            focus_counter = 0

        cv2.imshow("Live Focus Tracker", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            stop_tracking = True
            break

    cap.release()
    cv2.destroyAllWindows()

# 🔹 Streamlit Dashboard
def dashboard():
    st.set_page_config(page_title="Focus Tracker", layout="wide")
    st.title("🎯 Live Focus Tracker Dashboard")
    st.write("🔵 **Press `q` or `ESC` in the webcam window to stop tracking.**")

    focus_chart = st.line_chart([])
    focus_table = st.empty()
    prediction_output = st.empty()

    refresh_button = st.button("🔄 Refresh Data")

    while not stop_tracking:
        if len(focus_data) > 0:
            df = pd.DataFrame(focus_data, columns=["Timestamp", "Focus State"])
            if len(df) > 5:
                df["Focus State"] = df["Focus State"].rolling(5, min_periods=1).mean()

            focus_chart.line_chart(df["Focus State"])
            focus_table.dataframe(df.tail(10))

            # 🔹 ML Prediction
            latest_focus_state = np.array(df["Focus State"].tail(1)).reshape(-1, 1)
            predicted_focus = model.predict(latest_focus_state)
            prediction_output.write(f"📊 **Predicted Focus Level:** {'Focused' if predicted_focus[0] == 1 else 'Distracted'}")

        if refresh_button:
            st.rerun()

        time.sleep(2)

# 🔹 Run Everything
if __name__ == "__main__":
    threading.Thread(target=webcam_tracking, daemon=True).start()
    dashboard()
