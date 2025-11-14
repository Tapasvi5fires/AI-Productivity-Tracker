![Python](https://img.shields.io/badge/Python-3.x-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Mesh-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-brightgreen)
![Machine Learning](https://img.shields.io/badge/ML-RandomForest-yellow)
![Status](https://img.shields.io/badge/Status-Active-success)

# AI Productivity Tracker

🎯 **Live Focus Tracker using Computer Vision & ML**

## Table of Contents

* [About the Project](#about-the-project)
* [Features](#features)
* [Use Case & Motivation](#use-case--motivation)
* [Technology Stack](#technology-stack)
* [Architecture Overview](#architecture-overview)
* [Installation & Setup](#installation--setup)
* [Usage](#usage)
* [How It Works](#how-it-works)
* [Machine Learning Component](#machine-learning-component)
* [Data & Logging](#data--logging)
* [Dashboard](#dashboard)
* [Future Enhancements](#future-enhancements)
* [Contributing](#contributing)
* [License](#license)

---

## About the Project

**AI Productivity Tracker** is a desktop/web-hybrid application that monitors the user’s focus in real time via webcam, issues alerts when distraction is detected, logs focus data, and displays a live dashboard for insights. It uses face-mesh based eye tracking and a simple ML model to predict whether the user is “Focused” or “Distracted”.

## Features

* Real-time webcam capture and eye / face tracking using MediaPipe
* Distraction detection: alert pop-ups + sound when focus is lost or user looks away
* Continuous logging of focus state (timestamps + state) to CSV
* Embedded ML model (scikit‑learn Random Forest) to predict Focus vs Distracted
* Live dashboard built with Streamlit for visualization and monitoring
* Cross-platform support with desktop (OpenCV, pygame) and browser UI

## Use Case & Motivation

In modern work environments, distractions are ubiquitous. This tool is aimed at professionals, students, remote workers, and anyone who wants to enhance productivity by maintaining focus. By combining computer vision, alerting and analytics, the tool offers both real-time interruption prevention and longer-term insights into one’s focus patterns.

## Technology Stack

* **Language:** Python 3.x
* **Computer Vision:** OpenCV for webcam capture; MediaPipe FaceMesh for face / eye landmarks
* **Machine Learning:** scikit-learn (RandomForestClassifier)
* **UI / Dashboard:** Streamlit
* **Alerting:** Plyer for system notifications; pygame for playing alert sound
* **Data Storage:** Pandas to manage & export focus logs to CSV
* **Concurrency:** Python threading for parallel webcam capture and dashboard loop

## Architecture Overview

1. **Webcam Loop**: Continuously captures video frames, processes face/eye landmarks.
2. **Focus Detection Logic**: Determines if the user is looking away or eyes are closed beyond thresholds → mark as “distracted”.
3. **Alerting**: If distraction persists beyond a threshold counter, send a system notification + play sound.
4. **Data Logging**: Each observed state (focused/distraction) is timestamped and saved to a CSV (`focus_log.csv`).
5. **ML Component**: A random forest model predicts “Focused” vs “Distracted” based on historic features.
6. **Dashboard Loop**: Streamlit UI reads the logged data, displays charts, and shows real-time predictions.

## **Architecture Diagram**
```text
┌────────────────────────────┐
│        Webcam Input        │
└───────────────┬────────────┘
                ▼
   ┌────────────────────────┐
   │  OpenCV Frame Capture  │
   └─────────────┬──────────┘
                 ▼
┌──────────────────────────────────┐
│   MediaPipe FaceMesh Processing  │
└──────────────┬───────────────────┘
               ▼
┌──────────────────────────────────────┐
│ Eye Open Ratio + Face Orientation     │
│   (Distraction Detection Logic)       │
└───────────────┬──────────────────────┘
                ▼
   ┌────────────────────────────┐
   │ Focus State Classification │
   │    (RandomForest Model)    │
   └──────────────┬─────────────┘
                  ▼
┌────────────────────┬──────────────────────┬────────────────────────┐
│ Alerts / Sound     │ Focus Log (CSV)      │ Streamlit Dashboard    │
│ Notifications      │ Timestamp + State    │ Charts + Predictions   │
└────────────────────┴──────────────────────┴────────────────────────┘
```


## Installation & Setup

**Prerequisites**

* Python 3.8+ (recommend 3.9 or newer)
* Webcam connected

**Install dependencies**

```bash
git clone https://github.com/Tapasvi5fires/AI-Productivity-Tracker.git
cd AI-Productivity-Tracker
pip install -r requirements.txt
```

> *If `requirements.txt` doesn’t exist, install:*
> `pip install opencv-python mediapipe numpy pandas pygame joblib streamlit plyer scikit-learn`

**Alert sound file**
Place `alert.wav` in the project root (or adjust `ALERT_SOUND_PATH` accordingly).

**Model file**
`focus_predictor.pkl` will be generated automatically if not present – it uses a dummy fit in that case.

## Usage

### 1. Run the tracker

```bash
python main.py
```

This will:

* Launch a webcam window titled *Live Focus Tracker*
* Begin capturing and analyzing your focus state
* Press `q` or `ESC` in the webcam window to stop tracking

### 2. Open the dashboard

In a separate terminal window:

```bash
streamlit run main.py
```

or depending on project structure:

```bash
streamlit run dashboard.py
```

(Point it to the module containing the `dashboard()` function.)

### 3. Monitor

* The webcam window shows live video while the dashboard shows charts & predictions.
* Check `focus_log.csv` for historical data of timestamps & state.

## How It Works

* Video frames are captured using OpenCV.
* MediaPipe FaceMesh identifies facial landmarks including eye top/bottom and nose position.
* Eye openness is estimated by difference between eye-top and eye-bottom landmarks.
* If user’s nose x-coordinate moves outside [0.3, 0.7] normalized width → implies looking away → triggers alert.
* If average eye openness falls below half the computed baseline threshold → implies eyes closed/distracted → triggers.
* When focus state is “distracted” for `distraction_threshold` consecutive iterations → send alert.
* Each state (1 = focused, 0 = distracted) logged with timestamp.

## Machine Learning Component

* Model path: `focus_predictor.pkl`
* On first run, if no model file exists, the script trains a placeholder random forest on random data (this can be replaced by real labelled data).
* For inference: The rolling average of recent “Focus State” values is fed into the model to classify predicted focus level “Focused” or “Distracted”.
* In the dashboard, you will see the predicted focus level updating in near-real-time.

## Data & Logging

* Logged to: `focus_log.csv`
* Columns: `Timestamp`, `Focus State`
* The dashboard shows the rolling averaged focus state to smooth fluctuations.
* You can extend the data model to include additional features (e.g., gaze direction, blink rate, session duration) for richer analytics.

## Dashboard

* Built with Streamlit, configured with wide layout and page title “Live Focus Tracker Dashboard”.
* Components:

  * Line chart showing focus trend over time
  * Dataframe of recent focus states
  * Prediction output (ML model)
  * “Refresh” button to reload chart/data
* Users should keep the webcam window active, then view the dashboard in browser.

## Future Enhancements

Here are some ideas you may want to add:

* Use a **better labelled dataset** and train a more robust model (SVM, deep learning) for actual focus/distraction detection.
* Add **multiple features**: blink rate, head pose, gaze vector, ambient noise detection.
* Use **session tracking**: start/stop work sessions, auto-break reminders.
* Export analytics: weekly/monthly reports of focus patterns.
* Improve UI: use dashboard tabs, productivity score, goals, notifications summary.
* Use GPU/edge acceleration for faster face-mesh processing.
* Support **mobile webcam / multiple monitors**.
* Add user settings: adjust thresholds, enable/disable sound, customize alerts.

## Contributing

Thanks for your interest in contributing! Steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/YourFeatureName`
3. Commit your changes: `git commit -m "Add YourFeatureName"`
4. Push to the branch: `git push origin feature/YourFeatureName`
5. Open a Pull Request describing your changes.
   Please make sure you update this README, add tests where applicable, and maintain code style.

## License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute.

---

### For Recruiters

> **AI Productivity Tracker** demonstrates my abilities in:
>
> * Computer vision (OpenCV, MediaPipe) & real-time video analytics
> * Machine learning model integration (scikit-learn)
> * Full-stack Python development (backend tracking + front-end dashboard)
> * Data logging, visualization and UX design (Streamlit)
> * Handling concurrency (threading) and system notifications
> * Writing clean, maintainable code and delivering a meaningful productivity tool

I built this project end-to-end — from prototype to user-facing dashboard — showcasing how I can bring together AI, UX, and full-stack deployment.


