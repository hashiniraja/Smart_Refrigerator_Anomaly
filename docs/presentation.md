# Project Presentation Slides (PPT Content)

---

## Slide 1: Project Title
### Smart Refrigerator Temperature Anomaly Detection using IoT and DNN
- **Domain:** Artificial Intelligence / Internet of Things (Software Prototype)
- **Presenter:** Computer Science & Engineering Project Team
- **Key Concepts:** Time-Series Processing, Autoencoders, MOMENT Foundation Model, Flask Dashboard

---

## Slide 2: Problem Statement
- **Food Spoilage Risk:** Refrigerated items require strict 2°C–5°C storage.
- **Undetected Failures:** Door openings, gas leaks, and faulty relays cause undetected temperature deviations.
- **Current Limitations:** Basic thermostats only sound simple alarms after severe failure; they lack trend analysis or predictive alerts.

---

## Slide 3: Objectives
- Monitor 14 days of simulated refrigerator temperature time-series telemetry.
- Define normal vs. abnormal operating boundaries.
- Build a Deep Learning **Autoencoder** baseline model.
- Investigate the **MOMENT** foundation model for time-series AI.
- Serve a real-time responsive Flask Web Dashboard with alert banners.

---

## Slide 4: Proposed System
- **Software Prototype Pipeline:** Data Generator $\rightarrow$ Preprocessing $\rightarrow$ Autoencoder $\rightarrow$ MOMENT $\rightarrow$ Web Dashboard.
- **Unsupervised Anomaly Detection:** Train Autoencoder strictly on normal data to learn compressor cycle patterns.
- **Automated Alerts:** Trigger web alerts whenever reconstruction error exceeds the statistical threshold ($\mu + 3\sigma$).

---

## Slide 5: System Architecture
```text
Temperature Dataset ──> Preprocessing ──> Sliding Windows (10 steps)
                                                │
                       ┌────────────────────────┴────────────────────────┐
                       ▼                                                 ▼
             Autoencoder Baseline                              MOMENT Foundation Model
             Reconstruction Error                             Zero-Shot Anomaly Score
                       │                                                 │
                       └────────────────────────┬────────────────────────┘
                                                ▼
                                    Flask Backend & REST API
                                                │
                                                ▼
                                    Interactive Web Dashboard
```

---

## Slide 6: Dataset and Preprocessing
- **Dataset Size:** 14 Days @ 5-min interval = 4,032 records.
- **Normal Range:** 2.0°C – 5.0°C (Compressor sinusoidal cycles).
- **Abnormal Failure Scenarios:** Sudden Spike, Gradual Leak Increase, Sub-Zero Freezing Drop, Rapid Relay Chatter.
- **Preprocessing:** Missing value check (0 missing), timestamp conversion (`datetime64`), and sorting.

---

## Slide 7: Autoencoder + MOMENT Methodology
- **Autoencoder Architecture:** Dense(10) $\rightarrow$ Dense(16) $\rightarrow$ Dense(8 Bottleneck) $\rightarrow$ Dense(16) $\rightarrow$ Dense(10).
- **Training Strategy:** 40 epochs on NORMAL sequences only.
- **Learned Anomaly Threshold:** `0.004231` (MSE).
- **MOMENT Foundation Model:** Pre-trained Transformer evaluated for time-series anomaly detection.

---

## Slide 8: Web Dashboard
- **Backend:** Python Flask (`app/app.py`).
- **Frontend:** HTML5, CSS3, JavaScript, Chart.js.
- **Key Features:**
  - Real-time alert banner: `⚠ Temperature anomaly detected.`
  - Summary cards (Current Temp, Status, Anomaly Score, Last Updated).
  - Chart.js plots for Temperature Line and Reconstruction Error Curve.
  - Empirical Model Comparison table & Recent Reading logs.

---

## Slide 9: Results & Model Comparison

| Model | Precision | Recall | F1-Score | Detected Anomalies |
| :--- | :---: | :---: | :---: | :---: |
| **Autoencoder Baseline** | `0.8808` | `0.3260` | `0.4758` | `151` |
| **MOMENT Foundation Model** | *N/A* | *N/A* | *N/A* | *N/A (MSVC Dependency)* |

- **Observation:** Autoencoder achieves high precision (0.8808) on severe spikes and sub-zero drops.

---

## Slide 10: Conclusion & Future Enhancement
- **Conclusion:** A software-based AI system successfully detects temperature anomalies and provides visual alerts.
- **Future Enhancements:**
  1. Integrate ESP32 microcontroller with DS18B20 hardware sensor and active buzzer.
  2. Implement MQTT real-time streaming telemetry.
  3. Deploy Flask backend to cloud platforms (AWS/GCP) with SMS alerts.
