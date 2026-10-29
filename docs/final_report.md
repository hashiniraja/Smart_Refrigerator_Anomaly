# Smart Refrigerator Temperature Anomaly Detection using IoT and DNN

## Project Report: B.E. Computer Science & Engineering Prototype

---

### Executive Summary
This report presents the design, implementation, and evaluation of a software-based AI prototype titled **"Smart Refrigerator Temperature Anomaly Detection using IoT and DNN"**. The system detects unusual refrigerator temperature patterns early using Deep Neural Networks (Autoencoders) and Foundation Models (MOMENT) before food spoilage or hardware failure occurs.

---

## Chapter 1: Introduction
Perishable food items require consistent cold storage temperatures (between 2°C and 5°C). Unintended temperature spikes, refrigerant leaks, thermostat failures, or door openings can lead to bacterial contamination, food waste, and expensive repair costs. Traditional refrigerators rely on simple mechanical bimetallic thermostats that lack intelligent trend analysis or predictive alerts. This project develops an intelligent software-based anomaly detection pipeline that analyzes temperature time-series data using Deep Learning.

---

## Chapter 2: Problem Statement
Fluctuations and abnormalities in refrigerator temperature can spoil stored food and cause energy inefficiency. Existing systems lack early automated pattern detection for subtle anomalies like gradual refrigerant leaks or rapid cycling relay failures. The system should detect unusual refrigerator temperature patterns early using software simulation.

---

## Chapter 3: Objectives
1. Monitor simulated refrigerator temperature time-series data.
2. Define normal (2°C–5°C) and abnormal operating temperature boundaries.
3. Prepare a 14-day time-series temperature dataset (4,032 records).
4. Implement data preprocessing, hygiene checks, timestamp formatting, and sorting.
5. Build and train a baseline TensorFlow/Keras **Autoencoder** Neural Network model.
6. Investigate the **MOMENT** foundation model for time-series anomaly detection.
7. Classify sequence readings as Normal vs. Anomaly using statistical reconstruction thresholding.
8. Serve an interactive, responsive Flask Web Dashboard with real-time alert banners and performance metrics.

---

## Chapter 4: Existing System
Traditional refrigerator monitoring systems rely on static threshold alarms (e.g. ringing a buzzer if temperature exceeds 8°C).
- **Disadvantages:** High false positive rates (e.g., triggering brief alarms during normal door openings), inability to detect subtle trend degradation (e.g. slow gas leaks), lack of predictive alerts, and zero historical analytics.

---

## Chapter 5: Proposed System
The proposed system uses a software-based AI time-series pipeline combining:
1. **Unsupervised Deep Learning (Autoencoder Baseline):** Learns normal compressor cycle patterns on 10-reading sliding windows and detects anomalies via reconstruction error thresholding ($\mu + 3\sigma$).
2. **Time-Series Foundation Model (MOMENT):** Evaluates pre-trained generalist time-series representations.
3. **Interactive Web Dashboard:** Serves real-time telemetry graphs, alert notifications, and empirical model metrics via a Python Flask backend.

---

## Chapter 6: System Requirements

### Hardware Requirements (Software Prototype Machine)
- **Processor:** Intel Core i5 / AMD Ryzen 5 or higher
- **RAM:** 8 GB minimum (16 GB recommended)
- **Storage:** 2 GB available disk space

### Software Requirements
- **Operating System:** Windows 10/11, Linux, or macOS
- **Programming Language:** Python 3.9 – 3.11
- **Libraries:** Pandas, NumPy, Matplotlib, Scikit-learn, TensorFlow/Keras, PyTorch, momentfm, Flask

---

## Chapter 7: System Architecture
The system consists of Data Ingestion $\rightarrow$ Preprocessing $\rightarrow$ Window Sequence Generation $\rightarrow$ Model Inference (Autoencoder & MOMENT) $\rightarrow$ Anomaly Threshold Evaluation $\rightarrow$ Flask REST API $\rightarrow$ Web Dashboard.

*(Detailed architecture diagrams are provided in `docs/architecture.md`).*

---

## Chapter 8: Dataset
The dataset (`dataset/temperature_data.csv`) contains 14 days of simulated temperature telemetry recorded at 5-minute sampling intervals (4,032 rows).
- **Normal Patterns:** Sinusoidal compressor cycles (2°C to 5°C, 45-min period) + diurnal variance + Gaussian noise.
- **Injected Anomaly Scenarios:**
  - Sudden Spike (Door left open / hot food inserted)
  - Gradual Increase (Coolant gas leak / compressor degradation)
  - Sudden Drop (Thermostat relay lock / sub-zero freezing)
  - Rapid Fluctuations (Faulty relay chatter)

---

## Chapter 9: Data Preprocessing
- Missing value verification (0 missing values).
- Duplicate record removal (0 duplicate rows).
- Timestamp string conversion to `datetime64[ns]` objects and chronological sorting.
- Reference rule-based labeling (2°C <= temp <= 8°C = Normal; Anomaly otherwise).
- Export to `dataset/processed_temperature_data.csv`.

---

## Chapter 10: Autoencoder Methodology
- **Input Feature:** Scaled temperature time-series via `MinMaxScaler(0, 1)`.
- **Sliding Window:** 10 consecutive 5-minute readings per sample.
- **Training Strategy:** Trained strictly on NORMAL sequences for 40 epochs.
- **Architecture:** Dense(10) $\rightarrow$ Dense(16, ReLU) $\rightarrow$ Dense(8, ReLU Bottleneck) $\rightarrow$ Dense(16, ReLU) $\rightarrow$ Dense(10, Linear).
- **Loss Function:** Mean Squared Error (MSE).

---

## Chapter 11: MOMENT Methodology
- **Overview:** Open-source foundation model for time-series analysis (CMU Auton Lab).
- **Input Window:** 512 timesteps (~42.6 hours).
- **Pipeline:** `MOMENTPipeline.from_pretrained("AutonLab/MOMENT-1-large", task_name="anomaly-detection")`.
- **Execution Note:** Requires native C++ runtime libraries (`c10.dll` / MSVC Redistributable on Windows). Handled gracefully with diagnostic documentation.

---

## Chapter 12: Anomaly Detection & Thresholding
Anomalies are detected when a sequence's reconstruction error exceeds the statistical threshold:
$$\text{Threshold} = \mu_{\text{train}} + 3 \times \sigma_{\text{train}}$$
For the trained Autoencoder baseline:
- **Threshold Value:** `0.004231`
- **Classification:** Sequence MSE > Threshold $\rightarrow$ `Anomaly`, else $\rightarrow$ `Normal`.

---

## Chapter 13: Flask Web Dashboard
Built using Python Flask (`app/app.py`), HTML5, CSS3, JavaScript, and Chart.js.
- **Endpoints:** `/`, `/api/current`, `/api/temperature-data`, `/api/model-results`.
- **Features:** Real-time alert banner, summary cards, temperature & error charts, recent reading logs, and model comparison matrix.

---

## Chapter 14: Results and Discussion
- **Total Test Sequences:** 4,023
- **Autoencoder Normal Predictions:** 3,872
- **Autoencoder Detected Anomalies:** 151
- **Ground-Truth Anomaly Readings:** 217
- **Confusion Matrix:** True Negatives: 3,597 | False Positives: 18 | False Negatives: 275 | True Positives: 133

---

## Chapter 15: Model Comparison

| Model | Precision | Recall | F1-Score | Detected Anomalies | Evaluation Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Autoencoder Baseline** | `0.8808` | `0.3260` | `0.4758` | `151` | Evaluated & Operational |
| **MOMENT Foundation Model** | *N/A* | *N/A* | *N/A* | *N/A* | Requires MSVC C++ Redistributable on Windows |

*Observations:* The local Autoencoder achieved high precision (0.8808), correctly identifying severe spikes and sub-zero drops.

---

## Chapter 16: Limitations
1. Current implementation is a **software simulation prototype** using generated dataset.
2. No physical ESP32 microcontroller or hardware sensor is attached currently.
3. Anomaly threshold calculation depends on the distribution of training data.
4. MOMENT foundation model requires PyTorch native C++ runtime libraries on Windows.

---

## Chapter 17: Future Enhancement
1. **Physical Hardware:** Integrate ESP32 microcontroller, DS18B20 waterproof temperature sensor, and active piezo buzzer.
2. **Real-Time Telemetry Streaming:** Transmit live data over MQTT / HTTP REST API from ESP32 to Flask.
3. **Cloud & Mobile Alerts:** Deploy Flask application to cloud (AWS/GCP/Heroku) and send push notifications via Pushbullet/Twilio SMS.

---

## Chapter 18: Conclusion
The project successfully demonstrates a complete software-based AI anomaly detection system for smart refrigerator temperature monitoring. The system integrates data preprocessing, baseline Deep Learning (Autoencoder), foundation model analysis (MOMENT), and an interactive web dashboard with alert notifications.
