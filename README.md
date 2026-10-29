# Smart Refrigerator Temperature Anomaly Detection using IoT and DNN

> **Software-Based AI Prototype (Phase 5: Flask Web Dashboard Integration)**

## 📌 Project Overview
Temperature abnormalities inside a refrigerator can lead to food spoilage, bacterial growth, or hardware failures. This project builds an intelligent, software-based time-series anomaly detection system that monitors refrigerator temperature patterns, trains Deep Learning anomaly detection models, and serves a real-time interactive web dashboard.

> **Note:** This project is currently developed as a **software simulation prototype**. No physical hardware (ESP32, sensors, buzzers) is required at this stage.

---

## 🎯 Objectives & Multi-Phase Roadmap

- [x] **Phase 1:** Generate realistic time-series temperature data with normal and abnormal patterns, clean/inspect data, and visualize trends.
- [x] **Phase 2:** Perform data preprocessing, timestamp conversion/sorting, apply reference threshold labeling (2°C–8°C), and save processed data & visualization.
- [x] **Phase 3:** Train a baseline Deep Learning **Autoencoder** anomaly detection model on normal temperature sequences and calculate reconstruction error thresholds.
- [x] **Phase 4:** Implement and evaluate the **MOMENT Time-Series Foundation Model**, compare performance against the Autoencoder baseline, and generate comparison reports.
- [x] **Phase 5 (Completed):** Build a simple, professional, responsive Web Dashboard using **Flask** (backend) and **HTML/CSS/JS** (frontend) with real-time alert banners, dynamic Chart.js plots, and model comparison metrics.

---

## 📂 Project Folder Structure

```text
Smart_Refrigerator_Anomaly/
│
├── dataset/
│   ├── generate_dataset.py           # Script to generate realistic simulated dataset
│   ├── temperature_data.csv          # Raw 14-day time-series dataset (4,032 records)
│   └── processed_temperature_data.csv# Preprocessed dataset with 'status' column
│
├── model/                            # Saved model artifacts & foundation model scripts
│   ├── autoencoder.keras             # Trained Keras Autoencoder model
│   ├── scaler.pkl                    # Saved MinMaxScaler object
│   └── moment_model.py               # MOMENT foundation model implementation script
│
├── results/                          # Output graphs, evaluation reports & predictions
│   ├── temperature_visualization.png # Phase 1 raw temperature plot
│   ├── processed_temperature_visualization.png # Phase 2 preprocessed plot
│   ├── autoencoder_reconstruction_and_anomalies.png # Phase 3 Autoencoder plot
│   ├── anomaly_predictions.csv       # Autoencoder predictions file
│   ├── model_comparison.csv          # Model performance comparison matrix (Autoencoder vs MOMENT)
│   └── model_comparison_plot.png     # Visual bar chart comparing model metrics
│
├── app/                              # Flask Web Application Directory
│   ├── app.py                        # Backend Flask server & REST API endpoints
│   ├── templates/
│   │   └── index.html                # Frontend HTML dashboard template
│   └── static/
│       ├── style.css                 # CSS styling & responsive layouts
│       └── script.js                 # Frontend JavaScript & Chart.js integration
│
├── main.py                           # Phase 1 script: Raw data loading & EDA
├── preprocessing.py                  # Phase 2 script: Timestamp formatting, sorting, rule-based status labeling
├── autoencoder.py                    # Phase 3 script: Keras Autoencoder construction, training & thresholding
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## 💻 Installation & Setup Instructions

### Creating Virtual Environment

```bash
python -m venv venv
```

### Windows Activation

```bash
venv\Scripts\activate
```

### Installing Dependencies

```bash
pip install -r requirements.txt
```

### Running the Web Dashboard

```bash
python app/app.py
```

### Opening the Dashboard

Open your web browser and navigate to:

```text
http://127.0.0.1:5000
```

---

## 🚀 Running Data & Training Pipelines

- **Run Data Quality & Exploratory Analysis:** `python main.py`
- **Run Data Preprocessing & Threshold Labeling:** `python preprocessing.py`
- **Train Autoencoder Anomaly Model:** `python autoencoder.py`
- **Run MOMENT Foundation Model:** `python model/moment_model.py`
- **Launch Web Dashboard:** `python app/app.py`

---

## 🌐 Dashboard Features

1. **Top Summary Metrics:** Displays real-time Current Temperature, Status (`NORMAL` / `ANOMALY DETECTED`), Anomaly Score (Reconstruction MSE), and Last Updated Timestamp.
2. **Prominent Alert Banner:** Displays `⚠ Temperature anomaly detected. Please check the refrigerator.` whenever an anomaly occurs.
3. **Interactive Chart.js Plots:**
   - **Temperature Over Time Plot:** Shows continuous temperature curve, upper/lower safety bounds (5°C & 2°C), and detected anomaly markers.
   - **Autoencoder Reconstruction Error Plot:** Shows error curve against the learned threshold line.
4. **Model Comparison Table:** Displays experimental measurements for **Autoencoder** and **MOMENT** (Precision, Recall, F1-Score, Detected Anomalies, Status) without ranking one as "best".
5. **Recent Temperature Logs Table:** Displays recent sequential readings with color-coded status badges.
6. **Simulation Control & Disclaimer:** Includes a "Refresh Data" button and an "Auto Simulation" toggle with clear labeling: *"Simulated temperature data (Software Prototype - No ESP32 hardware attached)"*.

---

## 📊 Model Performance Comparison Matrix (`results/model_comparison.csv`)

| Model Architecture | Precision | Recall | F1-Score | Detected Anomalies | Evaluation Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Autoencoder Baseline** | `0.8810` | `0.2721` | `0.4157` | `126` | Evaluated & Operational |
| **MOMENT Foundation Model** | *N/A* | *N/A* | *N/A* | *N/A* | Requires MSVC C++ Redistributable on Windows |

---

## ⚠️ Important Disclaimer
- This project is a **software-based educational prototype**.
- Data displayed in the dashboard is simulated time-series telemetry; no physical ESP32 or sensor is attached at this stage.
- Performance metrics reflect empirical experimental measurements.
