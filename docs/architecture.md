# System Architecture Documentation

## 1. Current Software Prototype Architecture

```text
               ┌──────────────────────────────────────────┐
               │    Simulated Temperature Time-Series     │
               │        (dataset/temperature_data.csv)    │
               └────────────────────┬─────────────────────┘
                                    │
                                    ▼
               ┌──────────────────────────────────────────┐
               │             Data Preprocessing           │
               │   (Timestamp Conversion & Rule Labels)   │
               └────────────────────┬─────────────────────┘
                                    │
                                    ▼
               ┌──────────────────────────────────────────┐
               │       Time-Series Window Preparation     │
               │    (10-Step & 512-Step Sliding Windows)  │
               └───────────┬──────────────────┬───────────┘
                           │                  │
                           ▼                  ▼
              ┌───────────────────┐  ┌───────────────────┐
              │    Autoencoder    │  │   MOMENT Model    │
              │  Baseline Network │  │ Foundation Model  │
              └─────────┬─────────┘  └─────────┬─────────┘
                        │                      │
                        ▼                      ▼
              ┌───────────────────┐  ┌───────────────────┐
              │ Reconstruction    │  │ Anomaly Score     │
              │ Error Threshold   │  │ Calculation       │
              └─────────┬─────────┘  └─────────┬─────────┘
                        └──────────┬──────────┘
                                   │
                                   ▼
              ┌──────────────────────────────────────────┐
              │         Model Comparison Engine          │
              │   (results/final_model_comparison.csv)   │
              └────────────────────┬─────────────────────┘
                                   │
                                   ▼
              ┌──────────────────────────────────────────┐
              │           Flask Web Backend API          │
              │         (app/app.py - Port 5000)         │
              └────────────────────┬─────────────────────┘
                                   │
                                   ▼
              ┌──────────────────────────────────────────┐
              │          Interactive Web Dashboard       │
              │   (Summary Cards, Chart.js & Alert Banner)│
              └────────────────────┬─────────────────────┘
                                   │
                                   ▼
              ┌──────────────────────────────────────────┐
              │        Real-Time Anomaly Alert Banner    │
              └──────────────────────────────────────────┘
```

---

## 2. Future IoT Hardware Integration Architecture

> **Note:** The following architecture describes planned future hardware integration. The current system is a software simulation prototype.

```text
               ┌──────────────────────────────────────────┐
               │        DS18B20 Temperature Sensor        │
               └────────────────────┬─────────────────────┘
                                    │ (Digital Telemetry)
                                    ▼
               ┌──────────────────────────────────────────┐
               │           ESP32 Microcontroller          │
               │        (WiFi / MQTT Telemetry Client)    │
               └────────────────────┬─────────────────────┘
                                    │ (HTTP POST / MQTT Protocol)
                                    ▼
               ┌──────────────────────────────────────────┐
               │              Local Network               │
               └────────────────────┬─────────────────────┘
                                    │
                                    ▼
               ┌──────────────────────────────────────────┐
               │       Flask Web Application & API        │
               └────────────────────┬─────────────────────┘
                                    │
                                    ▼
               ┌──────────────────────────────────────────┐
               │          AI Inference Engine             │
               │   (Autoencoder / MOMENT Anomaly Score)   │
               └───────────┬──────────────────┬───────────┘
                           │                  │
                           ▼                  ▼
              ┌───────────────────┐  ┌───────────────────┐
              │  Web Dashboard UI │  │ Physical Buzzer   │
              │ (Visual Alerts)   │  │ (Hardware Alarm)  │
              └───────────────────┘  └───────────────────┘
```

---

## 3. Component Details
- **Temperature Dataset:** 14-day simulated temperature readings at 5-minute sampling intervals (4,032 records).
- **Data Preprocessing:** Cleans missing values, formats timestamps as `datetime64[ns]`, sorts chronologically, and applies a 2°C–8°C reference safety boundary.
- **Autoencoder Baseline:** Dense Deep Neural Network (10 → 16 → 8 → 16 → 10) trained on normal data to reconstruct temperature windows and detect anomalies via reconstruction error thresholds.
- **MOMENT Foundation Model:** Pre-trained time-series foundation model evaluated for zero-shot time-series anomaly detection.
- **Flask Backend & Web Dashboard:** Serves real-time interactive charts, summary metrics, model comparison reports, and alert notifications.
