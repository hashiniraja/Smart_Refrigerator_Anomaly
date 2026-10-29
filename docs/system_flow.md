# System Flowchart Documentation

## 1. System Execution Flowchart

```text
                           [ Start ]
                               │
                               ▼
               [ Collect / Load Temperature Data ]
                               │
                               ▼
                     [ Preprocess Data ]
          (Timestamp Conversion & Data Hygiene Check)
                               │
                               ▼
               [ Create Time-Series Windows ]
            (10-Step Sliding Window Sequences)
                               │
                               ▼
             [ Execute Autoencoder Inference ]
            (Calculate Reconstruction Error MSE)
                               │
                               ▼
               [ Execute MOMENT Evaluation ]
             (Pre-trained Foundation Model Score)
                               │
                               ▼
            [ Compare Error with Threshold Line ]
                               │
                     ┌─────────┴─────────┐
                     │ Is Score > Limit? │
                     └────┬─────────┬────┘
                          │         │
                 YES      │         │     NO
            ┌─────────────┘         └─────────────┐
            ▼                                     ▼
 [ Trigger Anomaly Status ]             [ Set Normal Status ]
[ Display Red Alert Banner ]           [ Display Green Status ]
            │                                     │
            └─────────────┬───────────────────────┘
                          │
                          ▼
            [ Serve Flask Web Dashboard ]
           (Charts, Tables, Model Comparison)
                          │
                          ▼
                       [ End ]
```

---

## 2. Process Flow Description
1. **Start:** The pipeline initiates either manually via terminal commands or automatically when accessing the web dashboard.
2. **Data Ingestion:** Reads the 14-day time-series dataset (`dataset/temperature_data.csv`).
3. **Preprocessing:** Verifies missing values, converts string timestamps to `datetime64`, sorts records chronologically, and categorizes reference range status (2°C–8°C).
4. **Sliding Window Preparation:** Transforms 1D temperature series into 2D overlapping window matrices (10 readings = 50 minutes).
5. **AI Inference & Error Calculation:**
   - Autoencoder reconstructs each sequence and computes Mean Squared Error (MSE).
   - Anomaly threshold is computed as $\mu_{\text{train}} + 3 \times \sigma_{\text{train}}$.
6. **Decision Logic:**
   - If sequence MSE > Threshold $\rightarrow$ Flagged as **ANOMALY DETECTED**.
   - If sequence MSE $\le$ Threshold $\rightarrow$ Flagged as **NORMAL**.
7. **Dashboard Update:** Flask REST API exposes current reading, time-series data, and model metrics to update Chart.js plots and UI summary cards.
