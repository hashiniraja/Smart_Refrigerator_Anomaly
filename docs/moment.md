# MOMENT Foundation Model Documentation

## 1. What is MOMENT?
**MOMENT** (Multivariate Time-series Open Foundation Models) is a family of open-source time-series foundation models developed by Carnegie Mellon University (Auton Lab). 

Pre-trained on large-scale time-series repositories spanning weather, health, energy, and industrial sensors, MOMENT acts as a generalist time-series backbone capable of zero-shot forecasting, classification, and anomaly detection.

---

## 2. Importance of Time-Series Analysis in Refrigeration
- Refrigerator temperature data is inherently **time-series data**—a sequence of observations recorded at regular time intervals (every 5 minutes).
- Thermal behavior depends strongly on past context: a reading of 6°C might be normal if the refrigerator is currently cooling down after a compressor trigger, but anomalous if drifting steadily upward.
- Time-series models evaluate context sequences rather than isolated static points.

---

## 3. How MOMENT is Used in This Project
1. **Input Data Format:** MOMENT operates on sequences of length 512 timesteps (~42.6 hours of 5-minute interval readings).
2. **Feature Input:** Tensor of shape `(batch_size, 1, 512)` representing univariate temperature telemetry.
3. **Anomaly Scoring:** MOMENT generates sequence reconstructions; reconstruction Mean Squared Error (MSE) serves as the anomaly score.
4. **Thresholding:** Sequences with anomaly scores exceeding $\mu + 3\sigma$ are flagged as anomalies.

---

## 4. Limitations & Experimental Observations
- **Computational Footprint:** MOMENT is a large Transformer foundation model requiring PyTorch and native C++ runtime libraries (`c10.dll` / MSVC Redistributable on Windows).
- **Sequence Length:** Requires longer context windows (512 timesteps vs. 10 timesteps for the local Autoencoder).
- **Performance Neutrality:** In software prototyping, pre-trained foundation models are evaluated side-by-side with lightweight task-specific Autoencoders. Neither model is declared universally superior; performance depends on window sizes, dataset characteristics, and deployment constraints.
