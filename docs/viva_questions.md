# Viva Voce Questions & Answers

## 1. What is this project about?
**Answer:** This project is a software-based AI system titled *"Smart Refrigerator Temperature Anomaly Detection using IoT and DNN"*. It monitors refrigerator temperature patterns, detects abnormalities (like door openings or cooling failure) using Deep Learning (Autoencoder) and Foundation Models (MOMENT), and alerts users via a web dashboard.

## 2. Why is refrigerator temperature monitoring important?
**Answer:** Food spoils quickly if refrigerator temperature strays outside the safe 2°C–5°C range. Early anomaly detection prevents food spoilage, bacterial growth, energy waste, and expensive compressor failure.

## 3. What is an anomaly?
**Answer:** An anomaly is a data reading or pattern that deviates significantly from normal expected behavior. In a refrigerator, normal behavior is a steady 2°C–5°C cycle; anomalies include sudden spikes (14°C), gradual upward drifts (11°C), or sub-zero freezing drops (-4°C).

## 4. What is anomaly detection?
**Answer:** It is an AI technique used to identify rare items, events, or observations that differ significantly from the majority of the data.

## 5. What is IoT (Internet of Things)?
**Answer:** IoT refers to a network of physical devices (like sensors and microcontrollers) connected to the internet that collect and share real-time telemetry data.

## 6. What is DNN (Deep Neural Network)?
**Answer:** A DNN is an artificial neural network with multiple hidden layers between the input and output layers, capable of learning complex non-linear patterns from data.

## 7. What is time-series data?
**Answer:** Time-series data is a sequence of data points recorded in chronological order at successive, equal intervals of time (e.g. temperature recorded every 5 minutes).

## 8. Why is temperature considered time-series data?
**Answer:** Because a single temperature reading alone lacks context; whether 6°C is normal depends on recent past readings and whether the cooling cycle is currently active.

## 9. What is an Autoencoder?
**Answer:** An Autoencoder is a Deep Learning neural network designed to compress input data into a small bottleneck layer (Encoder) and then reconstruct the original input (Decoder).

## 10. Why use an Autoencoder for anomaly detection?
**Answer:** The Autoencoder is trained strictly on *normal* data. When it sees normal data, it reconstructs it with low error. When it encounters an *abnormal* sequence, it cannot reconstruct it accurately, resulting in a high reconstruction error.

## 11. What is reconstruction error?
**Answer:** Reconstruction error is the Mean Squared Error (MSE) difference between the original input sequence and the Autoencoder's reconstructed output sequence.

## 12. What is an anomaly threshold?
**Answer:** It is a cutoff score calculated statistically ($\text{Threshold} = \mu + 3\sigma$ of normal training error). Any sequence with a reconstruction error higher than this threshold is classified as an anomaly.

## 13. What is MOMENT?
**Answer:** MOMENT is an open-source time-series foundation model developed by Carnegie Mellon University (Auton Lab), pre-trained on multi-domain time-series datasets for zero-shot forecasting and anomaly detection.

## 14. Why include MOMENT alongside the Autoencoder?
**Answer:** To compare a lightweight locally trained baseline (Autoencoder) with a modern pre-trained time-series foundation model (MOMENT).

## 15. What is Flask?
**Answer:** Flask is a lightweight Python web framework used to build web applications and REST API endpoints.

## 16. Why use Flask in this project?
**Answer:** Flask serves as the backend bridge that loads model prediction CSVs and delivers JSON telemetry to the interactive web dashboard.

## 17. Is the current project using real physical hardware?
**Answer:** No. The current implementation is a software-based simulation prototype using realistic simulated temperature time-series data. Hardware (ESP32, DS18B20 sensor, buzzer) is documented for future integration.

## 18. What are the limitations of the current system?
**Answer:** 
- It uses simulated dataset logs rather than live hardware sensors.
- The learned threshold depends on training data distribution.
- MOMENT requires native C++ runtime libraries (MSVC Redistributable on Windows).

## 19. How can ESP32 hardware be integrated later?
**Answer:** An ESP32 microcontroller with a DS18B20 temperature sensor can be programmed to send real-time temperature readings via HTTP POST or MQTT directly to Flask's `/api/current` endpoint.

## 20. What happens when an anomaly is detected on the dashboard?
**Answer:** The dashboard status changes to `ANOMALY DETECTED`, the badge turns red, red scatter markers appear on the temperature charts, and a prominent warning banner displays: *"⚠ Temperature anomaly detected. Please check the refrigerator."*
