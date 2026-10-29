# Dataset Documentation

## 1. Dataset Specifications
- **File Name:** `dataset/temperature_data.csv` & `dataset/processed_temperature_data.csv`
- **Total Duration:** 14 Days (Continuous simulation)
- **Sampling Interval:** 5 Minutes
- **Total Records (Rows):** 4,032
- **Data Features (Columns):**
  1. `timestamp`: Reading date and time (`YYYY-MM-DD HH:MM:SS`)
  2. `temperature`: Internal temperature reading in degrees Celsius (°C)
  3. `is_anomaly`: Ground-truth binary indicator (`0` = Normal, `1` = Anomaly)
  4. `anomaly_type`: Descriptive label (`Normal`, `Sudden Spike`, `Gradual Increase`, `Sudden Drop`, `Rapid Fluctuations`)
  5. `status`: Preprocessed rule-based reference classification (`Normal` for 2°C–8°C, `Anomaly` otherwise)

---

## 2. Normal Operating Pattern
Under normal operating conditions, refrigerator temperature fluctuates naturally within **2.0°C and 5.0°C**.
- **Compressor Cycling:** Oscillates in a sinusoidal pattern with a period of ~45 minutes.
- **Diurnal Variance:** Slight 24-hour heat-load fluctuation (±0.3°C).
- **Sensor Noise:** Simulated normal Gaussian noise ($\sigma = 0.15^\circ\text{C}$).

---

## 3. Injected Abnormal Scenarios
Real-world physical sensors in healthy refrigerators rarely emit failure data. To train and evaluate AI models effectively, four realistic abnormal failure scenarios were synthesized:

1. **Sudden Temperature Spike (Day 3):** Simulates the door being left wide open or hot food placed inside. Temperature spikes rapidly to ~14.3°C before slowly cooling back down.
2. **Gradual Temperature Increase (Day 6–7):** Simulates a gradual coolant gas leak or mechanical degradation. Temperature slowly rises from 3.5°C to ~11.5°C over 16 hours.
3. **Sudden Temperature Drop (Day 9):** Simulates a thermostat relay lock or continuous over-freezing failure. Temperature drops to sub-zero levels (-4.03°C).
4. **Rapid Fluctuations (Day 12):** Simulates a faulty defrost cycle or rapid relay chatter. Temperature swings erratically between 0°C and 9°C.

---

## 4. Rationale for Simulated Data
In early-stage AI software prototyping:
- Real physical hardware (ESP32/sensors) may not be available immediately.
- Operating a real refrigerator in broken states (leaving door open for days or draining refrigerant) is impractical and dangerous.
- Simulated time-series datasets provide a controlled, repeatable environment for baseline neural network validation before physical hardware deployment.
