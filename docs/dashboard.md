# Flask Web Dashboard Documentation

## 1. Overview
The web dashboard provides a user interface for monitoring refrigerator temperature, tracking AI anomaly detection scores, inspecting model comparison metrics, and receiving visual alert notifications.

---

## 2. Architecture & Stack
- **Backend Framework:** Python Flask (`app/app.py`)
- **Frontend Templates:** HTML5 (`app/templates/index.html`)
- **Styling:** Custom Responsive CSS3 with CSS Variables (`app/static/style.css`)
- **Client-Side Interactivity:** JavaScript ES6 & Chart.js (`app/static/script.js`)

---

## 3. REST API Routes (`app/app.py`)

| Endpoint | Method | Response | Description |
| :--- | :--- | :--- | :--- |
| `/` | `GET` | HTML Page | Renders the main dashboard interface |
| `/api/current` | `GET` | JSON Object | Delivers current simulated temperature, status, anomaly score, and alert text |
| `/api/temperature-data` | `GET` | JSON Arrays | Delivers time-series arrays for temperature & error charts |
| `/api/model-results` | `GET` | JSON Array | Delivers model evaluation comparison metrics |

---

## 4. Dashboard Visual Elements
1. **Summary Status Cards:** Displays current temperature (°C), status (`NORMAL` / `ANOMALY DETECTED`), reconstruction error score, and last updated timestamp.
2. **Dynamic Alert Banner:** Displays `⚠ Temperature anomaly detected. Please check the refrigerator.` when an anomaly occurs.
3. **Chart.js Graphs:**
   - **Temperature Timeline:** Plots temperature curve against 2°C–5°C safe zone boundaries and red anomaly dots.
   - **Reconstruction Error Curve:** Plots error scores against the learned threshold line.
4. **Model Comparison Table:** Displays Autoencoder and MOMENT precision, recall, F1-score, and anomaly counts.
5. **Simulation Controls:** "Refresh Data" button and "Auto Simulation" toggle with clear simulated telemetry disclaimers.
