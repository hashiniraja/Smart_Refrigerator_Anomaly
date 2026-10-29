
"""
Smart Refrigerator Temperature Anomaly Detection using IoT and DNN
File: app/app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from flask import Flask, render_template, jsonify, request

app = Flask(__name__, template_folder="templates", static_folder="static")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROCESSED_DATA_PATH = os.path.join(
    BASE_DIR, "dataset", "processed_temperature_data.csv"
)
RAW_DATA_PATH = os.path.join(BASE_DIR, "dataset", "temperature_data.csv")
ANOMALY_PREDICTIONS_PATH = os.path.join(
    BASE_DIR, "results", "anomaly_predictions.csv"
)
FINAL_COMPARISON_PATH = os.path.join(
    BASE_DIR, "results", "final_model_comparison.csv"
)
MODEL_COMPARISON_PATH = os.path.join(
    BASE_DIR, "results", "model_comparison.csv"
)
MODEL_PATH = os.path.join(BASE_DIR, "model", "autoencoder.keras")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")

loaded_scaler = None
loaded_model = None
trained_threshold = 0.008568

current_sim_index = 0
latest_iot_reading = None
iot_temperature_history = deque(maxlen=10)
iot_chart_history = deque(maxlen=300)

if os.path.exists(SCALER_PATH):
    try:
        loaded_scaler = joblib.load(SCALER_PATH)
        print("[OK] Scaler loaded successfully.")
    except Exception as e:
        print(f"[WARNING] Could not load scaler: {e}")

if os.path.exists(MODEL_PATH):
    try:
        loaded_model = tf.keras.models.load_model(
            MODEL_PATH, compile=False
        )
        print("[OK] Autoencoder loaded successfully.")
    except Exception as e:
        print(f"[WARNING] Could not load model: {e}")
else:
    print(f"[WARNING] Model not found: {MODEL_PATH}")

if os.path.exists(ANOMALY_PREDICTIONS_PATH):
    try:
        sample = pd.read_csv(ANOMALY_PREDICTIONS_PATH, nrows=5)
        if "threshold" in sample.columns and not sample.empty:
            trained_threshold = float(sample["threshold"].iloc[0])
    except Exception as e:
        print(f"[WARNING] Could not load threshold: {e}")

print(f"[INFO] Anomaly threshold: {trained_threshold:.6f}")


def get_current_kolkata_time():
    now = datetime.now(ZoneInfo("Asia/Kolkata"))
    return now.strftime("%d-%m-%Y %I:%M:%S %p")


def load_dataset():
    if os.path.exists(PROCESSED_DATA_PATH):
        df = pd.read_csv(PROCESSED_DATA_PATH)
    elif os.path.exists(RAW_DATA_PATH):
        df = pd.read_csv(RAW_DATA_PATH)
    elif os.path.exists(ANOMALY_PREDICTIONS_PATH):
        df = pd.read_csv(ANOMALY_PREDICTIONS_PATH)
        if "actual_temperature" in df.columns:
            df["temperature"] = df["actual_temperature"]
    else:
        return None

    if "temperature" not in df.columns:
        return None

    if "timestamp" not in df.columns:
        df["timestamp"] = ""

    return df


def predict_autoencoder_for_index(df, index):
    start_idx = max(0, index - 9)
    window = df["temperature"].iloc[start_idx:index + 1].to_numpy(
        dtype=float
    )

    if len(window) == 0:
        return 0.0, "NORMAL", False

    if len(window) < 10:
        window = np.pad(
            window, (10 - len(window), 0), mode="edge"
        )

    if loaded_scaler is not None and loaded_model is not None:
        try:
            scaled = loaded_scaler.transform(
                window.reshape(-1, 1)
            ).flatten()

            prediction = loaded_model.predict(
                scaled.reshape(1, 10), verbose=0
            )[0]

            score = float(np.mean(np.square(scaled - prediction)))
            anomaly = score >= trained_threshold

            return (
                score,
                "ANOMALY DETECTED" if anomaly else "NORMAL",
                anomaly,
            )
        except Exception as e:
            print(f"[WARNING] Autoencoder inference failed: {e}")

    temperature = float(window[-1])
    anomaly = temperature < 2.0 or temperature > 8.0

    return (
        0.0,
        "ANOMALY DETECTED" if anomaly else "NORMAL",
        anomaly,
    )


@app.route("/")
def dashboard():
    return render_template("index.html")


@app.route("/api/temperature", methods=["POST"])
def receive_temperature():
    global latest_iot_reading

    data = request.get_json(silent=True)

    if not isinstance(data, dict) or "temperature" not in data:
        return jsonify({
            "success": False,
            "message": "Provide temperature in JSON."
        }), 400

    try:
        temperature = float(data["temperature"])

        if not np.isfinite(temperature):
            raise ValueError("Temperature must be finite.")

        timestamp = get_current_kolkata_time()

        iot_temperature_history.append(temperature)

        score = 0.0
        method = "Temperature Range Check"

        if len(iot_temperature_history) == 10 and (
            loaded_scaler is not None and loaded_model is not None
        ):
            try:
                raw_window = np.array(
                    iot_temperature_history, dtype=float
                ).reshape(-1, 1)

                scaled_window = loaded_scaler.transform(
                    raw_window
                ).flatten()

                reconstructed = loaded_model.predict(
                    scaled_window.reshape(1, 10), verbose=0
                )[0]

                score = float(np.mean(
                    np.square(scaled_window - reconstructed)
                ))

                is_anomaly = score >= trained_threshold
                method = "Autoencoder"
            except Exception as e:
                print(f"[WARNING] Live model prediction failed: {e}")
                is_anomaly = temperature < 2.0 or temperature > 8.0
        else:
            is_anomaly = temperature < 2.0 or temperature > 8.0

        status = "ANOMALY DETECTED" if is_anomaly else "NORMAL"

        latest_iot_reading = {
            "timestamp": timestamp,
            "current_timestamp": timestamp,
            "historical_timestamp": "",
            "temperature": round(temperature, 2),
            "status": status,
            "prediction": "ANOMALY" if is_anomaly else "NORMAL",
            "is_anomaly": is_anomaly,
            "anomaly_score": round(score, 6),
            "threshold": round(trained_threshold, 6),
            "data_source": "Live IoT Sensor",
            "prediction_method": method,
            "alert_message": (
                "Temperature anomaly detected. Check the refrigerator."
                if is_anomaly else ""
            )
        }

        iot_chart_history.append({
            "timestamp": timestamp,
            "temperature": round(temperature, 2),
            "anomaly_score": round(score, 6),
            "threshold": trained_threshold,
            "status": status
        })

        print(
            f"[WOKWI] {temperature:.2f} °C | {status} | "
            f"Score: {score:.6f}"
        )

        return jsonify({"success": True, **latest_iot_reading})

    except (TypeError, ValueError):
        return jsonify({
            "success": False,
            "message": "Temperature must be a valid number."
        }), 400


@app.route("/api/current", methods=["GET"])
def get_current_reading():
    global current_sim_index

    if latest_iot_reading is not None:
        return jsonify(latest_iot_reading)

    df = load_dataset()

    if df is None or df.empty:
        return jsonify({
            "error": "No temperature dataset available.",
            "data_source": "No Data"
        }), 404

    requested_index = request.args.get("index", type=int)

    if requested_index is not None:
        current_sim_index = requested_index % len(df)
    else:
        current_sim_index = (current_sim_index + 1) % len(df)

    row = df.iloc[current_sim_index]
    temperature = float(row["temperature"])

    score, status, anomaly = predict_autoencoder_for_index(
        df, current_sim_index
    )

    timestamp = get_current_kolkata_time()

    return jsonify({
        "timestamp": timestamp,
        "current_timestamp": timestamp,
        "historical_timestamp": str(row.get("timestamp", "")),
        "temperature": round(temperature, 2),
        "status": status,
        "prediction": "ANOMALY" if anomaly else "NORMAL",
        "is_anomaly": anomaly,
        "anomaly_score": round(score, 6),
        "threshold": round(trained_threshold, 6),
        "data_source": "Simulated Temperature Data",
        "prediction_method": "Autoencoder",
        "simulation_index": current_sim_index,
        "total_records": len(df),
        "alert_message": (
            "Temperature anomaly detected. Check the refrigerator."
            if anomaly else ""
        )
    })


@app.route("/api/temperature-data", methods=["GET"])
def get_temperature_data():
    if iot_chart_history:
        readings = list(iot_chart_history)

        return jsonify({
            "timestamps": [r["timestamp"] for r in readings],
            "temperatures": [r["temperature"] for r in readings],
            "anomaly_scores": [r["anomaly_score"] for r in readings],
            "thresholds": [r["threshold"] for r in readings],
            "statuses": [r["status"] for r in readings]
        })

    df = load_dataset()

    if df is None or df.empty:
        return jsonify({"error": "No dataset found"}), 404

    limit = max(1, min(
        request.args.get("limit", default=300, type=int), 5000
    ))

    sliced = df.tail(limit)

    if os.path.exists(ANOMALY_PREDICTIONS_PATH):
        try:
            predictions = pd.read_csv(ANOMALY_PREDICTIONS_PATH)
            predictions = predictions.tail(limit)

            scores = predictions.get(
                "reconstruction_error",
                pd.Series([0.0] * len(predictions))
            ).fillna(0).astype(float).tolist()

            statuses = predictions.get(
                "predicted_status",
                pd.Series(["NORMAL"] * len(predictions))
            ).fillna("NORMAL").tolist()

            if "threshold" in predictions.columns and not predictions.empty:
                threshold = float(predictions["threshold"].iloc[0])
            else:
                threshold = trained_threshold
        except Exception as e:
            print(f"[WARNING] Chart prediction data unavailable: {e}")
            scores = [0.0] * len(sliced)
            statuses = ["NORMAL"] * len(sliced)
            threshold = trained_threshold
    else:
        scores = [0.0] * len(sliced)
        statuses = ["NORMAL"] * len(sliced)
        threshold = trained_threshold

    count = len(sliced)

    scores = (scores + [0.0] * count)[:count]
    statuses = (statuses + ["NORMAL"] * count)[:count]

    return jsonify({
        "timestamps": sliced["timestamp"].astype(str).tolist(),
        "temperatures": sliced["temperature"].astype(float).tolist(),
        "anomaly_scores": scores,
        "thresholds": [threshold] * count,
        "statuses": statuses
    })


@app.route("/api/model-results", methods=["GET"])
def get_model_results():
    if os.path.exists(FINAL_COMPARISON_PATH):
        target_path = FINAL_COMPARISON_PATH
    elif os.path.exists(MODEL_COMPARISON_PATH):
        target_path = MODEL_COMPARISON_PATH
    else:
        return jsonify({
            "status": "not_available",
            "message": "Model comparison file not found.",
            "models": []
        })

    try:
        df = pd.read_csv(target_path)
        records = []

        for _, row in df.iterrows():
            model_name = str(row.get("Model", ""))

            def clean_val(value):
                if pd.isna(value) or str(value).strip().lower() in (
                    "nan", "none", "n/a", ""
                ):
                    return "N/A"
                if isinstance(value, (float, np.floating)):
                    return round(float(value), 4)
                if isinstance(value, (int, np.integer)):
                    return int(value)
                return str(value)

            precision = clean_val(row.get("Precision", "N/A"))
            recall = clean_val(row.get("Recall", "N/A"))
            f1_score = clean_val(row.get("F1-score", "N/A"))
            detected = clean_val(row.get("Detected Anomalies", "N/A"))
            raw_status = str(row.get("Status", ""))

            if model_name.upper() == "MOMENT" and (
                precision == "N/A"
                or "MSVC" in raw_status
                or "Not evaluated" in raw_status
            ):
                status = "Not evaluated - environment dependency"
            else:
                status = raw_status or "Evaluated & Working"

            records.append({
                "Model": model_name,
                "Precision": precision,
                "Recall": recall,
                "F1-score": f1_score,
                "Detected Anomalies": detected,
                "Status": status
            })

        return jsonify({"status": "success", "models": records})

    except Exception as e:
        print(f"[ERROR] Model results: {e}")
        return jsonify({
            "status": "error",
            "message": "Failed to read model comparison file.",
            "models": []
        }), 500


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" SMART REFRIGERATOR ANOMALY DETECTION ")
    print("=" * 60)
    print("Dashboard: http://127.0.0.1:5000")
    print("IoT endpoint: POST /api/temperature")
    print("Press Ctrl+C to stop the server.\n")

    app.run(host="0.0.0.0", port=5000, debug=False)
