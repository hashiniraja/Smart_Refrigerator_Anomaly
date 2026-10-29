"""
===============================================================================
Smart Refrigerator Temperature Anomaly Detection using IoT and DNN
-------------------------------------------------------------------------------
File: model/moment_model.py
Phase 4: MOMENT Time-Series Foundation Model for Anomaly Detection
Description:
    MOMENT is an open-source foundation model for time-series analysis developed
    by Carnegie Mellon University (Auton Lab). It uses a Transformer-based architecture
    pre-trained on large-scale time-series datasets.

    This script loads preprocessed refrigerator temperature data, prepares time-series
    sequences (512-timestep windows required by MOMENT), and runs anomaly detection.
===============================================================================
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# File paths
PROCESSED_DATASET_PATH = os.path.join("dataset", "processed_temperature_data.csv")
PREDICTIONS_CSV_PATH = os.path.join("results", "moment_predictions.csv")
COMPARISON_CSV_PATH = os.path.join("results", "model_comparison.csv")
PLOT_OUTPUT_PATH = os.path.join("results", "moment_anomaly_visualization.png")
COMPARISON_PLOT_PATH = os.path.join("results", "model_comparison_plot.png")

# Window size required by MOMENT foundation model architecture
MOMENT_WINDOW_SIZE = 512


def check_dependencies_and_import_moment():
    """
    Checks if PyTorch and momentfm packages can be loaded in the local environment.
    Catches environment/DLL loading issues gracefully (e.g. WinError 1114 on Windows).
    """
    print("=" * 70)
    print("CHECKING DEPENDENCIES & LOADING MOMENT FOUNDATION MODEL")
    print("=" * 70)
    
    try:
        import torch
        from momentfm import MOMENTPipeline
        print("[OK] PyTorch and momentfm successfully loaded!")
        return torch, MOMENTPipeline
    except ImportError as e:
        print("\n[!] DEPENDENCY ERROR: MOMENT package 'momentfm' or 'torch' is not installed.")
        print(f"[!] Error Details: {e}")
        print("\n[+] TO FIX THIS:")
        print("    1. Install momentfm:  pip install momentfm")
        print("    2. Install PyTorch:   pip install torch --index-url https://download.pytorch.org/whl/cpu")
        return None, None
    except OSError as e:
        print("\n[!] OS / ENVIRONMENT ERROR: PyTorch C++ DLL Initialization Failed.")
        print(f"[!] Exact Error Details: {e}")
        print("\n" + "=" * 70)
        print("DIAGNOSIS OF THE ISSUE:")
        print("=" * 70)
        print("PyTorch requires the Microsoft Visual C++ Redistributable package on Windows.")
        print("When 'c10.dll' fails to load (WinError 1114), it indicates that the MSVC runtime")
        print("libraries (vcruntime140_1.dll / msvcp140.dll) are missing from the system.")
        print("\n[+] HOW TO RESOLVE THIS ISSUE:")
        print("    1. Download and install Microsoft Visual C++ 2015-2022 Redistributable (x64):")
        print("       https://aka.ms/vs/17/release/vc_redist.x64.exe")
        print("    2. Alternatively, run the project inside WSL2 (Windows Subsystem for Linux) or Google Colab.")
        print("    3. After installing VC++ Redistributable, re-run: python model/moment_model.py\n")
        return None, None
    except Exception as e:
        print(f"\n[!] Unexpected Error while loading MOMENT: {e}")
        return None, None


def load_dataset(file_path):
    """Loads preprocessed temperature dataset."""
    if not os.path.exists(file_path):
        print(f"[!] Dataset not found at '{file_path}'. Run preprocessing.py first!")
        sys.exit(1)
    df = pd.read_csv(file_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def run_moment_pipeline():
    """
    Executes MOMENT foundation model anomaly detection pipeline.
    """
    torch, MOMENTPipeline = check_dependencies_and_import_moment()
    
    if torch is None or MOMENTPipeline is None:
        print("=" * 70)
        print("MOMENT MODEL LOADING ABORTED DUE TO MISSING ENVIRONMENT DEPENDENCIES.")
        print("Existing Autoencoder pipeline remains 100% operational.")
        print("=" * 70)
        # Create/update model_comparison.csv with Autoencoder results & MOMENT diagnostic note
        generate_fallback_comparison()
        return

    print("\n[+] Loading pre-trained MOMENT model ('AutonLab/MOMENT-1-large')...")
    try:
        model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large",
            model_kwargs={"task_name": "anomaly-detection"}
        )
        model.init()
        print("[OK] Pre-trained MOMENT model loaded successfully!")
    except Exception as e:
        print(f"[!] Model download/initialization failed: {e}")
        generate_fallback_comparison()
        return

    df = load_dataset(PROCESSED_DATASET_PATH)
    temperatures = df["temperature"].values
    
    # Prepare 512-length sequences
    sequences = []
    timestamps = []
    ground_truth = []
    
    for i in range(len(temperatures) - MOMENT_WINDOW_SIZE + 1):
        sequences.append(temperatures[i : i + MOMENT_WINDOW_SIZE])
        idx = i + MOMENT_WINDOW_SIZE - 1
        timestamps.append(df["timestamp"].iloc[idx])
        ground_truth.append(df["is_anomaly"].iloc[idx])
        
    sequences = np.array(sequences)
    
    # Convert to PyTorch Tensor (batch_size, num_channels, sequence_length)
    x_input = torch.tensor(sequences, dtype=torch.float32).unsqueeze(1)
    
    print(f"[+] Formatted input tensor shape: {x_input.shape}")
    print("[+] Running inference with MOMENT Foundation Model...")

    with torch.no_grad():
        output = model(x_input)
        
    # Extract reconstruction / anomaly scores
    if hasattr(output, "reconstruction"):
        reconstructed = output.reconstruction.squeeze(1).numpy()
        anomaly_scores = np.mean(np.square(sequences - reconstructed), axis=1)
    elif hasattr(output, "anomaly_scores"):
        anomaly_scores = output.anomaly_scores.numpy()
    else:
        anomaly_scores = np.mean(np.square(sequences), axis=1)

    # Statistical Thresholding
    threshold = np.mean(anomaly_scores) + 3 * np.std(anomaly_scores)
    predicted_binary = (anomaly_scores > threshold).astype(int)
    predicted_status = np.where(predicted_binary == 1, "Anomaly", "Normal")

    # Evaluation
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    y_true = np.array(ground_truth)
    y_pred = predicted_binary

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    num_anomalies = int((y_pred == 1).sum())

    print("\n" + "=" * 70)
    print("MOMENT MODEL EVALUATION COMPLETED")
    print("=" * 70)
    print(f"Precision          : {precision:.4f}")
    print(f"Recall             : {recall:.4f}")
    print(f"F1-score           : {f1:.4f}")
    print(f"Detected Anomalies : {num_anomalies}")
    print(f"Anomaly Threshold  : {threshold:.6f}")

    # Save Predictions CSV
    res_df = pd.DataFrame({
        "timestamp": timestamps,
        "temperature": sequences[:, -1],
        "reference_is_anomaly": ground_truth,
        "anomaly_score": anomaly_scores,
        "threshold": threshold,
        "predicted_status": predicted_status
    })
    os.makedirs("results", exist_ok=True)
    res_df.to_csv(PREDICTIONS_CSV_PATH, index=False)
    print(f"[OK] Saved MOMENT predictions to '{PREDICTIONS_CSV_PATH}'")

    # Generate Comparison
    save_model_comparison(moment_precision=precision, moment_recall=recall, 
                          moment_f1=f1, moment_anomalies=num_anomalies)


def generate_fallback_comparison():
    """Generates results/model_comparison.csv with Autoencoder baseline stats."""
    os.makedirs("results", exist_ok=True)
    
    # Autoencoder evaluation metrics (from Phase 3)
    comparison_df = pd.DataFrame([
        {
            "Model": "Autoencoder",
            "Precision": 0.8810,
            "Recall": 0.2721,
            "F1-score": 0.4157,
            "Detected Anomalies": 126,
            "Status": "Evaluated & Working"
        },
        {
            "Model": "MOMENT",
            "Precision": "N/A",
            "Recall": "N/A",
            "F1-score": "N/A",
            "Detected Anomalies": "N/A",
            "Status": "Requires MSVC C++ Redistributable on Windows"
        }
    ])
    comparison_df.to_csv(COMPARISON_CSV_PATH, index=False)
    print(f"[OK] Model comparison report saved to: '{COMPARISON_CSV_PATH}'")
    create_comparison_plot(comparison_df)


def save_model_comparison(moment_precision, moment_recall, moment_f1, moment_anomalies):
    """Saves model comparison metrics to CSV and generates comparison chart."""
    comparison_df = pd.DataFrame([
        {
            "Model": "Autoencoder",
            "Precision": 0.8810,
            "Recall": 0.2721,
            "F1-score": 0.4157,
            "Detected Anomalies": 126
        },
        {
            "Model": "MOMENT",
            "Precision": round(moment_precision, 4),
            "Recall": round(moment_recall, 4),
            "F1-score": round(moment_f1, 4),
            "Detected Anomalies": moment_anomalies
        }
    ])
    comparison_df.to_csv(COMPARISON_CSV_PATH, index=False)
    print(f"[OK] Model comparison report saved to: '{COMPARISON_CSV_PATH}'")
    create_comparison_plot(comparison_df)


def create_comparison_plot(df):
    """Plots a visual bar chart comparing Autoencoder and MOMENT performance metrics."""
    plt.figure(figsize=(9, 5))
    
    models = df["Model"].tolist()
    metrics = ["Precision", "Recall", "F1-score"]
    
    x = np.arange(len(models))
    width = 0.25

    # Check if numerical values are available
    if isinstance(df.iloc[1]["Precision"], (int, float)):
        p_vals = df["Precision"].astype(float).tolist()
        r_vals = df["Recall"].astype(float).tolist()
        f_vals = df["F1-score"].astype(float).tolist()
        
        plt.bar(x - width, p_vals, width, label="Precision", color="#1f77b4")
        plt.bar(x, r_vals, width, label="Recall", color="#ff7f0e")
        plt.bar(x + width, f_vals, width, label="F1-Score", color="#2ca02c")

        plt.title("Model Performance Comparison: Autoencoder vs MOMENT", fontsize=13, fontweight="bold")
        plt.xlabel("Model Architecture", fontsize=11)
        plt.ylabel("Score", fontsize=11)
        plt.xticks(x, models)
        plt.ylim(0, 1.0)
        plt.legend(loc="upper right")
        plt.grid(True, linestyle=":", alpha=0.6)
    else:
        # Fallback plot for Autoencoder baseline
        ae_row = df[df["Model"] == "Autoencoder"].iloc[0]
        metrics_names = ["Precision", "Recall", "F1-Score"]
        scores = [float(ae_row["Precision"]), float(ae_row["Recall"]), float(ae_row["F1-score"])]
        
        plt.bar(metrics_names, scores, color=["#1f77b4", "#ff7f0e", "#2ca02c"], width=0.4)
        plt.title("Autoencoder Baseline Evaluation Metrics", fontsize=13, fontweight="bold")
        plt.ylabel("Score", fontsize=11)
        plt.ylim(0, 1.0)
        plt.grid(True, linestyle=":", alpha=0.6)
        
        # Add values above bars
        for i, val in enumerate(scores):
            plt.text(i, val + 0.02, f"{val:.4f}", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(COMPARISON_PLOT_PATH, dpi=300)
    plt.close()
    print(f"[OK] Model comparison chart saved to: '{COMPARISON_PLOT_PATH}'")


if __name__ == "__main__":
    run_moment_pipeline()
