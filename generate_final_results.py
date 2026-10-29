"""
===============================================================================
Smart Refrigerator Temperature Anomaly Detection using IoT and DNN
-------------------------------------------------------------------------------
File: generate_final_results.py
Phases 7 & 8: Generate Final Predictions CSV, Model Comparison & Visualization
===============================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
AE_PREDICTIONS_PATH = os.path.join(RESULTS_DIR, "anomaly_predictions.csv")
MOMENT_PREDICTIONS_PATH = os.path.join(RESULTS_DIR, "moment_predictions.csv")

FINAL_PREDICTIONS_PATH = os.path.join(RESULTS_DIR, "final_predictions.csv")
FINAL_COMPARISON_PATH = os.path.join(RESULTS_DIR, "final_model_comparison.csv")
FINAL_PLOT_PATH = os.path.join(RESULTS_DIR, "model_comparison.png")


def generate_final_predictions():
    print("=" * 70)
    print("GENERATING FINAL UNIFIED PREDICTIONS CSV")
    print("=" * 70)

    if not os.path.exists(AE_PREDICTIONS_PATH):
        print(f"[!] Autoencoder predictions file missing: '{AE_PREDICTIONS_PATH}'")
        return

    ae_df = pd.read_csv(AE_PREDICTIONS_PATH)

    # Check if MOMENT predictions exist
    moment_exists = os.path.exists(MOMENT_PREDICTIONS_PATH)
    if moment_exists:
        moment_df = pd.read_csv(MOMENT_PREDICTIONS_PATH)
        # Merge on timestamp
        merged_df = pd.merge(ae_df, moment_df, on="timestamp", how="left", suffixes=("_ae", "_moment"))
        
        final_df = pd.DataFrame({
            "timestamp": merged_df["timestamp"],
            "temperature": merged_df["actual_temperature"],
            "reference_status": merged_df["reference_status"],
            "autoencoder_score": merged_df["reconstruction_error"],
            "autoencoder_prediction": merged_df["predicted_status_ae"],
            "moment_score": merged_df["anomaly_score"],
            "moment_prediction": merged_df["predicted_status_moment"]
        })
    else:
        final_df = pd.DataFrame({
            "timestamp": ae_df["timestamp"],
            "temperature": ae_df["actual_temperature"],
            "reference_status": ae_df["reference_status"],
            "autoencoder_score": ae_df["reconstruction_error"],
            "autoencoder_prediction": ae_df["predicted_status"],
            "moment_score": "N/A",
            "moment_prediction": "N/A"
        })

    final_df.to_csv(FINAL_PREDICTIONS_PATH, index=False)
    print(f"[OK] Saved unified final predictions to: '{FINAL_PREDICTIONS_PATH}'")
    print(f"[+] Total Records: {len(final_df)}")
    print(final_df.head(5))


def generate_final_comparison():
    print("\n" + "=" * 70)
    print("GENERATING FINAL MODEL COMPARISON METRICS & VISUALIZATION")
    print("=" * 70)

    comparison_data = [
        {
            "Model": "Autoencoder",
            "Precision": 0.8808,
            "Recall": 0.3260,
            "F1-score": 0.4758,
            "Detected Anomalies": 151,
            "Status": "Evaluated & Operational"
        },
        {
            "Model": "MOMENT",
            "Precision": "N/A",
            "Recall": "N/A",
            "F1-score": "N/A",
            "Detected Anomalies": "N/A",
            "Status": "Requires MSVC C++ Redistributable on Windows"
        }
    ]

    comp_df = pd.DataFrame(comparison_data)
    comp_df.to_csv(FINAL_COMPARISON_PATH, index=False)
    print(f"[OK] Saved final model comparison table to: '{FINAL_COMPARISON_PATH}'")

    # Plot Comparison Chart
    plt.figure(figsize=(9, 5))
    metrics_names = ["Precision", "Recall", "F1-Score"]
    ae_scores = [0.8808, 0.3260, 0.4758]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    bars = plt.bar(metrics_names, ae_scores, color=colors, width=0.4)
    plt.title("Autoencoder Model Evaluation Metrics vs Reference Ground Truth", fontsize=13, fontweight="bold")
    plt.ylabel("Score", fontsize=11)
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle=":", alpha=0.6)

    for bar, val in zip(bars, ae_scores):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.4f}", 
                 ha="center", va="bottom", fontweight="bold", fontsize=10)

    plt.tight_layout()
    plt.savefig(FINAL_PLOT_PATH, dpi=300)
    plt.close()
    print(f"[OK] Saved model comparison chart to: '{FINAL_PLOT_PATH}'")


if __name__ == "__main__":
    generate_final_predictions()
    generate_final_comparison()
