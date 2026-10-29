"""
===============================================================================
Smart Refrigerator Temperature Anomaly Detection using IoT and DNN
-------------------------------------------------------------------------------
File: main.py
Version: 1.0 (Dataset Inspection & Visualization Baseline)
Description:
    This script loads the simulated refrigerator time-series dataset, performs
    data hygiene checks (missing values & duplicates), prints basic statistical
    summaries, and visualizes the temperature trends over time. The generated 
    plot is automatically saved to the 'results/' folder.
===============================================================================
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# File paths
DATASET_PATH = os.path.join("dataset", "temperature_data.csv")
RESULTS_DIR = "results"
PLOT_OUTPUT_PATH = os.path.join(RESULTS_DIR, "temperature_visualization.png")


def load_dataset(file_path):
    """Loads the temperature dataset from CSV into a Pandas DataFrame."""
    print("=" * 70)
    print("1. LOADING DATASET")
    print("=" * 70)
    
    if not os.path.exists(file_path):
        print(f"[!] File not found: '{file_path}'.")
        print("[!] Generating dataset automatically using dataset/generate_dataset.py ...")
        from dataset.generate_dataset import generate_refrigerator_dataset
        generate_refrigerator_dataset(file_path)

    df = pd.read_csv(file_path)
    # Convert timestamp column to proper datetime objects
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    print(f"[OK] Dataset successfully loaded from: '{file_path}'")
    print(f"[OK] Total Records (Rows): {len(df)}")
    print(f"[OK] Columns: {list(df.columns)}")
    print("\n--- First 5 Rows of Dataset ---")
    print(df.head())
    print("\n--- Last 5 Rows of Dataset ---")
    print(df.tail())
    return df


def inspect_data_quality(df):
    """Checks for missing values, nulls, and duplicate timestamps/records."""
    print("\n" + "=" * 70)
    print("2. DATA QUALITY & INTEGRITY CHECKS")
    print("=" * 70)
    
    # Check missing/null values
    missing_counts = df.isnull().sum()
    print("[+] Missing / Null Values per Column:")
    for col, count in missing_counts.items():
        status = "[OK] Clean" if count == 0 else f"[!] {count} missing"
        print(f"    - {col:<15}: {count} ({status})")

    # Check duplicate records
    duplicate_rows = df.duplicated().sum()
    duplicate_timestamps = df.duplicated(subset=["timestamp"]).sum()
    print(f"[+] Total Duplicate Rows: {duplicate_rows}")
    print(f"[+] Duplicate Timestamps: {duplicate_timestamps}")
    
    if missing_counts.sum() == 0 and duplicate_rows == 0:
        print("\n[OK] Data Hygiene Status: PASSED (No missing values or duplicate records detected)")
    else:
        print("\n[!] Data Hygiene Warning: Anomalous entries or duplicates found in raw CSV.")


def display_statistics(df):
    """Calculates and displays summary statistics of refrigerator temperatures."""
    print("\n" + "=" * 70)
    print("3. DESCRIPTIVE STATISTICS")
    print("=" * 70)
    
    stats = df["temperature"].describe()
    print("[+] Temperature Statistics (deg C):")
    print(f"    - Mean (Average)   : {stats['mean']:.2f} deg C")
    print(f"    - Std Deviation    : {stats['std']:.2f} deg C")
    print(f"    - Min Temperature  : {stats['min']:.2f} deg C")
    print(f"    - 25th Percentile   : {stats['25%']:.2f} deg C")
    print(f"    - Median (50%)     : {stats['50%']:.2f} deg C")
    print(f"    - 75th Percentile   : {stats['75%']:.2f} deg C")
    print(f"    - Max Temperature  : {stats['max']:.2f} deg C")

    # Breakdown by Anomaly Category (if available in simulated dataset)
    if "anomaly_type" in df.columns:
        print("\n[+] Breakdown of Reading Types:")
        counts = df["anomaly_type"].value_counts()
        for anomaly_name, count in counts.items():
            percentage = (count / len(df)) * 100
            print(f"    - {anomaly_name:<20}: {count} readings ({percentage:.2f}%)")


def plot_temperature_data(df, output_path):
    """Plots temperature vs time using Matplotlib and saves the output graph."""
    print("\n" + "=" * 70)
    print("4. GENERATING TEMPERATURE VISUALIZATION GRAPH")
    print("=" * 70)

    # Ensure results directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Setup plot figure
    plt.figure(figsize=(14, 6))
    
    # Plot continuous temperature reading line
    plt.plot(df["timestamp"], df["temperature"], color="#1f77b4", linewidth=1.2, label="Temperature (°C)")

    # Define normal operating boundary (2°C to 5°C)
    plt.axhline(y=5.0, color="orange", linestyle="--", linewidth=1.5, label="Upper Safe Limit (5°C)")
    plt.axhline(y=2.0, color="cyan", linestyle="--", linewidth=1.5, label="Lower Safe Limit (2°C)")
    
    # Fill normal safe zone range
    plt.axhspan(2.0, 5.0, color="green", alpha=0.1, label="Normal Safe Zone (2°C - 5°C)")

    # Highlight abnormal points if labeled in dataset
    if "is_anomaly" in df.columns:
        anomalies = df[df["is_anomaly"] == 1]
        plt.scatter(anomalies["timestamp"], anomalies["temperature"], 
                    color="red", s=12, label="Simulated Anomalies", zorder=3)

    # Title and Labels
    plt.title("Smart Refrigerator Temperature Monitoring (14-Day Simulation)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Timestamp (Date & Time)", fontsize=11, labelpad=8)
    plt.ylabel("Temperature (°C)", fontsize=11, labelpad=8)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()

    # Save visualization to file
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"[OK] Graph successfully plotted and saved to: '{output_path}'")


def main():
    print("\n" + "*" * 70)
    print(" SMART REFRIGERATOR TEMPERATURE ANOMALY DETECTION - FIRST VERSION ")
    print("*" * 70 + "\n")
    
    # Step 1: Load CSV Data
    df = load_dataset(DATASET_PATH)
    
    # Step 2: Quality Inspection
    inspect_data_quality(df)
    
    # Step 3: Statistical Analysis
    display_statistics(df)
    
    # Step 4: Data Visualization & Plot Export
    plot_temperature_data(df, PLOT_OUTPUT_PATH)
    
    print("\n" + "=" * 70)
    print("EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"-> Dataset: {DATASET_PATH}")
    print(f"-> Visualization saved at: {PLOT_OUTPUT_PATH}\n")


if __name__ == "__main__":
    main()
