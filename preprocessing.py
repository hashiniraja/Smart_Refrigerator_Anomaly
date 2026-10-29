"""
===============================================================================
Smart Refrigerator Temperature Anomaly Detection using IoT and DNN
-------------------------------------------------------------------------------
File: preprocessing.py
Phase 2: Data Preprocessing & Anomaly Label Preparation
Description:
    This script loads the raw simulated temperature time-series data, checks 
    data quality, converts timestamps, sorts records, applies a rule-based
    threshold (2 deg C to 8 deg C) to categorize readings into 'Normal' or 
    'Anomaly' status, saves the processed dataset to CSV, and generates an
    updated visualization graph.
===============================================================================
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# Define file paths
RAW_DATASET_PATH = os.path.join("dataset", "temperature_data.csv")
PROCESSED_DATASET_PATH = os.path.join("dataset", "processed_temperature_data.csv")
RESULTS_DIR = "results"
PLOT_OUTPUT_PATH = os.path.join(RESULTS_DIR, "processed_temperature_visualization.png")


def load_dataset(file_path):
    """
    Step 1: Load the raw CSV dataset into a Pandas DataFrame.
    """
    print("=" * 70)
    print("STEP 1: LOADING RAW DATASET")
    print("=" * 70)
    
    if not os.path.exists(file_path):
        print(f"[!] Raw dataset file not found at: '{file_path}'")
        print("[!] Running dataset generator script...")
        from dataset.generate_dataset import generate_refrigerator_dataset
        generate_refrigerator_dataset(file_path)

    df = pd.read_csv(file_path)
    print(f"[OK] Raw dataset loaded successfully from: '{file_path}'")
    return df


def inspect_raw_dataset(df):
    """
    Step 2: Check and report dataset statistics and hygiene metrics.
    """
    print("\n" + "=" * 70)
    print("STEP 2: DATA INSPECTION & SUMMARY METRICS")
    print("=" * 70)

    num_rows, num_cols = df.shape
    print(f"[+] Total Number of Rows    : {num_rows}")
    print(f"[+] Total Number of Columns : {num_cols}")
    print(f"[+] Column Names           : {list(df.columns)}")

    # Missing values check
    missing_values = df.isnull().sum()
    print("\n[+] Missing Values per Column:")
    for col, count in missing_values.items():
        status_msg = "Clean" if count == 0 else f"{count} missing!"
        print(f"    - {col:<15}: {count} ({status_msg})")

    # Duplicate rows check
    duplicate_count = df.duplicated().sum()
    print(f"[+] Duplicate Rows Count   : {duplicate_count}")

    # Temperature statistics
    min_temp = df["temperature"].min()
    max_temp = df["temperature"].max()
    avg_temp = df["temperature"].mean()

    print("\n[+] Temperature Summary Metrics:")
    print(f"    - Minimum Temperature : {min_temp:.2f} deg C")
    print(f"    - Maximum Temperature : {max_temp:.2f} deg C")
    print(f"    - Average Temperature : {avg_temp:.2f} deg C")


def convert_and_sort_timestamps(df):
    """
    Steps 3 & 4: Convert timestamp string column into datetime format and sort chronologically.
    """
    print("\n" + "=" * 70)
    print("STEPS 3 & 4: TIMESTAMP CONVERSION & SORTING")
    print("=" * 70)

    print("[+] Converting 'timestamp' column to pandas datetime format...")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    print("[+] Sorting dataset chronologically by timestamp...")
    df = df.sort_values(by="timestamp").reset_index(drop=True)

    print("[OK] Timestamp conversion and chronological sorting completed.")
    return df


def apply_rule_based_labeling(df):
    """
    Steps 5 & 6: Create rule-based reference labels:
        - Normal : 2 deg C <= temperature <= 8 deg C
        - Anomaly: temperature < 2 deg C OR temperature > 8 deg C
    Adds a new column named 'status'.
    """
    print("\n" + "=" * 70)
    print("STEPS 5 & 6: RULE-BASED ANOMALY LABELING")
    print("=" * 70)
    print("[+] Rule Definition:")
    print("    - Normal  : Temperature between 2.0 deg C and 8.0 deg C (inclusive)")
    print("    - Anomaly : Temperature < 2.0 deg C OR Temperature > 8.0 deg C")

    # Function to classify each temperature reading
    def classify_status(temp):
        if 2.0 <= temp <= 8.0:
            return "Normal"
        else:
            return "Anomaly"

    df["status"] = df["temperature"].apply(classify_status)

    # Report count breakdown
    status_counts = df["status"].value_counts()
    print("\n[+] Classification Results Summary:")
    for label, count in status_counts.items():
        percentage = (count / len(df)) * 100
        print(f"    - {label:<10}: {count} readings ({percentage:.2f}%)")

    print("\n--- First 5 Rows of Processed Dataset ---")
    print(df[["timestamp", "temperature", "status"]].head())

    return df


def save_processed_dataset(df, output_path):
    """
    Step 8: Save the processed dataset containing the new 'status' column.
    """
    print("\n" + "=" * 70)
    print("STEP 8: SAVING PROCESSED DATASET")
    print("=" * 70)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[OK] Processed dataset saved successfully to: '{output_path}'")


def plot_processed_visualization(df, output_path):
    """
    Steps 7 & 9: Create and save a visualization showing normal vs anomaly readings.
    """
    print("\n" + "=" * 70)
    print("STEPS 7 & 9: VISUALIZING PROCESSED DATA & SAVING GRAPH")
    print("=" * 70)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(14, 6))

    # Plot baseline temperature line
    plt.plot(df["timestamp"], df["temperature"], color="#7f7f7f", linewidth=0.8, alpha=0.7, label="Temperature Line")

    # Separate normal and anomaly points based on the new 'status' column
    normal_df = df[df["status"] == "Normal"]
    anomaly_df = df[df["status"] == "Anomaly"]

    # Scatter plot for Normal readings (Green)
    plt.scatter(normal_df["timestamp"], normal_df["temperature"], 
                color="#2ca02c", s=10, alpha=0.6, label="Normal (2 deg C - 8 deg C)")

    # Scatter plot for Anomaly readings (Red)
    plt.scatter(anomaly_df["timestamp"], anomaly_df["temperature"], 
                color="#d62728", s=20, label="Anomaly (<2 deg C or >8 deg C)", zorder=4)

    # Reference limit lines
    plt.axhline(y=8.0, color="red", linestyle="--", linewidth=1.2, label="Upper Threshold (8 deg C)")
    plt.axhline(y=2.0, color="blue", linestyle="--", linewidth=1.2, label="Lower Threshold (2 deg C)")

    # Highlight reference range
    plt.axhspan(2.0, 8.0, color="green", alpha=0.08, label="Acceptable Reference Range")

    # Graph details
    plt.title("Smart Refrigerator Temperature Preprocessing: Normal vs Anomaly Classification", 
              fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Timestamp (Date & Time)", fontsize=11, labelpad=8)
    plt.ylabel("Temperature (deg C)", fontsize=11, labelpad=8)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()

    # Save output plot image
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"[OK] Processed visualization graph saved to: '{output_path}'")


def main():
    print("\n" + "*" * 70)
    print(" SMART REFRIGERATOR ANOMALY DETECTION - PHASE 2: DATA PREPROCESSING ")
    print("*" * 70 + "\n")

    # Step 1: Load raw dataset
    df = load_dataset(RAW_DATASET_PATH)

    # Step 2: Inspect raw dataset
    inspect_raw_dataset(df)

    # Steps 3 & 4: Convert timestamps & sort
    df = convert_and_sort_timestamps(df)

    # Steps 5 & 6: Apply rule-based threshold labeling ('status')
    df = apply_rule_based_labeling(df)

    # Step 8: Save processed CSV
    save_processed_dataset(df, PROCESSED_DATASET_PATH)

    # Steps 7 & 9: Generate & save visualization plot
    plot_processed_visualization(df, PLOT_OUTPUT_PATH)

    print("\n" + "=" * 70)
    print("PHASE 2 PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"-> Processed CSV saved at: '{PROCESSED_DATASET_PATH}'")
    print(f"-> Visualization saved at : '{PLOT_OUTPUT_PATH}'\n")


if __name__ == "__main__":
    main()
