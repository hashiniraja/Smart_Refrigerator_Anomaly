import os
import numpy as np
import pandas as pd

def generate_refrigerator_dataset(output_path="dataset/temperature_data.csv"):
    """
    Generates a realistic time-series dataset of simulated refrigerator temperature readings.
    Includes normal operating patterns along with 4 distinct abnormal scenarios.
    """
    print("Generating simulated refrigerator temperature dataset...")
    
    # Set random seed for reproducible data generation
    np.random.seed(42)
    
    # Generate 14 days of data with 5-minute sampling interval
    # Total samples = 14 days * 24 hours/day * 12 readings/hour = 4032 records
    start_time = pd.Timestamp("2026-01-01 00:00:00")
    time_steps = 4032
    timestamps = [start_time + pd.Timedelta(minutes=5 * i) for i in range(time_steps)]
    
    # -------------------------------------------------------------
    # BASELINE NORMAL OPERATING PATTERN
    # Refrigerator baseline mean temperature: 3.5 °C
    # Regular compressor cooling cycles (~45 min period = 9 steps of 5 mins)
    # Diurnal ambient temperature variation (24 hour period = 288 steps)
    # Small sensor noise (standard deviation = 0.15 °C)
    # -------------------------------------------------------------
    time_indices = np.arange(time_steps)
    compressor_cycle = 1.0 * np.sin(2 * np.pi * time_indices / 9)
    diurnal_variation = 0.3 * np.sin(2 * np.pi * time_indices / 288)
    sensor_noise = np.random.normal(0, 0.15, size=time_steps)
    
    temperature = 3.5 + compressor_cycle + diurnal_variation + sensor_noise
    
    # Setup labels for training & validation in future phases
    is_anomaly = np.zeros(time_steps, dtype=int)
    anomaly_type = np.array(["Normal"] * time_steps, dtype=object)
    
    # -------------------------------------------------------------
    # INJECT ABNORMAL SCENARIOS
    # -------------------------------------------------------------
    
    # Scenario 1: Sudden Temperature Spike (e.g., Door left open / hot item added)
    # Timeframe: Day 3 (Step 800 to 836 -> ~3 hours)
    spike_indices = range(800, 836)
    for i, idx in enumerate(spike_indices):
        if i < 6:
            # Rapid jump up to ~14 °C
            temperature[idx] += (i + 1) * 1.8
        else:
            # Slow cooling back down after door is closed
            temperature[idx] += max(0, 10.8 - (i - 6) * 0.35)
        is_anomaly[idx] = 1
        anomaly_type[idx] = "Sudden Spike"
        
    # Scenario 2: Gradual Temperature Increase (e.g., Coolant leak / compressor degradation)
    # Timeframe: Day 6 to Day 7 (Step 1750 to 1950 -> ~16.6 hours)
    gradual_indices = range(1750, 1950)
    for i, idx in enumerate(gradual_indices):
        temperature[idx] += (i / len(gradual_indices)) * 7.5  # Drifts from ~3.5°C to ~11°C
        is_anomaly[idx] = 1
        anomaly_type[idx] = "Gradual Increase"
        
    # Scenario 3: Sudden Temperature Drop (e.g., Thermostat lock / over-freezing failure)
    # Timeframe: Day 9 (Step 2600 to 2672 -> ~6 hours)
    drop_indices = range(2600, 2672)
    for idx in drop_indices:
        temperature[idx] -= 6.5 + np.random.normal(0, 0.1)  # Drops to sub-zero (-3°C to -4.5°C)
        is_anomaly[idx] = 1
        anomaly_type[idx] = "Sudden Drop"
        
    # Scenario 4: Repeated Temperature Fluctuations (e.g., Faulty relay / erratic cycling)
    # Timeframe: Day 12 (Step 3450 to 3550 -> ~8.3 hours)
    fluct_indices = range(3450, 3550)
    for i, idx in enumerate(fluct_indices):
        temperature[idx] += 4.0 * np.sin(2 * np.pi * i / 4) + np.random.normal(0, 0.4)
        is_anomaly[idx] = 1
        anomaly_type[idx] = "Rapid Fluctuations"
        
    # Format values nicely
    temperature = np.round(temperature, 2)
    
    # Create DataFrame
    df = pd.DataFrame({
        "timestamp": timestamps,
        "temperature": temperature,
        "is_anomaly": is_anomaly,
        "anomaly_type": anomaly_type
    })
    
    # Ensure directory exists and export to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully created and saved to '{output_path}'")
    print(f"Total rows: {len(df)}")
    print(f"Normal readings: {(df['is_anomaly'] == 0).sum()}")
    print(f"Abnormal readings: {(df['is_anomaly'] == 1).sum()}\n")
    return df

if __name__ == "__main__":
    generate_refrigerator_dataset()
