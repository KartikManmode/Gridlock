"""Golden Ratio & Peak Stabilization (v19).

Refines the v15 champion (92.83%) by:
1. Tuned Ratio: 35% v1, 25% v5, 20% v6, 20% v4.
2. Global Hourly Anchoring: Prevents over-shooting by anchoring total predicted 
   hourly demand to the Day 48 baseline.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def main():
    print("Initializing Golden Ratio Blend v19...")
    
    # Define paths
    v1_path = Path("gemini_codes/submissions/gemini_v1_neural_boost_submission.csv")
    v5_path = Path("gemini_codes/pure_neural_boost_v5/gemini_v5_pure_neural_boost.csv")
    v6_path = Path("gemini_codes/anomaly_aware_v6/gemini_v6_anomaly_aware.csv")
    v4_path = Path("gemini_codes/submissions/old/gemini_v14_oracle.csv")
    
    # Load components
    v1 = pd.read_csv(v1_path).set_index("Index").rename(columns={"demand": "v1"})
    v5 = pd.read_csv(v5_path).set_index("Index").rename(columns={"demand": "v5"})
    v6 = pd.read_csv(v6_path).set_index("Index").rename(columns={"demand": "v6"})
    v4 = pd.read_csv(v4_path).set_index("Index").rename(columns={"demand": "v4"})
    
    # Align
    df = v1.join([v5, v6, v4], how="inner")
    
    # Apply Golden Ratio
    w1, w5, w6, w4 = 0.35, 0.25, 0.20, 0.20
    print(f"Applying Golden Ratio: v1={w1}, v5={w5}, v6={w6}, v4={w4}")
    
    df["demand"] = (
        df["v1"] * w1 +
        df["v5"] * w5 +
        df["v6"] * w6 +
        df["v4"] * w4
    )
    
    # --- PEAK STABILIZATION (Global Physical Anchor) ---
    print("Applying Global Peak Stabilization...")
    
    # We need the hours from the test set to do hourly anchoring
    test_raw = pd.read_csv("dataset/test.csv").set_index("Index")
    test_raw["hour"] = test_raw["timestamp"].str.split(":").str[0].astype(int)
    
    df = df.join(test_raw[["hour"]], how="left")
    
    # Load Day 48 baseline for hourly total demand
    train_raw = pd.read_csv("dataset/train.csv")
    train_raw["hour"] = train_raw["timestamp"].str.split(":").str[0].astype(int)
    day48_hourly_mean = train_raw[train_raw["day"] == 48].groupby("hour")["demand"].mean()
    
    # Calculate current predicted hourly means
    predicted_hourly_mean = df.groupby("hour")["demand"].mean()
    
    # Anchoring Logic:
    # If a predicted hour is more than 15% higher than Day 48, pull it back 5%
    for hour in df["hour"].unique():
        if hour in day48_hourly_mean.index:
            ratio = predicted_hourly_mean[hour] / day48_hourly_mean[hour]
            if ratio > 1.15:
                # Apply a slight dampening factor to the whole hour
                mask = df["hour"] == hour
                df.loc[mask, "demand"] *= 0.95
                print(f"  -> Dampened Hour {hour}: Ratio was {ratio:.2f}")
    
    # Final clipping
    df["demand"] = df["demand"].clip(0.0, 1.0)
    
    # Save
    output_path = Path("gemini_codes/submissions/gemini_v19_golden_ratio.csv")
    df[["demand"]].reset_index().to_csv(output_path, index=False)
    
    print(f"\nGolden Ratio Blend saved to: {output_path}")
    print(df["demand"].describe())

if __name__ == "__main__":
    main()
