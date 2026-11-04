"""Dynamic Meta-Fusion (v18).

Implements an adaptive blending strategy to break the 92.83% plateau.
Instead of static weights, it adjusts the contribution of v1, v5, v6, and v4 
based on the predicted demand intensity and model variance.

Logic:
1. Baseline: 30/30/20/20 (v15 champion).
2. Peak Boost: If max(predictions) > 0.7, increase weight of v6 (Shock-Aware) to 40%.
3. Uncertainty Anchor: If std(predictions) > 0.15, increase weight of v4 (Oracle) to 40%.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def main():
    print("Initializing Dynamic Meta-Fusion v18...")
    
    # Define paths
    v1_path = Path("gemini_codes/submissions/gemini_v1_neural_boost_submission.csv")
    v5_path = Path("gemini_codes/pure_neural_boost_v5/gemini_v5_pure_neural_boost.csv")
    v6_path = Path("gemini_codes/anomaly_aware_v6/gemini_v6_anomaly_aware.csv")
    v4_path = Path("gemini_codes/submissions/old/gemini_v14_oracle.csv")
    
    # Load Submissions
    print("Loading component models...")
    v1 = pd.read_csv(v1_path).set_index("Index").rename(columns={"demand": "v1"})
    v5 = pd.read_csv(v5_path).set_index("Index").rename(columns={"demand": "v5"})
    v6 = pd.read_csv(v6_path).set_index("Index").rename(columns={"demand": "v6"})
    v4 = pd.read_csv(v4_path).set_index("Index").rename(columns={"demand": "v4"})
    
    # Align
    df = v1.join([v5, v6, v4], how="inner")
    
    print(f"Total rows to process: {len(df)}")
    
    # Calculate row-wise stats for dynamic weighting
    preds = df[["v1", "v5", "v6", "v4"]]
    df["max_val"] = preds.max(axis=1)
    df["std_val"] = preds.std(axis=1)
    
    # Initialize weights
    # Default (v15): v1=0.3, v5=0.3, v6=0.2, v4=0.2
    w1 = np.full(len(df), 0.30)
    w5 = np.full(len(df), 0.30)
    w6 = np.full(len(df), 0.20)
    w4 = np.full(len(df), 0.20)
    
    # --- DYNAMIC RULES ---
    
    # Rule 1: High Demand Spike Detection
    # If any model sees a spike (>0.7), v6 (Anomaly Aware) is likely the most correct.
    spike_mask = df["max_val"] > 0.7
    w1[spike_mask] = 0.20
    w5[spike_mask] = 0.20
    w6[spike_mask] = 0.40 # Double the voice of the shock model
    w4[spike_mask] = 0.20
    
    # Rule 2: High Uncertainty Detection
    # If models disagree heavily (>0.15 std), we trust the Oracle (v4) more to prevent drift.
    uncertain_mask = df["std_val"] > 0.15
    w1[uncertain_mask] = 0.20
    w5[uncertain_mask] = 0.20
    w6[uncertain_mask] = 0.20
    w4[uncertain_mask] = 0.40 # Double the voice of the seasonal anchor
    
    print(f"Spikes detected: {spike_mask.sum()} rows")
    print(f"Uncertain rows detected: {uncertain_mask.sum()} rows")
    
    # Normalize weights row-wise just in case
    total_w = w1 + w5 + w6 + w4
    w1 /= total_w
    w5 /= total_w
    w6 /= total_w
    w4 /= total_w
    
    # Apply Dynamic Blend
    df["demand"] = (
        df["v1"] * w1 +
        df["v5"] * w5 +
        df["v6"] * w6 +
        df["v4"] * w4
    )
    
    # Final clipping
    df["demand"] = df["demand"].clip(0.0, 1.0)
    
    # Save
    output_path = Path("gemini_codes/submissions/gemini_v18_dynamic_meta_fusion.csv")
    df[["demand"]].reset_index().to_csv(output_path, index=False)
    
    print(f"\nDynamic Meta-Fusion saved to: {output_path}")
    
    # Stats
    print("\nBlend Statistics:")
    print(df["demand"].describe())

if __name__ == "__main__":
    main()
