"""Five-Star Meta-Ensemble (v15).

Combines all successful strategies into a single unified prediction:
1. v1 (Base Neural-Boost): 30% - The original raw SOTA.
2. v5 (Robust Neural-Boost): 30% - High-resolution physics.
3. v6 (Anomaly-Aware): 20% - Catching spikes.
4. v4 (Oracle Baseline): 20% - Seasonal anchor (Day 48).

Final Weights: 0.30*v1 + 0.30*v5 + 0.20*v6 + 0.20*v4
"""

import pandas as pd
import numpy as np
from pathlib import Path

def main():
    print("Initializing Five-Star Meta-Ensemble v15...")
    
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
    blend_df = v1.join([v5, v6, v4], how="inner")
    
    # Define Weights
    w1, w5, w6, w4 = 0.30, 0.30, 0.20, 0.20
    
    print(f"Applying Five-Star Weights: v1={w1}, v5={w5}, v6={w6}, v4={w4}")
    
    blend_df["demand"] = (
        blend_df["v1"] * w1 +
        blend_df["v5"] * w5 +
        blend_df["v6"] * w6 +
        blend_df["v4"] * w4
    )
    
    # Final clipping
    blend_df["demand"] = blend_df["demand"].clip(0.0, 1.0)
    
    # Save
    output_path = Path("gemini_codes/submissions/gemini_v15_fivestar_blend.csv")
    blend_df[["demand"]].reset_index().to_csv(output_path, index=False)
    
    print(f"\nFive-Star Meta-Ensemble saved to: {output_path}")
    
    # Quick Stats
    print("\nBlend Statistics:")
    print(blend_df["demand"].describe())

if __name__ == "__main__":
    main()
