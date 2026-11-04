"""Ultra-Blend (v14).

Fuses three distinct strategies to break the 92.56% plateau:
1. v5 (Robust Neural-Boost): 40% - The stable baseline.
2. v6 (Anomaly-Aware): 40% - Catching the high-demand spikes.
3. v4 (Oracle Baseline): 20% - The Day 48 seasonal anchor.

Final Weights: 0.40*v5 + 0.40*v6 + 0.20*v4
"""

import pandas as pd
import numpy as np
from pathlib import Path

def main():
    print("Initializing Ultra-Blend v14...")
    
    # Define paths
    v5_path = Path("gemini_codes/pure_neural_boost_v5/gemini_v5_pure_neural_boost.csv")
    v6_path = Path("gemini_codes/anomaly_aware_v6/gemini_v6_anomaly_aware.csv")
    v4_path = Path("gemini_codes/submissions/old/gemini_v14_oracle.csv")
    
    # Load Submissions
    print(f"Loading v5 (Robust) from {v5_path}")
    v5 = pd.read_csv(v5_path).set_index("Index").rename(columns={"demand": "v5"})
    
    print(f"Loading v6 (Anomaly) from {v6_path}")
    v6 = pd.read_csv(v6_path).set_index("Index").rename(columns={"demand": "v6"})
    
    print(f"Loading v4 (Oracle) from {v4_path}")
    v4 = pd.read_csv(v4_path).set_index("Index").rename(columns={"demand": "v4"})
    
    # Align
    blend_df = v5.join([v6, v4], how="inner")
    
    # Define Weights
    w5, w6, w4 = 0.40, 0.40, 0.20
    
    print(f"Applying Ultra-Blend ratios: v5={w5}, v6={w6}, v4={w4}")
    
    blend_df["demand"] = (
        blend_df["v5"] * w5 +
        blend_df["v6"] * w6 +
        blend_df["v4"] * w4
    )
    
    # Final clipping
    blend_df["demand"] = blend_df["demand"].clip(0.0, 1.0)
    
    # Save
    output_path = Path("gemini_codes/submissions/gemini_v14_ultra_blend.csv")
    blend_df[["demand"]].reset_index().to_csv(output_path, index=False)
    
    print(f"\nUltra-Blend saved to: {output_path}")
    
    # Quick Stats
    print("\nBlend Statistics:")
    print(blend_df["demand"].describe())

if __name__ == "__main__":
    main()
