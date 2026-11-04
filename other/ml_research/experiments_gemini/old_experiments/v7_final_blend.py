"""Final Meta-Blend (v7).

Aggregates three distinct architectural strategies:
1. v1: High-Signal Neural-Boost (SOTA 92.11%)
2. v5: Robust Pure Neural-Boost (92.02%)
3. v6: Anomaly-Aware / Shock-Aware Neural-Boost (91.58%)

Weighted Average: 0.40 * v1 + 0.40 * v5 + 0.20 * v6
"""

import pandas as pd
import numpy as np
from pathlib import Path

def main():
    print("Initializing Final Meta-Blend...")
    
    # Define paths
    v1_path = Path("gemini_codes/submissions/gemini_v1_neural_boost_submission.csv")
    v5_path = Path("gemini_codes/pure_neural_boost_v5/gemini_v5_pure_neural_boost.csv")
    v6_path = Path("gemini_codes/anomaly_aware_v6/gemini_v6_anomaly_aware.csv")
    
    # Load Submissions
    print(f"Loading v1 from {v1_path}")
    v1 = pd.read_csv(v1_path).set_index("Index").rename(columns={"demand": "v1"})
    
    print(f"Loading v5 from {v5_path}")
    v5 = pd.read_csv(v5_path).set_index("Index").rename(columns={"demand": "v5"})
    
    print(f"Loading v6 from {v6_path}")
    v6 = pd.read_csv(v6_path).set_index("Index").rename(columns={"demand": "v6"})
    
    # Align
    blend_df = v1.join([v5, v6], how="inner")
    
    # Define Weights
    # v1 and v5 are our strongest average predictors.
    # v6 provides the "shock absorption" for spikes.
    w1, w5, w6 = 0.40, 0.40, 0.20
    
    print(f"Applying weights: v1={w1}, v5={w5}, v6={w6}")
    
    blend_df["demand"] = (
        blend_df["v1"] * w1 +
        blend_df["v5"] * w5 +
        blend_df["v6"] * w6
    )
    
    # Final clipping
    blend_df["demand"] = blend_df["demand"].clip(0.0, 1.0)
    
    # Save
    output_path = Path("gemini_codes/submissions/gemini_v7_final_meta_blend.csv")
    blend_df[["demand"]].reset_index().to_csv(output_path, index=False)
    
    print(f"\nFinal Meta-Blend saved to: {output_path}")
    
    # Quick Stats
    print("\nBlend Statistics:")
    print(blend_df["demand"].describe())

if __name__ == "__main__":
    main()
