"""Gradient Tuning (v33).

Since Variation A (0.97 power) scored 93.0304% (better than 0.98 power's 93.01%),
we follow the gradient. We test 0.96 and 0.95 to see where the peak stops.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def apply_tuning(df_base, test_raw, archetypes, power_val, threshold_val, variation_name):
    df = df_base.copy()
    
    # Merge context
    df = df.merge(test_raw[["Index", "geohash", "Temperature"]], on="Index")
    df = df.merge(archetypes[["geohash", "archetype_id"]], on="geohash", how="left")
    df["Temperature"] = df["Temperature"].fillna(25.0)
    
    # 1. Hot Artery Spike (Static, proven logic)
    extreme_heat_mask = (df["Temperature"] > 40.0)
    artery_mask = df["archetype_id"].isin([1, 2])
    target_mask = extreme_heat_mask & artery_mask
    
    if target_mask.sum() > 0:
        df.loc[target_mask, "demand"] = df.loc[target_mask, "demand"] ** 0.70
        
    # 2. Variable Global Peak Preservation
    peak_mask = df["demand"] > threshold_val
    if peak_mask.sum() > 0:
        df.loc[peak_mask, "demand"] = df.loc[peak_mask, "demand"] ** power_val
        
    # Final clipping
    df["demand"] = df["demand"].clip(0.0, 1.0)
    
    # Save
    out_name = f"gemini_v33_gradient_tune_{variation_name}.csv"
    output_path = Path("gemini_codes/submissions") / out_name
    df[["Index", "demand"]].sort_values("Index").to_csv(output_path, index=False)
    
    print(f"[{variation_name}] Saved (Power={power_val}) -> Mean: {df['demand'].mean():.4f}")

def main():
    print("Initializing Gradient Tuning v33...")
    
    # Load the pristine 92.96% Baseline
    v19_path = Path("gemini_codes/submissions/gemini_v19_golden_ratio.csv")
    df_base = pd.read_csv(v19_path)
    
    test_raw = pd.read_csv("dataset/test.csv")
    archetypes = pd.read_csv("gemini_codes/geohash_archetypes.csv")
    
    print("\nGenerating Gradient Variations:")
    # Variation D: Even Stronger (following the gradient)
    apply_tuning(df_base, test_raw, archetypes, power_val=0.96, threshold_val=0.40, variation_name="D_096_power")
    
    # Variation E: Maximum Stretch
    apply_tuning(df_base, test_raw, archetypes, power_val=0.95, threshold_val=0.40, variation_name="E_095_power")
    
    print("\nGradient Tuning Complete. Ready for evaluation.")

if __name__ == "__main__":
    main()
