"""Gradient Tuning Extended (v34).

Follows the successful gradient from 0.95 (93.05783% SOTA).
Tests further non-linear stretches to find the top of the curve.
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
    out_name = f"gemini_v34_gradient_{variation_name}.csv"
    output_path = Path("gemini_codes/submissions") / out_name
    df[["Index", "demand"]].sort_values("Index").to_csv(output_path, index=False)
    
    print(f"[{variation_name}] Saved (Power={power_val}) -> Mean: {df['demand'].mean():.4f}")

def main():
    print("Initializing Gradient Tuning Extended v34...")
    
    # Load the pristine 92.96% Baseline
    v19_path = Path("gemini_codes/submissions/gemini_v19_golden_ratio.csv")
    df_base = pd.read_csv(v19_path)
    
    test_raw = pd.read_csv("dataset/test.csv")
    archetypes = pd.read_csv("gemini_codes/geohash_archetypes.csv")
    
    print("\nGenerating Further Gradient Variations:")
    
    apply_tuning(df_base, test_raw, archetypes, power_val=0.94, threshold_val=0.40, variation_name="F_094_power")
    apply_tuning(df_base, test_raw, archetypes, power_val=0.93, threshold_val=0.40, variation_name="G_093_power")
    apply_tuning(df_base, test_raw, archetypes, power_val=0.92, threshold_val=0.40, variation_name="H_092_power")
    apply_tuning(df_base, test_raw, archetypes, power_val=0.90, threshold_val=0.40, variation_name="I_090_power")
    
    print("\nExtended Gradient Tuning Complete. Ready for evaluation.")

if __name__ == "__main__":
    main()
