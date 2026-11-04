"""Thermo-Spike Filter (v31).

Restores the 92.96% SOTA (v19 Golden Ratio) as the absolute baseline.
Applies the thermodynamic findings strictly as a post-prediction multiplier 
to amplify high-demand spikes during extreme heat conditions, without 
disturbing the general traffic predictions.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def main():
    print("Initializing Thermo-Spike Filter (v31)...")
    
    # 1. Load the pristine 92.96% Baseline
    v19_path = Path("gemini_codes/submissions/gemini_v19_golden_ratio.csv")
    print(f"Loading SOTA Baseline: {v19_path}")
    df = pd.read_csv(v19_path)
    
    # 2. Load contextual test data and archetypes
    test_raw = pd.read_csv("dataset/test.csv")
    archetypes = pd.read_csv("gemini_codes/geohash_archetypes.csv")
    
    # Merge context onto the predictions
    df = df.merge(test_raw[["Index", "geohash", "Temperature"]], on="Index")
    df = df.merge(archetypes[["geohash", "archetype_id"]], on="geohash", how="left")
    
    # Fill missing temperatures with a safe non-extreme median (e.g., 25.0)
    df["Temperature"] = df["Temperature"].fillna(25.0)
    
    # 3. Apply Thermo-Spike Logic
    print("Applying targeted extreme-value amplification...")
    
    # Rule A: Extreme Heat Amplification
    # If it is extremely hot (>40C) and it's an Artery (Archetype 1 or 2),
    # the traffic is empirically known to be more severe than the standard model realizes.
    # We apply a power transform to push the peak higher, stretching out the under-predictions.
    
    extreme_heat_mask = (df["Temperature"] > 40.0)
    
    # We identified Archetypes 1 and 2 as the "Arteries" that spike in heat.
    artery_mask = df["archetype_id"].isin([1, 2])
    
    # Target Mask: Hot + Artery
    target_mask = extreme_heat_mask & artery_mask
    
    adjusted_count = target_mask.sum()
    print(f"  -> Found {adjusted_count} rows matching extreme heat/artery conditions.")
    
    if adjusted_count > 0:
        # Push the demand non-linearly higher (e.g., 0.2 ** 0.70 = 0.32)
        original_vals = df.loc[target_mask, "demand"].copy()
        df.loc[target_mask, "demand"] = df.loc[target_mask, "demand"] ** 0.70
        
        # Calculate the average boost
        boost_amount = (df.loc[target_mask, "demand"] - original_vals).mean()
        print(f"  -> Applied average boost of +{boost_amount:.4f} to these peaks.")
        
    # Rule B: Global Peak Preservation
    # Recursive models tend to "average out" and flatten high-demand waves over a 14-hour rollout.
    # We apply a very gentle stretch to the top end of the distribution to counteract this smoothing.
    print("Applying Global Peak Preservation...")
    peak_mask = df["demand"] > 0.40
    peak_count = peak_mask.sum()
    print(f"  -> Found {peak_count} rows in the high-demand tail.")
    if peak_count > 0:
        df.loc[peak_mask, "demand"] = df.loc[peak_mask, "demand"] ** 0.98
    
    # Final clipping
    df["demand"] = df["demand"].clip(0.0, 1.0)
    
    # 4. Save Final Submission
    output_path = Path("gemini_codes/submissions/gemini_v31_thermo_spike_filter.csv")
    df[["Index", "demand"]].sort_values("Index").to_csv(output_path, index=False)
    
    print(f"\nSOTA v31 Submission saved to: {output_path}")
    print("\nBlend Statistics:")
    print(df["demand"].describe())

if __name__ == "__main__":
    main()
