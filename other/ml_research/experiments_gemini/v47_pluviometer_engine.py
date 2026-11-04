"""Pluviometer Engine (v47).

Surgically refines the 93.065% SOTA (v34_G) by applying a bifurcated 
Rainy Weather adjustment based on the Core vs. Non-Core city map.

Rules:
1. Core City + Rainy: -1.4% dampening (Safety Thinning).
2. Non-Core (Mid/Outer) + Rainy: +5% boost (Mode-Shift Surge).
"""

import pandas as pd
import numpy as np
from pathlib import Path

def main():
    print("Initializing Pluviometer Engine v47...")
    
    # 1. Load the 93.065% Champion
    # (v34_gradient_G_093_power.csv contains the Golden Ratio blend + 0.93 Thermo-Spike)
    v34_path = Path("gemini_codes/submissions/gemini_v34_gradient_G_093_power.csv")
    if not v34_path.exists():
        v34_path = Path("gemini_codes/submissions/old/gemini_v34_gradient_G_093_power.csv")
        
    print(f"Loading champion from {v34_path}")
    df = pd.read_csv(v34_path)
    
    # 2. Load context: Weather and City Zones
    test_raw = pd.read_csv("dataset/test.csv")
    city_zones = pd.read_csv("gemini_codes/geohash_city_zones.csv")
    
    # Merge for conditional rules
    df = df.merge(test_raw[["Index", "geohash", "Weather"]], on="Index")
    df = df.merge(city_zones[["geohash", "zone_name"]], on="geohash", how="left")
    
    # Fill missing weather
    df["Weather"] = df["Weather"].fillna("Unknown")
    
    # 3. Apply Bifurcated Rainy Logic
    print("Applying Spatially-Conditional Rainy Adjustments...")
    
    # Rule 1: Core City Safety Thinning (-1.4%)
    core_rain_mask = (df["zone_name"] == "Core City") & (df["Weather"] == "Rainy")
    df.loc[core_rain_mask, "demand"] *= 0.986
    
    # Rule 2: Non-Core Mode-Shift Surge (+5%)
    non_core_rain_mask = (df["zone_name"] != "Core City") & (df["Weather"] == "Rainy")
    df.loc[non_core_rain_mask, "demand"] *= 1.05
    
    print(f"  -> Dampened {core_rain_mask.sum()} Core-Rainy rows.")
    print(f"  -> Boosted {non_core_rain_mask.sum()} Non-Core-Rainy rows.")
    
    # Final clipping
    df["demand"] = df["demand"].clip(0.0, 1.0)
    
    # 4. Save Final Refined SOTA
    output_path = Path("gemini_codes/submissions/gemini_v47_pluviometer.csv")
    df[["Index", "demand"]].sort_values("Index").to_csv(output_path, index=False)
    
    print(f"\nRefined SOTA v47 saved to: {output_path}")
    print(df["demand"].describe())

if __name__ == "__main__":
    main()
