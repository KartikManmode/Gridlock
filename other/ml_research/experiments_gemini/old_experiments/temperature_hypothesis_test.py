"""Thermodynamic Hypothesis Testing.

Tests four temperature-based hypotheses on the Gridlock training dataset:
1. Comfort Threshold (Inverted-U)
2. Heat Accumulation (Rolling Means)
3. Weather-Temp Coupling
4. Archetype-Temperature Interaction
"""

import pandas as pd
import numpy as np
from pathlib import Path

def main():
    print("Loading Training Data...")
    df = pd.read_csv("dataset/train.csv")
    archetypes = pd.read_csv("gemini_codes/geohash_archetypes.csv")
    df = df.merge(archetypes, on="geohash", how="left")
    
    df["Temperature"] = df["Temperature"].fillna(df["Temperature"].median())
    df["Weather"] = df["Weather"].fillna("Unknown")
    df["hour"] = df["timestamp"].str.split(":").str[0].astype(int)
    
    print("\n==================================================")
    print("HYPOTHESIS 1: The 'Comfort Threshold' (Inverted-U)")
    print("==================================================")
    # Bin temperatures to see demand curves
    df["temp_bin"] = pd.cut(df["Temperature"], bins=10)
    temp_profile = df.groupby("temp_bin")["demand"].mean()
    print("Average Demand by Temperature Range:")
    print(temp_profile)
    
    print("\nDoes demand peak around 25°C and drop at extremes?")
    peak_temp_bin = temp_profile.idxmax()
    print(f"-> Peak Demand occurs at: {peak_temp_bin}")
    
    
    print("\n==================================================")
    print("HYPOTHESIS 2: Weather-Temperature Coupling")
    print("==================================================")
    # Does 25C in Rain feel different than 25C in Clear?
    weather_temp_profile = df.groupby(["Weather", "temp_bin"])["demand"].mean().unstack()
    print("Demand Matrix: Weather vs. Temperature Range:")
    print(weather_temp_profile)
    
    
    print("\n==================================================")
    print("HYPOTHESIS 3: Thermo-Spatial Interactions (Archetypes)")
    print("==================================================")
    # Do Arteries react differently to heat than Residential zones?
    arch_temp_profile = df.groupby(["archetype_id", "temp_bin"])["demand"].mean().unstack()
    print("Demand Matrix: Archetype vs. Temperature Range:")
    print(arch_temp_profile)
    
    print("\n==================================================")
    print("HYPOTHESIS 4: Heat Accumulation (Rolling Means)")
    print("==================================================")
    # Are temperatures stable enough in this 2-day dataset to matter?
    print("Temperature Summary Stats:")
    print(f"Min Temp: {df['Temperature'].min():.2f}")
    print(f"Max Temp: {df['Temperature'].max():.2f}")
    print(f"Std Dev:  {df['Temperature'].std():.2f}")
    
    # Check if Temperature changes hour-by-hour
    hourly_temp = df.groupby(["day", "hour"])["Temperature"].mean()
    print("\nHourly Temperature Profile:")
    print(hourly_temp.head(12))

if __name__ == "__main__":
    main()
