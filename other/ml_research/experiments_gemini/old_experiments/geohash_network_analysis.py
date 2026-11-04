"""Geohash Network Analysis.

Analyzes the spatial connectivity and traffic flow between neighboring geohashes.
Goal: Understand how traffic propagates through the physical grid to engineer
continuous, non-overfitting spatial features.
"""

import pandas as pd
import numpy as np
import geohash
from scipy.stats import pearsonr

def main():
    print("Loading Training Data...")
    df = pd.read_csv("dataset/train.csv")
    
    print("\n--- 1. Grid Connectivity Analysis ---")
    all_gh = df["geohash"].unique()
    print(f"Total Unique Geohashes: {len(all_gh)}")
    
    # Calculate how many valid neighbors each geohash has in our dataset
    gh_to_neighbors = {gh: [n for n in geohash.neighbors(gh) if n in all_gh] for gh in all_gh}
    
    neighbor_counts = [len(n) for n in gh_to_neighbors.values()]
    print(f"Average active neighbors per cell: {np.mean(neighbor_counts):.2f}")
    print(f"Cells with 0 active neighbors (Islands): {neighbor_counts.count(0)}")
    print(f"Cells with all 8 active neighbors (Dense Core): {neighbor_counts.count(8)}")
    
    print("\n--- 2. Spatial-Temporal Correlation (The Flow of Traffic) ---")
    # To see if traffic flows from A to B, we need to compare a cell at T 
    # to its neighbors at T and T-1.
    
    # Create a time index
    df["hour"] = df["timestamp"].str.split(":").str[0].astype(int)
    df["minute"] = df["timestamp"].str.split(":").str[1].astype(int)
    df["slot_id"] = df["day"] * 96 + df["hour"] * 4 + df["minute"] // 15
    
    df = df.sort_values(["geohash", "slot_id"]).reset_index(drop=True)
    
    # Create lookup for fast neighbor querying
    lookup = df.set_index(["geohash", "slot_id"])["demand"].to_dict()
    
    # Sample a few active time slots to calculate correlation
    sample_slots = df["slot_id"].unique()[10:20] 
    
    sync_corrs = []  # Correlation at same time T
    flow_corrs = []  # Correlation of Neighbors at T-1 to Center at T
    
    for slot in sample_slots:
        center_vals = []
        sync_neighbor_vals = []
        flow_neighbor_vals = []
        
        for gh in all_gh:
            center_val = lookup.get((gh, slot), np.nan)
            if pd.isna(center_val): continue
                
            neighbors = gh_to_neighbors[gh]
            if not neighbors: continue
                
            # Sync (Same time)
            n_sync = [lookup.get((n, slot), np.nan) for n in neighbors]
            n_sync = [v for v in n_sync if not pd.isna(v)]
            
            # Flow (Previous time step)
            n_flow = [lookup.get((n, slot - 1), np.nan) for n in neighbors]
            n_flow = [v for v in n_flow if not pd.isna(v)]
            
            if n_sync and n_flow:
                center_vals.append(center_val)
                sync_neighbor_vals.append(np.mean(n_sync))
                flow_neighbor_vals.append(np.mean(n_flow))
                
        if len(center_vals) > 100:
            sync_r, _ = pearsonr(center_vals, sync_neighbor_vals)
            flow_r, _ = pearsonr(center_vals, flow_neighbor_vals)
            sync_corrs.append(sync_r)
            flow_corrs.append(flow_r)
            
    print(f"Average Synchronous Spatial Correlation (Center at T vs Neighbors at T):   {np.mean(sync_corrs):.4f}")
    print(f"Average Flow/Momentum Correlation (Center at T vs Neighbors at T-1): {np.mean(flow_corrs):.4f}")

if __name__ == "__main__":
    main()
