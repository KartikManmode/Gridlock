"""Behavioral Clustering of Geohashes (v16 Phase 1).

Analyzes Day 48 demand profiles for all geohashes and uses K-Means clustering 
to identify functional archetypes (e.g., arteries, residential, leisure).
Saves the mapping to 'gemini_codes/geohash_archetypes.csv'.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def main():
    print("Initializing Behavioral Clustering...")
    
    # Load Data
    train_df = pd.read_csv("dataset/train.csv")
    
    # Extract Hour and Filter for a full day (Day 48)
    train_df["hour"] = train_df["timestamp"].str.split(":").str[0].astype(int)
    day48 = train_df[train_df["day"] == 48].copy()
    
    # Pivot to create a 24-hour demand profile for each geohash
    # Columns: 0, 1, 2, ..., 23 (Hour)
    # Rows: Geohash
    profiles = day48.pivot_table(
        index="geohash", 
        columns="hour", 
        values="demand", 
        aggfunc="mean"
    ).fillna(0.0)
    
    print(f"Generated profiles for {len(profiles)} geohashes.")
    
    # Normalize profiles to focus on SHAPE rather than VOLUME
    # This allows a small artery and a large artery to be in the same cluster.
    scaler = StandardScaler()
    # Profiles scaled along the 24 hours for each geohash
    # .T ensures scaling is done per geohash (row-wise in original)
    profiles_scaled = scaler.fit_transform(profiles.T).T
    
    # Run K-Means
    # We choose 6 clusters as a starting point for archetypes
    n_clusters = 6
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(profiles_scaled)
    
    # Combine results
    result_df = pd.DataFrame({
        "geohash": profiles.index,
        "archetype_id": clusters
    })
    
    # Map back the raw average demand to help describe the clusters
    result_df["avg_demand"] = profiles.mean(axis=1).values
    
    # Save the mapping
    output_path = Path("gemini_codes/geohash_archetypes.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"Saved archetype mapping to: {output_path}")
    
    # Summary of Clusters
    print("\nArchetype Summary:")
    summary = result_df.groupby("archetype_id").agg({
        "geohash": "count",
        "avg_demand": "mean"
    }).rename(columns={"geohash": "node_count"})
    print(summary)
    
    # Interpretation logic (Optional but helpful)
    # We can look at the peak hours for each cluster centroid
    centroids = kmeans.cluster_centers_
    for i in range(n_clusters):
        peak_hour = np.argmax(centroids[i])
        print(f"Cluster {i}: Size={summary.loc[i, 'node_count']}, Peak Hour={peak_hour}:00")

if __name__ == "__main__":
    main()
