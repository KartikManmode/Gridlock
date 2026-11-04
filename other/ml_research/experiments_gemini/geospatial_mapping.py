"""Geospatial Mapping and Neighborhood Relations.

Analyzes the static structural topology of the Geohash grid.
Calculates graph centrality, clustering coefficients, and physical distances 
to find relationships between neighbors without relying on dynamic traffic data.
"""

import pandas as pd
import numpy as np
import geohash
import networkx as nx
from pathlib import Path

def decode_geohash(value):
    lat, lon = geohash.decode(str(value))
    return float(lat), float(lon)

def main():
    print("Loading Geohash coordinates from training data...")
    df = pd.read_csv("dataset/train.csv", usecols=["geohash"]).drop_duplicates()
    all_gh = set(df["geohash"].tolist())
    
    print(f"Total Unique Geohashes: {len(all_gh)}")
    
    # 1. Build the Adjacency Graph
    G = nx.Graph()
    for gh in all_gh:
        G.add_node(gh)
        
    edges = 0
    for gh in all_gh:
        neighbors = geohash.neighbors(gh)
        for n in neighbors:
            if n in all_gh:
                G.add_edge(gh, n)
                edges += 1
                
    print(f"Total Connections (Edges): {edges // 2}") # Undirected
    
    # 2. Structural Analysis
    print("\n--- Structural Analysis ---")
    
    # Degree (How many active neighbors does a cell have?)
    degrees = dict(G.degree())
    df["degree"] = df["geohash"].map(degrees)
    print(f"Average Neighbors: {df['degree'].mean():.2f}")
    
    # Eigenvector Centrality (Is this geohash a central hub connecting regions?)
    print("Calculating Eigenvector Centrality (identifying structural hubs)...")
    centrality = nx.eigenvector_centrality(G, max_iter=1000)
    df["centrality"] = df["geohash"].map(centrality)
    
    # Clustering Coefficient (Are a cell's neighbors also neighbors with each other?)
    print("Calculating Clustering Coefficient...")
    clustering = nx.clustering(G)
    df["clustering_coeff"] = df["geohash"].map(clustering)
    
    # PageRank (Does traffic *have* to pass through here?)
    print("Calculating PageRank (bottleneck identification)...")
    pagerank = nx.pagerank(G)
    df["pagerank"] = df["geohash"].map(pagerank)
    
    # 3. Save Node Features
    out_path = Path("gemini_codes/geohash_structural_features.csv")
    df.to_csv(out_path, index=False)
    
    print(f"\nStructural mapping complete. Saved to: {out_path}")
    print("\nSummary Statistics of Structural Relations:")
    print(df[["degree", "centrality", "clustering_coeff", "pagerank"]].describe())

if __name__ == "__main__":
    main()
