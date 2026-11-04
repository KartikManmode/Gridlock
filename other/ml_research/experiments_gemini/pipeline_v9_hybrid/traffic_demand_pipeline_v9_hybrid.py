"""Traffic demand prediction pipeline v9 - Hybrid GNN + GBDT.

This script resolves the catastrophic recursive drift of v8 by returning to a 
direct-prediction Gradient Boosting architecture, while using a PyTorch Graph 
Neural Network (GCN) to extract structural spatial embeddings for each geohash.

Architecture:
1. GNN: PyTorch GCN trained to predict average demand using physical geohash adjacency.
2. Feature Extractor: GNN hidden layer (16 dims) extracted as static spatial features.
3. GBDT Ensemble: LightGBM + CatBoost trained on tabular + GNN features for direct prediction.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gridlock")

import catboost as cb
import geohash
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler


RANDOM_STATE = 42
TARGET = "demand"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_timestamp(value: str) -> tuple[int, int]:
    hour, minute = str(value).split(":")
    return int(hour), int(minute)


def decode_geohash(value: str) -> tuple[float, float]:
    lat, lon = geohash.decode(str(value))
    return float(lat), float(lon)


# ==========================================
# PHASE 1: PYTORCH GNN FOR SPATIAL EMBEDDING
# ==========================================

class GCN(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 16):
        super().__init__()
        self.gc1 = nn.Linear(in_features, hidden_dim)
        self.gc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Simple graph convolution: D^-0.5 A D^-0.5 X W
        h1 = torch.relu(self.gc1(torch.matmul(adj, x)))
        emb = torch.relu(self.gc2(torch.matmul(adj, h1)))
        pred = self.out(emb)
        return emb, pred


def train_and_extract_gnn_embeddings(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Trains a GCN on the physical map of geohashes and extracts node embeddings."""
    print("\n--- Training PyTorch GNN for Spatial Embeddings ---")
    
    # Define Graph Nodes
    all_gh = sorted(list(set(train_df["geohash"].unique()) | set(test_df["geohash"].unique())))
    gh_to_idx = {gh: i for i, gh in enumerate(all_gh)}
    N = len(all_gh)
    
    # 1. Adjacency Matrix
    print(f"Building {N}x{N} Graph Adjacency Matrix...")
    adj = np.eye(N, dtype=np.float32)
    for gh in all_gh:
        idx = gh_to_idx[gh]
        for n_gh in geohash.neighbors(gh):
            if n_gh in gh_to_idx:
                adj[idx, gh_to_idx[n_gh]] = 1.0
                
    # Normalize Adjacency
    deg = adj.sum(axis=1)
    d_inv_sqrt = np.power(np.maximum(deg, 1.0), -0.5)
    adj_norm = d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]
    
    device = torch.device("cpu") # Force CPU for GNN to avoid MPS segfaults
    adj_t = torch.from_numpy(adj_norm).float().to(device)

    # 2. Node Features
    X = np.zeros((N, 3), dtype=np.float32)
    gh_counts = train_df["geohash"].value_counts().to_dict()
    for gh in all_gh:
        lat, lon = decode_geohash(gh)
        count = gh_counts.get(gh, 0.0)
        X[gh_to_idx[gh]] = [lat, lon, count]
        
    X = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-6)
    X_t = torch.from_numpy(X).float().to(device)

    # 3. Targets (Predict local mean demand to force structural learning)
    Y = np.zeros(N, dtype=np.float32)
    mask = np.zeros(N, dtype=np.bool_)
    
    gh_means = train_df.groupby("geohash")[TARGET].mean().to_dict()
    global_mean = float(train_df[TARGET].mean())
    
    for gh in all_gh:
        if gh in gh_means:
            Y[gh_to_idx[gh]] = gh_means[gh]
            mask[gh_to_idx[gh]] = True
        else:
            Y[gh_to_idx[gh]] = global_mean
            
    Y_t = torch.from_numpy(Y).float().unsqueeze(1).to(device)
    mask_t = torch.from_numpy(mask).bool().to(device)

    # 4. Train GCN
    model = GCN(in_features=3, hidden_dim=16).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.02)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(1, 201):
        opt.zero_grad()
        emb, pred = model(X_t, adj_t)
        loss = criterion(pred[mask_t], Y_t[mask_t])
        loss.backward()
        opt.step()
        if epoch % 50 == 0:
            print(f"  GNN Epoch {epoch:03d} | Structural Loss: {loss.item():.6f}")

    # 5. Extract Embeddings
    model.eval()
    with torch.no_grad():
        final_emb, _ = model(X_t, adj_t)
        emb_np = final_emb.numpy()
        
    emb_dict = {gh: emb_np[gh_to_idx[gh]] for gh in all_gh}
    print("GNN Embedding Extraction Complete.")
    return emb_dict


# ==========================================
# PHASE 2: TABULAR GBDT + HYBRID PIPELINE
# ==========================================

class HybridFeatureEngineer:
    """Replicates the 91.7% base logic while injecting the 16 GNN dimensions."""
    def __init__(self, gnn_embeddings: dict[str, np.ndarray], smoothing: float = 20.0):
        self.gnn_embeddings = gnn_embeddings
        self.smoothing = smoothing
        self.road_encoder = LabelEncoder()
        self.weather_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        self.global_demand = 0.5
        self.geo5_hour_mean = {}
        self.geo5_hour_count = {}
        
        self.num_cols = ["Temperature", "NumberofLanes", "lat", "lon", "lat_lon_product"]

    def fit(self, df: pd.DataFrame) -> "HybridFeatureEngineer":
        work = self._basic(df)
        self.global_demand = float(work[TARGET].mean())
        
        self.road_encoder.fit(work["RoadType"].astype(str))
        self.weather_encoder.fit(work["Weather"].astype(str))
        
        stats = work.groupby("geo5_hour")[TARGET].agg(["mean", "count"])
        self.geo5_hour_mean = stats["mean"].to_dict()
        self.geo5_hour_count = stats["count"].to_dict()
        
        feat = self.transform(df)
        self.scaler.fit(feat[self.num_cols].fillna(0))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        work = self._basic(df)
        
        # 1. Spatio-Temporal Target Encoding (The 91.7% secret weapon)
        m = work["geo5_hour"].map(self.geo5_hour_mean).fillna(self.global_demand)
        c = work["geo5_hour"].map(self.geo5_hour_count).fillna(0.0)
        work["geo5_hour_te"] = (c * m + self.smoothing * self.global_demand) / (c + self.smoothing)
        
        # 2. Native Categoricals
        work["RoadType"] = self.road_encoder.transform(work["RoadType"].astype(str))
        work["Weather"] = self.weather_encoder.transform(work["Weather"].astype(str))
        
        # 3. Inject GNN Embeddings
        gnn_feats = np.zeros((len(work), 16), dtype=np.float32)
        for i, gh in enumerate(work["geohash"]):
            gnn_feats[i] = self.gnn_embeddings.get(gh, np.zeros(16))
            
        for dim in range(16):
            work[f"gnn_emb_{dim}"] = gnn_feats[:, dim]
            
        # Compile final matrix
        cols = [
            "day", "hour", "minute_of_day", "hour_sin", "hour_cos", "minute_sin", "minute_cos",
            "is_rush_hour", "is_night", "Temperature", "NumberofLanes", "lat", "lon", "lat_lon_product",
            "geo5_hour_te", "RoadType", "Weather"
        ] + [f"gnn_emb_{d}" for d in range(16)]
        
        res = work[cols].copy().fillna(0)
        if hasattr(self.scaler, "mean_"):
            res[self.num_cols] = self.scaler.transform(res[self.num_cols])
            
        return res

    def _basic(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        work[["hour", "minute"]] = work["timestamp"].apply(parse_timestamp).tolist()
        work[["lat", "lon"]] = work["geohash"].apply(decode_geohash).tolist()
        
        work["lat_lon_product"] = work["lat"] * work["lon"]
        work["minute_of_day"] = work["hour"] * 60 + work["minute"]
        
        work["hour_sin"] = np.sin(2 * np.pi * work["hour"] / 24)
        work["hour_cos"] = np.cos(2 * np.pi * work["hour"] / 24)
        work["minute_sin"] = np.sin(2 * np.pi * work["minute_of_day"] / 1440)
        work["minute_cos"] = np.cos(2 * np.pi * work["minute_of_day"] / 1440)
        
        work["is_rush_hour"] = work["hour"].between(7, 9) | work["hour"].between(17, 19)
        work["is_night"] = (work["hour"] >= 22) | (work["hour"] <= 6)
        
        work["geohash5"] = work["geohash"].astype(str).str[:5]
        work["geo5_hour"] = work["geohash5"] + "__h" + work["hour"].astype(str)
        
        work["RoadType"] = work["RoadType"].fillna("Unknown")
        work["Weather"] = work["Weather"].fillna("Unknown")
        work["Temperature"] = work["Temperature"].fillna(25.0)
        return work


def find_optimal_weights(preds_list: list[np.ndarray], y_true: np.ndarray) -> np.ndarray:
    def loss_function(weights: np.ndarray) -> float:
        nw = weights / np.sum(weights)
        blend = np.sum([w * p for w, p in zip(nw, preds_list)], axis=0)
        return float(np.sqrt(mean_squared_error(y_true, np.clip(blend, 0.0, 1.0))))

    res = minimize(loss_function, np.ones(len(preds_list)) / len(preds_list), 
                   bounds=[(0, 1)] * len(preds_list), method="Nelder-Mead")
    return res.x / np.sum(res.x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("dataset/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("dataset/test.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/gemini_v9_hybrid_submission.csv"))
    args = parser.parse_args()

    seed_everything(RANDOM_STATE)
    
    print("Loading Data...")
    df = pd.read_csv(args.train).sort_values(["day", "timestamp"])
    test_df = pd.read_csv(args.test)
    
    # 1. GNN Embeddings
    gnn_embeddings = train_and_extract_gnn_embeddings(df, test_df)

    # 2. TimeSeries CV (To ensure robust weights)
    print("\n--- Running TimeSeries CV on Hybrid Matrix ---")
    tscv = TimeSeriesSplit(n_splits=3)
    lgbm_oof = np.zeros(len(df))
    cat_oof = np.zeros(len(df))
    valid_indices = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(df), 1):
        tr_df, va_df = df.iloc[train_idx], df.iloc[val_idx]
        
        fe = HybridFeatureEngineer(gnn_embeddings).fit(tr_df)
        X_tr, X_val = fe.transform(tr_df), fe.transform(va_df)
        y_tr, y_val = tr_df[TARGET], va_df[TARGET]
        
        # LGBM
        m_lgb = lgb.LGBMRegressor(n_estimators=1200, learning_rate=0.03, num_leaves=127, verbose=-1, random_state=RANDOM_STATE)
        m_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], categorical_feature=["RoadType", "Weather"],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        lgbm_oof[val_idx] = m_lgb.predict(X_val)
        
        # CatBoost
        m_cat = cb.CatBoostRegressor(iterations=1200, learning_rate=0.03, depth=6, verbose=False, random_seed=RANDOM_STATE)
        m_cat.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], cat_features=["RoadType", "Weather"],
                   early_stopping_rounds=50)
        cat_oof[val_idx] = m_cat.predict(X_val)
        valid_indices.extend(val_idx)

    y_true_val = df.iloc[valid_indices][TARGET].to_numpy()
    weights = find_optimal_weights([lgbm_oof[valid_indices], cat_oof[valid_indices]], y_true_val)
    print(f"Optimal Meta-Ensemble Weights: LGBM={weights[0]:.4f}, CatBoost={weights[1]:.4f}")
    
    cv_pred = np.clip(weights[0] * lgbm_oof[valid_indices] + weights[1] * cat_oof[valid_indices], 0.0, 1.0)
    print(f"Hybrid Validation RMSE: {math.sqrt(mean_squared_error(y_true_val, cv_pred)):.6f}")
    print(f"Hybrid Validation R2:   {r2_score(y_true_val, cv_pred):.6f}")

    # 3. Final Train & Direct Inference (No recursive drift!)
    print("\n--- Training Final Models & Generating Submission ---")
    fe = HybridFeatureEngineer(gnn_embeddings).fit(df)
    X_full = fe.transform(df)
    X_test = fe.transform(test_df)
    
    m_lgb = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=127, verbose=-1, random_state=RANDOM_STATE)
    m_lgb.fit(X_full, df[TARGET], categorical_feature=["RoadType", "Weather"])
    
    m_cat = cb.CatBoostRegressor(iterations=800, learning_rate=0.03, depth=6, verbose=False, random_seed=RANDOM_STATE)
    m_cat.fit(X_full, df[TARGET], cat_features=["RoadType", "Weather"])
    
    test_preds = np.clip(weights[0] * m_lgb.predict(X_test) + weights[1] * m_cat.predict(X_test), 0.0, 1.0)
    
    # Save
    sub_df = pd.DataFrame({"Index": test_df["Index"], "demand": test_preds})
    sub_df.to_csv(args.output, index=False)
    print(f"Direct Tabular Submission saved to {args.output}")

if __name__ == "__main__":
    main()
