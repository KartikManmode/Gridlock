"""Phase 4 Strategy 1: Zero-Inflated (Two-Stage) Hybrid Model.

Handles massive sparsity in traffic data by using a LightGBM Classifier to predict
the probability of traffic > 0, and a CatBoost Regressor trained ONLY on active traffic.
Integrates GNN spatial embeddings from v9 to prevent recursive drift.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gridlock")

import catboost as cb
import geohash
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler

import torch
import torch.nn as nn

RANDOM_STATE = 42
TARGET = "demand"
ZERO_THRESHOLD = 0.02


def seed_everything(seed: int) -> None:
    import random
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

# --- Re-use GNN for Embeddings ---
class GCN(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 16):
        super().__init__()
        self.gc1 = nn.Linear(in_features, hidden_dim)
        self.gc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h1 = torch.relu(self.gc1(torch.matmul(adj, x)))
        emb = torch.relu(self.gc2(torch.matmul(adj, h1)))
        pred = self.out(emb)
        return emb, pred

def get_gnn_embeddings(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, np.ndarray]:
    all_gh = sorted(list(set(train_df["geohash"].unique()) | set(test_df["geohash"].unique())))
    gh_to_idx = {gh: i for i, gh in enumerate(all_gh)}
    N = len(all_gh)
    
    adj = np.eye(N, dtype=np.float32)
    for gh in all_gh:
        idx = gh_to_idx[gh]
        for n_gh in geohash.neighbors(gh):
            if n_gh in gh_to_idx:
                adj[idx, gh_to_idx[n_gh]] = 1.0
                
    deg = adj.sum(axis=1)
    d_inv_sqrt = np.power(np.maximum(deg, 1.0), -0.5)
    adj_norm = d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]
    
    device = torch.device("cpu")
    adj_t = torch.from_numpy(adj_norm).float().to(device)

    X = np.zeros((N, 3), dtype=np.float32)
    gh_counts = train_df["geohash"].value_counts().to_dict()
    for gh in all_gh:
        lat, lon = decode_geohash(gh)
        X[gh_to_idx[gh]] = [lat, lon, gh_counts.get(gh, 0.0)]
        
    X = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-6)
    X_t = torch.from_numpy(X).float().to(device)

    Y = np.zeros(N, dtype=np.float32)
    mask = np.zeros(N, dtype=np.bool_)
    gh_means = train_df.groupby("geohash")[TARGET].mean().to_dict()
    g_mean = float(train_df[TARGET].mean())
    
    for gh in all_gh:
        if gh in gh_means:
            Y[gh_to_idx[gh]] = gh_means[gh]
            mask[gh_to_idx[gh]] = True
        else:
            Y[gh_to_idx[gh]] = g_mean
            
    Y_t = torch.from_numpy(Y).float().unsqueeze(1).to(device)
    mask_t = torch.from_numpy(mask).bool().to(device)

    model = GCN(in_features=3, hidden_dim=16).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.02)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(150):
        opt.zero_grad()
        emb, pred = model(X_t, adj_t)
        loss = criterion(pred[mask_t], Y_t[mask_t])
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        final_emb, _ = model(X_t, adj_t)
        emb_np = final_emb.numpy()
        
    return {gh: emb_np[gh_to_idx[gh]] for gh in all_gh}


class ZeroInflatedFE:
    def __init__(self, gnn_embeddings: dict[str, np.ndarray]):
        self.gnn_embeddings = gnn_embeddings
        self.road_enc = LabelEncoder()
        self.wea_enc = LabelEncoder()
        self.scaler = StandardScaler()
        self.global_demand = 0.5
        self.te_map = {}
        self.num_cols = ["Temperature", "NumberofLanes", "lat", "lon", "lat_lon_product"]

    def fit(self, df: pd.DataFrame):
        work = self._basic(df)
        self.global_demand = float(work[TARGET].mean())
        
        self.road_enc.fit(work["RoadType"].astype(str))
        self.wea_enc.fit(work["Weather"].astype(str))
        
        stats = work.groupby("geo5_hour")[TARGET].agg(["mean", "count"])
        self.te_map = ((stats["count"] * stats["mean"] + 20.0 * self.global_demand) / 
                       (stats["count"] + 20.0)).to_dict()
        
        feat = self.transform(df)
        self.scaler.fit(feat[self.num_cols].fillna(0))
        return self

    def transform(self, df: pd.DataFrame):
        work = self._basic(df)
        work["geo5_hour_te"] = work["geo5_hour"].map(self.te_map).fillna(self.global_demand)
        work["RoadType"] = self.road_enc.transform(work["RoadType"].astype(str))
        work["Weather"] = self.wea_enc.transform(work["Weather"].astype(str))
        
        gnn_feats = np.zeros((len(work), 16), dtype=np.float32)
        for i, gh in enumerate(work["geohash"]):
            gnn_feats[i] = self.gnn_embeddings.get(gh, np.zeros(16))
            
        for dim in range(16):
            work[f"gnn_emb_{dim}"] = gnn_feats[:, dim]
            
        cols = ["day", "hour", "minute_of_day", "hour_sin", "hour_cos", "is_rush_hour", 
                "is_night", "Temperature", "NumberofLanes", "lat", "lon", "lat_lon_product",
                "geo5_hour_te", "RoadType", "Weather"] + [f"gnn_emb_{d}" for d in range(16)]
        
        res = work[cols].copy().fillna(0)
        if hasattr(self.scaler, "mean_"):
            res[self.num_cols] = self.scaler.transform(res[self.num_cols])
        return res

    def _basic(self, df: pd.DataFrame):
        work = df.copy()
        work[["hour", "minute"]] = work["timestamp"].apply(parse_timestamp).tolist()
        work[["lat", "lon"]] = work["geohash"].apply(decode_geohash).tolist()
        work["lat_lon_product"] = work["lat"] * work["lon"]
        work["minute_of_day"] = work["hour"] * 60 + work["minute"]
        work["hour_sin"] = np.sin(2 * np.pi * work["hour"] / 24)
        work["hour_cos"] = np.cos(2 * np.pi * work["hour"] / 24)
        work["is_rush_hour"] = work["hour"].between(7, 9) | work["hour"].between(17, 19)
        work["is_night"] = (work["hour"] >= 22) | (work["hour"] <= 6)
        work["geo5_hour"] = work["geohash"].astype(str).str[:5] + "__h" + work["hour"].astype(str)
        work["RoadType"] = work["RoadType"].fillna("Unknown")
        work["Weather"] = work["Weather"].fillna("Unknown")
        work["Temperature"] = work["Temperature"].fillna(25.0)
        return work


class TwoStageModel:
    def __init__(self):
        # Classifier aims for high recall on positive traffic
        self.clf = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, verbose=-1, random_state=RANDOM_STATE)
        # Regressor trained purely on active demand signals
        self.reg = cb.CatBoostRegressor(iterations=1000, learning_rate=0.03, depth=6, verbose=False, random_seed=RANDOM_STATE)

    def fit(self, X, y):
        # Stage 1
        y_bin = (y > ZERO_THRESHOLD).astype(int)
        self.clf.fit(X, y_bin, categorical_feature=["RoadType", "Weather"])
        
        # Stage 2
        active_mask = y > ZERO_THRESHOLD
        X_active = X[active_mask]
        y_active = y[active_mask]
        self.reg.fit(X_active, y_active, cat_features=["RoadType", "Weather"])

    def predict(self, X):
        prob = self.clf.predict_proba(X)[:, 1]
        vol = self.reg.predict(X)
        return np.clip(prob * vol, 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("dataset/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("dataset/test.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/gemini_v10_zero_inflated.csv"))
    args = parser.parse_args()

    seed_everything(RANDOM_STATE)
    df = pd.read_csv(args.train).sort_values(["day", "timestamp"])
    test_df = pd.read_csv(args.test)
    
    gnn_emb = get_gnn_embeddings(df, test_df)

    tscv = TimeSeriesSplit(n_splits=3)
    scores = []
    
    print("\n--- Training Zero-Inflated Model ---")
    for train_idx, val_idx in tscv.split(df):
        tr_df, va_df = df.iloc[train_idx], df.iloc[val_idx]
        fe = ZeroInflatedFE(gnn_emb).fit(tr_df)
        X_tr, X_val = fe.transform(tr_df), fe.transform(va_df)
        
        model = TwoStageModel()
        model.fit(X_tr, tr_df[TARGET])
        preds = model.predict(X_val)
        
        score = r2_score(va_df[TARGET], preds)
        scores.append(score)
        print(f"Fold R2: {score:.6f}")
        
    print(f"Zero-Inflated CV R2: {np.mean(scores):.6f}")

    fe = ZeroInflatedFE(gnn_emb).fit(df)
    model = TwoStageModel()
    model.fit(fe.transform(df), df[TARGET])
    
    pd.DataFrame({
        "Index": test_df["Index"], 
        "demand": model.predict(fe.transform(test_df))
    }).to_csv(args.output, index=False)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
