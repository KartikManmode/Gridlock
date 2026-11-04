"""Anomaly-Aware Neural-Boosted Framework (v6).

Supercharges v5 with "Extreme Point Training" to break the 95% barrier:
1. Adversarial Jitter: Randomly corrupts 5% of training sequences to simulate shocks (road blocks).
2. Asymmetric Sample Weighting: 2x weight on high demand (>0.7) and sudden shifts (>0.2).
3. Shock Features: weather_changed, demand_velocity, demand_acceleration.
4. High-Res Spatial/Temporal Embeddings (128 dims).
"""

import argparse
import math
import os
import random
from pathlib import Path

import catboost as cb
import geohash
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gridlock")

RANDOM_STATE = 42
TARGET = "demand"
SEQ_LEN = 4
SHOCK_WEIGHT = 2.0  # Increased priority for anomalies

def seed_everything(seed: int = RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_timestamp(value: str) -> tuple[int, int]:
    hour, minute = str(value).split(":")
    return int(hour), int(minute)

def decode_geohash(value: str) -> tuple[float, float]:
    lat, lon = geohash.decode(str(value))
    return float(lat), float(lon)

# ==========================================
# 1. DATA PREPROCESSING & SHOCK FEATURES
# ==========================================

def prepare_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    print("Preparing Anomaly-Aware Spatio-Temporal Dataset...")
    
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["is_test"] = 0
    test_df["is_test"] = 1
    test_df[TARGET] = np.nan
    test_df["original_index"] = test_df["Index"]
    train_df["original_index"] = np.arange(len(train_df)) * -1
    
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Base Features
    full_df["hour"] = full_df["timestamp"].str.split(":").str[0].astype(int)
    full_df["minute"] = full_df["timestamp"].str.split(":").str[1].astype(int)
    coords = pd.DataFrame(full_df["geohash"].apply(decode_geohash).tolist(), index=full_df.index, columns=["lat", "lon"])
    full_df["lat"] = coords["lat"]
    full_df["lon"] = coords["lon"]
    
    full_df["minute_of_day"] = full_df["hour"] * 60 + full_df["minute"]
    full_df["slot_id"] = full_df["day"] * 96 + full_df["minute_of_day"] // 15
    full_df["hour_sin"] = np.sin(2 * np.pi * full_df["hour"] / 24)
    full_df["hour_cos"] = np.cos(2 * np.pi * full_df["hour"] / 24)
    full_df["minute_sin"] = np.sin(2 * np.pi * full_df["minute_of_day"] / 1440)
    full_df["minute_cos"] = np.cos(2 * np.pi * full_df["minute_of_day"] / 1440)
    
    full_df["RoadType"] = full_df["RoadType"].fillna("Unknown")
    full_df["Weather"] = full_df["Weather"].fillna("Unknown")
    full_df["Temperature"] = full_df["Temperature"].fillna(25.0)
    
    # Categorical
    le_road = LabelEncoder()
    le_weather = LabelEncoder()
    full_df["RoadType_idx"] = le_road.fit_transform(full_df["RoadType"])
    full_df["Weather_idx"] = le_weather.fit_transform(full_df["Weather"])
    
    # Shock Feature 1: Weather Change
    full_df = full_df.sort_values(["geohash", "slot_id"])
    full_df["prev_weather"] = full_df.groupby("geohash")["Weather_idx"].shift(1)
    full_df["weather_changed"] = (full_df["Weather_idx"] != full_df["prev_weather"]).astype(int)
    
    scaler_coords = StandardScaler()
    full_df[["lat", "lon"]] = scaler_coords.fit_transform(full_df[["lat", "lon"]])
    
    # Temporal Lags
    print("Generating Temporal Lags and Shock Signals...")
    lag_cols = [TARGET, "hour_sin", "hour_cos", "minute_sin", "minute_cos"]
    for col in lag_cols:
        for lag in range(1, SEQ_LEN + 1):
            full_df[f"{col}_lag{lag}"] = full_df[col].shift(lag)
            
    # Shock Feature 2: Velocity & Acceleration
    full_df["demand_v"] = full_df[f"{TARGET}_lag1"] - full_df[f"{TARGET}_lag2"]
    full_df["demand_a"] = full_df["demand_v"] - (full_df[f"{TARGET}_lag2"] - full_df[f"{TARGET}_lag3"])
    
    # Mask out geohash changes
    for lag in range(1, SEQ_LEN + 1):
        mask = full_df["geohash"] != full_df["geohash"].shift(lag)
        for col in lag_cols:
            full_df.loc[mask, f"{col}_lag{lag}"] = np.nan
        if lag == 1:
            full_df.loc[mask, ["weather_changed", "demand_v", "demand_a"]] = 0
            
    print("Data preparation complete.")
    return full_df, scaler_coords, le_road, le_weather

# ==========================================
# 2. PYTORCH NEURAL EXTRACTOR WITH JITTER
# ==========================================

class FourierFeatures(nn.Module):
    def __init__(self, in_features: int, out_features: int, scale: float = 10.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(in_features, out_features // 2) * scale, requires_grad=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = (2.0 * math.pi * x) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class NeuralFeatureExtractor(nn.Module):
    def __init__(self, spatial_dim: int = 64, temporal_dim: int = 64, flat_seq_features: int = 20):
        super().__init__()
        self.spatial = nn.Sequential(
            FourierFeatures(in_features=2, out_features=128),
            nn.Linear(128, spatial_dim * 2), nn.GELU(),
            nn.Linear(spatial_dim * 2, spatial_dim)
        )
        self.temporal = nn.Sequential(
            nn.Linear(flat_seq_features, temporal_dim * 2), nn.GELU(),
            nn.Linear(temporal_dim * 2, temporal_dim)
        )
        self.head = nn.Sequential(
            nn.Linear(spatial_dim + temporal_dim, 64), nn.GELU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, coords: torch.Tensor, flat_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s_emb = self.spatial(coords)
        t_emb = self.temporal(flat_seq)
        combined = torch.cat([s_emb, t_emb], dim=-1)
        pred = self.head(combined)
        return combined, pred

def get_tensors_from_df(df: pd.DataFrame, device: torch.device):
    coords = df[["lat", "lon"]].to_numpy(dtype=np.float32)
    seq_cols = [TARGET, "hour_sin", "hour_cos", "minute_sin", "minute_cos"]
    lag_arrays = [df[[f"{col}_lag{lag}" for col in seq_cols]].fillna(0.0).to_numpy(dtype=np.float32) for lag in range(SEQ_LEN, 0, -1)]
    seqs = np.hstack(lag_arrays)
    coords_t = torch.from_numpy(coords).to(device)
    seqs_t = torch.from_numpy(seqs).to(device)
    if TARGET in df.columns and not df[TARGET].isna().all():
        targets_t = torch.from_numpy(df[TARGET].to_numpy(dtype=np.float32)[:, None]).to(device)
        return coords_t, seqs_t, targets_t
    return coords_t, seqs_t, None

def train_pytorch_extractor_with_jitter(train_df: pd.DataFrame, device: torch.device, epochs: int = 10):
    print(f"\n--- Phase 1: Training PyTorch with Adversarial Jitter ({epochs} epochs) ---")
    coords_t, seqs_t, targets_t = get_tensors_from_df(train_df, device)
    model = NeuralFeatureExtractor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    batch_size = 512
    num_samples = len(coords_t)
    indices = np.arange(num_samples)
    
    model.train()
    for epoch in range(1, epochs + 1):
        np.random.shuffle(indices)
        total_loss = 0.0
        for start_idx in range(0, num_samples, batch_size):
            batch_idx = indices[start_idx:start_idx + batch_size]
            c_b, s_batch, t_b = coords_t[batch_idx], seqs_t[batch_idx].clone(), targets_t[batch_idx]
            
            # --- ADVERSARIAL JITTER (Simulate Road Blocks/Accidents) ---
            if random.random() < 0.2:  # Jitter 20% of batches
                jitter_mask = torch.rand(len(s_batch)) < 0.05 # Jitter 5% of rows in batch
                s_batch[jitter_mask, 0] = 0.0 # Force current lag to 0 (road block)
            
            optimizer.zero_grad()
            _, preds = model(c_b, s_batch)
            loss = criterion(preds, t_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"PyTorch Epoch {epoch} | Loss: {total_loss/(num_samples/batch_size):.6f}")
    return model

@torch.no_grad()
def extract_embeddings(model: nn.Module, df: pd.DataFrame, device: torch.device):
    model.eval()
    coords_t, seqs_t, _ = get_tensors_from_df(df, device)
    batch_size = 1024
    embeddings = []
    for start_idx in range(0, len(coords_t), batch_size):
        c_b, s_b = coords_t[start_idx:start_idx+batch_size], seqs_t[start_idx:start_idx+batch_size]
        emb, _ = model(c_b, s_b)
        embeddings.append(emb.cpu().numpy())
    return np.concatenate(embeddings, axis=0)

# ==========================================
# 3. GBDT WITH ASYMMETRIC WEIGHTING
# ==========================================

def calculate_sample_weights(df: pd.DataFrame):
    weights = np.ones(len(df))
    # Rule 1: Prioritize high demand spikes
    weights[df[TARGET] > 0.7] *= SHOCK_WEIGHT
    # Rule 2: Prioritize sudden shifts (shocks)
    shift_signal = (df[TARGET] - df[f"{TARGET}_lag1"]).abs()
    weights[shift_signal > 0.2] *= SHOCK_WEIGHT
    return weights

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("dataset/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("dataset/test.csv"))
    parser.add_argument("--output", type=Path, default=Path("gemini_codes/anomaly_aware_v6/gemini_v6_anomaly_aware.csv"))
    args = parser.parse_args()

    seed_everything(RANDOM_STATE)
    device = torch.device("cpu")
    
    train_raw = pd.read_csv(args.train)
    test_raw = pd.read_csv(args.test)
    
    full_df, scaler_coords, le_road, le_wea = prepare_data(train_raw, test_raw)
    train_full = full_df[full_df["is_test"] == 0].copy()
    test_full = full_df[full_df["is_test"] == 1].copy()
    train_clean = train_full.dropna(subset=[f"{TARGET}_lag{SEQ_LEN}"]).copy()
    
    # 1. Jitter Training
    pt_model = train_pytorch_extractor_with_jitter(train_clean, device, epochs=10)
    train_emb = extract_embeddings(pt_model, train_clean, device)
    
    # 2. Build Hybrid Features
    shock_features = ["Temperature", "NumberofLanes", "RoadType_idx", "Weather_idx", 
                      "hour_sin", "hour_cos", "weather_changed", "demand_v", "demand_a"]
    X_train_full = np.hstack([train_clean[shock_features].values, train_emb])
    y_train = train_clean[TARGET].values
    sample_weights = calculate_sample_weights(train_clean)

    # 3. Chronological CV
    tscv = TimeSeriesSplit(n_splits=3)
    r2_scores = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_full), 1):
        X_tr, X_val = X_train_full[train_idx], X_train_full[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        w_tr = sample_weights[train_idx]
        
        # Huber Objective + Sample Weights
        lgb_fold = lgb.LGBMRegressor(objective='huber', n_estimators=600, learning_rate=0.03, num_leaves=127, verbose=-1, random_state=RANDOM_STATE)
        lgb_fold.fit(X_tr, y_tr, sample_weight=w_tr)
        
        cat_fold = cb.CatBoostRegressor(iterations=600, learning_rate=0.03, depth=6, verbose=False, random_seed=RANDOM_STATE)
        cat_fold.fit(X_tr, y_tr, sample_weight=w_tr)
        
        preds = np.clip((lgb_fold.predict(X_val) + cat_fold.predict(X_val)) / 2.0, 0.0, 1.0)
        r2 = r2_score(y_val, preds)
        r2_scores.append(r2)
        print(f"  -> CV Fold {fold} R2: {r2:.5f}")
    print(f"Avg CV R2: {np.mean(r2_scores):.5f}")

    # 4. Final Train and Rollout
    print("\n--- Training Final Anomaly-Aware Models ---")
    lgb_final = lgb.LGBMRegressor(objective='huber', n_estimators=1000, learning_rate=0.02, num_leaves=127, verbose=-1, random_state=RANDOM_STATE)
    lgb_final.fit(X_train_full, y_train, sample_weight=sample_weights)
    cat_final = cb.CatBoostRegressor(iterations=1000, learning_rate=0.02, depth=6, verbose=False, random_seed=RANDOM_STATE)
    cat_final.fit(X_train_full, y_train, sample_weight=sample_weights)
    
    test_slots = sorted(test_full["slot_id"].unique())
    dynamic_df = full_df.copy().set_index(["geohash", "slot_id"])
    predictions = []
    
    for slot in test_slots:
        current_mask = (test_full["slot_id"] == slot)
        slot_df = test_full[current_mask].copy()
        if slot_df.empty: continue
            
        for lag in range(1, SEQ_LEN + 1):
            lag_slot = slot - lag
            slot_df[f"{TARGET}_lag{lag}"] = slot_df["geohash"].apply(
                lambda gh: dynamic_df.at[(gh, lag_slot), TARGET] if (gh, lag_slot) in dynamic_df.index else 0.0
            )
        
        slot_df["demand_v"] = slot_df[f"{TARGET}_lag1"] - slot_df[f"{TARGET}_lag2"]
        slot_df["demand_a"] = slot_df["demand_v"] - (slot_df[f"{TARGET}_lag2"] - slot_df[f"{TARGET}_lag3"])
        
        slot_emb = extract_embeddings(pt_model, slot_df, device)
        X_slot = np.hstack([slot_df[shock_features].values, slot_emb])
        
        final_pred = np.clip((lgb_final.predict(X_slot) + cat_final.predict(X_slot)) / 2.0, 0.0, 1.0)
        for i, gh in enumerate(slot_df["geohash"]):
            dynamic_df.at[(gh, slot), TARGET] = final_pred[i]
        for i, orig_idx in enumerate(slot_df["original_index"]):
            predictions.append({"Index": int(orig_idx), "demand": final_pred[i]})
            
    sub_df = pd.DataFrame(predictions).sort_values("Index")
    sub_df.to_csv(args.output, index=False)
    print(f"Anomaly-Aware Submission saved to {args.output}")

if __name__ == "__main__":
    main()
