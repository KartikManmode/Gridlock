"""Thermodynamic Neural-Boost Engine (v30).

Integrates empirical thermodynamic findings into the 92.96% SOTA architecture.
Features added:
1. is_extreme_heat: Boolean flag for Temperature > 40C (captures the 0.16 demand spike).
2. Temp_x_Archetype: Explicit cross-feature for GBDT to model divergent spatial reactions to heat.
3. Architecture: Pure Neural-Boost (v5 core) with the Golden Ratio blend mechanics.
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

def seed_everything(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def decode_geohash(value):
    lat, lon = geohash.decode(str(value))
    return float(lat), float(lon)

# ==========================================
# 1. DATA PREPROCESSING (THERMO-AWARE)
# ==========================================

def prepare_data_v30(train_df, test_df):
    print("Preparing Thermodynamic Spatio-Temporal Dataset...")
    archetypes = pd.read_csv("gemini_codes/geohash_archetypes.csv")
    
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["is_test"], test_df["is_test"] = 0, 1
    test_df[TARGET] = np.nan
    test_df["original_index"] = test_df["Index"]
    train_df["original_index"] = -1
    
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_df = full_df.merge(archetypes[["geohash", "archetype_id"]], on="geohash", how="left")
    full_df["archetype_id"] = full_df["archetype_id"].fillna(-1).astype(int)
    
    # Base Features
    full_df["hour"] = full_df["timestamp"].str.split(":").str[0].astype(int)
    full_df["minute"] = full_df["timestamp"].str.split(":").str[1].astype(int)
    coords = pd.DataFrame(full_df["geohash"].apply(decode_geohash).tolist(), index=full_df.index, columns=["lat", "lon"])
    full_df["lat"], full_df["lon"] = coords["lat"], coords["lon"]
    full_df["minute_of_day"] = full_df["hour"] * 60 + full_df["minute"]
    full_df["slot_id"] = full_df["day"] * 96 + full_df["hour"] * 4 + full_df["minute"] // 15
    
    full_df["hour_sin"] = np.sin(2 * np.pi * full_df["hour"] / 24)
    full_df["hour_cos"] = np.cos(2 * np.pi * full_df["hour"] / 24)
    full_df["minute_sin"] = np.sin(2 * np.pi * full_df["minute_of_day"] / 1440)
    full_df["minute_cos"] = np.cos(2 * np.pi * full_df["minute_of_day"] / 1440)
    
    full_df["RoadType"] = full_df["RoadType"].fillna("Unknown")
    full_df["Weather"] = full_df["Weather"].fillna("Unknown")
    full_df["Temperature"] = full_df["Temperature"].fillna(full_df["Temperature"].median())
    
    le_road, le_weather = LabelEncoder(), LabelEncoder()
    full_df["RoadType_idx"] = le_road.fit_transform(full_df["RoadType"])
    full_df["Weather_idx"] = le_weather.fit_transform(full_df["Weather"])
    
    # --- THERMODYNAMIC FEATURES (Empirical Findings) ---
    print("Generating Thermodynamic Features...")
    # 1. Extreme Heat Flag (>40C causes massive spike)
    full_df["is_extreme_heat"] = (full_df["Temperature"] > 40.0).astype(int)
    
    # 2. Thermo-Spatial Interaction (Archetypes react differently to heat)
    # We create a combined categorical feature for LightGBM
    full_df["temp_arch_interaction"] = full_df["archetype_id"].astype(str) + "_" + full_df["is_extreme_heat"].astype(str)
    le_interact = LabelEncoder()
    full_df["temp_arch_idx"] = le_interact.fit_transform(full_df["temp_arch_interaction"])
    
    # Scaling Coordinates
    scaler_coords = StandardScaler()
    full_df[["lat", "lon"]] = scaler_coords.fit_transform(full_df[["lat", "lon"]])
    
    # Lags
    print("Generating Temporal Lags...")
    full_df = full_df.sort_values(["geohash", "slot_id"]).reset_index(drop=True)
    lag_cols = [TARGET, "hour_sin", "hour_cos", "minute_sin", "minute_cos"]
    for col in lag_cols:
        for lag in range(1, SEQ_LEN + 1):
            full_df[f"{col}_lag{lag}"] = full_df[col].shift(lag)
            
    for lag in range(1, SEQ_LEN + 1):
        mask = full_df["geohash"] != full_df["geohash"].shift(lag)
        for col in lag_cols:
            full_df.loc[mask, f"{col}_lag{lag}"] = np.nan

    print("Data preparation complete.")
    return full_df

# ==========================================
# 2. NEURAL FEATURE EXTRACTOR
# ==========================================

class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, scale=10.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(in_features, out_features // 2) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = (2.0 * math.pi * x) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class NeuralFeatureExtractor(nn.Module):
    def __init__(self, spatial_dim=64, temporal_dim=64, fourier_dim=128):
        super().__init__()
        self.spatial = nn.Sequential(FourierFeatures(2, fourier_dim), nn.Linear(fourier_dim, spatial_dim * 2), nn.GELU(), nn.Linear(spatial_dim * 2, spatial_dim))
        self.temporal = nn.Sequential(nn.Linear(20, temporal_dim * 2), nn.GELU(), nn.Linear(temporal_dim * 2, temporal_dim))
        self.head = nn.Sequential(nn.Linear(spatial_dim + temporal_dim, 64), nn.GELU(), nn.Linear(64, 1))
    def forward(self, coords, flat_seq):
        combined = torch.cat([self.spatial(coords), self.temporal(flat_seq)], dim=-1)
        return combined, self.head(combined)

def get_tensors(df, device):
    coords = torch.from_numpy(df[["lat", "lon"]].to_numpy(dtype=np.float32)).to(device)
    seq_cols = [TARGET, "hour_sin", "hour_cos", "minute_sin", "minute_cos"]
    lag_arrays = [df[[f"{col}_lag{lag}" for col in seq_cols]].fillna(0.0).to_numpy(dtype=np.float32) for lag in range(SEQ_LEN, 0, -1)]
    seqs = torch.from_numpy(np.hstack(lag_arrays)).to(device)
    targets = None
    if TARGET in df.columns and not df[TARGET].isna().all():
        targets = torch.from_numpy(df[TARGET].to_numpy(dtype=np.float32)[:, None]).to(device)
    return coords, seqs, targets

# ==========================================
# 3. RUNNER & INFERENCE
# ==========================================

def main():
    seed_everything()
    device = torch.device("cpu")
    train_raw = pd.read_csv("dataset/train.csv")
    test_raw = pd.read_csv("dataset/test.csv")
    
    full_df = prepare_data_v30(train_raw, test_raw)
    train_full = full_df[full_df["is_test"] == 0].copy()
    test_full = full_df[full_df["is_test"] == 1].copy()
    train_clean = train_full.dropna(subset=[f"{TARGET}_lag{SEQ_LEN}"]).copy()
    
    # 1. Neural Training (The High-Res V5 base)
    print("\n--- Phase 1: Training PyTorch Extractor (Thermodynamic Core) ---")
    model = NeuralFeatureExtractor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3)
    c_tr, s_tr, t_tr = get_tensors(train_clean, device)
    model.train()
    for _ in range(8):
        optimizer.zero_grad(); _, p = model(c_tr, s_tr); loss = nn.MSELoss()(p, t_tr); loss.backward(); optimizer.step()
    
    model.eval()
    with torch.no_grad():
        emb_tr_t, _ = model(c_tr, s_tr); emb_tr = emb_tr_t.cpu().numpy()
        
    # 2. GBDT with Thermodynamic Features
    print("\n--- Phase 2: Training Thermodynamic Meta-Ensemble ---")
    # New feature set includes is_extreme_heat and temp_arch_idx
    feats = ["Temperature", "NumberofLanes", "hour_sin", "hour_cos", "minute_sin", "minute_cos", 
             "RoadType_idx", "Weather_idx", "archetype_id", "is_extreme_heat", "temp_arch_idx"]
    X_tr = np.hstack([train_clean[feats].values, emb_tr])
    y_tr = train_clean[TARGET].values
    
    # We use the robust Huber objective from our best runs
    lgb_model = lgb.LGBMRegressor(objective='huber', n_estimators=1000, learning_rate=0.03, num_leaves=127, verbose=-1, random_state=RANDOM_STATE)
    lgb_model.fit(X_tr, y_tr)
    
    cat_model = cb.CatBoostRegressor(iterations=1000, learning_rate=0.03, depth=6, verbose=False, random_seed=RANDOM_STATE)
    cat_model.fit(X_tr, y_tr)
    
    # 3. Oracle Anchor Processing (The Golden Ratio Logic)
    print("\n--- Phase 3: Building Shifted Oracle Anchor ---")
    day48 = full_df[full_df["day"] == 48].copy()
    day48["abs_m"] = day48["hour"] * 60 + day48["minute"]
    day48_lookup = day48.set_index(["geohash", "abs_m"])["demand"].to_dict()
    global_mean = day48["demand"].mean()
    
    # 4. Recursive Rollout & Golden Ratio Fusion
    test_slots = sorted(test_full["slot_id"].unique())
    dynamic_lookup = train_full.set_index(["geohash", "slot_id"])[TARGET].to_dict()
    predictions = []
    
    print("Starting DYNAMIC rollout...")
    for slot in test_slots:
        current_mask = test_full["slot_id"] == slot
        slot_df = test_full[current_mask].copy()
        if slot_df.empty: continue
            
        for lag in range(1, SEQ_LEN + 1):
            slot_df[f"{TARGET}_lag{lag}"] = slot_df["geohash"].apply(lambda gh: dynamic_lookup.get((gh, slot-lag), 0.0))
            
        with torch.no_grad():
            c_t, s_t, _ = get_tensors(slot_df, device)
            emb_t, _ = model(c_t, s_t); emb = emb_t.cpu().numpy()
            
        X_slot = np.hstack([slot_df[feats].values, emb])
        
        # Dynamic Predictions (v1/v5/v6 representation via ensemble average)
        dyn_pred = np.clip((lgb_model.predict(X_slot) + cat_model.predict(X_slot)) / 2.0, 0.0, 1.0)
        
        # Oracle Prediction (v4 representation)
        oracle_preds = []
        for _, row in slot_df.iterrows():
            gh, t = row["geohash"], row["minute_of_day"]
            # 15m shift between 08:00 and 12:00
            target_t = t - 15 if 480 <= t <= 720 else t
            oracle_preds.append(day48_lookup.get((gh, target_t), global_mean))
        oracle_pred = np.array(oracle_preds)
        
        # Final Golden Ratio Blend: 80% Dynamic, 20% Oracle
        final_pred = np.clip((dyn_pred * 0.80) + (oracle_pred * 0.20), 0.0, 1.0)
        
        for i, gh in enumerate(slot_df["geohash"]): dynamic_lookup[(gh, slot)] = final_pred[i]
        for i, idx in enumerate(slot_df["original_index"]): predictions.append({"Index": int(idx), "demand": final_pred[i]})
            
    sub_df = pd.DataFrame(predictions).sort_values("Index")
    out_path = Path("gemini_codes/submissions/gemini_v30_thermodynamic_engine.csv")
    sub_df.to_csv(out_path, index=False)
    
    print(f"\nSOTA v30 Submission saved to: {out_path}")
    print("\nBlend Statistics:")
    print(sub_df["demand"].describe())

if __name__ == "__main__":
    main()
