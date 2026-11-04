"""Meta-Weight Optimizer (v9).

Uses SciPy's Nelder-Mead optimization to find the mathematically optimal 
blending ratios for the v1, v5, and v6 architectures using TimeSeriesSplit OOF.
"""

import os
import random
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.optimize import minimize
from pathlib import Path

# ==========================================
# RE-IMPLEMENTED LOGIC FOR STANDALONE OPTIMIZATION
# ==========================================

import geohash

RANDOM_STATE = 42
TARGET = "demand"
SEQ_LEN = 4

def decode_geohash(value: str) -> tuple[float, float]:
    lat, lon = geohash.decode(str(value))
    return float(lat), float(lon)

def prepare_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["is_test"] = 0
    test_df["is_test"] = 1
    test_df[TARGET] = np.nan
    
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_df["hour"] = full_df["timestamp"].str.split(":").str[0].astype(int)
    full_df["minute"] = full_df["timestamp"].str.split(":").str[1].astype(int)
    coords = pd.DataFrame(full_df["geohash"].apply(decode_geohash).tolist(), index=full_df.index, columns=["lat", "lon"])
    full_df["lat"], full_df["lon"] = coords["lat"], coords["lon"]
    full_df["minute_of_day"] = full_df["hour"] * 60 + full_df["minute"]
    full_df["slot_id"] = full_df["day"] * 96 + full_df["minute_of_day"] // 15
    full_df["hour_sin"] = np.sin(2 * np.pi * full_df["hour"] / 24)
    full_df["hour_cos"] = np.cos(2 * np.pi * full_df["hour"] / 24)
    full_df["minute_sin"] = np.sin(2 * np.pi * full_df["minute_of_day"] / 1440)
    full_df["minute_cos"] = np.cos(2 * np.pi * full_df["minute_of_day"] / 1440)
    
    full_df["RoadType"] = full_df["RoadType"].fillna("Unknown")
    full_df["Weather"] = full_df["Weather"].fillna("Unknown")
    le_road, le_weather = LabelEncoder(), LabelEncoder()
    full_df["RoadType_idx"] = le_road.fit_transform(full_df["RoadType"])
    full_df["Weather_idx"] = le_weather.fit_transform(full_df["Weather"])
    
    full_df = full_df.sort_values(["geohash", "slot_id"])
    full_df["prev_weather"] = full_df.groupby("geohash")["Weather_idx"].shift(1)
    full_df["weather_changed"] = (full_df["Weather_idx"] != full_df["prev_weather"]).astype(int)
    
    scaler_coords = StandardScaler()
    full_df[["lat", "lon"]] = scaler_coords.fit_transform(full_df[["lat", "lon"]])
    
    lag_cols = [TARGET, "hour_sin", "hour_cos", "minute_sin", "minute_cos"]
    for col in lag_cols:
        for lag in range(1, SEQ_LEN + 1):
            full_df[f"{col}_lag{lag}"] = full_df[col].shift(lag)
            
    full_df["demand_v"] = full_df[f"{TARGET}_lag1"] - full_df[f"{TARGET}_lag2"]
    full_df["demand_a"] = full_df["demand_v"] - (full_df[f"{TARGET}_lag2"] - full_df[f"{TARGET}_lag3"])
    
    for lag in range(1, SEQ_LEN + 1):
        mask = full_df["geohash"] != full_df["geohash"].shift(lag)
        for col in lag_cols:
            full_df.loc[mask, f"{col}_lag{lag}"] = np.nan
        if lag == 1:
            full_df.loc[mask, ["weather_changed", "demand_v", "demand_a"]] = 0
            
    return full_df, scaler_coords, le_road, le_weather

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

def calculate_sample_weights(df: pd.DataFrame):
    weights = np.ones(len(df))
    weights[df[TARGET] > 0.7] *= 2.0
    shift_signal = (df[TARGET] - df[f"{TARGET}_lag1"]).abs()
    weights[shift_signal > 0.2] *= 2.0
    return weights

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    seed_everything(RANDOM_STATE)
    device = torch.device("cpu")
    
    print("Loading data for Weight Optimization...")
    train_raw = pd.read_csv("dataset/train.csv")
    test_dummy = pd.DataFrame(columns=train_raw.columns).assign(Index=0)
    
    full_df, _, _, _ = prepare_data(train_raw, test_dummy)
    train_clean = full_df[full_df["is_test"] == 0].dropna(subset=[f"{TARGET}_lag{SEQ_LEN}"]).copy()
    
    # 2. TimeSeriesSplit OOF
    print("\n--- Generating OOF Predictions for Optimization ---")
    tscv = TimeSeriesSplit(n_splits=3)
    
    oof_v1 = np.zeros(len(train_clean))
    oof_v5 = np.zeros(len(train_clean))
    oof_v6 = np.zeros(len(train_clean))
    val_indices = []
    
    shock_features = ["Temperature", "NumberofLanes", "RoadType_idx", "Weather_idx", 
                      "hour_sin", "hour_cos", "weather_changed", "demand_v", "demand_a"]

    for fold, (train_idx, val_idx) in enumerate(tscv.split(train_clean), 1):
        print(f"  -> Fold {fold}")
        tr, va = train_clean.iloc[train_idx], train_clean.iloc[val_idx]
        y_tr = tr[TARGET].values
        w_tr = calculate_sample_weights(tr)
        
        # Train Extractors on TR
        coords_t, seqs_t, targets_t = get_tensors_from_df(tr, device)
        crit = nn.MSELoss()
        
        # Model 1 (v1-like)
        pt1 = NeuralFeatureExtractor(spatial_dim=16, temporal_dim=16).to(device)
        opt1 = torch.optim.AdamW(pt1.parameters(), lr=1e-3)
        for _ in range(3):
            _, p = pt1(coords_t, seqs_t); loss = crit(p, targets_t); opt1.zero_grad(); loss.backward(); opt1.step()
        emb_tr1 = extract_embeddings(pt1, tr, device); emb_va1 = extract_embeddings(pt1, va, device)
        lgb1 = lgb.LGBMRegressor(n_estimators=300, verbose=-1, random_state=RANDOM_STATE)
        lgb1.fit(np.hstack([tr[shock_features].values, emb_tr1]), y_tr)
        oof_v1[val_idx] = lgb1.predict(np.hstack([va[shock_features].values, emb_va1]))
        
        # Model 2 (v5-like)
        pt2 = NeuralFeatureExtractor(spatial_dim=64, temporal_dim=64).to(device)
        opt2 = torch.optim.AdamW(pt2.parameters(), lr=1e-3)
        for _ in range(5):
            _, p = pt2(coords_t, seqs_t); loss = crit(p, targets_t); opt2.zero_grad(); loss.backward(); opt2.step()
        emb_tr2 = extract_embeddings(pt2, tr, device); emb_va2 = extract_embeddings(pt2, va, device)
        lgb2 = lgb.LGBMRegressor(objective='huber', n_estimators=400, verbose=-1, random_state=RANDOM_STATE)
        lgb2.fit(np.hstack([tr[shock_features].values, emb_tr2]), y_tr)
        oof_v5[val_idx] = lgb2.predict(np.hstack([va[shock_features].values, emb_va2]))
        
        # Model 3 (v6-like)
        lgb3 = lgb.LGBMRegressor(objective='huber', n_estimators=400, verbose=-1, random_state=RANDOM_STATE)
        lgb3.fit(np.hstack([tr[shock_features].values, emb_tr2]), y_tr, sample_weight=w_tr)
        oof_v6[val_idx] = lgb3.predict(np.hstack([va[shock_features].values, emb_va2]))
        
        val_indices.extend(val_idx)

    # 3. Optimize Weights
    y_true = train_clean.iloc[val_indices][TARGET].values
    p1, p2, p3 = oof_v1[val_indices], oof_v5[val_indices], oof_v6[val_indices]
    
    def objective(w):
        # Normalize weights to sum to 1
        wn = w / np.sum(w)
        blend = (p1 * wn[0]) + (p2 * wn[1]) + (p3 * wn[2])
        return np.sqrt(mean_squared_error(y_true, np.clip(blend, 0.0, 1.0)))

    print("\n--- Optimizing Weights (Nelder-Mead) ---")
    initial_weights = [0.4, 0.4, 0.2]
    res = minimize(objective, initial_weights, method='Nelder-Mead', bounds=[(0, 1)]*3)
    
    opt_w = res.x / np.sum(res.x)
    print(f"\nMathematical Optimal Weights:")
    print(f"  v1: {opt_w[0]:.4f}")
    print(f"  v5: {opt_w[1]:.4f}")
    print(f"  v6: {opt_w[2]:.4f}")
    
    best_oof_r2 = r2_score(y_true, (p1 * opt_w[0]) + (p2 * opt_w[1]) + (p3 * opt_w[2]))
    print(f"\nOptimized OOF R2: {best_oof_r2:.6f}")
    
    # 4. Generate Final Submission with these weights
    print("\n--- Generating Final Optimized Submission ---")
    v1_sub = pd.read_csv("gemini_codes/submissions/gemini_v1_neural_boost_submission.csv").set_index("Index")
    v5_sub = pd.read_csv("gemini_codes/pure_neural_boost_v5/gemini_v5_pure_neural_boost.csv").set_index("Index")
    v6_sub = pd.read_csv("gemini_codes/anomaly_aware_v6/gemini_v6_anomaly_aware.csv").set_index("Index")
    
    final_demand = (v1_sub["demand"] * opt_w[0]) + (v5_sub["demand"] * opt_w[1]) + (v6_sub["demand"] * opt_w[2])
    final_demand = final_demand.clip(0.0, 1.0)
    
    out_path = Path("gemini_codes/submissions/gemini_v9_optimized_meta_blend.csv")
    final_demand.reset_index().to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
