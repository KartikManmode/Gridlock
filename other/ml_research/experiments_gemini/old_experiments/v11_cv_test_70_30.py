"""Meta-CV Validation: 70/30 Ratio (v11).

Evaluates the specific 70% v5 / 30% v6 ratio on an honest chronological holdout:
1. Split train_clean: 80% train / 20% holdout (chronological).
2. Train v5 (High-Res) and v6 (Shock-Aware) on the 80%.
3. Generate predictions for the 20% holdout.
4. Blend them: 0.70*v5 + 0.30*v6.
5. Report final R2.
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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path

import geohash

RANDOM_STATE = 42
TARGET = "demand"
SEQ_LEN = 4

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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

def main():
    seed_everything(RANDOM_STATE)
    device = torch.device("cpu")
    
    print("Loading data for 70/30 Meta-CV...")
    train_raw = pd.read_csv("dataset/train.csv")
    test_dummy = pd.DataFrame(columns=train_raw.columns).assign(Index=0)
    
    full_df, _, _, _ = prepare_data(train_raw, test_dummy)
    train_full = full_df[full_df["is_test"] == 0].copy()
    train_clean = train_full.dropna(subset=[f"{TARGET}_lag{SEQ_LEN}"]).copy()
    
    # Chronological Split
    split_idx = int(len(train_clean) * 0.8)
    tr_df = train_clean.iloc[:split_idx].copy()
    ho_df = train_clean.iloc[split_idx:].copy()
    
    y_tr = tr_df[TARGET].values
    y_ho = ho_df[TARGET].values
    weights_tr = calculate_sample_weights(tr_df)
    shock_features = ["Temperature", "NumberofLanes", "RoadType_idx", "Weather_idx", 
                      "hour_sin", "hour_cos", "weather_changed", "demand_v", "demand_a"]

    # 1. Train v5 Extractor & GBDT
    print("\n--- Training Model v5 (High-Res) ---")
    model_v5_pt = NeuralFeatureExtractor(spatial_dim=64, temporal_dim=64).to(device)
    coords_t, seqs_t, targets_t = get_tensors_from_df(tr_df, device)
    opt = torch.optim.AdamW(model_v5_pt.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    for _ in range(5):
        _, p = model_v5_pt(coords_t, seqs_t); loss = crit(p, targets_t); opt.zero_grad(); loss.backward(); opt.step()
    
    emb_tr5 = extract_embeddings(model_v5_pt, tr_df, device)
    emb_ho5 = extract_embeddings(model_v5_pt, ho_df, device)
    lgb_v5 = lgb.LGBMRegressor(objective='huber', n_estimators=500, verbose=-1, random_state=RANDOM_STATE)
    lgb_v5.fit(np.hstack([tr_df[shock_features].values, emb_tr5]), y_tr)
    preds_v5 = lgb_v5.predict(np.hstack([ho_df[shock_features].values, emb_ho5]))

    # 2. Train v6 GBDT
    print("\n--- Training Model v6 (Shock-Aware) ---")
    lgb_v6 = lgb.LGBMRegressor(objective='huber', n_estimators=500, verbose=-1, random_state=RANDOM_STATE)
    lgb_v6.fit(np.hstack([tr_df[shock_features].values, emb_tr5]), y_tr, sample_weight=weights_tr)
    preds_v6 = lgb_v6.predict(np.hstack([ho_df[shock_features].values, emb_ho5]))

    # 3. Final Blend
    print("\n--- Final Meta-Blend CV Score (70/30 Ratio) ---")
    final_blend = (preds_v5 * 0.70) + (preds_v6 * 0.30)
    final_blend = np.clip(final_blend, 0.0, 1.0)
    
    r2 = r2_score(y_ho, final_blend)
    rmse = math.sqrt(mean_squared_error(y_ho, final_blend))
    
    print(f"Meta-Blend (70/30) Holdout RMSE: {rmse:.6f}")
    print(f"Meta-Blend (70/30) Holdout R2:   {r2:.6f}")

if __name__ == "__main__":
    main()
