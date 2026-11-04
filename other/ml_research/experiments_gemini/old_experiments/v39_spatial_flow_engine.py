"""Spatial Flow Engine (v39).

Integrates dynamic spatial momentum (`neighbor_mean_lag1`) into the PyTorch 
and GBDT models to prevent static spatial overfitting.
Includes an internal CV phase to validate rollout stability.
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
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path
import geohash

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
# 1. DATA PREPROCESSING
# ==========================================

def prepare_data(train_df, test_df):
    print("Preparing Spatial Flow Dataset...")
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
    
    # Time and Space
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
    
    # Categoricals
    full_df["RoadType"] = full_df["RoadType"].fillna("Unknown")
    full_df["Weather"] = full_df["Weather"].fillna("Unknown")
    le_road, le_weather = LabelEncoder(), LabelEncoder()
    full_df["RoadType_idx"] = le_road.fit_transform(full_df["RoadType"])
    full_df["Weather_idx"] = le_weather.fit_transform(full_df["Weather"])
    full_df["Temperature"] = full_df["Temperature"].fillna(25.0)
    
    # Scaling Coordinates
    scaler_coords = StandardScaler()
    full_df[["lat", "lon"]] = scaler_coords.fit_transform(full_df[["lat", "lon"]])
    
    # Lags
    full_df = full_df.sort_values(["geohash", "slot_id"]).reset_index(drop=True)
    lag_cols = [TARGET, "hour_sin", "hour_cos", "minute_sin", "minute_cos"]
    for col in lag_cols:
        for lag in range(1, SEQ_LEN + 1):
            full_df[f"{col}_lag{lag}"] = full_df[col].shift(lag)
            
    for lag in range(1, SEQ_LEN + 1):
        mask = full_df["geohash"] != full_df["geohash"].shift(lag)
        for col in lag_cols:
            full_df.loc[mask, f"{col}_lag{lag}"] = np.nan
            
    # Physical Momentum
    full_df["demand_v"] = full_df[f"{TARGET}_lag1"] - full_df[f"{TARGET}_lag2"]
    full_df["demand_a"] = full_df["demand_v"] - (full_df[f"{TARGET}_lag2"] - full_df[f"{TARGET}_lag3"])
    
    # NEIGHBOR DYNAMIC INFLOW
    print("Generating Dynamic Spatial Flow Features...")
    all_gh = full_df["geohash"].unique()
    gh_to_neighbors = {gh: [n for n in geohash.neighbors(gh) if n in all_gh] for gh in all_gh}
    
    lookup_lag1 = full_df.set_index(["geohash", "slot_id"])[f"{TARGET}_lag1"].to_dict()
    
    n_mean_lag1 = []
    for _, row in full_df.iterrows():
        gh = row["geohash"]
        slot = row["slot_id"]
        neighbors = gh_to_neighbors.get(gh, [])
        vals = [lookup_lag1.get((n, slot), np.nan) for n in neighbors]
        vals = [v for v in vals if not pd.isna(v)]
        n_mean_lag1.append(np.mean(vals) if vals else np.nan)
        
    full_df["neighbor_mean_lag1"] = n_mean_lag1
    
    # Oracle Seasonality (Day 48 Anchor)
    day48 = full_df[full_df["day"] == 48].copy()
    day48["geohash_hour"] = day48["geohash"].astype(str) + "__h" + day48["hour"].astype(str)
    global_mean = float(day48[TARGET].mean())
    gh_stats = day48.groupby("geohash_hour")[TARGET].agg(["mean", "count"])
    gh_te_map = ((gh_stats["count"] * gh_stats["mean"] + 10.0 * global_mean) / (gh_stats["count"] + 10.0)).to_dict()
    
    full_df["geohash_hour"] = full_df["geohash"].astype(str) + "__h" + full_df["hour"].astype(str)
    full_df["geohash_hour_te"] = full_df["geohash_hour"].map(gh_te_map).fillna(global_mean)

    return full_df, scaler_coords, gh_te_map, global_mean, gh_to_neighbors

# ==========================================
# 2. NEURAL EXTRACTOR
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
        
        # Temporal has 20 seq features + 1 neighbor flow feature = 21
        self.temporal = nn.Sequential(nn.Linear(21, temporal_dim * 2), nn.GELU(), nn.Linear(temporal_dim * 2, temporal_dim))
        self.head = nn.Sequential(nn.Linear(spatial_dim + temporal_dim, 64), nn.GELU(), nn.Linear(64, 1))
        
    def forward(self, coords, flat_seq, n_flow):
        s_emb = self.spatial(coords)
        t_combined = torch.cat([flat_seq, n_flow], dim=-1)
        t_emb = self.temporal(t_combined)
        
        combined = torch.cat([s_emb, t_emb], dim=-1)
        return combined, self.head(combined)

def get_tensors(df, device, seq_len):
    coords = torch.from_numpy(df[["lat", "lon"]].to_numpy(dtype=np.float32)).to(device)
    
    seq_cols = [TARGET, "hour_sin", "hour_cos", "minute_sin", "minute_cos"]
    lag_arrays = [df[[f"{col}_lag{lag}" for col in seq_cols]].fillna(0.0).to_numpy(dtype=np.float32) for lag in range(seq_len, 0, -1)]
    seqs = torch.from_numpy(np.hstack(lag_arrays)).to(device)
    
    n_flow = torch.from_numpy(df[["neighbor_mean_lag1"]].fillna(0.0).to_numpy(dtype=np.float32)).to(device)
    
    targets = None
    if TARGET in df.columns and not df[TARGET].isna().all():
        targets = torch.from_numpy(df[TARGET].to_numpy(dtype=np.float32)[:, None]).to(device)
    return coords, seqs, n_flow, targets

@torch.no_grad()
def extract_embeddings(model, df, device, seq_len):
    model.eval()
    coords_t, seqs_t, n_flow_t, _ = get_tensors(df, device, seq_len)
    batch_size = 1024
    embeddings = []
    for start in range(0, len(coords_t), batch_size):
        c_b, s_b, nf_b = coords_t[start:start+batch_size], seqs_t[start:start+batch_size], n_flow_t[start:start+batch_size]
        emb, _ = model(c_b, s_b, nf_b)
        embeddings.append(emb.cpu().numpy())
    return np.concatenate(embeddings, axis=0)

# ==========================================
# 3. RUNNER & ROLLOUT
# ==========================================

def train_pt_model(train_clean, device, seq_len, epochs=10):
    model = NeuralFeatureExtractor(spatial_dim=128, temporal_dim=64).to(device) 
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3)
    c_tr, s_tr, nf_tr, y_tr = get_tensors(train_clean, device, seq_len)
    model.train()
    for e in range(1, epochs + 1):
        optimizer.zero_grad()
        _, p = model(c_tr, s_tr, nf_tr)
        loss = nn.MSELoss()(p, y_tr)
        loss.backward(); optimizer.step()
        if e % 2 == 0: print(f"  PyTorch Epoch {e} | Loss: {loss.item():.5f}")
    return model

class UnifiedModel:
    pass

def rollout(model, full_df, feats, seq_len, name, device, gh_te_map, global_mean, gh_to_neighbors, is_cv=False):
    if is_cv:
        test_full = full_df[(full_df["is_test"] == 0) & (full_df["slot_id"] >= 4608 + 75)].copy()
    else:
        test_full = full_df[full_df["is_test"] == 1].copy()
        
    test_slots = sorted(test_full["slot_id"].unique())
    dynamic_lookup = full_df[full_df["is_test"] == 0].set_index(["geohash", "slot_id"])[TARGET].to_dict()
    
    results = []
    print(f"Starting Rollout for {name}...")
    for slot in test_slots:
        current_mask = test_full["slot_id"] == slot
        slot_df = test_full[current_mask].copy()
        if slot_df.empty: continue
            
        for lag in range(1, seq_len + 1):
            slot_df[f"{TARGET}_lag{lag}"] = slot_df["geohash"].apply(lambda gh: dynamic_lookup.get((gh, slot-lag), 0.0))
            
        slot_df["demand_v"] = slot_df[f"{TARGET}_lag1"] - slot_df[f"{TARGET}_lag2"]
        slot_df["demand_a"] = slot_df["demand_v"] - (slot_df[f"{TARGET}_lag2"] - slot_df[f"{TARGET}_lag3"])
        
        # Dynamic Neighbor Flow
        n_mean_lag1 = []
        for gh in slot_df["geohash"]:
            neighbors = gh_to_neighbors.get(gh, [])
            vals = [dynamic_lookup.get((n, slot-1), np.nan) for n in neighbors]
            vals = [v for v in vals if not pd.isna(v)]
            n_mean_lag1.append(np.mean(vals) if vals else 0.0)
        slot_df["neighbor_mean_lag1"] = n_mean_lag1
        
        # Oracle Temporal Offset (15m shift on Sunday morning)
        oracle_preds = []
        for _, row in slot_df.iterrows():
            gh, t = row["geohash"], row["minute_of_day"]
            target_t = t - 15 if 480 <= t <= 720 else t
            key = gh + "__h" + str(target_t // 60)
            oracle_preds.append(gh_te_map.get(key, global_mean))
        slot_df["geohash_hour_te"] = oracle_preds
        
        emb = extract_embeddings(model.pt, slot_df, device, seq_len)
        X_slot = np.hstack([slot_df[feats].values, emb])
        
        preds = np.clip(model.gbdt.predict(X_slot), 0.0, 1.0)
        
        for i, gh in enumerate(slot_df["geohash"]): dynamic_lookup[(gh, slot)] = preds[i]
        
        if is_cv:
            for i, idx in enumerate(slot_df.index): 
                results.append({"Index": int(idx), "pred_demand": preds[i], "actual_demand": slot_df.iloc[i]["demand"]})
        else:
            for i, idx in enumerate(slot_df["original_index"]): 
                results.append({"Index": int(idx), "demand": preds[i]})
            
    return pd.DataFrame(results)

def main():
    seed_everything()
    device = torch.device("cpu")
    train_raw = pd.read_csv("dataset/train.csv")
    test_raw = pd.read_csv("dataset/test.csv")
    
    full_df, _, gh_te_map, global_mean, gh_to_neighbors = prepare_data(train_raw, test_raw)
    
    # Common Features
    feats = ["Temperature", "NumberofLanes", 
             "hour_sin", "hour_cos", "minute_sin", "minute_cos", 
             "demand_v", "demand_a", "neighbor_mean_lag1", "geohash_hour_te", "archetype_id"]
             
    print("\n==========================================")
    print("PHASE 1: CROSS VALIDATION (75/25 Split)")
    print("==========================================")
    
    cv_train = full_df[(full_df["is_test"] == 0) & (full_df["slot_id"] < 4608 + 75)].dropna(subset=[f"{TARGET}_lag{SEQ_LEN}"]).copy()
    
    print("Training CV PyTorch...")
    cv_pt = train_pt_model(cv_train, device, SEQ_LEN, epochs=8)
    cv_emb = extract_embeddings(cv_pt, cv_train, device, SEQ_LEN)
    cv_X = np.hstack([cv_train[feats].values, cv_emb])
    cv_y = cv_train[TARGET].values
    
    print("Training CV GBDT...")
    cv_lgb = lgb.LGBMRegressor(objective='huber', n_estimators=1000, learning_rate=0.03, num_leaves=127, verbose=-1, random_state=RANDOM_STATE)
    cv_lgb.fit(cv_X, cv_y)
    
    cv_cat = cb.CatBoostRegressor(iterations=1000, learning_rate=0.03, depth=6, verbose=False, random_seed=RANDOM_STATE)
    cv_cat.fit(cv_X, cv_y)
    
    class EnsembleWrapper:
        def __init__(self, lgb, cat):
            self.lgb = lgb
            self.cat = cat
        def predict(self, X):
            return (self.lgb.predict(X) + self.cat.predict(X)) / 2.0
            
    cv_model = UnifiedModel()
    cv_model.pt = cv_pt
    cv_model.gbdt = EnsembleWrapper(cv_lgb, cv_cat)
    
    cv_res = rollout(cv_model, full_df, feats, SEQ_LEN, "CV Rollout", device, gh_te_map, global_mean, gh_to_neighbors, is_cv=True)
    
    cv_r2 = r2_score(cv_res["actual_demand"], cv_res["pred_demand"])
    print(f"\n>>> INTERNAL CV R2 SCORE: {cv_r2:.5f} <<<")
    
    print("\n==========================================")
    print("PHASE 2: FULL TRAINING & FINAL SUBMISSION")
    print("==========================================")
    
    train_clean = full_df[(full_df["is_test"] == 0)].dropna(subset=[f"{TARGET}_lag{SEQ_LEN}"]).copy()
    
    print("Training Full PyTorch...")
    pt_full = train_pt_model(train_clean, device, SEQ_LEN, epochs=8)
    emb_full = extract_embeddings(pt_full, train_clean, device, SEQ_LEN)
    X_full = np.hstack([train_clean[feats].values, emb_full])
    y_full = train_clean[TARGET].values
    
    print("Training Full GBDT...")
    lgb_full = lgb.LGBMRegressor(objective='huber', n_estimators=1000, learning_rate=0.03, num_leaves=127, verbose=-1, random_state=RANDOM_STATE)
    lgb_full.fit(X_full, y_full)
    
    cat_full = cb.CatBoostRegressor(iterations=1000, learning_rate=0.03, depth=6, verbose=False, random_seed=RANDOM_STATE)
    cat_full.fit(X_full, y_full)
    
    model_full = UnifiedModel()
    model_full.pt = pt_full
    model_full.gbdt = EnsembleWrapper(lgb_full, cat_full)
    
    final_res = rollout(model_full, full_df, feats, SEQ_LEN, "Final Rollout", device, gh_te_map, global_mean, gh_to_neighbors, is_cv=False)
    final_res = final_res.set_index("Index")
    
    # --- POST PROCESSING (SOTA 0.93 Thermo-Spike) ---
    print("\n--- Applying SOTA Thermo-Spike Tuning ---")
    test_raw = test_raw.set_index("Index")
    final_res = final_res.join(test_raw[["Temperature"]], how="left")
    final_res["Temperature"] = final_res["Temperature"].fillna(25.0)
    
    final_res = final_res.reset_index().merge(full_df[["original_index", "archetype_id"]].rename(columns={"original_index": "Index"}), on="Index", how="left").set_index("Index")
    
    extreme_heat = final_res["Temperature"] > 40.0
    artery = final_res["archetype_id"].isin([1, 2])
    target_mask = extreme_heat & artery
    
    if target_mask.sum() > 0:
        final_res.loc[target_mask, "demand"] = final_res.loc[target_mask, "demand"] ** 0.70
        
    peak_mask = final_res["demand"] > 0.40
    if peak_mask.sum() > 0:
        final_res.loc[peak_mask, "demand"] = final_res.loc[peak_mask, "demand"] ** 0.93
        
    final_res["demand"] = final_res["demand"].clip(0.0, 1.0)
    
    out_path = Path("gemini_codes/submissions/gemini_v39_spatial_flow.csv")
    final_res[["demand"]].reset_index().to_csv(out_path, index=False)
    print(f"\nSOTA v39 Submission saved to: {out_path}")

if __name__ == "__main__":
    main()
