"""Neural-Boosted Spatio-Temporal Framework (v4) - Stabilized Embeddings.

Addresses the low chronological CV R² of v3 by:
1. Training the PyTorch Feature Extractor **once globally** on `train_clean`.
2. Extracting these stable neural embeddings for the entire `train_clean` dataset.
3. Performing GBDT K-Fold CV using these globally derived neural embeddings.
4. Integrating `geohash_hour_te` feature.
5. Two-Stage (Zero-Inflated) GBDT backend.
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
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gridlock")

RANDOM_STATE = 42
TARGET = "demand"
SEQ_LEN = 4  # Look back 1 hour (4 x 15 mins)
ZERO_THRESHOLD = 0.015

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
# 1. DATA PREPROCESSING
# ==========================================

def prepare_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    print("Preparing Continuous Spatio-Temporal Dataset...")
    
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    train_df["is_test"] = 0
    test_df["is_test"] = 1
    test_df[TARGET] = np.nan
    
    # Store original test index for final submission alignment
    test_df["original_index"] = test_df["Index"]
    train_df["original_index"] = np.arange(len(train_df)) * -1 # Dummy for train
    
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
    
    # Categorical Encoding
    le_road = LabelEncoder()
    le_weather = LabelEncoder()
    full_df["RoadType_idx"] = le_road.fit_transform(full_df["RoadType"])
    full_df["Weather_idx"] = le_weather.fit_transform(full_df["Weather"])
    
    # Scaling Coordinates
    scaler_coords = StandardScaler()
    full_df[["lat", "lon"]] = scaler_coords.fit_transform(full_df[["lat", "lon"]])
    
    # Create Lags efficiently
    print("Generating Temporal Lags...")
    full_df = full_df.sort_values(["geohash", "slot_id"]).reset_index(drop=True)
    
    lag_cols = [TARGET, "hour_sin", "hour_cos", "minute_sin", "minute_cos"]
    
    # Avoid groupby.shift for memory efficiency
    for col in lag_cols:
        for lag in range(1, SEQ_LEN + 1):
            full_df[f"{col}_lag{lag}"] = full_df[col].shift(lag)
            
    # Mask out rows where the geohash changed due to shifting
    for lag in range(1, SEQ_LEN + 1):
        mask = full_df["geohash"] != full_df["geohash"].shift(lag)
        for col in lag_cols:
            full_df.loc[mask, f"{col}_lag{lag}"] = np.nan
            
    print("Data preparation complete.")
    return full_df, scaler_coords, le_road, le_weather

# ==========================================
# 2. GNN Embeddings
# ==========================================
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

def get_gnn_embeddings(full_data: pd.DataFrame, target_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Generates GNN embeddings for all geohashes present in full_data,
       trained using targets from target_df."""
    
    full_data = full_data.copy()
    if "geohash5" not in full_data.columns:
        full_data["geohash5"] = full_data["geohash"].astype(str).str[:5]
    if "geo5_hour" not in full_data.columns:
        full_data["geo5_hour"] = full_data["geohash5"] + "__h" + full_data["hour"].astype(str)

    target_df = target_df.copy()
    if "geohash5" not in target_df.columns:
        target_df["geohash5"] = target_df["geohash"].astype(str).str[:5]
    if "geo5_hour" not in target_df.columns:
        target_df["geo5_hour"] = target_df["geohash5"] + "__h" + target_df["hour"].astype(str)

    all_gh = sorted(list(full_data["geohash"].unique()))
    gh_to_idx = {gh: i for i, gh in enumerate(all_gh)}
    N = len(all_gh)
    
    # Adjacency Matrix
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

    global_mean = float(target_df[TARGET].mean())
    geo5_hour_stats = target_df.groupby("geo5_hour")[TARGET].agg(["mean", "count"])
    smoothing = 10.0
    geo5_hour_te_map = ((geo5_hour_stats["count"] * geo5_hour_stats["mean"] + smoothing * global_mean) / 
                        (geo5_hour_stats["count"] + smoothing)).to_dict()

    X_nodes = []
    for gh_node in all_gh:
        lat, lon = decode_geohash(gh_node)
        temp_df = full_data[full_data["geohash"] == gh_node]
        if not temp_df.empty:
            avg_geo5_hour_te = temp_df["geo5_hour"].map(geo5_hour_te_map).mean()
            if pd.isna(avg_geo5_hour_te): avg_geo5_hour_te = global_mean
        else:
            avg_geo5_hour_te = global_mean
        X_nodes.append([lat, lon, avg_geo5_hour_te])
        
    X = np.array(X_nodes, dtype=np.float32)
    X = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-6)
    X_t = torch.from_numpy(X).float().to(device)

    Y = np.zeros(N, dtype=np.float32)
    mask = np.zeros(N, dtype=np.bool_)
    gh_means = target_df.groupby("geohash")[TARGET].mean().to_dict()
    
    for gh in all_gh:
        if gh in gh_means:
            Y[gh_to_idx[gh]] = gh_means[gh]
            mask[gh_to_idx[gh]] = True
        else:
            Y[gh_to_idx[gh]] = global_mean
            
    Y_t = torch.from_numpy(Y).float().unsqueeze(1).to(device)
    mask_t = torch.from_numpy(mask).bool().to(device)

    model = GCN(in_features=3, hidden_dim=16).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.02)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(100):
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


# ==========================================
# 3. PYTORCH NEURAL EXTRACTOR MODULES
# ==========================================

class FourierFeatures(nn.Module):
    def __init__(self, in_features: int, out_features: int, scale: float = 10.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(in_features, out_features // 2) * scale, requires_grad=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = (2.0 * math.pi * x) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class SpatialMapper(nn.Module):
    def __init__(self, gnn_embed_dim: int, fourier_dim: int = 64, hidden_dim: int = 32):
        super().__init__()
        self.fourier = FourierFeatures(in_features=2, out_features=fourier_dim)
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim + gnn_embed_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
    def forward(self, coords: torch.Tensor, gnn_emb: torch.Tensor) -> torch.Tensor:
        fourier_feats = self.fourier(coords)
        combined_feats = torch.cat([fourier_feats, gnn_emb], dim=-1)
        return self.mlp(combined_feats)

class TemporalGRU(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 32):
        super().__init__()
        self.gru = nn.GRU(in_features, hidden_dim, batch_first=True)
        
    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(seq)
        return h_n.squeeze(0)

class NeuralFeatureExtractor(nn.Module):
    def __init__(self, num_road_types: int, num_weather_types: int, 
                 gnn_embed_dim: int = 16, spatial_dim: int = 32, temporal_dim: int = 32, 
                 seq_features: int = 5, embed_dim: int = 8):
        super().__init__()
        self.road_embed = nn.Embedding(num_road_types + 1, embed_dim)
        self.weather_embed = nn.Embedding(num_weather_types + 1, embed_dim)
        
        self.spatial = SpatialMapper(gnn_embed_dim=gnn_embed_dim, hidden_dim=spatial_dim)
        self.temporal = TemporalGRU(in_features=seq_features, hidden_dim=temporal_dim)
        
        self.feature_combiner = nn.Sequential(
            nn.Linear(spatial_dim + temporal_dim + 2 * embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32)
        )
        self.head = nn.Sequential(
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, coords: torch.Tensor, gnn_emb: torch.Tensor, seq: torch.Tensor, 
                road_idx: torch.Tensor, weather_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        s_emb = self.spatial(coords, gnn_emb)
        t_emb = self.temporal(seq)
        r_emb = self.road_embed(road_idx)
        w_emb = self.weather_embed(weather_idx)
        
        combined_features = torch.cat([s_emb, t_emb, r_emb, w_emb], dim=-1)
        final_embedding = self.feature_combiner(combined_features)
        pred = self.head(final_embedding)
        return final_embedding, pred

# ==========================================
# 4. TENSOR CONVERSION
# ==========================================

def get_tensors_from_df(df: pd.DataFrame, gnn_embeddings: dict[str, np.ndarray], device: torch.device):
    coords = df[["lat", "lon"]].to_numpy(dtype=np.float32)
    gnn_emb_feats = np.array([gnn_embeddings.get(gh, np.zeros(16)) for gh in df["geohash"]], dtype=np.float32)
    
    seq_cols = [TARGET, "hour_sin", "hour_cos", "minute_sin", "minute_cos"]
    seq_data_list = []
    for lag in range(SEQ_LEN, 0, -1):
        lag_cols = [f"{col}_lag{lag}" for col in seq_cols]
        seq_data_list.append(df[lag_cols].fillna(0.0).to_numpy(dtype=np.float32))
    seqs = np.stack(seq_data_list, axis=1)
    
    road_idx = df["RoadType_idx"].to_numpy(dtype=np.int64)
    weather_idx = df["Weather_idx"].to_numpy(dtype=np.int64)
    
    df["geohash_hour"] = df["geohash"].astype(str) + "__h" + df["hour"].astype(str)
    global_mean = float(df[TARGET].mean()) if TARGET in df.columns else 0.5
    geohash_hour_stats = df.groupby("geohash_hour")[TARGET].agg(["mean", "count"])
    smoothing = 10.0
    geohash_hour_te_map = ((geohash_hour_stats["count"] * geohash_hour_stats["mean"] + smoothing * global_mean) / 
                            (geohash_hour_stats["count"] + smoothing)).to_dict()
    gh_hour_te = df["geohash_hour"].map(geohash_hour_te_map).fillna(global_mean).to_numpy(dtype=np.float32)[:, None]


    coords_t = torch.from_numpy(coords).to(device)
    gnn_emb_t = torch.from_numpy(gnn_emb_feats).to(device)
    seqs_t = torch.from_numpy(seqs).to(device)
    road_idx_t = torch.from_numpy(road_idx).to(device)
    weather_idx_t = torch.from_numpy(weather_idx).to(device)
    gh_hour_te_t = torch.from_numpy(gh_hour_te).to(device)
    
    if TARGET in df.columns and not df[TARGET].isna().all():
        targets = df[TARGET].to_numpy(dtype=np.float32)[:, None]
        targets_t = torch.from_numpy(targets).to(device)
        return coords_t, gnn_emb_t, seqs_t, road_idx_t, weather_idx_t, gh_hour_te_t, targets_t
    return coords_t, gnn_emb_t, seqs_t, road_idx_t, weather_idx_t, gh_hour_te_t, None

# ==========================================
# 5. TRAINING ROUTINES
# ==========================================

def train_pytorch_extractor(train_df: pd.DataFrame, gnn_embeddings: dict[str, np.ndarray],
                            num_road_types: int, num_weather_types: int, device: torch.device,
                            pytorch_epochs: int = 5):
    print(f"\n--- Phase 1: Training PyTorch Neural Feature Extractor ({pytorch_epochs} epochs) ---")
    
    coords_t, gnn_emb_t, seqs_t, road_idx_t, weather_idx_t, gh_hour_te_t, targets_t = get_tensors_from_df(train_df, gnn_embeddings, device)
    
    model = NeuralFeatureExtractor(
        num_road_types=num_road_types, num_weather_types=num_weather_types
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.5e-4)
    criterion = nn.MSELoss()
    
    batch_size = 256
    num_samples = len(coords_t)
    indices = np.arange(num_samples)
    
    model.train()
    for epoch in range(1, pytorch_epochs + 1):
        np.random.shuffle(indices)
        total_loss = 0.0
        
        for start_idx in range(0, num_samples, batch_size):
            batch_idx = indices[start_idx:start_idx + batch_size]
            
            c_batch = coords_t[batch_idx]
            g_batch = gnn_emb_t[batch_idx]
            s_batch = seqs_t[batch_idx]
            r_batch = road_idx_t[batch_idx]
            w_batch = weather_idx_t[batch_idx]
            t_batch = targets_t[batch_idx]
            
            optimizer.zero_grad()
            _, preds = model(c_batch, g_batch, s_batch, r_batch, w_batch)
            loss = criterion(preds, t_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"PyTorch Epoch {epoch} | Loss: {total_loss/(num_samples/batch_size):.5f}")
        
    return model

@torch.no_grad()
def extract_embeddings(model: nn.Module, df: pd.DataFrame, gnn_embeddings: dict[str, np.ndarray], device: torch.device):
    print(f"Extracting embeddings for {len(df)} rows...")
    model.eval()
    
    coords_t, gnn_emb_t, seqs_t, road_idx_t, weather_idx_t, gh_hour_te_t, _ = get_tensors_from_df(df, gnn_embeddings, device)
    
    batch_size = 512
    num_samples = len(coords_t)
    embeddings = []
    
    for start_idx in range(0, num_samples, batch_size):
        c_batch = coords_t[start_idx:start_idx + batch_size]
        g_batch = gnn_emb_t[start_idx:start_idx + batch_size]
        s_batch = seqs_t[start_idx:start_idx + batch_size]
        r_batch = road_idx_t[start_idx:start_idx + batch_size]
        w_batch = weather_idx_t[start_idx:start_idx + batch_size]
        
        emb, _ = model(c_batch, g_batch, s_batch, r_batch, w_batch)
        embeddings.append(emb.cpu().numpy())
        
    return np.concatenate(embeddings, axis=0)

# ==========================================
# 6. GBDT & RECURSIVE INFERENCE
# ==========================================

class TwoStageGBDTModel:
    def __init__(self):
        self.clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, verbose=-1, random_state=RANDOM_STATE)
        self.reg = cb.CatBoostRegressor(iterations=800, learning_rate=0.05, depth=6, verbose=False, random_seed=RANDOM_STATE)

    def fit(self, X, y):
        y_bin = (y > ZERO_THRESHOLD).astype(int)
        self.clf.fit(X, y_bin)
        active_mask = y > ZERO_THRESHOLD
        self.reg.fit(X[active_mask], y[active_mask])

    def predict(self, X):
        prob = self.clf.predict_proba(X)[:, 1]
        vol = self.reg.predict(X)
        return np.clip(prob * vol, 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("dataset/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("dataset/test.csv"))
    parser.add_argument("--output", type=Path, default=Path("gemini_codes/neural_boost_v4/gemini_v4_neural_boost_submission.csv"))
    parser.add_argument("--pytorch-epochs", type=int, default=5)
    args = parser.parse_args()

    seed_everything(RANDOM_STATE)
    device = torch.device("cpu")
    
    print("Loading datasets...")
    train_raw = pd.read_csv(args.train)
    test_raw = pd.read_csv(args.test)
    
    full_df, scaler_coords, le_road, le_wea = prepare_data(train_raw, test_raw)
    num_road_types = len(le_road.classes_)
    num_weather_types = len(le_wea.classes_)
    
    train_full = full_df[full_df["is_test"] == 0].copy()
    test_full = full_df[full_df["is_test"] == 1].copy()
    
    train_clean = train_full.dropna(subset=[f"{TARGET}_lag{SEQ_LEN}"]).copy()
    
    # 1. Global Neural Training (The key to v4)
    print("\n--- GNN Embedding Generation (Global) ---")
    gnn_embeddings_global = get_gnn_embeddings(train_clean, train_clean)
    
    pt_model_global = train_pytorch_extractor(train_clean, gnn_embeddings_global, 
                                             num_road_types, num_weather_types, device, 
                                             pytorch_epochs=args.pytorch_epochs)
                                             
    # 2. Extract stable global embeddings
    print("\n--- Extracting Stable Global Embeddings ---")
    X_emb_global = extract_embeddings(pt_model_global, train_clean, gnn_embeddings_global, device)
    
    # Generate static tabular features
    train_clean["geohash_hour"] = train_clean["geohash"].astype(str) + "__h" + train_clean["hour"].astype(str)
    global_mean_full = float(train_clean[TARGET].mean())
    gh_stats = train_clean.groupby("geohash_hour")[TARGET].agg(["mean", "count"])
    gh_te_map = ((gh_stats["count"] * gh_stats["mean"] + 10.0 * global_mean_full) / (gh_stats["count"] + 10.0)).to_dict()
    X_gh_te = train_clean["geohash_hour"].map(gh_te_map).fillna(global_mean_full).to_numpy(dtype=np.float32)[:, None]
    
    base_features_cols = ["Temperature", "NumberofLanes", "hour_sin", "hour_cos", "minute_sin", "minute_cos"]
    X_base = train_clean[base_features_cols].values
    X_cat = train_clean[["RoadType_idx", "Weather_idx"]].values
    
    X_gbdt_global = np.hstack([X_base, X_cat, X_gh_te, X_emb_global])
    y_gbdt_global = train_clean[TARGET].values

    # 3. K-Fold CV on the stable tabular set
    print("\n--- Phase 2: K-Fold Cross-Validation for GBDT ---")
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    r2_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_gbdt_global), 1):
        print(f"  -> CV Fold {fold}")
        X_tr, X_val = X_gbdt_global[train_idx], X_gbdt_global[val_idx]
        y_tr, y_val = y_gbdt_global[train_idx], y_gbdt_global[val_idx]
        
        gbdt_model = TwoStageGBDTModel()
        gbdt_model.fit(X_tr, y_tr)
        preds = gbdt_model.predict(X_val)
        
        r2 = r2_score(y_val, preds)
        r2_scores.append(r2)
        print(f"    R2: {r2:.5f}")
        
    print(f"\nAverage K-Fold GBDT CV R2: {np.mean(r2_scores):.5f} +/- {np.std(r2_scores):.5f}")

    # 4. Final Train and Recursive Inference
    print("\n--- Phase 3: Training Final Meta-Ensemble & Recursive Rollout Inference on Test Set ---")
    final_gbdt_model = TwoStageGBDTModel()
    final_gbdt_model.fit(X_gbdt_global, y_gbdt_global)
    
    test_slots = sorted(test_full["slot_id"].unique())
    dynamic_df = full_df.copy()
    dynamic_df = dynamic_df.set_index(["geohash", "slot_id"])
    predictions = []
    
    for slot_idx, slot in enumerate(test_slots):
        current_mask = (test_full["slot_id"] == slot)
        slot_df = test_full[current_mask].copy()
        
        if slot_df.empty:
            continue
            
        for lag in range(1, SEQ_LEN + 1):
            lag_slot = slot - lag
            slot_df[f"{TARGET}_lag{lag}"] = slot_df["geohash"].apply(
                lambda gh: dynamic_df.at[(gh, lag_slot), TARGET] if (gh, lag_slot) in dynamic_df.index else 0.0
            )
            for col in ["hour_sin", "hour_cos", "minute_sin", "minute_cos"]:
                 slot_df[f"{col}_lag{lag}"] = slot_df[f"{col}_lag{lag}"].fillna(0.0)
        
        X_test_slot_emb = extract_embeddings(pt_model_global, slot_df, gnn_embeddings_global, device)
        
        slot_df["geohash_hour"] = slot_df["geohash"].astype(str) + "__h" + slot_df["hour"].astype(str)
        X_test_slot_gh_hour_te = slot_df["geohash_hour"].map(gh_te_map).fillna(global_mean_full).to_numpy(dtype=np.float32)[:, None]
        
        X_test_slot_base = slot_df[base_features_cols].values
        X_test_slot_cat = slot_df[["RoadType_idx", "Weather_idx"]].values
        X_test_slot_gbdt = np.hstack([X_test_slot_base, X_test_slot_cat, X_test_slot_gh_hour_te, X_test_slot_emb])
        
        final_pred = final_gbdt_model.predict(X_test_slot_gbdt)
        
        for i, gh in enumerate(slot_df["geohash"]):
            dynamic_df.at[(gh, slot), TARGET] = final_pred[i]
            
        for i, orig_idx in enumerate(slot_df["original_index"]):
            predictions.append({"Index": int(orig_idx), "demand": final_pred[i]})
            
    print("\nSaving Submission...")
    sub_df = pd.DataFrame(predictions).sort_values("Index")
    out_path = Path(args.output)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    sub_df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
