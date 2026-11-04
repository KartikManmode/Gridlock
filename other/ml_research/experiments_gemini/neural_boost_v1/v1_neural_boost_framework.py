"""Neural-Boosted Spatio-Temporal Framework (v1).

Combines PyTorch (for Geometric and Temporal Feature Extraction) with 
Gradient Boosting (for robust tabular prediction).

Features:
1. Fourier Feature Mapping for precise 2D spatial boundaries (Lat/Lon).
2. GRU Temporal Encoder for lag sequences and cyclical time.
3. LightGBM + CatBoost Meta-Ensemble trained on neural embeddings.
4. Recursive "Anti-Drift" Inference Loop for the test set.
"""

import argparse
import math
import os
import random
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gridlock")

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

RANDOM_STATE = 42
TARGET = "demand"
SEQ_LEN = 4  # Look back 1 hour (4 x 15 mins)

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
# 1. PYTORCH NEURAL EXTRACTOR MODULES
# ==========================================

class FourierFeatures(nn.Module):
    """Maps low-dim coordinates (lat, lon) to high-frequency sine/cosine features."""
    def __init__(self, in_features: int, out_features: int, scale: float = 10.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(in_features, out_features // 2) * scale, requires_grad=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = (2.0 * math.pi * x) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class SpatialMapper(nn.Module):
    """Maps 2D World (Lat, Lon) to a dense Spatial State."""
    def __init__(self, fourier_dim: int = 32, hidden_dim: int = 16):
        super().__init__()
        self.fourier = FourierFeatures(in_features=2, out_features=fourier_dim)
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        features = self.fourier(coords)
        return self.mlp(features)

class TemporalMLP(nn.Module):
    """Processes flattened sequence of lags and cyclical time into a Momentum State.
    Uses MLP instead of GRU to avoid PyTorch CPU segmentation faults on Apple Silicon.
    """
    def __init__(self, in_features: int, hidden_dim: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
    def forward(self, flat_seq: torch.Tensor) -> torch.Tensor:
        return self.mlp(flat_seq)

class NeuralFeatureExtractor(nn.Module):
    """End-to-End PyTorch model for demand prediction. Used for embedding extraction."""
    def __init__(self, spatial_dim: int = 16, temporal_dim: int = 16, flat_seq_features: int = 20):
        super().__init__()
        self.spatial = SpatialMapper(fourier_dim=64, hidden_dim=spatial_dim)
        self.temporal = TemporalMLP(in_features=flat_seq_features, hidden_dim=temporal_dim)
        self.head = nn.Sequential(
            nn.Linear(spatial_dim + temporal_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, coords: torch.Tensor, flat_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s_emb = self.spatial(coords)
        t_emb = self.temporal(flat_seq)
        combined = torch.cat([s_emb, t_emb], dim=-1)
        pred = self.head(combined)
        return combined, pred

# ==========================================
# 2. DATA PROCESSING & SEQUENCING
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

def get_tensors_from_df(df: pd.DataFrame, device: torch.device):
    coords = df[["lat", "lon"]].to_numpy(dtype=np.float32)
    seq_cols = [TARGET, "hour_sin", "hour_cos", "minute_sin", "minute_cos"]
    lag_arrays = [df[[f"{col}_lag{lag}" for col in seq_cols]].fillna(0.0).to_numpy(dtype=np.float32) for lag in range(SEQ_LEN, 0, -1)]
    seqs = np.hstack(lag_arrays)
    
    coords_t = torch.from_numpy(coords).to(device)
    seqs_t = torch.from_numpy(seqs).to(device)
    
    if TARGET in df.columns and not df[TARGET].isna().all():
        targets = df[TARGET].to_numpy(dtype=np.float32)[:, None]
        targets_t = torch.from_numpy(targets).to(device)
        return coords_t, seqs_t, targets_t
    return coords_t, seqs_t, None

# ==========================================
# 3. TRAINING ROUTINES
# ==========================================

def train_pytorch_extractor(train_df: pd.DataFrame, device: torch.device):
    print("\n--- Phase 1: Training PyTorch Neural Feature Extractor ---")
    
    print("  -> Getting tensors...")
    coords_t, seqs_t, targets_t = get_tensors_from_df(train_df, device)
    
    print("  -> Initializing Model...")
    model = NeuralFeatureExtractor(spatial_dim=16, temporal_dim=16).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    batch_size = 256
    num_samples = len(coords_t)
    indices = np.arange(num_samples)
    
    print("  -> Entering Train Loop...")
    model.train()
    for epoch in range(1, 4):  # Quick pre-training
        np.random.shuffle(indices)
        total_loss = 0.0
        
        for start_idx in range(0, num_samples, batch_size):
            if start_idx == 0 and epoch == 1:
                print("      -> First batch slice...")
            batch_idx = indices[start_idx:start_idx + batch_size]
            
            c_batch = coords_t[batch_idx]
            s_batch = seqs_t[batch_idx]
            t_batch = targets_t[batch_idx]
            
            if start_idx == 0 and epoch == 1:
                print("      -> Forward pass...")
            optimizer.zero_grad()
            _, preds = model(c_batch, s_batch)
            loss = criterion(preds, t_batch)
            
            if start_idx == 0 and epoch == 1:
                print("      -> Backward pass...")
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"PyTorch Epoch {epoch} | Loss: {total_loss/(num_samples/batch_size):.5f}")
        
    return model

@torch.no_grad()
def extract_embeddings(model: nn.Module, df: pd.DataFrame, device: torch.device):
    """Extracts the 32-dim [Spatial, Temporal] embedding for the DataFrame."""
    print(f"Extracting embeddings for {len(df)} rows...")
    model.eval()
    
    coords_t, seqs_t, _ = get_tensors_from_df(df, device)
    
    batch_size = 512
    num_samples = len(coords_t)
    embeddings = []
    
    for start_idx in range(0, num_samples, batch_size):
        c_batch = coords_t[start_idx:start_idx + batch_size]
        s_batch = seqs_t[start_idx:start_idx + batch_size]
        
        emb, _ = model(c_batch, s_batch)
        embeddings.append(emb.cpu().numpy())
        
    return np.concatenate(embeddings, axis=0)

# ==========================================
# 4. GBDT & RECURSIVE INFERENCE
# ==========================================

def main():
    seed_everything(RANDOM_STATE)
    device = torch.device("cpu") # For stability
    
    print("Loading datasets...")
    train_raw = pd.read_csv("dataset/train.csv")
    test_raw = pd.read_csv("dataset/test.csv")
    
    full_df, scaler_coords, le_road, le_wea = prepare_data(train_raw, test_raw)
    
    print("Splitting data back...")
    train_full = full_df[full_df["is_test"] == 0].copy()
    test_full = full_df[full_df["is_test"] == 1].copy()
    
    print("Dropping NaNs...")
    train_clean = train_full.dropna(subset=[f"{TARGET}_lag{SEQ_LEN}"]).copy()
    
    print("Starting training...")
    pt_model = train_pytorch_extractor(train_clean, device)
    
    # 2. Extract Embeddings for Training GBDT
    print("\n--- Extracting Neural Embeddings ---")
    train_emb = extract_embeddings(pt_model, train_clean, device)
    
    # Build Tabular Training Set with Embeddings
    base_features = ["Temperature", "NumberofLanes", "RoadType_idx", "Weather_idx", 
                     "hour_sin", "hour_cos", "minute_sin", "minute_cos"]
    
    X_train_base = train_clean[base_features].values
    X_train_full_gbdt = np.hstack([X_train_base, train_emb])
    y_train_gbdt = train_clean[TARGET].values
    
    print("\n--- Phase 2: K-Fold Cross-Validation for Meta-Ensemble (LightGBM + CatBoost) ---")
    
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    r2_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full_gbdt), 1):
        print(f"  -> Fold {fold}")
        X_tr, X_val = X_train_full_gbdt[train_idx], X_train_full_gbdt[val_idx]
        y_tr, y_val = y_train_gbdt[train_idx], y_train_gbdt[val_idx]
        
        lgb_model_fold = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=63, verbose=-1, random_state=RANDOM_STATE)
        lgb_model_fold.fit(X_tr, y_tr)
        lgb_preds = lgb_model_fold.predict(X_val)
        
        cat_model_fold = cb.CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6, verbose=False, random_seed=RANDOM_STATE)
        cat_model_fold.fit(X_tr, y_tr)
        cat_preds = cat_model_fold.predict(X_val)
        
        # Simple average ensemble for CV
        final_preds = np.clip((lgb_preds + cat_preds) / 2.0, 0.0, 1.0)
        
        r2 = r2_score(y_val, final_preds)
        r2_scores.append(r2)
        print(f"    R2: {r2:.5f}")
        
    print(f"\nAverage GBDT CV R2: {np.mean(r2_scores):.5f} +/- {np.std(r2_scores):.5f}")
    
    print("\n--- Phase 3: Training Final Meta-Ensemble & Recursive Rollout Inference on Test Set ---")
    # Train final GBDT models on full data
    lgb_model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63, verbose=-1, random_state=RANDOM_STATE)
    lgb_model.fit(X_train_full_gbdt, y_train_gbdt)
    
    cat_model = cb.CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, verbose=False, random_seed=RANDOM_STATE)
    cat_model.fit(X_train_full_gbdt, y_train_gbdt)
    
    # We must predict test set slot by slot chronologically to update the lags
    
    # Sort test set by time slots
    test_slots = sorted(test_full["slot_id"].unique())
    
    # We need a dynamic dataframe that we can update. We use full_df up to the test set start.
    dynamic_df = full_df.copy()
    dynamic_df = dynamic_df.set_index(["geohash", "slot_id"])
    
    predictions = []
    
    for slot in test_slots:
        # Get current slot data
        current_mask = (full_df["is_test"] == 1) & (full_df["slot_id"] == slot)
        slot_df = full_df[current_mask].copy()
        
        # We must re-calculate lags dynamically for this slot
        # For each geohash in slot_df, grab the last SEQ_LEN targets
        for lag in range(1, SEQ_LEN + 1):
            lag_slot = slot - lag
            # Map the demand from the dynamic index
            slot_df[f"{TARGET}_lag{lag}"] = slot_df["geohash"].apply(
                lambda gh: dynamic_df.at[(gh, lag_slot), TARGET] if (gh, lag_slot) in dynamic_df.index else 0.0
            )
            
        # Now we have updated lags, we can extract embeddings
        slot_emb = extract_embeddings(pt_model, slot_df, device)
        
        X_test_base = slot_df[base_features].values
        X_test_full = np.hstack([X_test_base, slot_emb])
        
        # Predict
        lgb_pred = lgb_model.predict(X_test_full)
        cat_pred = cat_model.predict(X_test_full)
        final_pred = np.clip((lgb_pred + cat_pred) / 2.0, 0.0, 1.0)
        
        # Update dynamic dataframe with predictions
        for i, gh in enumerate(slot_df["geohash"]):
            dynamic_df.at[(gh, slot), TARGET] = final_pred[i]
            
        # Store for submission
        for i, orig_idx in enumerate(slot_df["original_index"]):
            predictions.append({"Index": int(orig_idx), "demand": final_pred[i]})
            
    print("\nSaving Submission...")
    sub_df = pd.DataFrame(predictions).sort_values("Index")
    out_path = Path("gemini_codes/gemini_v1_neural_boost_submission.csv")
    out_path.parent.mkdir(exist_ok=True, parents=True)
    sub_df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
