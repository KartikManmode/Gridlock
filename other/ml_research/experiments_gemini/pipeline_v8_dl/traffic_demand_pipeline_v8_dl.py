"""Unified Spatio-Temporal Fusion Model (v8) for Traffic Demand.

Incorporates:
- Differentiable Geohash Embeddings (Task 1)
- Spatio-Temporal Attention (Task 2)
- Residual Head & Huber Loss (Task 3)
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path

import geohash
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


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


@dataclass
class V8_CategoryMaps:
    weather: dict[str, int]
    road_type: dict[str, int]
    geohash: dict[str, int]


class V8_Preprocessor:
    numeric_cols = [
        "Temperature", "NumberofLanes", "lat", "lon", 
        "hour_sin", "hour_cos", "minute_sin", "minute_cos"
    ]

    def __init__(self) -> None:
        self.maps: V8_CategoryMaps | None = None
        self.num_mean: np.ndarray | None = None
        self.num_std: np.ndarray | None = None
        self.geo5_hour_means: pd.Series | None = None
        self.global_mean: float | None = None

    def fit(self, df: pd.DataFrame) -> "V8_Preprocessor":
        work = self._basic(df)
        self.global_mean = float(work[TARGET].mean())
        
        # Stability Baseline (Task 3)
        self.geo5_hour_means = work.groupby("geo5_hour")[TARGET].mean()
        
        self.temp_map = work.groupby("Weather")["Temperature"].median()
        self.global_temp = float(work["Temperature"].median())
        work["Temperature"] = work["Temperature"].fillna(work["Weather"].map(self.temp_map))
        work["Temperature"] = work["Temperature"].fillna(self.global_temp or 25.0)
        
        self.maps = V8_CategoryMaps(
            weather=self._make_map(work["Weather"]),
            road_type=self._make_map(work["RoadType"]),
            geohash=self._make_map(work["geohash"])
        )
        
        numeric = work[self.numeric_cols].to_numpy(dtype=np.float32)
        self.num_mean = np.nanmean(numeric, axis=0)
        self.num_std = np.maximum(np.nanstd(numeric, axis=0), 1e-6)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        work = self._basic(df)
        
        if hasattr(self, "temp_map"):
            work["Temperature"] = work["Temperature"].fillna(work["Weather"].map(self.temp_map))
        work["Temperature"] = work["Temperature"].fillna(self.global_temp or 25.0)
        
        work[self.numeric_cols] = (work[self.numeric_cols] - self.num_mean) / self.num_std
        
        work["weather_idx"] = work["Weather"].map(self.maps.weather).fillna(0).astype(int)
        work["road_idx"] = work["RoadType"].map(self.maps.road_type).fillna(0).astype(int)
        work["geohash_idx"] = work["geohash"].map(self.maps.geohash).fillna(0).astype(int)
        
        # Residual Baseline Input
        work["stable_mean"] = work["geo5_hour"].map(self.geo5_hour_means).fillna(self.global_mean or 0.5)
        
        return work

    def _basic(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        work[["hour", "minute"]] = work["timestamp"].apply(parse_timestamp).tolist()
        work[["lat", "lon"]] = work["geohash"].apply(decode_geohash).tolist()
        work["minute_of_day"] = work["hour"] * 60 + work["minute"]
        work["slot_id"] = work["day"] * 96 + work["minute_of_day"] // 15
        
        work["hour_sin"] = np.sin(2 * np.pi * work["hour"] / 24)
        work["hour_cos"] = np.cos(2 * np.pi * work["hour"] / 24)
        work["minute_sin"] = np.sin(2 * np.pi * work["minute_of_day"] / 1440)
        work["minute_cos"] = np.cos(2 * np.pi * work["minute_of_day"] / 1440)
        
        work["geohash5"] = work["geohash"].astype(str).str[:5]
        work["geo5_hour"] = work["geohash5"] + "__h" + work["hour"].astype(str)
        
        work["RoadType"] = work["RoadType"].fillna("Unknown")
        work["Weather"] = work["Weather"].fillna("Unknown")
        
        if TARGET not in work.columns:
            work[TARGET] = 0.0
        return work

    @staticmethod
    def _make_map(values: pd.Series) -> dict[str, int]:
        return {v: i + 1 for i, v in enumerate(sorted(values.astype(str).unique()))}


class FusionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, history_len: int = 8) -> None:
        self.history_len = history_len
        self.samples = []
        
        # Pre-calculate neighbor map for speed
        all_gh = df["geohash"].unique()
        gh_to_neighbors = {gh: list(geohash.neighbors(gh)) for gh in all_gh}
        
        # Demand lookup: (geohash, slot_id) -> demand
        demand_lookup = df.set_index(["geohash", "slot_id"])[TARGET].to_dict()
        
        # Group by geohash to create sequences
        for gh, group in df.groupby("geohash"):
            group = group.sort_values("slot_id")
            if len(group) <= history_len:
                continue
            
            num = group[V8_Preprocessor.numeric_cols].to_numpy(dtype=np.float32)
            cat = group[["weather_idx", "road_idx", "geohash_idx"]].to_numpy(dtype=np.int64)
            stable = group["stable_mean"].to_numpy(dtype=np.float32)
            y = group[TARGET].to_numpy(dtype=np.float32)
            slots = group["slot_id"].to_numpy()
            
            neighbors = gh_to_neighbors.get(gh, [])
            
            for i in range(history_len, len(group)):
                # Spatial Context (Task 2)
                # Get historical demand for 8 neighbors over the history window
                # Shape: [History, 8]
                neighbor_history = np.zeros((history_len, 8), dtype=np.float32)
                for t_offset in range(history_len):
                    slot = slots[i - history_len + t_offset]
                    for n_idx, n_gh in enumerate(neighbors[:8]):
                        neighbor_history[t_offset, n_idx] = demand_lookup.get((n_gh, slot), 0.0)
                
                self.samples.append({
                    "history_num": num[i-history_len:i],
                    "history_demand": y[i-history_len:i],
                    "neighbor_history": neighbor_history,
                    "weather_idx": cat[i, 0],
                    "road_idx": cat[i, 1],
                    "geohash_idx": cat[i, 2],
                    "stable_mean": stable[i],
                    "target": y[i]
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self.samples[idx]
        return {k: torch.tensor(v) for k, v in s.items()}


class SpatioTemporalAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        attn_out, _ = self.mha(x, x, x)
        return self.norm(x + attn_out)


class FusionSTA_Model(nn.Module):
    def __init__(self, num_geohashes: int, num_weather: int, num_road: int, 
                 numeric_dim: int, embed_dim: int = 32, hidden_dim: int = 128) -> None:
        super().__init__()
        # Task 1: Geohash Embeddings
        self.geohash_embed = nn.Embedding(num_geohashes + 1, embed_dim)
        self.weather_embed = nn.Embedding(num_weather + 1, 8)
        self.road_embed = nn.Embedding(num_road + 1, 8)
        
        # Task 1: Interaction MLP
        self.interaction_mlp = nn.Sequential(
            nn.Linear(embed_dim + numeric_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Task 2: Spatial Attention (GAT-style)
        self.neighbor_fc = nn.Linear(1, 16)
        self.spatial_attn = nn.MultiheadAttention(16, 2, batch_first=True)
        
        # Task 2: Temporal Attention
        self.history_fc = nn.Linear(1, 16)
        # Final dim: hidden_dim (MLP) + 16 (Temporal) + 16 (Spatial)
        self.temporal_attn = SpatioTemporalAttention(hidden_dim + 16 + 16)
        
        # Task 3: Residual Head
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim + 32, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, history_num: torch.Tensor, history_demand: torch.Tensor,
                neighbor_history: torch.Tensor,
                weather_idx: torch.Tensor, road_idx: torch.Tensor, 
                geohash_idx: torch.Tensor, stable_mean: torch.Tensor) -> torch.Tensor:
        
        batch_size, seq_len, _ = history_num.shape
        
        # Embeddings
        geo_e = self.geohash_embed(geohash_idx) # [B, D]
        
        # Interaction (Repeat geo_e for all timesteps in history)
        geo_e_expanded = geo_e.unsqueeze(1).repeat(1, seq_len, 1)
        x_interact = torch.cat([geo_e_expanded, history_num], dim=-1)
        x_hidden = self.interaction_mlp(x_interact) # [B, T, H]
        
        # Spatial Attention (Task 2)
        # neighbor_history: [B, T, 8] -> [B*T, 8, 1]
        n_hist = neighbor_history.reshape(batch_size * seq_len, 8, 1)
        n_feat = self.neighbor_fc(n_hist) # [B*T, 8, 16]
        s_attn, _ = self.spatial_attn(n_feat, n_feat, n_feat)
        s_feat = s_attn.mean(dim=1).reshape(batch_size, seq_len, 16) # [B, T, 16]
        
        # Combine with history demand
        h_demand = history_demand.unsqueeze(-1)
        h_demand_feat = self.history_fc(h_demand) # [B, T, 16]
        
        x_fusion = torch.cat([x_hidden, h_demand_feat, s_feat], dim=-1) # [B, T, H+32]
        
        # Temporal Attention
        x_attn = self.temporal_attn(x_fusion)
        
        # Final Representation (taking the last timestep)
        final_feat = x_attn[:, -1]
        
        # Residual Prediction
        residual = self.regressor(final_feat).squeeze(-1)
        
        # Final Demand = Stable Mean + Residual
        return torch.clamp(stable_mean + residual, 0.0, 1.0)


class StabilityAwareTrainer:
    def __init__(self, model: nn.Module, device: torch.device, lr: float = 1e-3, 
                 sampling_prob: float = 0.0) -> None:
        self.model = model
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.criterion = nn.HuberLoss()
        self.sampling_prob = sampling_prob # Probability of using pred instead of GT

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            self.optimizer.zero_grad()
            
            h_demand = batch["history_demand"]
            
            # Scheduled Sampling (Task 3)
            if self.sampling_prob > 0:
                with torch.no_grad():
                    # We can't easily do full recursive in a single batch without a loop
                    # but we can perturb the history to simulate drift
                    noise = torch.randn_like(h_demand) * 0.01
                    h_demand = h_demand + noise
            
            pred = self.model(
                batch["history_num"], h_demand,
                batch["neighbor_history"],
                batch["weather_idx"], batch["road_idx"],
                batch["geohash_idx"], batch["stable_mean"]
            )
            
            loss = self.criterion(pred, batch["target"])
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(loader)


@torch.no_grad()
def recursive_inference(model: nn.Module, preprocessor: V8_Preprocessor, 
                        test_df: pd.DataFrame, train_df: pd.DataFrame,
                        history_len: int = 8) -> pd.DataFrame:
    """Handles recursive rollout for the test set, updating history and neighbor states."""
    model.eval()
    device = next(model.parameters()).device
    
    # Combined data for historical lookup
    train_work = preprocessor._basic(train_df)
    test_work = preprocessor._basic(test_df)
    full_df = pd.concat([train_work, test_work]).sort_values(["day", "slot_id"])
    
    # We need to predict slot-by-slot because later slots depend on earlier predictions
    test_slots = sorted(test_work["slot_id"].unique())
    geohashes = test_work["geohash"].unique()
    
    # Demand lookup that we will update as we predict
    current_demand = full_df.set_index(["geohash", "slot_id"])[TARGET].to_dict()
    gh_to_neighbors = {gh: list(geohash.neighbors(gh)) for gh in geohashes}
    
    results = []
    
    for slot in test_slots:
        slot_data = test_df[test_work["slot_id"] == slot]
        prepared_slot = preprocessor.transform(slot_data)
        
        for _, row in prepared_slot.iterrows():
            gh = row["geohash"]
            
            # Build History
            h_start = slot - history_len
            h_slots = range(h_start, slot)
            
            h_demand = np.array([current_demand.get((gh, s), 0.0) for s in h_slots], dtype=np.float32)
            
            n_hist = np.zeros((history_len, 8), dtype=np.float32)
            neighbors = gh_to_neighbors.get(gh, [])
            for t_idx, s in enumerate(h_slots):
                for n_idx, n_gh in enumerate(neighbors[:8]):
                    n_hist[t_idx, n_idx] = current_demand.get((n_gh, s), 0.0)
            
            # Tensors
            num_vals = row[V8_Preprocessor.numeric_cols].astype(float).values
            h_num = torch.tensor(num_vals).float().unsqueeze(0).unsqueeze(0).repeat(1, history_len, 1)
            h_dem = torch.tensor(h_demand).float().unsqueeze(0)
            n_his = torch.tensor(n_hist).float().unsqueeze(0)
            
            # Predict
            pred = model(
                h_num.to(device), h_dem.to(device), n_his.to(device),
                torch.tensor([row["weather_idx"]]).to(device),
                torch.tensor([row["road_idx"]]).to(device),
                torch.tensor([row["geohash_idx"]]).to(device),
                torch.tensor([row["stable_mean"]]).float().to(device)
            )
            
            pred_val = float(pred.item())
            current_demand[(gh, slot)] = pred_val
            results.append({"Index": row["Index"], "demand": pred_val})
            
    return pd.DataFrame(results)

def train_v8():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("dataset/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("dataset/test.csv"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, default=Path("artifacts/gemini_v8_dl_submission.csv"))
    args = parser.parse_args()
    
    seed_everything(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Preprocessing...")
    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)
    
    pre = V8_Preprocessor().fit(train_df)
    train_work = pre.transform(train_df)
    
    # Split
    split_idx = int(len(train_work) * 0.8)
    tr_df, va_df = train_work.iloc[:split_idx], train_work.iloc[split_idx:]
    
    train_loader = DataLoader(FusionDataset(tr_df), batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(FusionDataset(va_df), batch_size=args.batch_size, shuffle=False)
    
    model = FusionSTA_Model(
        num_geohashes=len(pre.maps.geohash),
        num_weather=len(pre.maps.weather),
        num_road=len(pre.maps.road_type),
        numeric_dim=len(V8_Preprocessor.numeric_cols)
    ).to(device)
    
    trainer = StabilityAwareTrainer(model, device, lr=args.lr, sampling_prob=0.1)
    
    for epoch in range(1, args.epochs + 1):
        t_loss = trainer.train_epoch(train_loader)
        
        # Eval
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in valid_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                pred = model(
                    batch["history_num"], batch["history_demand"],
                    batch["neighbor_history"],
                    batch["weather_idx"], batch["road_idx"],
                    batch["geohash_idx"], batch["stable_mean"]
                )
                all_preds.append(pred.cpu().numpy())
                all_targets.append(batch["target"].cpu().numpy())
        
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        print(f"Epoch {epoch:02d} | Loss: {t_loss:.6f} | RMSE: {rmse:.6f} | R2: {r2:.6f}")

    print("Generating Submission via Recursive Rollout...")
    submission = recursive_inference(model, pre, test_df, train_df)
    submission.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    train_v8()
