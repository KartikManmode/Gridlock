"""Spatial-temporal graph neural network for traffic demand.

This script treats geohashes as graph nodes and 15-minute intervals as graph
snapshots. It trains a simple GCN + GRU model with strict chronological
validation: windows whose prediction time is earlier are always used before
later windows.

Example:
    python3 codex_codes/dl_graph_model.py --train dataset/train.csv --epochs 20
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


TARGET = "demand"
RANDOM_STATE = 42


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_timestamp(value: str) -> tuple[int, int]:
    hour, minute = str(value).split(":")
    return int(hour), int(minute)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work[["hour", "minute"]] = work["timestamp"].apply(parse_timestamp).tolist()
    work["minute_of_day"] = work["hour"] * 60 + work["minute"]
    work["slot_id"] = work["day"] * 96 + work["minute_of_day"] // 15
    work["time_sin"] = np.sin(2.0 * np.pi * work["minute_of_day"] / 1440.0)
    work["time_cos"] = np.cos(2.0 * np.pi * work["minute_of_day"] / 1440.0)
    work["LargeVehicles"] = (work["LargeVehicles"] == "Allowed").astype(float)
    work["Landmarks"] = (work["Landmarks"] == "Yes").astype(float)
    work["NumberofLanes"] = work["NumberofLanes"].astype(float)
    work["Temperature"] = work["Temperature"].astype(float)
    work["Weather"] = work["Weather"].fillna("Unknown")
    return work


def fill_temperature(train: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    weather_median = train.groupby("Weather")["Temperature"].median()
    global_median = float(train["Temperature"].median())
    work["Temperature"] = work["Temperature"].fillna(work["Weather"].map(weather_median))
    work["Temperature"] = work["Temperature"].fillna(global_median)
    return work


def geohash_neighbors(value: str) -> set[str]:
    if hasattr(geohash, "neighbors"):
        return set(geohash.neighbors(value))
    neighbors: set[str] = set()
    for direction in ("n", "s", "e", "w"):
        try:
            first = geohash.adjacent(value, direction)
            neighbors.add(first)
            if direction in {"n", "s"}:
                neighbors.add(geohash.adjacent(first, "e"))
                neighbors.add(geohash.adjacent(first, "w"))
            else:
                neighbors.add(geohash.adjacent(first, "n"))
                neighbors.add(geohash.adjacent(first, "s"))
        except Exception:
            continue
    return neighbors


def build_adjacency(geohashes: list[str], device: torch.device) -> torch.Tensor:
    index = {g: i for i, g in enumerate(geohashes)}
    adjacency = np.eye(len(geohashes), dtype=np.float32)
    for source in geohashes:
        src_idx = index[source]
        for neighbor in geohash_neighbors(source):
            dst_idx = index.get(neighbor)
            if dst_idx is not None:
                adjacency[src_idx, dst_idx] = 1.0
                adjacency[dst_idx, src_idx] = 1.0
    degree = adjacency.sum(axis=1)
    degree_inv_sqrt = np.power(np.maximum(degree, 1.0), -0.5)
    normalized = degree_inv_sqrt[:, None] * adjacency * degree_inv_sqrt[None, :]
    return torch.tensor(normalized, dtype=torch.float32, device=device)


@dataclass
class SnapshotTensors:
    x: np.ndarray
    y: np.ndarray
    slots: np.ndarray
    geohashes: list[str]
    feature_names: list[str]


def build_snapshots(train_df: pd.DataFrame) -> SnapshotTensors:
    work = add_time_features(train_df)
    work = fill_temperature(work, work)
    geohashes = sorted(work["geohash"].unique())
    slots = np.array(sorted(work["slot_id"].unique()))
    node_index = {g: i for i, g in enumerate(geohashes)}
    slot_index = {s: i for i, s in enumerate(slots)}

    feature_names = [
        "observed_demand",
        "Temperature",
        "NumberofLanes",
        "LargeVehicles",
        "Landmarks",
        "time_sin",
        "time_cos",
    ]
    x = np.zeros((len(slots), len(geohashes), len(feature_names)), dtype=np.float32)
    y = np.zeros((len(slots), len(geohashes)), dtype=np.float32)

    rows = work[
        ["slot_id", "geohash", TARGET, "Temperature", "NumberofLanes", "LargeVehicles", "Landmarks", "time_sin", "time_cos"]
    ]
    for row in rows.itertuples(index=False):
        t = slot_index[row.slot_id]
        n = node_index[row.geohash]
        demand = float(np.clip(getattr(row, TARGET), 0.0, 1.0))
        x[t, n, 0] = demand
        x[t, n, 1:] = [
            row.Temperature,
            row.NumberofLanes,
            row.LargeVehicles,
            row.Landmarks,
            row.time_sin,
            row.time_cos,
        ]
        y[t, n] = demand

    numeric = x[:, :, 1:5]
    mean = numeric.reshape(-1, numeric.shape[-1]).mean(axis=0)
    std = numeric.reshape(-1, numeric.shape[-1]).std(axis=0)
    x[:, :, 1:5] = (numeric - mean) / np.maximum(std, 1e-6)
    return SnapshotTensors(x=x, y=y, slots=slots, geohashes=geohashes, feature_names=feature_names)


class GraphWindowDataset(Dataset):
    def __init__(self, snapshots: SnapshotTensors, indices: np.ndarray, history: int) -> None:
        self.snapshots = snapshots
        self.indices = indices
        self.history = history

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        target_t = int(self.indices[idx])
        start = target_t - self.history
        x = self.snapshots.x[start:target_t]
        y = self.snapshots.y[target_t]
        return torch.from_numpy(x), torch.from_numpy(y)


class GraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # x: batch, nodes, features
        propagated = torch.einsum("ij,bjf->bif", adjacency, x)
        return self.proj(propagated)


class STGCN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, adjacency: torch.Tensor, dropout: float) -> None:
        super().__init__()
        self.register_buffer("adjacency", adjacency)
        self.gcn = GraphConvolution(input_dim, hidden_dim)
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: batch, history, nodes, features
        batch, history, nodes, features = x.shape
        spatial_steps = []
        for t in range(history):
            h_t = torch.relu(self.gcn(x[:, t], self.adjacency))
            spatial_steps.append(h_t)
        h = torch.stack(spatial_steps, dim=2)  # batch, nodes, history, hidden
        h = h.reshape(batch * nodes, history, -1)
        encoded, _ = self.temporal(h)
        last = self.dropout(encoded[:, -1])
        pred = self.head(last).reshape(batch, nodes)
        return torch.clamp(pred, 0.0, 1.0)


def chronological_indices(num_slots: int, history: int, valid_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    all_indices = np.arange(history, num_slots)
    split = int(len(all_indices) * (1.0 - valid_fraction))
    return all_indices[:split], all_indices[split:]


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    criterion = nn.MSELoss()
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * x.size(0)
    return total_loss / max(len(loader.dataset), 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        pred = model(x.to(device)).cpu().numpy()
        preds.append(pred.reshape(-1))
        targets.append(y.numpy().reshape(-1))
    y_pred = np.clip(np.concatenate(preds), 0.0, 1.0)
    y_true = np.clip(np.concatenate(targets), 0.0, 1.0)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return rmse, r2_score(y_true, y_pred)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("dataset/train.csv"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--valid-fraction", type=float, default=0.2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    seed_everything(RANDOM_STATE)
    device = torch.device(args.device)
    train_df = pd.read_csv(args.train)
    snapshots = build_snapshots(train_df)
    train_idx, valid_idx = chronological_indices(len(snapshots.slots), args.history, args.valid_fraction)
    adjacency = build_adjacency(snapshots.geohashes, device)

    train_loader = DataLoader(GraphWindowDataset(snapshots, train_idx, args.history), batch_size=args.batch_size, shuffle=False)
    valid_loader = DataLoader(GraphWindowDataset(snapshots, valid_idx, args.history), batch_size=args.batch_size, shuffle=False)
    model = STGCN(len(snapshots.feature_names), args.hidden_dim, adjacency, dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        valid_rmse, valid_r2 = evaluate(model, valid_loader, device)
        print(
            f"epoch={epoch:03d} train_mse={train_loss:.6f} "
            f"valid_rmse={valid_rmse:.6f} valid_r2={valid_r2:.6f}"
        )


if __name__ == "__main__":
    main()
