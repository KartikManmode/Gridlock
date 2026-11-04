"""Deep per-geohash sequence model for traffic demand.

Rows are grouped by geohash, sorted chronologically, converted into sliding
windows, and trained with shuffle=False so validation remains future-facing.

Example:
    python3 codex_codes/dl_sequence_model.py --train dataset/train.csv --epochs 20
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


def decode_geohash(value: str) -> tuple[float, float]:
    lat, lon = geohash.decode(str(value))
    return float(lat), float(lon)


@dataclass
class CategoryMaps:
    weather: dict[str, int]
    road_type: dict[str, int]
    geohash_prefix: dict[str, int]


class SequencePreprocessor:
    numeric_cols = [
        "observed_demand",
        "Temperature",
        "NumberofLanes",
        "LargeVehicles",
        "Landmarks",
        "hour_sin",
        "hour_cos",
        "minute_sin",
        "minute_cos",
        "lat",
        "lon",
    ]

    def __init__(self) -> None:
        self.maps: CategoryMaps | None = None
        self.numeric_mean: np.ndarray | None = None
        self.numeric_std: np.ndarray | None = None
        self.temp_by_weather: pd.Series | None = None
        self.global_temp: float | None = None

    def fit(self, df: pd.DataFrame) -> "SequencePreprocessor":
        work = self._basic(df)
        self.temp_by_weather = work.groupby("Weather")["Temperature"].median()
        self.global_temp = float(work["Temperature"].median())
        work = self._impute_temperature(work)
        self.maps = CategoryMaps(
            weather=self._make_map(work["Weather"]),
            road_type=self._make_map(work["RoadType"]),
            geohash_prefix=self._make_map(work["geohash_prefix"]),
        )
        numeric = work[self.numeric_cols].to_numpy(dtype=np.float32)
        self.numeric_mean = numeric.mean(axis=0)
        self.numeric_std = np.maximum(numeric.std(axis=0), 1e-6)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.maps is None or self.numeric_mean is None or self.numeric_std is None:
            raise RuntimeError("SequencePreprocessor must be fit before transform.")
        work = self._basic(df)
        work = self._impute_temperature(work)
        work[self.numeric_cols] = (work[self.numeric_cols] - self.numeric_mean) / self.numeric_std
        work["weather_id"] = work["Weather"].map(self.maps.weather).fillna(0).astype(int)
        work["road_type_id"] = work["RoadType"].map(self.maps.road_type).fillna(0).astype(int)
        work["geohash_prefix_id"] = work["geohash_prefix"].map(self.maps.geohash_prefix).fillna(0).astype(int)
        return work

    def _basic(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        work[["hour", "minute"]] = work["timestamp"].apply(parse_timestamp).tolist()
        work[["lat", "lon"]] = work["geohash"].apply(decode_geohash).tolist()
        work["minute_of_day"] = work["hour"] * 60 + work["minute"]
        work["slot_id"] = work["day"] * 96 + work["minute_of_day"] // 15
        work["hour_sin"] = np.sin(2.0 * np.pi * work["hour"] / 24.0)
        work["hour_cos"] = np.cos(2.0 * np.pi * work["hour"] / 24.0)
        work["minute_sin"] = np.sin(2.0 * np.pi * work["minute_of_day"] / 1440.0)
        work["minute_cos"] = np.cos(2.0 * np.pi * work["minute_of_day"] / 1440.0)
        work["RoadType"] = work["RoadType"].fillna("Unknown")
        work["Weather"] = work["Weather"].fillna("Unknown")
        work["LargeVehicles"] = (work["LargeVehicles"] == "Allowed").astype(float)
        work["Landmarks"] = (work["Landmarks"] == "Yes").astype(float)
        work["NumberofLanes"] = work["NumberofLanes"].astype(float)
        work["geohash_prefix"] = work["geohash"].astype(str).str[:5]
        if TARGET in work.columns:
            work["observed_demand"] = work[TARGET].clip(0.0, 1.0)
        else:
            work["observed_demand"] = 0.0
        return work.sort_values(["slot_id", "geohash"]).reset_index(drop=True)

    def _impute_temperature(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        if self.temp_by_weather is None or self.global_temp is None:
            raise RuntimeError("Temperature imputers are not fitted.")
        work["Temperature"] = work["Temperature"].fillna(work["Weather"].map(self.temp_by_weather))
        work["Temperature"] = work["Temperature"].fillna(self.global_temp)
        return work

    @staticmethod
    def _make_map(values: pd.Series) -> dict[str, int]:
        return {value: i + 1 for i, value in enumerate(sorted(values.astype(str).unique()))}


class SlidingSequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, sequence_length: int) -> None:
        self.sequence_length = sequence_length
        self.samples: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]] = []
        ordered = df.sort_values(["geohash", "slot_id"])
        for _, group in ordered.groupby("geohash", sort=False):
            group = group.sort_values("slot_id")
            if len(group) <= sequence_length:
                continue
            numeric = group[SequencePreprocessor.numeric_cols].to_numpy(dtype=np.float32)
            categorical = group[["weather_id", "road_type_id", "geohash_prefix_id"]].to_numpy(dtype=np.int64)
            targets = group[TARGET].clip(0.0, 1.0).to_numpy(dtype=np.float32)
            target_slots = group["slot_id"].to_numpy(dtype=np.int64)
            for end in range(sequence_length, len(group)):
                self.samples.append(
                    (
                        int(target_slots[end]),
                        numeric[end - sequence_length:end],
                        categorical[end - sequence_length:end, 0],
                        categorical[end - sequence_length:end, 1],
                        categorical[end - sequence_length:end, 2],
                        float(targets[end]),
                    )
                )
        self.samples.sort(key=lambda sample: sample[0])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _, numeric, weather, road_type, prefix, target = self.samples[idx]
        return (
            torch.from_numpy(numeric),
            torch.from_numpy(weather),
            torch.from_numpy(road_type),
            torch.from_numpy(prefix),
            torch.tensor(target, dtype=torch.float32),
        )


class TrafficLSTM(nn.Module):
    def __init__(self, numeric_dim: int, weather_size: int, road_size: int, prefix_size: int, hidden_dim: int) -> None:
        super().__init__()
        self.weather_embedding = nn.Embedding(weather_size + 1, 4, padding_idx=0)
        self.road_embedding = nn.Embedding(road_size + 1, 4, padding_idx=0)
        self.prefix_embedding = nn.Embedding(prefix_size + 1, 8, padding_idx=0)
        input_dim = numeric_dim + 4 + 4 + 8
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, dropout=0.1, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

    def forward(self, numeric: torch.Tensor, weather: torch.Tensor, road_type: torch.Tensor, prefix: torch.Tensor) -> torch.Tensor:
        x = torch.cat(
            [
                numeric,
                self.weather_embedding(weather),
                self.road_embedding(road_type),
                self.prefix_embedding(prefix),
            ],
            dim=-1,
        )
        encoded, _ = self.lstm(x)
        pred = self.head(encoded[:, -1]).squeeze(-1)
        return torch.clamp(pred, 0.0, 1.0)


def split_dataset(dataset: SlidingSequenceDataset, valid_fraction: float) -> tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    split = int(len(dataset) * (1.0 - valid_fraction))
    train_idx = list(range(split))
    valid_idx = list(range(split, len(dataset)))
    return torch.utils.data.Subset(dataset, train_idx), torch.utils.data.Subset(dataset, valid_idx)


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    criterion = nn.MSELoss()
    total = 0.0
    for numeric, weather, road_type, prefix, target in loader:
        numeric = numeric.to(device)
        weather = weather.to(device)
        road_type = road_type.to(device)
        prefix = prefix.to(device)
        target = target.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(numeric, weather, road_type, prefix)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * numeric.size(0)
    return total / max(len(loader.dataset), 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    preds, targets = [], []
    for numeric, weather, road_type, prefix, target in loader:
        pred = model(numeric.to(device), weather.to(device), road_type.to(device), prefix.to(device)).cpu().numpy()
        preds.append(pred)
        targets.append(target.numpy())
    y_pred = np.clip(np.concatenate(preds), 0.0, 1.0)
    y_true = np.clip(np.concatenate(targets), 0.0, 1.0)
    return math.sqrt(mean_squared_error(y_true, y_pred)), r2_score(y_true, y_pred)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("dataset/train.csv"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--valid-fraction", type=float, default=0.2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    seed_everything(RANDOM_STATE)
    raw = pd.read_csv(args.train)
    preprocessor = SequencePreprocessor().fit(raw)
    prepared = preprocessor.transform(raw)
    dataset = SlidingSequenceDataset(prepared, args.sequence_length)
    train_ds, valid_ds = split_dataset(dataset, args.valid_fraction)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)

    assert preprocessor.maps is not None
    model = TrafficLSTM(
        numeric_dim=len(SequencePreprocessor.numeric_cols),
        weather_size=len(preprocessor.maps.weather),
        road_size=len(preprocessor.maps.road_type),
        prefix_size=len(preprocessor.maps.geohash_prefix),
        hidden_dim=args.hidden_dim,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, torch.device(args.device))
        valid_rmse, valid_r2 = evaluate(model, valid_loader, torch.device(args.device))
        print(
            f"epoch={epoch:03d} train_mse={train_loss:.6f} "
            f"valid_rmse={valid_rmse:.6f} valid_r2={valid_r2:.6f}"
        )


if __name__ == "__main__":
    main()
