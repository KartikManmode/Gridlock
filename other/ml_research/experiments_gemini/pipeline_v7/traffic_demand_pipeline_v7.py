"""Traffic demand prediction pipeline v7 - Production Grade Ensemble.

Implements:
1. Chronological TimeSeriesSplit Cross-Validation
2. Native Categorical Handling for GBDTs
3. Spatial Neighborhood Features (Demand of 8 adjacent cells)
4. Smoothed Spatio-Temporal Target Encoding (OOF-like logic)
5. Dynamic Ensemble Weight Optimization (LGBM + CatBoost)
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gridlock")

import geohash
import lightgbm as lgb
import catboost as cb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler


RANDOM_STATE = 42
TARGET = "demand"


@dataclass
class Metrics:
    rmse: float
    mae: float
    r2: float


def parse_timestamp(value: str) -> tuple[int, int]:
    hour, minute = str(value).split(":")
    return int(hour), int(minute)


def decode_geohash(value: str) -> tuple[float, float]:
    lat, lon = geohash.decode(str(value))
    return float(lat), float(lon)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _build_neighbor_lookup(geohash_means: pd.Series, global_mean: float) -> pd.Series:
    unique_geohashes = geohash_means.index.tolist()
    lookup = {}
    for g in unique_geohashes:
        try:
            neighbors = geohash.neighbors(g)
            neighbor_demands = [geohash_means.get(n, global_mean) for n in neighbors]
            lookup[g] = np.mean(neighbor_demands)
        except:
            lookup[g] = global_mean
    return pd.Series(lookup)


class TrafficFeatureEngineerV7:
    def __init__(self, smoothing: float = 25.0) -> None:
        self.smoothing = smoothing
        self.road_encoder = LabelEncoder()
        self.weather_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        self.temp_map: pd.Series | None = None
        self.global_temp: float | None = None
        self.global_demand: float | None = None
        
        self.geohash_stats: dict[str, pd.Series] = {}
        self.neighbor_lookup: pd.Series | None = None
        
        self.numeric_to_scale = [
            "Temperature", "NumberofLanes", "lat", "lon", 
            "geohash_count", "neighbor_demand"
        ]

    def fit(self, df: pd.DataFrame) -> "TrafficFeatureEngineerV7":
        work = self._preprocess(df)
        
        # Imputation
        self.temp_map = work.groupby("Weather")["Temperature"].median()
        self.global_temp = float(work["Temperature"].median())
        self.global_demand = float(work[TARGET].mean())
        
        # Encoders
        self.road_encoder.fit(work["RoadType"].astype(str))
        self.weather_encoder.fit(work["Weather"].astype(str))
        
        # Aggregates
        for col in ["geohash", "geohash5", "geohash4", "geo5_hour", "geohash_hour"]:
            stats = work.groupby(col)[TARGET].agg(["mean", "count"])
            self.geohash_stats[col] = stats
            
        self.neighbor_lookup = _build_neighbor_lookup(
            self.geohash_stats["geohash"]["mean"], self.global_demand
        )
        
        features = self.transform(df)
        self.scaler.fit(features[self.numeric_to_scale].fillna(0))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        work = self._preprocess(df)
        
        # Impute
        if self.temp_map is not None:
            work["Temperature"] = work["Temperature"].fillna(work["Weather"].map(self.temp_map))
        work["Temperature"] = work["Temperature"].fillna(self.global_temp or 25.0)
        
        # Target Encoding
        for col in ["geohash", "geohash5", "geo5_hour", "geohash_hour"]:
            if col in self.geohash_stats:
                stats = self.geohash_stats[col]
                m = work[col].map(stats["mean"]).fillna(self.global_demand or 0.5)
                c = work[col].map(stats["count"]).fillna(0)
                work[f"{col}_te"] = (c * m + self.smoothing * (self.global_demand or 0.5)) / (c + self.smoothing)
            else:
                work[f"{col}_te"] = self.global_demand or 0.5
                
        # Spatial
        work["neighbor_demand"] = work["geohash"].map(self.neighbor_lookup).fillna(self.global_demand or 0.5)
        work["geohash_count"] = work["geohash"].map(self.geohash_stats["geohash"]["count"]).fillna(0)
        
        # Matrix
        cols = [
            "day", "hour", "minute", "minute_of_day", "hour_sin", "hour_cos",
            "minute_sin", "minute_cos", "is_rush_hour", "is_night", "Temperature", 
            "NumberofLanes", "lat", "lon", "lat_bin", "lon_bin", "lat_plus_lon",
            "lat_lon_product", "geohash_count", "neighbor_demand", "geohash_te", 
            "geohash5_te", "geo5_hour_te", "geohash_hour_te"
        ]
        
        features = work[cols].copy()
        features["RoadType"] = self.road_encoder.transform(work["RoadType"].astype(str))
        features["Weather"] = self.weather_encoder.transform(work["Weather"].astype(str))
        
        if hasattr(self.scaler, "mean_"):
            features[self.numeric_to_scale] = self.scaler.transform(features[self.numeric_to_scale].fillna(0))
            
        return features

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        work[["hour", "minute"]] = work["timestamp"].apply(parse_timestamp).tolist()
        work[["lat", "lon"]] = work["geohash"].apply(decode_geohash).tolist()
        
        work["RoadType"] = work["RoadType"].fillna("Unknown")
        work["Weather"] = work["Weather"].fillna("Unknown")
        
        work["minute_of_day"] = work["hour"] * 60 + work["minute"]
        work["hour_sin"] = np.sin(2 * np.pi * work["hour"] / 24)
        work["hour_cos"] = np.cos(2 * np.pi * work["hour"] / 24)
        work["minute_sin"] = np.sin(2 * np.pi * work["minute_of_day"] / 1440)
        work["minute_cos"] = np.cos(2 * np.pi * work["minute_of_day"] / 1440)
        
        work["is_rush_hour"] = work["hour"].between(7, 9) | work["hour"].between(17, 19)
        work["is_night"] = (work["hour"] >= 22) | (work["hour"] <= 6)
        
        # Spatial Bins
        work["lat_bin"] = np.digitize(work["lat"], bins=np.linspace(-90, 90, 20))
        work["lon_bin"] = np.digitize(work["lon"], bins=np.linspace(-180, 180, 20))
        work["lat_plus_lon"] = work["lat"] + work["lon"]
        work["lat_lon_product"] = work["lat"] * work["lon"]
        
        work["geohash5"] = work["geohash"].astype(str).str[:5]
        work["geohash4"] = work["geohash"].astype(str).str[:4]
        work["geo5_hour"] = work["geohash5"] + "__h" + work["hour"].astype(str)
        work["geohash_hour"] = work["geohash"].astype(str) + "__h" + work["hour"].astype(str)
        
        return work


def find_optimal_weights(preds_list: list[np.ndarray], y_true: np.ndarray) -> np.ndarray:
    def loss_function(weights: np.ndarray) -> float:
        normalized_weights = weights / np.sum(weights)
        blend = np.sum([w * p for w, p in zip(normalized_weights, preds_list)], axis=0)
        return float(np.sqrt(mean_squared_error(y_true, np.clip(blend, 0.0, 1.0))))

    res = minimize(loss_function, np.ones(len(preds_list)) / len(preds_list), 
                   bounds=[(0, 1)] * len(preds_list), method="Nelder-Mead")
    return res.x / np.sum(res.x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("dataset/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("dataset/test.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/gemini_v7_submission.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.train).sort_values(["day", "timestamp"])
    test_df = pd.read_csv(args.test)

    # Cross Validation
    tscv = TimeSeriesSplit(n_splits=5)
    lgbm_oof = np.zeros(len(df))
    cat_oof = np.zeros(len(df))
    valid_indices = []

    print("Running TimeSeries CV...")
    for train_idx, val_idx in tscv.split(df):
        train_fold, val_fold = df.iloc[train_idx], df.iloc[val_idx]
        fe = TrafficFeatureEngineerV7().fit(train_fold)
        X_tr, X_val = fe.transform(train_fold), fe.transform(val_fold)
        y_tr, y_val = train_fold[TARGET], val_fold[TARGET]
        
        # LGBM
        m_lgb = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, num_leaves=63, verbose=-1)
        m_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], categorical_feature=["RoadType", "Weather"],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        lgbm_oof[val_idx] = m_lgb.predict(X_val)
        
        # CatBoost
        m_cat = cb.CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6, verbose=False)
        m_cat.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], cat_features=["RoadType", "Weather"],
                   early_stopping_rounds=50)
        cat_oof[val_idx] = m_cat.predict(X_val)
        valid_indices.extend(val_idx)

    y_true_val = df.iloc[valid_indices][TARGET].to_numpy()
    weights = find_optimal_weights([lgbm_oof[valid_indices], cat_oof[valid_indices]], y_true_val)
    print(f"Optimal Weights: LGBM={weights[0]:.4f}, CatBoost={weights[1]:.4f}")

    # Final Train
    print("Training final models on full data...")
    fe = TrafficFeatureEngineerV7().fit(df)
    X_full = fe.transform(df)
    X_test = fe.transform(test_df)
    
    m_lgb = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63, verbose=-1)
    m_lgb.fit(X_full, df[TARGET], categorical_feature=["RoadType", "Weather"])
    
    m_cat = cb.CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, verbose=False)
    m_cat.fit(X_full, df[TARGET], cat_features=["RoadType", "Weather"])
    
    test_preds = np.clip(weights[0] * m_lgb.predict(X_test) + weights[1] * m_cat.predict(X_test), 0.0, 1.0)
    
    pd.DataFrame({"Index": test_df["Index"], "demand": test_preds}).to_csv(args.output, index=False)
    print(f"Submission saved to {args.output}")

if __name__ == "__main__":
    main()
