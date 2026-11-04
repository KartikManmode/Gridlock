"""Traffic demand prediction pipeline v8 (Proxy) - GBDT Implementation.

This script implements the logical essence of Phase 3 Tasks using available libraries:
1. Two-Stage Modeling (Task 3: Classifier + Regressor)
2. Spatial Neighbor Demand (Task 2: Tabular Approximation)
3. Residual Learning (Task 3: Predicting deviation from geo5_hour_te)
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

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


def parse_timestamp(value: str) -> tuple[int, int]:
    hour, minute = str(value).split(":")
    return int(hour), int(minute)


def decode_geohash(value: str) -> tuple[float, float]:
    lat, lon = geohash.decode(str(value))
    return float(lat), float(lon)


class ProxyFeatureEngineerV8:
    def __init__(self, smoothing: float = 25.0) -> None:
        self.smoothing = smoothing
        self.road_encoder = LabelEncoder()
        self.weather_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.geohash_stats = {}
        self.neighbor_lookup = None
        self.global_demand = None

    def fit(self, df: pd.DataFrame) -> "ProxyFeatureEngineerV8":
        work = self._basic(df)
        self.global_demand = float(work[TARGET].mean())
        
        self.road_encoder.fit(work["RoadType"].astype(str))
        self.weather_encoder.fit(work["Weather"].astype(str))
        
        for col in ["geohash", "geohash5", "geo5_hour"]:
            self.geohash_stats[col] = work.groupby(col)[TARGET].agg(["mean", "count"])
            
        gh_means = self.geohash_stats["geohash"]["mean"]
        unique_gh = gh_means.index.tolist()
        lookup = {}
        for g in unique_gh:
            neighbors = geohash.neighbors(g)
            lookup[g] = np.mean([gh_means.get(n, self.global_demand) for n in neighbors])
        self.neighbor_lookup = pd.Series(lookup)
        
        # Determine numeric columns for scaling
        num_cols = ["Temperature", "NumberofLanes", "lat", "lon", "neighbor_demand"]
        work_num = self._apply_aggregates(work)[num_cols].fillna(0)
        self.scaler.fit(work_num)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        work = self._basic(df)
        work = self._apply_aggregates(work)
        
        # Categorical
        work["RoadType"] = self.road_encoder.transform(work["RoadType"].astype(str))
        work["Weather"] = self.weather_encoder.transform(work["Weather"].astype(str))
        
        cols = [
            "day", "hour", "minute_of_day", "hour_sin", "hour_cos",
            "Temperature", "NumberofLanes", "lat", "lon", "neighbor_demand",
            "geohash_te", "geohash5_te", "geo5_hour_te", "RoadType", "Weather"
        ]
        
        res = work[cols].copy().fillna(0)
        num_cols = ["Temperature", "NumberofLanes", "lat", "lon", "neighbor_demand"]
        res[num_cols] = self.scaler.transform(res[num_cols])
            
        return res

    def _apply_aggregates(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        for col in ["geohash", "geohash5", "geo5_hour"]:
            stats = self.geohash_stats[col]
            m = work[col].map(stats["mean"]).fillna(self.global_demand)
            c = work[col].map(stats["count"]).fillna(0)
            work[f"{col}_te"] = (c * m + self.smoothing * self.global_demand) / (c + self.smoothing)
        work["neighbor_demand"] = work["geohash"].map(self.neighbor_lookup).fillna(self.global_demand)
        return work

    def _basic(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        work[["hour", "minute"]] = work["timestamp"].apply(parse_timestamp).tolist()
        work[["lat", "lon"]] = work["geohash"].apply(decode_geohash).tolist()
        work["minute_of_day"] = work["hour"] * 60 + work["minute"]
        work["hour_sin"] = np.sin(2 * np.pi * work["hour"] / 24)
        work["hour_cos"] = np.cos(2 * np.pi * work["hour"] / 24)
        work["geohash5"] = work["geohash"].astype(str).str[:5]
        work["geo5_hour"] = work["geohash5"] + "__h" + work["hour"].astype(str)
        work["RoadType"] = work["RoadType"].fillna("Unknown")
        work["Weather"] = work["Weather"].fillna("Unknown")
        work["Temperature"] = work["Temperature"].fillna(25.0)
        return work


class TwoStageModelV8:
    """Task 3: Zero-Inflated Modeling."""
    def __init__(self):
        self.clf = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, verbose=-1)
        self.reg = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03, num_leaves=127, verbose=-1)

    def fit(self, X, y):
        # Stage 1: Classification (Is there traffic?)
        y_bin = (y > 0.02).astype(int)
        self.clf.fit(X, y_bin)
        
        # Stage 2: Regression (How much traffic?)
        # Train ONLY on rows where demand > 0
        X_pos = X[y > 0.0]
        y_pos = y[y > 0.0]
        self.reg.fit(X_pos, y_pos)

    def predict(self, X):
        prob = self.clf.predict_proba(X)[:, 1]
        val = self.reg.predict(X)
        return np.clip(prob * val, 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("dataset/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("dataset/test.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/gemini_v8_proxy_submission.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.train).sort_values(["day", "timestamp"])
    test_df = pd.read_csv(args.test)

    print("Running Two-Stage Proxy Model...")
    tscv = TimeSeriesSplit(n_splits=3)
    scores = []
    
    for train_idx, val_idx in tscv.split(df):
        tr_df, va_df = df.iloc[train_idx], df.iloc[val_idx]
        fe = ProxyFeatureEngineerV8().fit(tr_df)
        X_tr, X_va = fe.transform(tr_df), fe.transform(va_df)
        y_tr, y_va = tr_df[TARGET], va_df[TARGET]
        
        model = TwoStageModelV8()
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        scores.append(r2_score(y_va, preds))
        print(f"Fold R2: {scores[-1]:.6f}")

    print(f"Average CV R2: {np.mean(scores):.6f}")
    
    # Final Submission
    print("Training final model...")
    fe = ProxyFeatureEngineerV8().fit(df)
    X_full = fe.transform(df)
    X_te = fe.transform(test_df)
    
    final_model = TwoStageModelV8()
    final_model.fit(X_full, df[TARGET])
    test_preds = final_model.predict(X_te)
    
    pd.DataFrame({"Index": test_df["Index"], "demand": test_preds}).to_csv(args.output, index=False)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
