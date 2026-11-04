"""Advanced traffic demand model with spatio-temporal features.

Examples:
    python3 advanced_traffic_model.py --local-score
    python3 advanced_traffic_model.py --predict-test dataset/test.csv --output artifacts/advanced_submission.csv

This file intentionally leaves traffic_demand_pipeline.py and linear_reg.py intact.
It adds higher-signal features:
    - cyclic time features
    - geohash prefixes
    - road/weather interactions
    - smoothed target encoding
    - exact previous-slot lags
    - same-time previous-day seasonal lags
    - neighboring-geohash historical demand
    - recursive test prediction so earlier test predictions can support later lags
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gridlock")

import geohash
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


RANDOM_STATE = 42
TARGET = "demand"
MISSING = "Missing"


@dataclass
class Metrics:
    rmse: float
    mae: float
    r2: float

    @property
    def score(self) -> float:
        return float(np.clip(self.r2 * 100.0, 0.0, 100.0))


def parse_timestamp(value: str) -> tuple[int, int]:
    hour, minute = str(value).split(":")
    return int(hour), int(minute)


def decode_geohash(value: str) -> tuple[float, float]:
    lat, lon = geohash.decode(str(value))
    return float(lat), float(lon)


@lru_cache(maxsize=None)
def cached_neighbors(value: str) -> tuple[str, ...]:
    return tuple(geohash.neighbors(value))


def previous_day_slot(day: int, slot: int, steps: int) -> tuple[int, int]:
    absolute = day * 96 + slot - steps
    return absolute // 96, absolute % 96


def mode_or_default(series: pd.Series, default: str) -> str:
    mode = series.dropna().mode()
    return str(mode.iloc[0]) if not mode.empty else default


def compute_metrics(y_true: pd.Series, predictions: np.ndarray) -> Metrics:
    clipped = np.clip(predictions, 0.0, 1.0)
    return Metrics(
        rmse=float(np.sqrt(mean_squared_error(y_true, clipped))),
        mae=float(mean_absolute_error(y_true, clipped)),
        r2=float(r2_score(y_true, clipped)),
    )


def print_metrics(name: str, metrics: Metrics) -> None:
    print(f"{name} RMSE:  {metrics.rmse:.6f}")
    print(f"{name} MAE:   {metrics.mae:.6f}")
    print(f"{name} R2:    {metrics.r2:.6f}")
    print(f"{name} Score: {metrics.score:.2f} / 100")


class AdvancedFeatureEngineer:
    def __init__(self, smoothing: float = 50.0) -> None:
        self.smoothing = smoothing
        self.road_mode: str | None = None
        self.weather_mode: str | None = None
        self.temperature_by_weather: pd.Series | None = None
        self.global_temperature: float | None = None
        self.global_target: float | None = None
        self.target_maps: dict[str, pd.Series] = {}
        self.count_maps: dict[str, pd.Series] = {}
        self.dummy_columns: list[str] = []
        self.feature_columns: list[str] = []
        self.scaler = StandardScaler()
        self.scale_columns = [
            "Temperature",
            "NumberofLanes",
            "lat",
            "lon",
            "geohash_count",
            "geohash5_count",
            "geohash4_count",
        ]
        self.target_encoded_columns = [
            "geohash",
            "geohash5",
            "geohash4",
            "road_weather",
            "weather_hour",
            "road_lanes",
            "geo5_hour",
            "geo4_hour",
            "geohash_hour",
        ]

    def fit(self, df: pd.DataFrame) -> "AdvancedFeatureEngineer":
        work = self._base_features(df, fit=True)
        self.global_target = float(work[TARGET].mean())
        self.temperature_by_weather = work.groupby("Weather")["Temperature"].median()
        self.global_temperature = float(work["Temperature"].median())

        for column in self.target_encoded_columns:
            stats = work.groupby(column)[TARGET].agg(["count", "mean"])
            smooth = (
                stats["count"] * stats["mean"] + self.smoothing * self.global_target
            ) / (stats["count"] + self.smoothing)
            self.target_maps[column] = smooth
            self.count_maps[column] = stats["count"]

        work = self._impute_temperature(work)
        static = self._static_features(work, fit=True)
        dynamic = self._dynamic_features(work, history=work)
        features = pd.concat([static, dynamic], axis=1)
        self.feature_columns = features.columns.tolist()
        self.scaler.fit(features[self.scale_columns])
        return self

    def transform(self, df: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
        work = self._base_features(df, fit=False)
        work = self._impute_temperature(work)
        static = self._static_features(work, fit=False)
        dynamic = self._dynamic_features(work, history=history)
        features = pd.concat([static, dynamic], axis=1)
        features = features.reindex(columns=self.feature_columns, fill_value=0.0)
        features[self.scale_columns] = self.scaler.transform(features[self.scale_columns])
        return features

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df, history=df)

    def _base_features(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        work = df.copy()
        work[["hour", "minute"]] = work["timestamp"].apply(parse_timestamp).tolist()
        work[["lat", "lon"]] = work["geohash"].apply(decode_geohash).tolist()

        if fit:
            self.road_mode = mode_or_default(work["RoadType"], MISSING)
            self.weather_mode = mode_or_default(work["Weather"], MISSING)

        work["RoadType"] = work["RoadType"].fillna(self.road_mode or MISSING).astype(str)
        work["Weather"] = work["Weather"].fillna(self.weather_mode or MISSING).astype(str)
        work["LargeVehicles_allowed"] = (work["LargeVehicles"] == "Allowed").astype(int)
        work["Landmarks_yes"] = (work["Landmarks"] == "Yes").astype(int)

        work["minute_of_day"] = work["hour"] * 60 + work["minute"]
        work["slot"] = (work["minute_of_day"] // 15).astype(int)
        work["hour_sin"] = np.sin(2 * np.pi * work["hour"] / 24)
        work["hour_cos"] = np.cos(2 * np.pi * work["hour"] / 24)
        work["time_sin"] = np.sin(2 * np.pi * work["minute_of_day"] / 1440)
        work["time_cos"] = np.cos(2 * np.pi * work["minute_of_day"] / 1440)
        work["is_rush_hour"] = (
            work["hour"].between(7, 9) | work["hour"].between(17, 19)
        ).astype(int)
        work["is_night"] = ((work["hour"] >= 22) | (work["hour"] <= 6)).astype(int)
        work["is_weekend"] = (work["day"] % 7).isin([5, 6]).astype(int)

        work["geohash5"] = work["geohash"].astype(str).str[:5]
        work["geohash4"] = work["geohash"].astype(str).str[:4]
        work["road_weather"] = work["RoadType"] + "__" + work["Weather"]
        work["weather_hour"] = work["Weather"] + "__h" + work["hour"].astype(str)
        work["road_lanes"] = work["RoadType"] + "__lanes" + work["NumberofLanes"].astype(str)
        work["geo5_hour"] = work["geohash5"] + "__h" + work["hour"].astype(str)
        work["geo4_hour"] = work["geohash4"] + "__h" + work["hour"].astype(str)
        work["geohash_hour"] = work["geohash"].astype(str) + "__h" + work["hour"].astype(str)
        work["lanes_x_large_allowed"] = (
            work["NumberofLanes"] * work["LargeVehicles_allowed"]
        )
        return work

    def _impute_temperature(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        if self.temperature_by_weather is None or self.global_temperature is None:
            raise RuntimeError("Feature engineer must be fit before transform.")
        weather_medians = work["Weather"].map(self.temperature_by_weather)
        work["Temperature_missing"] = work["Temperature"].isna().astype(int)
        work["Temperature"] = work["Temperature"].fillna(weather_medians)
        work["Temperature"] = work["Temperature"].fillna(self.global_temperature)
        return work

    def _static_features(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        numeric = df[
            [
                "day",
                "hour",
                "minute",
                "minute_of_day",
                "slot",
                "hour_sin",
                "hour_cos",
                "time_sin",
                "time_cos",
                "is_rush_hour",
                "is_night",
                "is_weekend",
                "LargeVehicles_allowed",
                "Landmarks_yes",
                "lanes_x_large_allowed",
                "Temperature_missing",
                "Temperature",
                "NumberofLanes",
                "lat",
                "lon",
            ]
        ].copy()

        for column in self.target_encoded_columns:
            target_map = self.target_maps.get(column)
            count_map = self.count_maps.get(column)
            fallback = self.global_target if self.global_target is not None else 0.0
            numeric[f"{column}_te"] = df[column].map(target_map).fillna(fallback)
            numeric[f"{column}_count"] = np.log1p(df[column].map(count_map).fillna(0.0))

        dummies = pd.get_dummies(
            df[["RoadType", "Weather", "road_weather"]], drop_first=False
        )
        if fit:
            self.dummy_columns = dummies.columns.tolist()
        else:
            dummies = dummies.reindex(columns=self.dummy_columns, fill_value=0)
        return pd.concat([numeric, dummies.astype(int)], axis=1)

    def _dynamic_features(self, df: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
        prepared_history = self._prepare_history(history)
        exact = prepared_history.set_index(["geohash", "day", "slot"])[TARGET].to_dict()
        prefix5 = prepared_history.groupby(["geohash5", "day", "slot"])[TARGET].mean().to_dict()
        prefix4 = prepared_history.groupby(["geohash4", "day", "slot"])[TARGET].mean().to_dict()
        fallback = self.global_target if self.global_target is not None else float(history[TARGET].mean())

        rows: list[dict[str, float]] = []
        for row in df.itertuples(index=False):
            day = int(row.day)
            slot = int(row.slot)
            ghash = str(row.geohash)
            g5 = str(row.geohash5)
            g4 = str(row.geohash4)

            lag1_day, lag1_slot = previous_day_slot(day, slot, 1)
            lag2_day, lag2_slot = previous_day_slot(day, slot, 2)
            lag3_day, lag3_slot = previous_day_slot(day, slot, 3)
            seasonal_day, seasonal_slot = day - 1, slot

            lag1 = exact.get((ghash, lag1_day, lag1_slot), fallback)
            lag2 = exact.get((ghash, lag2_day, lag2_slot), fallback)
            lag3 = exact.get((ghash, lag3_day, lag3_slot), fallback)
            seasonal = exact.get((ghash, seasonal_day, seasonal_slot), fallback)
            rolling3 = float(np.mean([lag1, lag2, lag3]))

            neighbor_prev_values = [
                exact[(neighbor, lag1_day, lag1_slot)]
                for neighbor in cached_neighbors(ghash)
                if (neighbor, lag1_day, lag1_slot) in exact
            ]
            neighbor_seasonal_values = [
                exact[(neighbor, seasonal_day, seasonal_slot)]
                for neighbor in cached_neighbors(ghash)
                if (neighbor, seasonal_day, seasonal_slot) in exact
            ]

            rows.append(
                {
                    "lag_1": lag1,
                    "lag_2": lag2,
                    "rolling3": rolling3,
                    "seasonal_lag_1d": seasonal,
                    "prefix5_seasonal_lag_1d": prefix5.get(
                        (g5, seasonal_day, seasonal_slot), fallback
                    ),
                    "prefix4_seasonal_lag_1d": prefix4.get(
                        (g4, seasonal_day, seasonal_slot), fallback
                    ),
                    "neighbor_lag_1_mean": float(np.mean(neighbor_prev_values))
                    if neighbor_prev_values
                    else fallback,
                    "neighbor_seasonal_lag_1d_mean": float(
                        np.mean(neighbor_seasonal_values)
                    )
                    if neighbor_seasonal_values
                    else fallback,
                    "neighbor_lag_1_count": float(len(neighbor_prev_values)),
                    "neighbor_seasonal_count": float(len(neighbor_seasonal_values)),
                }
            )
        return pd.DataFrame(rows, index=df.index)

    def _prepare_history(self, history: pd.DataFrame) -> pd.DataFrame:
        work = self._base_features(history, fit=False)
        return work[["geohash", "geohash5", "geohash4", "day", "slot", TARGET]].copy()


class GeneralizerFeatureEngineer:
    """Coordinate-first, no-lag feature set for chronological validation."""

    def __init__(self) -> None:
        self.road_mode: str | None = None
        self.weather_mode: str | None = None
        self.temperature_by_weather: pd.Series | None = None
        self.global_temperature: float | None = None
        self.lat_edges: np.ndarray | None = None
        self.lon_edges: np.ndarray | None = None
        self.dummy_columns: list[str] = []
        self.feature_columns: list[str] = []
        self.scaler = StandardScaler()
        self.scale_columns = [
            "Temperature",
            "NumberofLanes",
            "lat",
            "lon",
            "lat_lon",
            "lat_sq",
            "lon_sq",
            "temp_x_rush",
            "lanes_x_rush",
        ]

    def fit(self, df: pd.DataFrame) -> "GeneralizerFeatureEngineer":
        work = self._base_features(df, fit=True)
        self.temperature_by_weather = work.groupby("Weather")["Temperature"].median()
        self.global_temperature = float(work["Temperature"].median())
        self.lat_edges = np.linspace(work["lat"].min(), work["lat"].max(), 9)
        self.lon_edges = np.linspace(work["lon"].min(), work["lon"].max(), 9)
        work = self._impute_temperature(work)
        work = self._spatial_bins(work)
        features = self._encode(work, fit=True)
        self.feature_columns = features.columns.tolist()
        self.scaler.fit(features[self.scale_columns])
        return self

    def transform(self, df: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.DataFrame:
        work = self._base_features(df, fit=False)
        work = self._impute_temperature(work)
        work = self._spatial_bins(work)
        features = self._encode(work, fit=False)
        features = features.reindex(columns=self.feature_columns, fill_value=0.0)
        features[self.scale_columns] = self.scaler.transform(features[self.scale_columns])
        return features

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)

    def _base_features(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        work = df.copy()
        work[["hour", "minute"]] = work["timestamp"].apply(parse_timestamp).tolist()
        work[["lat", "lon"]] = work["geohash"].apply(decode_geohash).tolist()

        if fit:
            self.road_mode = mode_or_default(work["RoadType"], MISSING)
            self.weather_mode = mode_or_default(work["Weather"], MISSING)

        work["RoadType"] = work["RoadType"].fillna(self.road_mode or MISSING).astype(str)
        work["Weather"] = work["Weather"].fillna(self.weather_mode or MISSING).astype(str)
        work["LargeVehicles_allowed"] = (work["LargeVehicles"] == "Allowed").astype(int)
        work["Landmarks_yes"] = (work["Landmarks"] == "Yes").astype(int)

        minute_of_day = work["hour"] * 60 + work["minute"]
        work["hour_sin"] = np.sin(2 * np.pi * work["hour"] / 24)
        work["hour_cos"] = np.cos(2 * np.pi * work["hour"] / 24)
        work["time_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
        work["time_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)
        work["is_rush_hour"] = (
            work["hour"].between(7, 9) | work["hour"].between(17, 19)
        ).astype(int)
        work["is_night"] = ((work["hour"] >= 22) | (work["hour"] <= 6)).astype(int)
        work["is_weekend"] = (work["day"] % 7).isin([5, 6]).astype(int)
        work["lanes_x_large_allowed"] = (
            work["NumberofLanes"] * work["LargeVehicles_allowed"]
        )
        work["lat_lon"] = work["lat"] * work["lon"]
        work["lat_sq"] = work["lat"] ** 2
        work["lon_sq"] = work["lon"] ** 2
        work["temp_x_rush"] = work["Temperature"].fillna(0) * work["is_rush_hour"]
        work["lanes_x_rush"] = work["NumberofLanes"] * work["is_rush_hour"]
        work["weather_rush"] = (
            work["Weather"] + "__rush" + work["is_rush_hour"].astype(str)
        )
        work["road_rush"] = (
            work["RoadType"] + "__rush" + work["is_rush_hour"].astype(str)
        )
        return work

    def _impute_temperature(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        if self.temperature_by_weather is None or self.global_temperature is None:
            raise RuntimeError("Feature engineer must be fit before transform.")
        weather_medians = work["Weather"].map(self.temperature_by_weather)
        work["Temperature_missing"] = work["Temperature"].isna().astype(int)
        work["Temperature"] = work["Temperature"].fillna(weather_medians)
        work["Temperature"] = work["Temperature"].fillna(self.global_temperature)
        work["temp_x_rush"] = work["Temperature"] * work["is_rush_hour"]
        return work

    def _spatial_bins(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        if self.lat_edges is None or self.lon_edges is None:
            raise RuntimeError("Spatial bins are not fitted.")
        lat_bin = np.digitize(work["lat"], self.lat_edges[1:-1], right=False)
        lon_bin = np.digitize(work["lon"], self.lon_edges[1:-1], right=False)
        work["lat_bin"] = lat_bin.astype(str)
        work["lon_bin"] = lon_bin.astype(str)
        work["spatial_grid"] = work["lat_bin"] + "__" + work["lon_bin"]
        return work

    def _encode(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        numeric = df[
            [
                "hour_sin",
                "hour_cos",
                "time_sin",
                "time_cos",
                "is_rush_hour",
                "is_night",
                "is_weekend",
                "LargeVehicles_allowed",
                "Landmarks_yes",
                "lanes_x_large_allowed",
                "Temperature_missing",
                "Temperature",
                "NumberofLanes",
                "lat",
                "lon",
                "lat_lon",
                "lat_sq",
                "lon_sq",
                "temp_x_rush",
                "lanes_x_rush",
            ]
        ].copy()
        dummies = pd.get_dummies(
            df[["RoadType", "Weather", "weather_rush", "road_rush", "spatial_grid"]],
            drop_first=False,
        )
        if fit:
            self.dummy_columns = dummies.columns.tolist()
        else:
            dummies = dummies.reindex(columns=self.dummy_columns, fill_value=0)
        return pd.concat([numeric, dummies.astype(int)], axis=1)


class StableMemoryFeatureEngineer(GeneralizerFeatureEngineer):
    """No-lag model that keeps smoothed geohash/time memory for submission stability."""

    def __init__(self, smoothing: float = 35.0) -> None:
        super().__init__()
        self.smoothing = smoothing
        self.global_target: float | None = None
        self.target_maps: dict[str, pd.Series] = {}
        self.count_maps: dict[str, pd.Series] = {}
        self.memory_columns = [
            "geohash",
            "geohash5",
            "geohash4",
            "timestamp",
            "geo5_hour",
            "geo4_hour",
            "geohash_hour",
            "road_weather",
            "weather_hour",
            "weather_rush",
            "spatial_grid",
        ]

    def fit(self, df: pd.DataFrame) -> "StableMemoryFeatureEngineer":
        work = self._base_features(df, fit=True)
        self.temperature_by_weather = work.groupby("Weather")["Temperature"].median()
        self.global_temperature = float(work["Temperature"].median())
        self.lat_edges = np.linspace(work["lat"].min(), work["lat"].max(), 9)
        self.lon_edges = np.linspace(work["lon"].min(), work["lon"].max(), 9)
        work = self._impute_temperature(work)
        work = self._spatial_bins(work)
        work = self._add_memory_keys(work)
        self.global_target = float(work[TARGET].mean())

        for column in self.memory_columns:
            stats = work.groupby(column)[TARGET].agg(["count", "mean"])
            smooth = (
                stats["count"] * stats["mean"] + self.smoothing * self.global_target
            ) / (stats["count"] + self.smoothing)
            self.target_maps[column] = smooth
            self.count_maps[column] = stats["count"]

        features = self._encode(work, fit=True)
        self.feature_columns = features.columns.tolist()
        self.scaler.fit(features[self.scale_columns])
        return self

    def transform(self, df: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.DataFrame:
        work = self._base_features(df, fit=False)
        work = self._impute_temperature(work)
        work = self._spatial_bins(work)
        work = self._add_memory_keys(work)
        features = self._encode(work, fit=False)
        features = features.reindex(columns=self.feature_columns, fill_value=0.0)
        features[self.scale_columns] = self.scaler.transform(features[self.scale_columns])
        return features

    def _base_features(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        work = super()._base_features(df, fit)
        work["geohash5"] = work["geohash"].astype(str).str[:5]
        work["geohash4"] = work["geohash"].astype(str).str[:4]
        work["road_weather"] = work["RoadType"] + "__" + work["Weather"]
        work["weather_hour"] = work["Weather"] + "__h" + work["hour"].astype(str)
        return work

    def _add_memory_keys(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        work["geo5_hour"] = work["geohash5"] + "__h" + work["hour"].astype(str)
        work["geo4_hour"] = work["geohash4"] + "__h" + work["hour"].astype(str)
        work["geohash_hour"] = work["geohash"].astype(str) + "__h" + work["hour"].astype(str)
        return work

    def _encode(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        features = super()._encode(df, fit)
        fallback = self.global_target if self.global_target is not None else 0.0
        for column in self.memory_columns:
            target_map = self.target_maps.get(column)
            count_map = self.count_maps.get(column)
            features[f"{column}_te"] = df[column].map(target_map).fillna(fallback)
            features[f"{column}_count"] = np.log1p(df[column].map(count_map).fillna(0.0))
        return features


def chronological_split(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = raw.copy()
    work[["hour", "minute"]] = work["timestamp"].apply(parse_timestamp).tolist()
    work["slot"] = (work["hour"] * 60 + work["minute"]) // 15

    max_day = int(work["day"].max())
    day_slots = sorted(work.loc[work["day"] == max_day, "slot"].unique())
    if len(day_slots) < 2:
        ordered = work.sort_values(["day", "slot", "Index"])
        split_at = int(len(ordered) * 0.8)
        return ordered.iloc[:split_at].copy(), ordered.iloc[split_at:].copy()

    midpoint = day_slots[len(day_slots) // 2 - 1]
    train_mask = (work["day"] < max_day) | (
        (work["day"] == max_day) & (work["slot"] <= midpoint)
    )
    valid_mask = (work["day"] == max_day) & (work["slot"] > midpoint)
    train_df = raw.loc[train_mask].copy()
    valid_df = raw.loc[valid_mask].copy()
    if valid_df.empty:
        ordered = work.sort_values(["day", "slot", "Index"])
        split_at = int(len(ordered) * 0.8)
        return ordered.iloc[:split_at].copy(), ordered.iloc[split_at:].copy()
    return train_df, valid_df


def make_feature_engineer(
    feature_set: str,
) -> AdvancedFeatureEngineer | GeneralizerFeatureEngineer | StableMemoryFeatureEngineer:
    if feature_set == "advanced":
        return AdvancedFeatureEngineer()
    if feature_set == "generalizer":
        return GeneralizerFeatureEngineer()
    if feature_set == "stable":
        return StableMemoryFeatureEngineer()
    raise ValueError(f"Unknown feature set: {feature_set}")


def build_lgbm() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1800,
        learning_rate=0.025,
        num_leaves=127,
        min_child_samples=10,
        subsample=0.95,
        colsample_bytree=0.95,
        reg_alpha=0.0,
        reg_lambda=0.5,
        random_state=RANDOM_STATE,
        n_jobs=4,
        verbose=-1,
    )


def build_xgb() -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=900,
        learning_rate=0.035,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        random_state=RANDOM_STATE,
        n_jobs=4,
        tree_method="hist",
    )


def build_rf() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=220,
        max_depth=22,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=4,
    )


def fit_models(x_train: pd.DataFrame, y_train: pd.Series, model_type: str) -> dict[str, Any]:
    builders = {
        "lgbm": build_lgbm,
        "xgb": build_xgb,
        "rf": build_rf,
    }
    if model_type == "ensemble":
        selected = ["lgbm", "xgb", "rf"]
    else:
        selected = [model_type]

    models: dict[str, Any] = {}
    for name in selected:
        print(f"Training {name.upper()}...")
        model = builders[name]()
        model.fit(x_train, y_train)
        models[name] = model
    return models


def predict_models(models: dict[str, Any], x_data: pd.DataFrame) -> np.ndarray:
    predictions = {name: np.clip(model.predict(x_data), 0.0, 1.0) for name, model in models.items()}
    if set(predictions) == {"lgbm", "xgb", "rf"}:
        # Slightly adjusted weights for better generalization
        return (
            0.50 * predictions["lgbm"]
            + 0.30 * predictions["xgb"]
            + 0.20 * predictions["rf"]
        )
    return next(iter(predictions.values()))


def run_local_score(train_path: Path, model_type: str, feature_set: str, split: str) -> Metrics:
    raw = pd.read_csv(train_path)
    if split == "chronological":
        train_df, valid_df = chronological_split(raw)
    elif split == "random":
        train_df, valid_df = train_test_split(raw, test_size=0.2, random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unknown split: {split}")

    print(f"Validation split: {split}")
    print(f"Train rows: {len(train_df)} | Valid rows: {len(valid_df)}")
    print(
        "Train day/time range: "
        f"{train_df['day'].min()} {train_df['timestamp'].iloc[0]} -> "
        f"{train_df['day'].max()} {train_df['timestamp'].iloc[-1]}"
    )
    print(
        "Valid day/time range: "
        f"{valid_df['day'].min()} {valid_df['timestamp'].iloc[0]} -> "
        f"{valid_df['day'].max()} {valid_df['timestamp'].iloc[-1]}"
    )

    fe = make_feature_engineer(feature_set)
    x_train = fe.fit_transform(train_df)
    x_valid = fe.transform(valid_df, history=train_df)
    y_train = train_df[TARGET]
    y_valid = valid_df[TARGET]

    models = fit_models(x_train, y_train, model_type=model_type)
    predictions = predict_models(models, x_valid)
    metrics = compute_metrics(y_valid, predictions)
    print_metrics(f"{feature_set.title()} {model_type.upper()} local validation", metrics)
    return metrics


def recursive_test_predictions(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fe: AdvancedFeatureEngineer,
    models: dict[str, Any],
) -> np.ndarray:
    history = train_df.copy()
    predicted_parts: list[pd.Series] = []
    test_work = test_df.copy()
    test_work[["hour", "minute"]] = test_work["timestamp"].apply(parse_timestamp).tolist()
    test_work["slot"] = (test_work["hour"] * 60 + test_work["minute"]) // 15
    test_work = test_work.sort_values(["day", "slot", "Index"])

    for (_, _), group in test_work.groupby(["day", "slot"], sort=True):
        x_group = fe.transform(group, history=history)
        preds = np.clip(predict_models(models, x_group), 0.0, 1.0)
        pred_series = pd.Series(preds, index=group.index)
        predicted_parts.append(pred_series)

        observed = group.copy()
        observed[TARGET] = preds
        history = pd.concat([history, observed], ignore_index=True)

    predictions = pd.concat(predicted_parts).sort_index()
    return predictions.to_numpy()


def write_submission(
    train_path: Path,
    test_path: Path,
    output_path: Path,
    model_type: str,
    feature_set: str,
) -> pd.DataFrame:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    fe = make_feature_engineer(feature_set)
    x_train = fe.fit_transform(train_df)
    y_train = train_df[TARGET]
    models = fit_models(x_train, y_train, model_type=model_type)

    if feature_set == "advanced":
        predictions = recursive_test_predictions(train_df, test_df, fe, models)
    else:
        x_test = fe.transform(test_df, history=train_df)
        predictions = np.clip(predict_models(models, x_test), 0.0, 1.0)
    submission = pd.DataFrame(
        {
            "Index": test_df["Index"].to_numpy()
            if "Index" in test_df.columns
            else np.arange(len(test_df)),
            TARGET: predictions,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Saved {len(submission)} predictions to {output_path}")
    print(submission.head().to_string(index=False))
    return submission


def blend_submission_files(
    primary_path: Path,
    secondary_path: Path,
    output_path: Path,
    primary_weight: float,
) -> pd.DataFrame:
    primary = pd.read_csv(primary_path)
    secondary = pd.read_csv(secondary_path)
    required = {"Index", TARGET}
    if not required.issubset(primary.columns) or not required.issubset(secondary.columns):
        raise ValueError("Both submission files must contain Index and demand columns.")

    merged = primary.merge(secondary, on="Index", suffixes=("_primary", "_secondary"))
    if len(merged) != len(primary) or len(merged) != len(secondary):
        raise ValueError("Submission files do not have matching Index values.")

    secondary_weight = 1.0 - primary_weight
    blended = pd.DataFrame(
        {
            "Index": merged["Index"],
            TARGET: np.clip(
                primary_weight * merged[f"{TARGET}_primary"]
                + secondary_weight * merged[f"{TARGET}_secondary"],
                0.0,
                1.0,
            ),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    blended.to_csv(output_path, index=False)
    print(
        f"Saved blend to {output_path} "
        f"({primary_weight:.2f} * {primary_path} + {secondary_weight:.2f} * {secondary_path})"
    )
    print(blended.head().to_string(index=False))
    return blended


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Advanced traffic demand model")
    parser.add_argument("--train", type=Path, default=Path("dataset/train.csv"))
    parser.add_argument("--predict-test", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/advanced_submission.csv"))
    parser.add_argument("--local-score", action="store_true")
    parser.add_argument("--blend-primary", type=Path, help="Anchor submission CSV to blend")
    parser.add_argument("--blend-secondary", type=Path, help="Secondary submission CSV to blend")
    parser.add_argument(
        "--blend-primary-weight",
        type=float,
        default=0.85,
        help="Weight for --blend-primary; secondary gets 1 - this value",
    )
    parser.add_argument(
        "--feature-set",
        choices=["stable", "generalizer", "advanced"],
        default="stable",
        help="stable uses smoothed geohash/time memory without lags; generalizer removes memory; advanced keeps lags.",
    )
    parser.add_argument(
        "--split",
        choices=["chronological", "random"],
        default="chronological",
        help="Validation split for --local-score.",
    )
    parser.add_argument(
        "--model",
        choices=["lgbm", "xgb", "rf", "ensemble"],
        default="lgbm",
        help="Model family to train. Ensemble blends LGBM, XGB, and RandomForest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.blend_primary or args.blend_secondary:
        if not args.blend_primary or not args.blend_secondary:
            raise ValueError("--blend-primary and --blend-secondary must be used together")
        blend_submission_files(
            args.blend_primary,
            args.blend_secondary,
            args.output,
            args.blend_primary_weight,
        )
        return

    if args.local_score:
        run_local_score(args.train, args.model, args.feature_set, args.split)

    if args.predict_test:
        write_submission(
            args.train,
            args.predict_test,
            args.output,
            args.model,
            args.feature_set,
        )

    if not args.local_score and not args.predict_test:
        raise SystemExit("Use --local-score and/or --predict-test.")


if __name__ == "__main__":
    main()
