"""Zero-inflated two-stage traffic demand model.

Stage 1 estimates P(demand > threshold). Stage 2 estimates positive demand
only on rows above the threshold. The final bounded prediction is:

    probability_of_traffic * predicted_positive_demand

Example:
    python3 codex_codes/two_stage_model.py --train dataset/train.csv --test dataset/test.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import geohash
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


TARGET = "demand"
RANDOM_STATE = 42


def parse_timestamp(value: str) -> tuple[int, int]:
    hour, minute = str(value).split(":")
    return int(hour), int(minute)


def decode_geohash(value: str) -> tuple[float, float]:
    lat, lon = geohash.decode(str(value))
    return float(lat), float(lon)


@dataclass
class TwoStageModels:
    classifier: lgb.LGBMClassifier
    regressor: lgb.LGBMRegressor
    feature_engineer: "FeatureEngineer"
    threshold: float


class FeatureEngineer:
    numeric_to_scale = ["Temperature", "NumberofLanes", "lat", "lon"]

    def __init__(self) -> None:
        self.road_mode: str | None = None
        self.weather_mode: str | None = None
        self.temp_by_weather: pd.Series | None = None
        self.global_temp: float | None = None
        self.dummy_columns: list[str] | None = None
        self.scaler = StandardScaler()

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        work = self._basic(df, fit_modes=True)
        self.temp_by_weather = work.groupby("Weather")["Temperature"].median()
        self.global_temp = float(work["Temperature"].median())
        work = self._impute_temperature(work)
        features = self._encode(work, fit_columns=True)
        self.scaler.fit(features[self.numeric_to_scale])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.dummy_columns is None:
            raise RuntimeError("FeatureEngineer must be fit before transform.")
        work = self._basic(df, fit_modes=False)
        work = self._impute_temperature(work)
        features = self._encode(work, fit_columns=False)
        features[self.numeric_to_scale] = self.scaler.transform(features[self.numeric_to_scale])
        return features

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def _basic(self, df: pd.DataFrame, fit_modes: bool) -> pd.DataFrame:
        work = df.copy()
        work[["hour", "minute"]] = work["timestamp"].apply(parse_timestamp).tolist()
        work[["lat", "lon"]] = work["geohash"].apply(decode_geohash).tolist()
        if fit_modes:
            self.road_mode = self._mode(work["RoadType"], "Unknown")
            self.weather_mode = self._mode(work["Weather"], "Unknown")
        work["RoadType"] = work["RoadType"].fillna(self.road_mode)
        work["Weather"] = work["Weather"].fillna(self.weather_mode)
        work["minute_of_day"] = work["hour"] * 60 + work["minute"]
        work["slot_id"] = work["day"] * 96 + work["minute_of_day"] // 15
        work["hour_sin"] = np.sin(2.0 * np.pi * work["hour"] / 24.0)
        work["hour_cos"] = np.cos(2.0 * np.pi * work["hour"] / 24.0)
        work["minute_sin"] = np.sin(2.0 * np.pi * work["minute_of_day"] / 1440.0)
        work["minute_cos"] = np.cos(2.0 * np.pi * work["minute_of_day"] / 1440.0)
        work["LargeVehicles"] = (work["LargeVehicles"] == "Allowed").astype(int)
        work["Landmarks"] = (work["Landmarks"] == "Yes").astype(int)
        work["geohash5"] = work["geohash"].astype(str).str[:5]
        work["geohash4"] = work["geohash"].astype(str).str[:4]
        work["road_weather"] = work["RoadType"] + "__" + work["Weather"]
        work["lat_lon_product"] = work["lat"] * work["lon"]
        return work

    def _impute_temperature(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        if self.temp_by_weather is None or self.global_temp is None:
            raise RuntimeError("Temperature imputers are not fitted.")
        work["Temperature"] = work["Temperature"].fillna(work["Weather"].map(self.temp_by_weather))
        work["Temperature"] = work["Temperature"].fillna(self.global_temp)
        return work

    def _encode(self, df: pd.DataFrame, fit_columns: bool) -> pd.DataFrame:
        base_cols = [
            "day",
            "hour",
            "minute",
            "minute_of_day",
            "slot_id",
            "hour_sin",
            "hour_cos",
            "minute_sin",
            "minute_cos",
            "LargeVehicles",
            "Landmarks",
            "Temperature",
            "NumberofLanes",
            "lat",
            "lon",
            "lat_lon_product",
        ]
        features = df[base_cols].copy()
        dummies = pd.get_dummies(df[["RoadType", "Weather", "geohash5", "geohash4", "road_weather"]], drop_first=False)
        if fit_columns:
            self.dummy_columns = dummies.columns.tolist()
        else:
            dummies = dummies.reindex(columns=self.dummy_columns, fill_value=0)
        return pd.concat([features, dummies.astype(int)], axis=1)

    @staticmethod
    def _mode(series: pd.Series, default: str) -> str:
        mode = series.dropna().mode()
        return str(mode.iloc[0]) if not mode.empty else default


def build_classifier() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1200,
        learning_rate=0.025,
        num_leaves=63,
        min_child_samples=30,
        subsample=0.9,
        colsample_bytree=0.85,
        reg_alpha=0.01,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=4,
        verbose=-1,
    )


def build_regressor() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1600,
        learning_rate=0.025,
        num_leaves=127,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.85,
        reg_alpha=0.01,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=4,
        verbose=-1,
    )


def fit_two_stage(train_df: pd.DataFrame, threshold: float) -> TwoStageModels:
    fe = FeatureEngineer()
    x = fe.fit_transform(train_df)
    y = train_df[TARGET].clip(0.0, 1.0)
    y_binary = (y > threshold).astype(int)
    classifier = build_classifier()
    classifier.fit(x, y_binary)

    positive_mask = y > threshold
    regressor = build_regressor()
    regressor.fit(x.loc[positive_mask], y.loc[positive_mask])
    return TwoStageModels(classifier=classifier, regressor=regressor, feature_engineer=fe, threshold=threshold)


def predict_two_stage(models: TwoStageModels, df: pd.DataFrame) -> np.ndarray:
    x = models.feature_engineer.transform(df)
    prob_traffic = models.classifier.predict_proba(x)[:, 1]
    positive_demand = models.regressor.predict(x)
    return np.clip(prob_traffic * positive_demand, 0.0, 1.0)


def validate(train_df: pd.DataFrame, threshold: float, valid_fraction: float) -> None:
    train_part, valid_part = train_test_split(train_df, test_size=valid_fraction, shuffle=False)
    models = fit_two_stage(train_part, threshold)
    valid_y = valid_part[TARGET].clip(0.0, 1.0)
    valid_binary = (valid_y > threshold).astype(int)
    x_valid = models.feature_engineer.transform(valid_part)
    prob = models.classifier.predict_proba(x_valid)[:, 1]
    preds = predict_two_stage(models, valid_part)
    print(f"valid_logloss={log_loss(valid_binary, prob):.6f}")
    print(f"valid_rmse={np.sqrt(mean_squared_error(valid_y, preds)):.6f}")
    print(f"valid_r2={r2_score(valid_y, preds):.6f}")


def write_submission(models: TwoStageModels, test_df: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    preds = predict_two_stage(models, test_df)
    submission = pd.DataFrame({"Index": test_df["Index"], TARGET: preds})
    submission.to_csv(output, index=False)
    print(f"wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("dataset/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("dataset/test.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/two_stage_submission.csv"))
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--valid-fraction", type=float, default=0.2)
    parser.add_argument("--no-submit", action="store_true")
    args = parser.parse_args()

    train_df = pd.read_csv(args.train)
    validate(train_df, args.threshold, args.valid_fraction)
    if not args.no_submit and args.test.exists():
        models = fit_two_stage(train_df, args.threshold)
        test_df = pd.read_csv(args.test)
        write_submission(models, test_df, args.output)


if __name__ == "__main__":
    main()
