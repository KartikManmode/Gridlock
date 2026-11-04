"""Advanced Traffic Demand Prediction Pipeline (v2).
Final version with strict index alignment and full feature parity + improvements.
"""

import geohash
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

RANDOM_STATE = 42
TARGET = "demand"

def parse_ts(v):
    h, m = str(v).split(":")
    return int(h), int(m)

def engineer_v2(df, train_meta=None):
    work = df.copy()
    work[["hour", "minute"]] = work["timestamp"].apply(parse_ts).tolist()
    work[["lat", "lon"]] = work["geohash"].apply(lambda x: geohash.decode(str(x))).tolist()
    
    # Temporal Features
    work["minute_of_day"] = work["hour"] * 60 + work["minute"]
    work["hour_sin"] = np.sin(2 * np.pi * work["hour"] / 24)
    work["hour_cos"] = np.cos(2 * np.pi * work["hour"] / 24)
    work["minute_sin"] = np.sin(2 * np.pi * work["minute_of_day"] / 1440)
    work["minute_cos"] = np.cos(2 * np.pi * work["minute_of_day"] / 1440)
    work["is_rush_hour"] = (work["hour"].between(7, 9) | work["hour"].between(17, 19)).astype(int)
    work["is_night"] = ((work["hour"] >= 22) | (work["hour"] <= 6)).astype(int)
    
    # Categorical & Binary
    work["LargeVehicles"] = (work["LargeVehicles"] == "Allowed").astype(int)
    work["Landmarks"] = (work["Landmarks"] == "Yes").astype(int)
    for col in ["RoadType", "Weather"]:
        work[col] = work[col].fillna("Unknown")
    
    # Lags (Crucial for 96%+ score)
    work["orig_order"] = range(len(work))
    work = work.sort_values(["geohash", "day", "hour", "minute"])
    group = work.groupby(["geohash", "day"])[TARGET]
    work["lag_1"] = group.shift(1).fillna(0)
    work["lag_2"] = group.shift(2).fillna(0)
    work["roll_mean_3"] = group.shift(1).rolling(3, min_periods=1).mean().fillna(0)
    work = work.sort_values("orig_order").drop(columns=["orig_order"])
    
    if train_meta is not None:
        # Transform
        work["geo_mean"] = work["geohash"].map(train_meta['geo_means']).fillna(train_meta['global_mean'])
        work["neighbor_mean"] = work["geohash"].map(train_meta['neighbor_means']).fillna(train_meta['global_mean'])
        
        # Dummies parity
        dummies = pd.get_dummies(work[["RoadType", "Weather"]], drop_first=False)
        dummies = dummies.reindex(columns=train_meta['dummy_cols'], fill_value=0)
        work = pd.concat([work, dummies.astype(int)], axis=1)
    else:
        # Fit
        global_mean = work[TARGET].mean()
        geo_means = work.groupby("geohash")[TARGET].mean()
        
        neighbor_means = {}
        for gh in work["geohash"].unique():
            neighbors = geohash.neighbors(gh)
            n_demands = [geo_means[n] for n in neighbors if n in geo_means]
            neighbor_means[gh] = np.mean(n_demands) if n_demands else global_mean
            
        work["geo_mean"] = work["geohash"].map(geo_means)
        work["neighbor_mean"] = work["geohash"].map(neighbor_means)
        
        dummies = pd.get_dummies(work[["RoadType", "Weather"]], drop_first=False)
        dummy_cols = dummies.columns.tolist()
        work = pd.concat([work, dummies.astype(int)], axis=1)
        
        meta = {
            'global_mean': global_mean,
            'geo_means': geo_means,
            'neighbor_means': neighbor_means,
            'dummy_cols': dummy_cols
        }
        return work, meta
        
    return work

def run_v2():
    print("Loading data...")
    df = pd.read_csv("dataset/train.csv")
    train_raw, valid_raw = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
    
    print("Engineering features (v2 full parity)...")
    train_df, meta = engineer_v2(train_raw)
    valid_df = engineer_v2(valid_raw, train_meta=meta)
    
    # Exclude non-numeric/metadata
    exclude = ["geohash", "timestamp", "RoadType", "Weather", TARGET]
    cols = [c for c in train_df.columns if c not in exclude]
    
    X_train = train_df[cols]
    y_train = train_df[TARGET]
    X_valid = valid_df[cols]
    y_valid = valid_df[TARGET]
    
    xgb_params = {
        'subsample': 0.85, 'n_estimators': 900, 'max_depth': 8, 
        'learning_rate': 0.1, 'colsample_bytree': 1.0,
        'random_state': RANDOM_STATE, 'n_jobs': 4, 'tree_method': 'hist'
    }
    
    print(f"Training Tuned XGBoost on {len(cols)} features...")
    model = XGBRegressor(**xgb_params)
    model.fit(X_train, y_train)
    
    preds = np.clip(model.predict(X_valid), 0, 1)
    rmse = np.sqrt(mean_squared_error(y_valid, preds))
    r2 = r2_score(y_valid, preds)
    
    print(f"\nV2 Final Optimized Results:")
    print(f"RMSE: {rmse:.6f}")
    print(f"R2 Score: {r2*100:.2f}")

if __name__ == "__main__":
    run_v2()
