"""Phase 4 Strategy 2: Time-of-Day Expert Routing (MoE).

Routes traffic predictions to specialized experts:
- Rush Expert: 07:00-09:00 and 17:00-19:00
- Night Expert: 22:00-06:00
- Normal Expert: Everything else
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gridlock")

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit

# We reuse the ZeroInflatedFE from v10 to keep the code modular and DRY,
# but we wrap the modeling in an MoE router.
from traffic_demand_pipeline_v10_zero_inflated import (
    RANDOM_STATE, TARGET, ZeroInflatedFE, get_gnn_embeddings, seed_everything
)


class MoERouter:
    def __init__(self):
        # We use a simple LightGBM model for each expert
        self.rush_expert = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=127, verbose=-1, random_state=RANDOM_STATE)
        self.night_expert = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=63, verbose=-1, random_state=RANDOM_STATE)
        self.normal_expert = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03, num_leaves=127, verbose=-1, random_state=RANDOM_STATE)
        self._fitted = {"rush": False, "night": False, "normal": False}

    def _get_masks(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Based on the FE, we have 'is_rush_hour' and 'is_night' as features
        rush_mask = X["is_rush_hour"] == 1
        night_mask = X["is_night"] == 1
        normal_mask = ~(rush_mask | night_mask)
        return rush_mask, night_mask, normal_mask

    def fit(self, X: pd.DataFrame, y: pd.Series):
        rush, night, norm = self._get_masks(X)
        
        if rush.any():
            self.rush_expert.fit(X[rush], y[rush], categorical_feature=["RoadType", "Weather"])
            self._fitted["rush"] = True
        if night.any():
            self.night_expert.fit(X[night], y[night], categorical_feature=["RoadType", "Weather"])
            self._fitted["night"] = True
        if norm.any():
            self.normal_expert.fit(X[norm], y[norm], categorical_feature=["RoadType", "Weather"])
            self._fitted["normal"] = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        rush, night, norm = self._get_masks(X)
        preds = np.zeros(len(X), dtype=np.float32)
        
        if rush.any() and self._fitted["rush"]:
            preds[rush] = self.rush_expert.predict(X[rush])
        if night.any() and self._fitted["night"]:
            preds[night] = self.night_expert.predict(X[night])
        if norm.any() and self._fitted["normal"]:
            preds[norm] = self.normal_expert.predict(X[norm])
            
        return np.clip(preds, 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("dataset/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("dataset/test.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/gemini_v11_moe.csv"))
    args = parser.parse_args()

    seed_everything(RANDOM_STATE)
    df = pd.read_csv(args.train).sort_values(["day", "timestamp"])
    test_df = pd.read_csv(args.test)
    
    gnn_emb = get_gnn_embeddings(df, test_df)

    tscv = TimeSeriesSplit(n_splits=3)
    scores = []
    
    print("\n--- Training Mixture of Experts (MoE) ---")
    for train_idx, val_idx in tscv.split(df):
        tr_df, va_df = df.iloc[train_idx], df.iloc[val_idx]
        fe = ZeroInflatedFE(gnn_emb).fit(tr_df)
        
        # We need X as a DataFrame to route based on features
        X_tr, X_val = fe.transform(tr_df), fe.transform(va_df)
        
        model = MoERouter()
        model.fit(X_tr, tr_df[TARGET])
        preds = model.predict(X_val)
        
        score = r2_score(va_df[TARGET], preds)
        scores.append(score)
        print(f"Fold R2: {score:.6f}")
        
    print(f"MoE CV R2: {np.mean(scores):.6f}")

    fe = ZeroInflatedFE(gnn_emb).fit(df)
    model = MoERouter()
    model.fit(fe.transform(df), df[TARGET])
    
    pd.DataFrame({
        "Index": test_df["Index"], 
        "demand": model.predict(fe.transform(test_df))
    }).to_csv(args.output, index=False)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
