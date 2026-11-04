import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from linear_reg import engineer_features, make_models, get_feature_config, TARGET
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def compute_metrics(y_true, y_pred):
    clipped = np.clip(y_pred, 0.0, 1.0)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, clipped))),
        "mae": float(mean_absolute_error(y_true, clipped)),
        "r2": float(r2_score(y_true, clipped)),
    }

def run_test():
    print("Loading data...")
    df = pd.read_csv("dataset/train.csv")
    
    print("Splitting data (80/20)...")
    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)
    
    print("Engineering features...")
    train_feat = engineer_features(train_df)
    valid_feat = engineer_features(valid_df)
    
    config = get_feature_config()
    models = make_models(config)
    
    # We only want to test LinearRegression as requested
    name = "LinearRegression"
    pipeline = models[name]
    
    print(f"Training {name}...")
    feature_columns = config.numeric + config.one_hot + config.target_encoded
    pipeline.fit(train_feat[feature_columns], train_feat[TARGET])
    
    print(f"Evaluating {name}...")
    preds = pipeline.predict(valid_feat[feature_columns])
    metrics = compute_metrics(valid_feat[TARGET], preds)
    
    print(f"\n{name} Results:")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"MAE:  {metrics['mae']:.6f}")
    print(f"R2:   {metrics['r2']:.6f}")
    print(f"Score out of 100: {metrics['r2']*100:.2f}")

if __name__ == "__main__":
    run_test()
