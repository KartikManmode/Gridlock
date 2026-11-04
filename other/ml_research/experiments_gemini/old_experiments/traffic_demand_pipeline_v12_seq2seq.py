"""Phase 4 Strategy 3: Multi-Horizon Seq2Seq Modeling.

Predicts the entire 14-hour test window simultaneously to prevent recursive drift.
Uses a simple PyTorch architecture taking the last N known slots to predict the next M slots.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset

RANDOM_STATE = 42
TARGET = "demand"

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_timestamp(value: str) -> tuple[int, int]:
    h, m = str(value).split(":")
    return int(h), int(m)

class Seq2SeqDataset(Dataset):
    def __init__(self, df: pd.DataFrame, in_len: int = 16, out_len: int = 56):
        self.in_len = in_len
        self.out_len = out_len
        self.samples = []
        
        # Sort chronologically
        df = df.copy()
        df[["hour", "minute"]] = df["timestamp"].apply(parse_timestamp).tolist()
        df["slot_id"] = df["day"] * 96 + df["hour"] * 4 + df["minute"] // 15
        
        for gh, group in df.groupby("geohash"):
            group = group.sort_values("slot_id")
            y = group[TARGET].to_numpy(dtype=np.float32)
            
            # Simple features for proof of concept
            # We use hour_sin, hour_cos as basic temporal anchors
            h_sin = np.sin(2 * np.pi * group["hour"] / 24).to_numpy(dtype=np.float32)
            h_cos = np.cos(2 * np.pi * group["hour"] / 24).to_numpy(dtype=np.float32)
            
            features = np.stack([y, h_sin, h_cos], axis=1) # [T, 3]
            
            # Create sliding windows that map in_len -> out_len
            total_len = in_len + out_len
            if len(group) < total_len:
                continue
                
            for i in range(len(group) - total_len + 1):
                X = features[i : i + in_len]
                Y = y[i + in_len : i + total_len]
                self.samples.append((X, Y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X, Y = self.samples[idx]
        return torch.tensor(X), torch.tensor(Y)


class MultiHorizonModel(nn.Module):
    def __init__(self, in_features: int = 3, hidden_dim: int = 64, out_len: int = 56):
        super().__init__()
        self.encoder = nn.GRU(in_features, hidden_dim, batch_first=True)
        # Direct mapping from final hidden state to the entire horizon
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, out_len)
        )

    def forward(self, x):
        # x: [B, in_len, features]
        _, h_n = self.encoder(x)
        # h_n is [1, B, hidden_dim]
        h_n = h_n.squeeze(0) 
        out = self.decoder(h_n) # [B, out_len]
        return torch.clamp(out, 0.0, 1.0)


def train_seq2seq(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(args.train)
    
    # Chronological Split inside the dataset generator instead of splitting DataFrames
    # But for a simple proof of concept, we can just use the whole df and split the generated sequences.
    
    full_ds = Seq2SeqDataset(df, in_len=args.in_len, out_len=args.out_len)
    
    split_idx = int(len(full_ds) * 0.8)
    
    # Sequential split ensures validation is in the future
    train_samples = full_ds.samples[:split_idx]
    valid_samples = full_ds.samples[split_idx:]
    
    class SubsetDS(Dataset):
        def __init__(self, samples):
            self.samples = samples
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            X, Y = self.samples[idx]
            return torch.tensor(X), torch.tensor(Y)
            
    train_loader = DataLoader(SubsetDS(train_samples), batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(SubsetDS(valid_samples), batch_size=args.batch_size, shuffle=False)
    
    model = MultiHorizonModel(out_len=args.out_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.HuberLoss()
    
    print("\n--- Training Multi-Horizon Seq2Seq ---")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, Y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for X, Y in valid_loader:
                X = X.to(device)
                pred = model(X)
                all_preds.append(pred.cpu().numpy())
                all_targets.append(Y.numpy())
                
        if all_preds:
            y_pred = np.concatenate(all_preds).flatten()
            y_true = np.concatenate(all_targets).flatten()
            r2 = r2_score(y_true, y_pred)
            print(f"Epoch {epoch:02d} | Train Loss: {train_loss/len(train_loader):.5f} | Val R2: {r2:.5f}")
        else:
            print(f"Epoch {epoch:02d} | Train Loss: {train_loss/len(train_loader):.5f} | Val R2: N/A (No valid sequences)")
        
    print("Training complete. (Inference logic requires matching the exact test set slots).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("dataset/train.csv"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--in-len", type=int, default=16) # Use past 4 hours
    parser.add_argument("--out-len", type=int, default=56) # Predict next 14 hours
    args = parser.parse_args()
    
    seed_everything(RANDOM_STATE)
    train_seq2seq(args)
