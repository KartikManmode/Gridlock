import pandas as pd
import numpy as np
import geohash

def add_time_features(df):
    work = df.copy()
    def parse_timestamp(value):
        hour, minute = str(value).split(":")
        return int(hour), int(minute)
    
    work[["hour", "minute"]] = work["timestamp"].apply(parse_timestamp).tolist()
    work["slot_15m"] = work["hour"] * 4 + work["minute"] // 15
    work["time_minutes"] = work["hour"] * 60 + work["minute"]
    work["hour_sin"] = np.sin(2 * np.pi * work["hour"] / 24)
    work["hour_cos"] = np.cos(2 * np.pi * work["hour"] / 24)
    return work

def add_geohash_features(df):
    work = df.copy()
    def decode_geohash(value):
        lat, lon = geohash.decode(str(value))
        return float(lat), float(lon)
    
    work[["latitude", "longitude"]] = work["geohash"].apply(decode_geohash).tolist()
    work["geohash_prefix_4"] = work["geohash"].str[:4]
    work["geohash_prefix_5"] = work["geohash"].str[:5]
    return work
