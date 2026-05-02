#!/usr/bin/env python
"""Stronger ETA trainer: zone-pair priors + calendar features + XGBoost (MAE objective).

Produces `model.pkl` as a versioned bundle that `predict.py` understands.

Prerequisites:
    python data/download_data.py

Run:
    python baseline-copy.py

Fast iteration (subset rows, custom artifact path):
    ETA_SAMPLE_FRAC=0.05 ETA_MODEL_PATH=model_copy.pkl python baseline-copy.py

Baseline (~6 features) is ~351s Dev MAE; this script typically improves materially by
adding pair-level duration priors and richer time encoding.
"""

from __future__ import annotations

import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

DATA_DIR = Path(__file__).parent / "data"
MODEL_PATH = Path(os.environ.get("ETA_MODEL_PATH", Path(__file__).parent / "model.pkl"))

# Zones are 1–265; index 0 unused for cleaner direct indexing.
_MAX_ZONE = 266

FEATURE_ORDER = [
    "pickup_zone",
    "dropoff_zone",
    "hour",
    "dow",
    "month",
    "passenger_count",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "pair_prior_sec",
    "log1p_pair_count",
    "pickup_prior_sec",
    "haversine_dist",
    "manhattan_dist",
    "bearing",
]

def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * 6371 * np.arcsin(np.sqrt(d)) # km
    return h

def manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b

def bearing_array(lat1, lng1, lat2, lng2):
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))


def build_aggregate_arrays(train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """OD pair medians/counts and pickup-level fallback medians (train only)."""
    global_med = float(train["duration_seconds"].median())

    pair_median = np.full((_MAX_ZONE, _MAX_ZONE), np.nan, dtype=np.float32)
    pair_cnt = np.zeros((_MAX_ZONE, _MAX_ZONE), dtype=np.int32)

    grp = train.groupby(["pickup_zone", "dropoff_zone"], sort=False)["duration_seconds"]
    cnt = grp.size()
    med = grp.median()
    pu_idx = cnt.index.get_level_values(0).astype(np.int32).to_numpy()
    do_idx = cnt.index.get_level_values(1).astype(np.int32).to_numpy()
    pair_cnt[pu_idx, do_idx] = cnt.to_numpy().astype(np.int32)
    pair_median[pu_idx, do_idx] = med.to_numpy().astype(np.float32)

    pickup_median = np.full(_MAX_ZONE, np.nan, dtype=np.float32)
    ps = train.groupby("pickup_zone")["duration_seconds"].median()
    ids = ps.index.astype(np.int32).to_numpy()
    pickup_median[ids] = ps.to_numpy().astype(np.float32)
    # Cold-start pickups → global
    pickup_median = np.where(np.isnan(pickup_median), global_med, pickup_median).astype(np.float32)

    return pair_median, pair_cnt, pickup_median, global_med


def engineer_features(
    df: pd.DataFrame,
    pair_median: np.ndarray,
    pair_cnt: np.ndarray,
    pickup_median: np.ndarray,
    zone_lat: np.ndarray,
    zone_lon: np.ndarray,
) -> pd.DataFrame:
    ts = pd.to_datetime(df["requested_at"])
    hour = ts.dt.hour.astype(np.int32)
    dow = ts.dt.dayofweek.astype(np.int32)
    month = ts.dt.month.astype(np.int32)
    pu = df["pickup_zone"].astype(np.int32).to_numpy()
    do = df["dropoff_zone"].astype(np.int32).to_numpy()

    two_pi_h = 2.0 * np.pi * hour.to_numpy(dtype=np.float64) / 24.0
    pm = pair_median[pu, do]
    pc = pair_cnt[pu, do]
    has_pair = pc > 0
    pup = pickup_median[pu].astype(np.float64)
    pair_prior = np.where(has_pair, np.nan_to_num(pm, nan=pup), pup).astype(np.float64)
    log1p_pc = np.where(has_pair, np.log1p(pc.astype(np.float64)), 0.0)

    pu_lat = zone_lat[pu]
    pu_lon = zone_lon[pu]
    do_lat = zone_lat[do]
    do_lon = zone_lon[do]

    h_dist = haversine_array(pu_lat, pu_lon, do_lat, do_lon)
    m_dist = manhattan_distance(pu_lat, pu_lon, do_lat, do_lon)
    bearing = bearing_array(pu_lat, pu_lon, do_lat, do_lon)

    out = pd.DataFrame(
        {
            "pickup_zone": pu,
            "dropoff_zone": do,
            "hour": hour.to_numpy(),
            "dow": dow.to_numpy(),
            "month": month.to_numpy(),
            "passenger_count": df["passenger_count"].astype(np.int32).to_numpy(),
            "is_weekend": (dow >= 5).astype(np.int8).to_numpy(),
            "hour_sin": np.sin(two_pi_h),
            "hour_cos": np.cos(two_pi_h),
            "pair_prior_sec": pair_prior,
            "log1p_pair_count": log1p_pc,
            "pickup_prior_sec": pickup_median[pu].astype(np.float64),
            "haversine_dist": h_dist,
            "manhattan_dist": m_dist,
            "bearing": bearing,
        }
    )
    return out[FEATURE_ORDER]


def main() -> None:
    train_path = DATA_DIR / "train.parquet"
    dev_path = DATA_DIR / "dev.parquet"
    for p in (train_path, dev_path):
        if not p.exists():
            raise SystemExit(f"Missing {p.name}. Run `python data/download_data.py` first.")

    print("Loading data...")
    train = pd.read_parquet(train_path)
    dev = pd.read_parquet(dev_path)
    frac = os.environ.get("ETA_SAMPLE_FRAC")
    if frac:
        f = float(frac)
        train = train.sample(frac=f, random_state=42)
        dev = dev.sample(frac=f, random_state=43)
        print(f"  ETA_SAMPLE_FRAC={f} (subsampled for fast iteration)")
    print(f"  train: {len(train):,} rows")
    print(f"  dev:   {len(dev):,} rows")

    print("Loading zone coordinates...")
    coords_df = pd.read_csv(DATA_DIR / "zone_coords.csv")
    zone_lat = np.full(_MAX_ZONE, np.nan, dtype=np.float32)
    zone_lon = np.full(_MAX_ZONE, np.nan, dtype=np.float32)
    ids = coords_df["zone_id"].astype(np.int32).to_numpy()
    mask = ids < _MAX_ZONE
    ids = ids[mask]
    zone_lat[ids] = coords_df["latitude"].to_numpy(dtype=np.float32)[mask]
    zone_lon[ids] = coords_df["longitude"].to_numpy(dtype=np.float32)[mask]

    print("Computing OD / pickup aggregates from train...")
    t0 = time.time()
    pair_median, pair_cnt, pickup_median, global_med = build_aggregate_arrays(train)
    print(f"  done in {time.time() - t0:.1f}s")

    print("Building feature matrices...")
    X_train = engineer_features(train, pair_median, pair_cnt, pickup_median, zone_lat, zone_lon)
    y_train = train["duration_seconds"].to_numpy(dtype=np.float64)
    X_dev = engineer_features(dev, pair_median, pair_cnt, pickup_median, zone_lat, zone_lon)
    y_dev = dev["duration_seconds"].to_numpy(dtype=np.float64)

    print("\nTraining XGBoost (objective tuned for MAE)...")
    model = xgb.XGBRegressor(
        n_estimators=2500,
        max_depth=12,
        learning_rate=0.04,
        min_child_weight=4,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.2,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        objective="reg:absoluteerror",
        early_stopping_rounds=80,
    )
    t1 = time.time()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_dev, y_dev)],
        verbose=False,
    )
    print(f"  trained in {time.time() - t1:.0f}s (best_iteration={model.best_iteration})")

    if hasattr(model, "get_booster"):
        model.get_booster().feature_names = None

    preds = model.predict(X_dev)
    mae = float(np.mean(np.abs(preds - y_dev)))
    print(f"\nDev MAE: {mae:.1f} seconds")

    bundle = {
        "version": 2,
        "xgb": model,
        "pair_median": pair_median,
        "pair_cnt": pair_cnt,
        "pickup_median": pickup_median,
        "global_median": global_med,
        "feature_order": FEATURE_ORDER,
        "zone_lat": zone_lat,
        "zone_lon": zone_lon,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Saved bundle to {MODEL_PATH}")


if __name__ == "__main__":
    main()
