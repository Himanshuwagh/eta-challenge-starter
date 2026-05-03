#!/usr/bin/env python
"""ETA trainer v3: spatial + weather + holidays + time-conditioned priors + smoothed encoding.

Produces `model.pkl` as a versioned bundle that `predict.py` understands.

Prerequisites:
    python data/download_data.py
    python extract_coords.py          # creates data/zone_coords.csv
    data/weather_hourly.csv            # must exist

Run:
    python baseline-copy.py

Fast iteration (subset rows, custom artifact path):
    ETA_SAMPLE_FRAC=0.05 ETA_MODEL_PATH=model_copy.pkl python baseline-copy.py
"""

from __future__ import annotations

import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm

class TqdmCallback(xgb.callback.TrainingCallback):
    def __init__(self, total):
        self.pbar = tqdm(total=total, desc="Boosting")
    def after_iteration(self, model, epoch, evals_log):
        self.pbar.update(1)
        if evals_log and "validation_0" in evals_log:
            # show latest metric in progress bar
            metric_name = list(evals_log["validation_0"].keys())[0]
            val = evals_log["validation_0"][metric_name][-1]
            self.pbar.set_postfix({metric_name: f"{val:.1f}"})
        return False

DATA_DIR = Path(__file__).parent / "data"
MODEL_PATH = Path(os.environ.get("ETA_MODEL_PATH", Path(__file__).parent / "model.pkl"))

# Zones are 1–265; index 0 unused for cleaner direct indexing.
_MAX_ZONE = 266
_SMOOTHING_WEIGHT = 50  # Bayesian smoothing weight for rare zone pairs

# --- US Federal Holidays + key NYC dates for 2023-2024 ---
_HOLIDAYS = {
    # 2023
    (1, 1), (1, 2),     # New Year's Day (observed)
    (1, 16),            # MLK Day
    (2, 20),            # Presidents' Day
    (5, 29),            # Memorial Day
    (6, 19),            # Juneteenth
    (7, 4),             # Independence Day
    (9, 4),             # Labor Day
    (10, 9),            # Columbus Day
    (11, 10),           # Veterans Day (observed)
    (11, 23), (11, 24), # Thanksgiving + Black Friday
    (12, 25),           # Christmas Day
    (12, 31),           # New Year's Eve
    # 2024 (eval set is early 2024)
    (1, 1),             # New Year's Day
    (1, 15),            # MLK Day
    (2, 19),            # Presidents' Day
}

# Christmas / New Year week: Dec 22 - Jan 2 (traffic is very different)
def _is_holiday_period(month: np.ndarray, day: np.ndarray) -> np.ndarray:
    """Returns True for dates in the Christmas-New Year corridor."""
    return ((month == 12) & (day >= 22)) | ((month == 1) & (day <= 2))


FEATURE_ORDER = [
    "pickup_zone", "dropoff_zone", "hour", "dow", "month", "passenger_count",
    "is_weekend", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "pair_prior_sec", "pair_prior_smoothed", "log1p_pair_count",
    "pickup_prior_sec", "dropoff_prior_sec", "pair_hour_prior_sec",
    "haversine_dist", "manhattan_dist", "bearing",
    "is_holiday", "is_holiday_period", "day_of_year", "minute_of_day",
    "is_morning_rush", "is_evening_rush",
    "pu_bor", "do_bor", "is_cross_borough", "is_airport",
    "days_to_christmas", "days_to_newyear", "speed_proxy",
    "temp_c", "precip_mm", "wind_speed_kmh", "visibility_km", "is_raining",
    "rain_x_airport", "rain_x_dist",
]

_BOROUGH_MAP = {}
try:
    _zl = pd.read_csv(DATA_DIR / "zone_lookup.csv")
    for _, r in _zl.iterrows():
        zid = int(r["LocationID"])
        bor = str(r["Borough"]).strip()
        _BOROUGH_MAP[zid] = {"Manhattan": 0, "Brooklyn": 1, "Queens": 2, "Bronx": 3, "Staten Island": 4, "EWR": 5}.get(bor, 6)
except Exception: pass


# ── Geometry helpers ─────────────────────────────────────────────────────────

def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * 6371 * np.arcsin(np.sqrt(d))  # km
    return h

def manhattan_distance(lat1, lng1, lat2, lng2):
    # NYC grid is rotated ~29 degrees clockwise from true North
    # We rotate the coordinates counter-clockwise by 29 degrees to align the grid with N/S E/W
    theta = np.radians(29.0)
    c, s = np.cos(theta), np.sin(theta)
    
    # Rotate coordinates
    r_lat1 = lat1 * c - lng1 * s
    r_lng1 = lat1 * s + lng1 * c
    r_lat2 = lat2 * c - lng2 * s
    r_lng2 = lat2 * s + lng2 * c
    
    # Approximate Manhattan distance on the rotated grid
    # 1 degree lat ~ 111.1 km. 1 degree lon ~ 111.1 * cos(lat) km
    lat_dist = np.abs(r_lat2 - r_lat1) * 111.1
    avg_lat = np.radians((lat1 + lat2) / 2.0)
    lng_dist = np.abs(r_lng2 - r_lng1) * 111.1 * np.cos(avg_lat)
    
    return lat_dist + lng_dist

def bearing_array(lat1, lng1, lat2, lng2):
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))


# ── Aggregate builders ───────────────────────────────────────────────────────

def build_aggregate_arrays(train: pd.DataFrame):
    """Build all lookup arrays from training data only."""
    
    # --- Clip outliers for aggregates ---
    # Trips > 99th pct (~4000s) or < 60s skew the medians significantly for rare pairs.
    # We clip the values temporarily just for the grouping operations.
    clip_upper = float(train["duration_seconds"].quantile(0.99))
    clip_lower = 60.0
    clipped_duration = train["duration_seconds"].clip(clip_lower, clip_upper)
    
    # We create a temporary DataFrame for aggregation to avoid modifying the original 'train' DataFrame
    agg_df = train[["pickup_zone", "dropoff_zone", "requested_at"]].copy()
    agg_df["duration_seconds"] = clipped_duration

    global_med = float(agg_df["duration_seconds"].median())

    # --- OD pair median / count ---
    pair_median = np.full((_MAX_ZONE, _MAX_ZONE), np.nan, dtype=np.float32)
    pair_cnt = np.zeros((_MAX_ZONE, _MAX_ZONE), dtype=np.int32)

    grp = agg_df.groupby(["pickup_zone", "dropoff_zone"], sort=False)["duration_seconds"]
    cnt = grp.size()
    med = grp.median()
    pu_idx = cnt.index.get_level_values(0).astype(np.int32).to_numpy()
    do_idx = cnt.index.get_level_values(1).astype(np.int32).to_numpy()
    pair_cnt[pu_idx, do_idx] = cnt.to_numpy().astype(np.int32)
    pair_median[pu_idx, do_idx] = med.to_numpy().astype(np.float32)

    # --- Pickup zone median ---
    pickup_median = np.full(_MAX_ZONE, np.nan, dtype=np.float32)
    ps = agg_df.groupby("pickup_zone")["duration_seconds"].median()
    ids = ps.index.astype(np.int32).to_numpy()
    pickup_median[ids] = ps.to_numpy().astype(np.float32)
    pickup_median = np.where(np.isnan(pickup_median), global_med, pickup_median).astype(np.float32)

    # --- Dropoff zone median ---
    dropoff_median = np.full(_MAX_ZONE, np.nan, dtype=np.float32)
    ds = agg_df.groupby("dropoff_zone")["duration_seconds"].median()
    ids_d = ds.index.astype(np.int32).to_numpy()
    dropoff_median[ids_d] = ds.to_numpy().astype(np.float32)
    dropoff_median = np.where(np.isnan(dropoff_median), global_med, dropoff_median).astype(np.float32)

    # --- Time-conditioned pair prior: median per (PU, DO, 1-hour block) ---
    # 24 bins: 0, 1, 2, ..., 23
    ts = pd.to_datetime(agg_df["requested_at"])
    agg_df["hour_bin"] = ts.dt.hour.astype(np.int8)
    
    pair_hour_median = np.full((_MAX_ZONE, _MAX_ZONE, 24), np.nan, dtype=np.float32)
    grp_h = agg_df.groupby(["pickup_zone", "dropoff_zone", "hour_bin"], sort=False)["duration_seconds"]
    med_h = grp_h.median()
    ph_pu = med_h.index.get_level_values(0).astype(np.int32).to_numpy()
    ph_do = med_h.index.get_level_values(1).astype(np.int32).to_numpy()
    ph_hb = med_h.index.get_level_values(2).astype(np.int32).to_numpy()
    pair_hour_median[ph_pu, ph_do, ph_hb] = med_h.to_numpy().astype(np.float32)

    return pair_median, pair_cnt, pickup_median, dropoff_median, pair_hour_median, global_med


def build_weather_lookup(weather_path: Path) -> pd.DataFrame:
    """Load weather CSV and prepare for merge by rounded hour."""
    wdf = pd.read_csv(weather_path)
    wdf["datetime"] = pd.to_datetime(wdf["datetime"])
    wdf = wdf.set_index("datetime").sort_index()
    # Forward-fill any gaps
    full_range = pd.date_range(wdf.index.min(), wdf.index.max(), freq="h")
    wdf = wdf.reindex(full_range).ffill().bfill()
    wdf.index.name = "datetime"
    return wdf


# ── Feature engineering ──────────────────────────────────────────────────────

def engineer_features(
    df: pd.DataFrame,
    pair_median: np.ndarray,
    pair_cnt: np.ndarray,
    pickup_median: np.ndarray,
    dropoff_median: np.ndarray,
    pair_hour_median: np.ndarray,
    global_med: float,
    zone_lat: np.ndarray,
    zone_lon: np.ndarray,
    weather_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    ts = pd.to_datetime(df["requested_at"])
    hour = ts.dt.hour.astype(np.int32)
    minute = ts.dt.minute.astype(np.int32)
    dow = ts.dt.dayofweek.astype(np.int32)
    month = ts.dt.month.astype(np.int32)
    day = ts.dt.day.astype(np.int32)
    day_of_year = ts.dt.dayofyear.astype(np.int32)
    pu = df["pickup_zone"].astype(np.int32).to_numpy()
    do = df["dropoff_zone"].astype(np.int32).to_numpy()

    hour_np = hour.to_numpy()
    dow_np = dow.to_numpy()
    month_np = month.to_numpy()
    day_np = day.to_numpy()
    minute_np = minute.to_numpy()

    # Cyclical encodings
    two_pi_h = 2.0 * np.pi * hour_np.astype(np.float64) / 24.0
    two_pi_d = 2.0 * np.pi * dow_np.astype(np.float64) / 7.0

    # --- Pair priors ---
    pm = pair_median[pu, do]
    pc = pair_cnt[pu, do]
    has_pair = pc > 0
    pup = pickup_median[pu].astype(np.float64)
    pair_prior = np.where(has_pair, np.nan_to_num(pm, nan=pup), pup).astype(np.float64)
    log1p_pc = np.where(has_pair, np.log1p(pc.astype(np.float64)), 0.0)

    # --- Smoothed pair prior (Bayesian shrinkage) ---
    w = _SMOOTHING_WEIGHT
    pair_smoothed = np.where(
        has_pair,
        (pc * np.nan_to_num(pm, nan=pup) + w * pup) / (pc + w),
        pup,
    ).astype(np.float64)

    # --- Time-conditioned pair prior ---
    hour_bin = hour_np.astype(np.int32)
    phm = pair_hour_median[pu, do, hour_bin]
    # Fall back to overall pair median, then pickup median
    pair_hour_prior = np.where(
        ~np.isnan(phm), phm,
        np.where(has_pair, np.nan_to_num(pm, nan=pup), pup)
    ).astype(np.float64)

    # --- Spatial features ---
    pu_lat = zone_lat[pu]
    pu_lon = zone_lon[pu]
    do_lat = zone_lat[do]
    do_lon = zone_lon[do]

    h_dist = haversine_array(pu_lat, pu_lon, do_lat, do_lon)
    m_dist = manhattan_distance(pu_lat, pu_lon, do_lat, do_lon)
    bearing = bearing_array(pu_lat, pu_lon, do_lat, do_lon)

    # --- Holiday features ---
    month_day_tuples = set(zip(month_np.tolist(), day_np.tolist()))
    is_hol_arr = np.array([(m, d) in _HOLIDAYS for m, d in zip(month_np, day_np)], dtype=np.int8)
    is_hol_period = _is_holiday_period(month_np, day_np).astype(np.int8)

    # --- Rush hour flags ---
    is_weekday = dow_np < 5
    is_morning_rush = (is_weekday & (hour_np >= 7) & (hour_np <= 10)).astype(np.int8)
    is_evening_rush = (is_weekday & (hour_np >= 16) & (hour_np <= 19)).astype(np.int8)

    # --- Minute of day ---
    minute_of_day = (hour_np * 60 + minute_np).astype(np.int32)
    # --- Borough & Airport ---
    pu_bor = np.array([_BOROUGH_MAP.get(z, 6) for z in pu], dtype=np.int8)
    do_bor = np.array([_BOROUGH_MAP.get(z, 6) for z in do], dtype=np.int8)
    is_cross = (pu_bor != do_bor).astype(np.int8)
    is_airport = (np.isin(pu, [1, 132, 138]) | np.isin(do, [1, 132, 138])).astype(np.int8)

    # --- Holiday Proximity ---
    days_to_xmas = np.where(month_np == 12, np.abs(day_np - 25), np.where(month_np == 1, day_np + 6, 180)).astype(np.float32)
    days_to_ny = np.where(month_np == 12, 31 - day_np, np.where(month_np == 1, day_np, 180)).astype(np.float32)

    # --- Speed Proxy ---
    speed_proxy = np.where(pair_prior > 0, h_dist / (pair_prior / 3600.0 + 1e-6), 0.0)

    # --- Weather features ---
    if weather_df is not None:
        rounded_hour = ts.dt.floor("h")
        # Use vectorized reindex for speed
        weather_vals = weather_df.reindex(rounded_hour).values
        temp_c = weather_vals[:, 0].astype(np.float32)
        precip_mm = weather_vals[:, 1].astype(np.float32)
        # humidity is col 2, we skip it (correlated with others)
        wind_speed_kmh = weather_vals[:, 3].astype(np.float32)
        visibility_km = weather_vals[:, 4].astype(np.float32)
        is_raining = (precip_mm > 0.1).astype(np.int8)
        # Fill NaNs with median values for safety
        temp_c = np.nan_to_num(temp_c, nan=15.0)
        precip_mm = np.nan_to_num(precip_mm, nan=0.0)
        wind_speed_kmh = np.nan_to_num(wind_speed_kmh, nan=10.0)
        visibility_km = np.nan_to_num(visibility_km, nan=16.0)
    else:
        n = len(df)
        temp_c = np.full(n, 15.0, dtype=np.float32)
        precip_mm = np.zeros(n, dtype=np.float32)
        wind_speed_kmh = np.full(n, 10.0, dtype=np.float32)
        visibility_km = np.full(n, 16.0, dtype=np.float32)
        is_raining = np.zeros(n, dtype=np.int8)

    # --- Interactions ---
    rain_x_airport = (is_raining * is_airport).astype(np.float32)
    rain_x_dist = (is_raining * h_dist).astype(np.float32)

    out = pd.DataFrame(
        {
            "pickup_zone": pu,
            "dropoff_zone": do,
            "hour": hour_np,
            "dow": dow_np,
            "month": month_np,
            "passenger_count": df["passenger_count"].astype(np.int32).to_numpy(),
            "is_weekend": (dow_np >= 5).astype(np.int8),
            "hour_sin": np.sin(two_pi_h),
            "hour_cos": np.cos(two_pi_h),
            "dow_sin": np.sin(two_pi_d),
            "dow_cos": np.cos(two_pi_d),
            "pair_prior_sec": pair_prior,
            "pair_prior_smoothed": pair_smoothed,
            "log1p_pair_count": log1p_pc,
            "pickup_prior_sec": pickup_median[pu].astype(np.float64),
            "dropoff_prior_sec": dropoff_median[do].astype(np.float64),
            "pair_hour_prior_sec": pair_hour_prior,
            "haversine_dist": h_dist,
            "manhattan_dist": m_dist,
            "bearing": bearing,
            "is_holiday": is_hol_arr,
            "is_holiday_period": is_hol_period,
            "day_of_year": day_of_year.to_numpy(),
            "minute_of_day": minute_of_day,
            "is_morning_rush": is_morning_rush,
            "is_evening_rush": is_evening_rush,
            "pu_bor": pu_bor,
            "do_bor": do_bor,
            "is_cross_borough": is_cross,
            "is_airport": is_airport,
            "days_to_christmas": days_to_xmas,
            "days_to_newyear": days_to_ny,
            "speed_proxy": speed_proxy,
            "temp_c": temp_c,
            "precip_mm": precip_mm,
            "wind_speed_kmh": wind_speed_kmh,
            "visibility_km": visibility_km,
            "is_raining": is_raining,
            "rain_x_airport": rain_x_airport,
            "rain_x_dist": rain_x_dist,
        }
    )
    return out[FEATURE_ORDER]


# ── Main ─────────────────────────────────────────────────────────────────────

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

    weather_path = DATA_DIR / "weather_hourly.csv"
    weather_df = None
    if weather_path.exists():
        print("Loading weather data...")
        weather_df = build_weather_lookup(weather_path)
        print(f"  weather range: {weather_df.index.min()} → {weather_df.index.max()}")
    else:
        print("  ⚠ weather_hourly.csv not found — weather features will be zeroed")

    print("Computing aggregates from train...")
    t0 = time.time()
    pair_median, pair_cnt, pickup_median, dropoff_median, pair_hour_median, global_med = \
        build_aggregate_arrays(train)
    print(f"  done in {time.time() - t0:.1f}s")

    print("Building feature matrices...")
    X_train = engineer_features(
        train, pair_median, pair_cnt, pickup_median, dropoff_median,
        pair_hour_median, global_med, zone_lat, zone_lon, weather_df,
    )
    y_train = train["duration_seconds"].to_numpy(dtype=np.float64)
    X_dev = engineer_features(
        dev, pair_median, pair_cnt, pickup_median, dropoff_median,
        pair_hour_median, global_med, zone_lat, zone_lon, weather_df,
    )
    y_dev = dev["duration_seconds"].to_numpy(dtype=np.float64)

    print(f"\nFeature count: {X_train.shape[1]}")
    print(f"Features: {list(X_train.columns)}")

    def check_gpu():
        try:
            xgb.XGBRegressor(tree_method='gpu_hist').get_booster()
            return True
        except:
            return False

    use_gpu = check_gpu()
    print(f"\nTraining XGBoost (objective tuned for MAE) using {'GPU' if use_gpu else 'CPU'}...")
    
    xgb_params = {
        "n_estimators": 3000,
        "max_depth": 10,
        "learning_rate": 0.03,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.5,
        "reg_alpha": 0.1,
        "gamma": 0.1,
        "random_state": 42,
        "objective": "reg:absoluteerror",
        "early_stopping_rounds": 100,
    }
    if use_gpu:
        xgb_params.update({"tree_method": "gpu_hist", "predictor": "gpu_predictor"})
    else:
        xgb_params.update({"tree_method": "hist", "n_jobs": -1})

    model = xgb.XGBRegressor(**xgb_params, callbacks=[TqdmCallback(xgb_params["n_estimators"])])
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
        "version": 6,
        "xgb": model,
        "pair_median": pair_median,
        "pair_cnt": pair_cnt,
        "pickup_median": pickup_median,
        "dropoff_median": dropoff_median,
        "pair_hour_median": pair_hour_median,
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
