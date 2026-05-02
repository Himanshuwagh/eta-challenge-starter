#!/usr/bin/env python
"""
ETA Trainer v5 — Multi-GBDT Stacking (XGBoost + LightGBM + CatBoost)

Strategy 1 from deep analysis: stack 3 diverse GBDT models with Ridge meta-learner.
Also incorporates: borough features, enhanced holiday features, Huber loss, outlier clipping.

Run:
    python train_v5_stack.py
Fast iteration:
    ETA_SAMPLE_FRAC=0.05 python train_v5_stack.py
"""
from __future__ import annotations
import os, pickle, time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from joblib import Parallel, delayed

def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    try:
        import xgboost as xgb
        xgb.XGBRegressor(tree_method='gpu_hist').get_booster()
        return "cuda"
    except:
        pass
    return "cpu"

USE_GPU = check_gpu() == "cuda"
print(f"Detected Device: {'GPU' if USE_GPU else 'CPU'}")

DATA_DIR = Path(__file__).parent / "data"
MODEL_PATH = Path(os.environ.get("ETA_MODEL_PATH", Path(__file__).parent / "model.pkl"))
_MAX_ZONE = 266
_SMOOTHING_WEIGHT = 50

# ── Borough lookup ────────────────────────────────────────────────────────────
_BOROUGH_MAP = {}
_SERVICE_MAP = {}
try:
    _zl = pd.read_csv(DATA_DIR / "zone_lookup.csv")
    for _, r in _zl.iterrows():
        zid = int(r["LocationID"])
        bor = str(r["Borough"]).strip()
        svc = str(r.get("service_zone", "")).strip()
        _BOROUGH_MAP[zid] = {"Manhattan": 0, "Brooklyn": 1, "Queens": 2,
                              "Bronx": 3, "Staten Island": 4, "EWR": 5}.get(bor, 6)
        _SERVICE_MAP[zid] = {"Yellow Zone": 0, "Boro Zone": 1, "EWR": 2, "Airports": 3}.get(svc, 4)
except Exception:
    pass

_HOLIDAYS = {
    (1,1),(1,2),(1,15),(1,16),(2,19),(2,20),(5,29),(6,19),(7,4),
    (9,4),(10,9),(11,10),(11,23),(11,24),(12,25),(12,31),
}

# ── Geometry ──────────────────────────────────────────────────────────────────
def haversine(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    d = np.sin((lat2-lat1)*0.5)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lng2-lng1)*0.5)**2
    return 2 * 6371 * np.arcsin(np.sqrt(d))

def manhattan_dist(lat1, lng1, lat2, lng2):
    return haversine(lat1, lng1, lat1, lng2) + haversine(lat1, lng1, lat2, lng1)

def bearing(lat1, lng1, lat2, lng2):
    dl = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(dl) * np.cos(lat2)
    x = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dl)
    return np.degrees(np.arctan2(y, x))

# ── Aggregates ────────────────────────────────────────────────────────────────
def build_aggregates(train):
    global_med = float(train["duration_seconds"].median())
    pair_median = np.full((_MAX_ZONE, _MAX_ZONE), np.nan, dtype=np.float32)
    pair_cnt = np.zeros((_MAX_ZONE, _MAX_ZONE), dtype=np.int32)

    grp = train.groupby(["pickup_zone","dropoff_zone"], sort=False)["duration_seconds"]
    cnt, med = grp.size(), grp.median()
    pu_i = cnt.index.get_level_values(0).astype(np.int32).to_numpy()
    do_i = cnt.index.get_level_values(1).astype(np.int32).to_numpy()
    pair_cnt[pu_i, do_i] = cnt.to_numpy().astype(np.int32)
    pair_median[pu_i, do_i] = med.to_numpy().astype(np.float32)

    pickup_median = np.full(_MAX_ZONE, np.nan, np.float32)
    ps = train.groupby("pickup_zone")["duration_seconds"].median()
    pickup_median[ps.index.astype(np.int32)] = ps.to_numpy(np.float32)
    pickup_median = np.where(np.isnan(pickup_median), global_med, pickup_median).astype(np.float32)

    dropoff_median = np.full(_MAX_ZONE, np.nan, np.float32)
    ds = train.groupby("dropoff_zone")["duration_seconds"].median()
    dropoff_median[ds.index.astype(np.int32)] = ds.to_numpy(np.float32)
    dropoff_median = np.where(np.isnan(dropoff_median), global_med, dropoff_median).astype(np.float32)

    ts = pd.to_datetime(train["requested_at"])
    tmp = train.assign(hbin=(ts.dt.hour // 3).astype(np.int8))
    pair_hour_med = np.full((_MAX_ZONE, _MAX_ZONE, 8), np.nan, np.float32)
    gh = tmp.groupby(["pickup_zone","dropoff_zone","hbin"], sort=False)["duration_seconds"].median()
    ph_pu = gh.index.get_level_values(0).astype(np.int32).to_numpy()
    ph_do = gh.index.get_level_values(1).astype(np.int32).to_numpy()
    ph_hb = gh.index.get_level_values(2).astype(np.int32).to_numpy()
    pair_hour_med[ph_pu, ph_do, ph_hb] = gh.to_numpy(np.float32)

    # DOW-conditioned pair priors
    tmp2 = train.assign(dow=ts.dt.dayofweek.astype(np.int8))
    pair_dow_med = np.full((_MAX_ZONE, _MAX_ZONE, 7), np.nan, np.float32)
    gd = tmp2.groupby(["pickup_zone","dropoff_zone","dow"], sort=False)["duration_seconds"].median()
    pd_pu = gd.index.get_level_values(0).astype(np.int32).to_numpy()
    pd_do = gd.index.get_level_values(1).astype(np.int32).to_numpy()
    pd_dw = gd.index.get_level_values(2).astype(np.int32).to_numpy()
    pair_dow_med[pd_pu, pd_do, pd_dw] = gd.to_numpy(np.float32)

    return pair_median, pair_cnt, pickup_median, dropoff_median, pair_hour_med, pair_dow_med, global_med

def build_weather(path):
    wdf = pd.read_csv(path)
    wdf["datetime"] = pd.to_datetime(wdf["datetime"])
    wdf = wdf.set_index("datetime").sort_index()
    full = pd.date_range(wdf.index.min(), wdf.index.max(), freq="h")
    return wdf.reindex(full).ffill().bfill()

# ── Feature extraction ────────────────────────────────────────────────────────
def extract_features(df, pair_median, pair_cnt, pickup_median, dropoff_median,
                     pair_hour_med, pair_dow_med, global_med, zone_lat, zone_lon, weather_df=None):
    ts = pd.to_datetime(df["requested_at"])
    hour = ts.dt.hour.to_numpy(np.int32)
    minute = ts.dt.minute.to_numpy(np.int32)
    dow = ts.dt.dayofweek.to_numpy(np.int32)
    month = ts.dt.month.to_numpy(np.int32)
    day = ts.dt.day.to_numpy(np.int32)
    doy = ts.dt.dayofyear.to_numpy(np.int32)
    woy = ts.dt.isocalendar().week.to_numpy(np.int32)
    pu = df["pickup_zone"].to_numpy(np.int32)
    do = df["dropoff_zone"].to_numpy(np.int32)

    # Pair priors
    pm = pair_median[pu, do]
    pc = pair_cnt[pu, do]
    has = pc > 0
    pup = pickup_median[pu].astype(np.float64)
    dop = dropoff_median[do].astype(np.float64)
    pair_prior = np.where(has, np.nan_to_num(pm, nan=pup), pup)
    pair_smoothed = np.where(has, (pc*np.nan_to_num(pm,nan=pup)+_SMOOTHING_WEIGHT*pup)/(pc+_SMOOTHING_WEIGHT), pup)
    log1p_pc = np.where(has, np.log1p(pc.astype(np.float64)), 0.0)
    hbin = hour // 3
    phm = pair_hour_med[pu, do, hbin]
    pair_hour_prior = np.where(~np.isnan(phm), phm, pair_prior)
    # DOW prior
    pdm = pair_dow_med[pu, do, dow]
    pair_dow_prior = np.where(~np.isnan(pdm), pdm, pair_prior)

    # Spatial
    pu_lat, pu_lon = zone_lat[pu], zone_lon[pu]
    do_lat, do_lon = zone_lat[do], zone_lon[do]
    h_dist = haversine(pu_lat, pu_lon, do_lat, do_lon)
    m_dist = manhattan_dist(pu_lat, pu_lon, do_lat, do_lon)
    bear = bearing(pu_lat, pu_lon, do_lat, do_lon)

    # Borough
    pu_bor = np.array([_BOROUGH_MAP.get(z, 6) for z in pu], np.int32)
    do_bor = np.array([_BOROUGH_MAP.get(z, 6) for z in do], np.int32)
    is_cross_borough = (pu_bor != do_bor).astype(np.float32)
    is_airport = np.isin(pu, [1,132,138]) | np.isin(do, [1,132,138])
    is_airport = is_airport.astype(np.float32)

    # Holiday
    is_hol = np.array([(m,d) in _HOLIDAYS for m,d in zip(month,day)], np.float32)
    is_xmas_week = (((month==12)&(day>=22))|((month==1)&(day<=2))).astype(np.float32)
    days_to_xmas = np.where(month==12, np.abs(day-25), np.where(month==1, day+6, 180)).astype(np.float32)
    days_to_ny = np.where(month==12, 31-day, np.where(month==1, day, 180)).astype(np.float32)

    # Time
    is_wd = dow < 5
    is_mr = (is_wd & (hour>=7) & (hour<=10)).astype(np.float32)
    is_er = (is_wd & (hour>=16) & (hour<=19)).astype(np.float32)
    min_of_day = (hour*60+minute).astype(np.float32)
    two_pi_h = 2*np.pi*hour/24.0
    two_pi_d = 2*np.pi*dow/7.0
    two_pi_w = 2*np.pi*woy/52.0

    # Speed proxy
    speed_proxy = np.where(pair_prior > 0, h_dist / (pair_prior / 3600.0 + 1e-6), 0.0)

    # Weather
    n = len(df)
    if weather_df is not None:
        rounded = ts.dt.floor("h")
        wv = weather_df.reindex(rounded).values
        temp_c = np.nan_to_num(wv[:,0], nan=15.0).astype(np.float32)
        precip = np.nan_to_num(wv[:,1], nan=0.0).astype(np.float32)
        humidity = np.nan_to_num(wv[:,2], nan=60.0).astype(np.float32)
        wind = np.nan_to_num(wv[:,3], nan=10.0).astype(np.float32)
        vis = np.nan_to_num(wv[:,4], nan=16.0).astype(np.float32)
        is_rain = (precip > 0.1).astype(np.float32)
    else:
        temp_c = np.full(n,15.0,np.float32)
        precip = np.zeros(n,np.float32)
        humidity = np.full(n,60.0,np.float32)
        wind = np.full(n,10.0,np.float32)
        vis = np.full(n,16.0,np.float32)
        is_rain = np.zeros(n,np.float32)

    # Interactions
    rain_x_airport = is_rain * is_airport
    rain_x_dist = is_rain * h_dist.astype(np.float32)

    feats = np.column_stack([
        pu, do, hour, dow, month,
        df["passenger_count"].to_numpy(np.float32),
        (dow>=5).astype(np.float32),
        np.sin(two_pi_h), np.cos(two_pi_h),
        np.sin(two_pi_d), np.cos(two_pi_d),
        np.sin(two_pi_w), np.cos(two_pi_w),
        pair_prior, pair_smoothed, log1p_pc,
        pup, dop, pair_hour_prior, pair_dow_prior,
        h_dist, m_dist, bear,
        is_hol, is_xmas_week, days_to_xmas, days_to_ny,
        doy.astype(np.float32), min_of_day,
        is_mr, is_er,
        pu_bor.astype(np.float32), do_bor.astype(np.float32),
        is_cross_borough, is_airport,
        speed_proxy.astype(np.float32),
        temp_c, precip, humidity, wind, vis, is_rain,
        rain_x_airport, rain_x_dist,
        woy.astype(np.float32),
    ]).astype(np.float32)
    return feats

FEATURE_NAMES = [
    "pu","do","hour","dow","month","pax","is_wknd",
    "hour_sin","hour_cos","dow_sin","dow_cos","woy_sin","woy_cos",
    "pair_prior","pair_smoothed","log1p_pc","pickup_med","dropoff_med",
    "pair_hour_prior","pair_dow_prior",
    "h_dist","m_dist","bearing",
    "is_hol","is_xmas_week","days_to_xmas","days_to_ny",
    "doy","min_of_day","is_mr","is_er",
    "pu_bor","do_bor","is_cross_borough","is_airport",
    "speed_proxy",
    "temp_c","precip","humidity","wind","vis","is_rain",
    "rain_x_airport","rain_x_dist","woy",
]

CAT_INDICES = [0, 1, 2, 3, 4, 31, 32]  # pu,do,hour,dow,month,pu_bor,do_bor

# ── Training ──────────────────────────────────────────────────────────────────
def main():
    train_path = DATA_DIR / "train.parquet"
    dev_path = DATA_DIR / "dev.parquet"
    for p in (train_path, dev_path):
        if not p.exists():
            raise SystemExit(f"Missing {p.name}")

    print("Loading data...")
    train = pd.read_parquet(train_path)
    dev = pd.read_parquet(dev_path)
    frac = os.environ.get("ETA_SAMPLE_FRAC")
    if frac:
        f = float(frac)
        train = train.sample(frac=f, random_state=42)
        dev = dev.sample(frac=f, random_state=43)
        print(f"  ETA_SAMPLE_FRAC={f}")
    print(f"  train: {len(train):,} | dev: {len(dev):,}")

    # Clip outliers for aggregate computation (not target)
    clip_val = float(train["duration_seconds"].quantile(0.99))
    print(f"  Clipping aggregates at 99th pct: {clip_val:.0f}s")
    train_clipped = train.assign(duration_seconds=train["duration_seconds"].clip(upper=clip_val))

    # Zone coords
    coords = pd.read_csv(DATA_DIR / "zone_coords.csv")
    zone_lat = np.full(_MAX_ZONE, 40.75, np.float32)
    zone_lon = np.full(_MAX_ZONE, -73.98, np.float32)
    ids = coords["zone_id"].to_numpy(np.int32)
    m = ids < _MAX_ZONE
    zone_lat[ids[m]] = coords["latitude"].to_numpy(np.float32)[m]
    zone_lon[ids[m]] = coords["longitude"].to_numpy(np.float32)[m]

    # Weather
    weather_df = None
    wp = DATA_DIR / "weather_hourly.csv"
    if wp.exists():
        print("Loading weather...")
        weather_df = build_weather(wp)

    # Aggregates from clipped data
    print("Computing aggregates...")
    aggs = build_aggregates(train_clipped)
    pair_median, pair_cnt, pickup_median, dropoff_median, pair_hour_med, pair_dow_med, global_med = aggs

    # Features
    print("Extracting features...")
    X_tr = extract_features(train, *aggs, zone_lat, zone_lon, weather_df)
    y_tr = train["duration_seconds"].to_numpy(np.float64)
    X_dev = extract_features(dev, *aggs, zone_lat, zone_lon, weather_df)
    y_dev = dev["duration_seconds"].to_numpy(np.float64)
    print(f"  Features: {X_tr.shape[1]}")

    # ── Model 1: XGBoost (Huber loss) ─────────────────────────────────────
    print("\n══ Model 1: XGBoost ══")
    t0 = time.time()
    # ── Model 1: XGBoost ──────────────────────────────────────────────────
    print("\n══ Model 1: XGBoost ══")
    t0 = time.time()
    xgb_params = {
        "n_estimators": 4000, "max_depth": 10, "learning_rate": 0.03,
        "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.7,
        "reg_lambda": 2.0, "reg_alpha": 0.1, "gamma": 0.1,
        "random_state": 42,
        "objective": "reg:pseudohubererror", "huber_slope": 500,
        "early_stopping_rounds": 100,
    }
    if USE_GPU:
        xgb_params.update({"tree_method": "gpu_hist", "predictor": "gpu_predictor"})
    else:
        xgb_params.update({"tree_method": "hist", "n_jobs": -1})

    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_dev, y_dev)], verbose=False)
    xgb_pred = xgb_model.predict(X_dev)
    xgb_mae = float(np.mean(np.abs(xgb_pred - y_dev)))
    print(f"  XGB MAE: {xgb_mae:.1f}s ({time.time()-t0:.0f}s, iter={xgb_model.best_iteration})")
    if hasattr(xgb_model, "get_booster"):
        xgb_model.get_booster().feature_names = None

    # ── Model 2: LightGBM ────────────────────────────────────────────────
    print("\n══ Model 2: LightGBM ══")
    t0 = time.time()
    lgb_params = {
        "objective": "huber", "alpha": 500,
        "num_leaves": 255, "max_depth": -1,
        "learning_rate": 0.03, "n_estimators": 4000,
        "min_child_samples": 20, "subsample": 0.8,
        "colsample_bytree": 0.7, "reg_lambda": 2.0,
        "reg_alpha": 0.1, "random_state": 43,
        "verbose": -1,
    }
    if USE_GPU:
        lgb_params.update({"device": "gpu", "gpu_use_dp": False})
    else:
        lgb_params.update({"n_jobs": -1})

    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(X_tr, y_tr, eval_set=[(X_dev, y_dev)],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    lgb_pred = lgb_model.predict(X_dev)
    lgb_mae = float(np.mean(np.abs(lgb_pred - y_dev)))
    print(f"  LGB MAE: {lgb_mae:.1f}s ({time.time()-t0:.0f}s, iter={lgb_model.best_iteration_})")

    # ── Model 3: CatBoost ────────────────────────────────────────────────
    print("\n══ Model 3: CatBoost ══")
    t0 = time.time()
    cat_cols = [FEATURE_NAMES[i] for i in CAT_INDICES]
    X_tr_df = pd.DataFrame(X_tr, columns=FEATURE_NAMES)
    X_dev_df = pd.DataFrame(X_dev, columns=FEATURE_NAMES)
    for c in cat_cols:
        X_tr_df[c] = X_tr_df[c].astype(int)
        X_dev_df[c] = X_dev_df[c].astype(int)

    cat_params = {
        "iterations": 2000, "depth": 6, "learning_rate": 0.08,
        "l2_leaf_reg": 3.0, "random_seed": 44,
        "loss_function": "MAE", "eval_metric": "MAE",
        "early_stopping_rounds": 80, "verbose": 200,
        "cat_features": cat_cols,
    }
    if USE_GPU:
        cat_params.update({"task_type": "GPU", "devices": "0"})
    else:
        cat_params.update({"thread_count": -1})

    cat_model = CatBoostRegressor(**cat_params)
    cat_model.fit(X_tr_df, y_tr, eval_set=(X_dev_df, y_dev))
    cat_pred = cat_model.predict(X_dev_df)
    cat_mae = float(np.mean(np.abs(cat_pred - y_dev)))
    print(f"  CAT MAE: {cat_mae:.1f}s ({time.time()-t0:.0f}s, iter={cat_model.best_iteration_})")

    # ── Meta-learner: Ridge on OOF ────────────────────────────────────────
    print("\n══ Stacking: Ridge Meta-Learner ══")
    # Simple dev-set stacking (for speed)
    stack_X = np.column_stack([xgb_pred, lgb_pred, cat_pred])
    meta = Ridge(alpha=1.0)
    meta.fit(stack_X, y_dev)
    meta_pred = meta.predict(stack_X)
    meta_mae = float(np.mean(np.abs(meta_pred - y_dev)))
    print(f"  Meta MAE (on dev, optimistic): {meta_mae:.1f}s")
    print(f"  Weights: XGB={meta.coef_[0]:.3f}, LGB={meta.coef_[1]:.3f}, CAT={meta.coef_[2]:.3f}, bias={meta.intercept_:.1f}")

    # Also try simple average
    avg_pred = (xgb_pred + lgb_pred + cat_pred) / 3
    avg_mae = float(np.mean(np.abs(avg_pred - y_dev)))
    print(f"  Simple avg MAE: {avg_mae:.1f}s")

    # Grid search pairwise
    best_w, best_ens_mae = None, float("inf")
    for w1 in np.arange(0.1, 0.8, 0.1):
        for w2 in np.arange(0.1, 0.8-w1, 0.1):
            w3 = 1.0 - w1 - w2
            if w3 < 0.05: continue
            p = w1*xgb_pred + w2*lgb_pred + w3*cat_pred
            mae = float(np.mean(np.abs(p - y_dev)))
            if mae < best_ens_mae:
                best_ens_mae = mae
                best_w = (w1, w2, w3)
    print(f"  Grid-search best: XGB={best_w[0]:.1f}, LGB={best_w[1]:.1f}, CAT={best_w[2]:.1f} → MAE: {best_ens_mae:.1f}s")

    print(f"\n══ Summary ══")
    print(f"  XGBoost:    {xgb_mae:.1f}s")
    print(f"  LightGBM:   {lgb_mae:.1f}s")
    print(f"  CatBoost:   {cat_mae:.1f}s")
    print(f"  Simple avg: {avg_mae:.1f}s")
    print(f"  Grid best:  {best_ens_mae:.1f}s")

    # Save bundle
    bundle = {
        "version": 5,
        "xgb_model": xgb_model,
        "lgb_model": lgb_model,
        "cat_model": cat_model,
        "ensemble_weights": best_w,
        "cat_features": CAT_INDICES,
        "pair_median": pair_median,
        "pair_cnt": pair_cnt,
        "pickup_median": pickup_median,
        "dropoff_median": dropoff_median,
        "pair_hour_median": pair_hour_med,
        "pair_dow_median": pair_dow_med,
        "global_median": global_med,
        "zone_lat": zone_lat,
        "zone_lon": zone_lon,
        "feature_names": FEATURE_NAMES,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nSaved v5 bundle to {MODEL_PATH}")

if __name__ == "__main__":
    main()
