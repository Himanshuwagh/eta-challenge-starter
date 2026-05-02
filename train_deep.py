#!/usr/bin/env python
"""
ETA Deep Learning Trainer v4 — Entity Embedding MLP + XGBoost Ensemble

Architecture (inspired by Uber's DeepETA + Kaggle entity embedding papers):
  - pickup_zone  → Embedding(266, 16)
  - dropoff_zone → Embedding(266, 16)
  - hour         → Embedding(24, 8)
  - dow          → Embedding(7, 4)
  - Concat(embeddings, continuous_features) → MLP(512→256→128→64→1)
  - BatchNorm + GELU activations + Dropout

Strategy:
  1. Train the NN with entity embeddings (CPU or MPS on Apple Silicon).
  2. Extract the trained zone embeddings (16-dim vectors per zone).
  3. Feed those embeddings into a fresh XGBoost model as additional features.
  4. Ensemble NN + XGBoost predictions for the final model.pkl bundle.

Run:
    python train_deep.py

Fast iteration (5% data):
    ETA_SAMPLE_FRAC=0.05 ETA_MODEL_PATH=model.pkl python train_deep.py
"""

from __future__ import annotations

import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

DATA_DIR = Path(__file__).parent / "data"
MODEL_PATH = Path(os.environ.get("ETA_MODEL_PATH", Path(__file__).parent / "model.pkl"))
DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
_MAX_ZONE = 266
_SMOOTHING_WEIGHT = 50

print(f"[deep] Using device: {DEVICE}")

# ── US Holidays (same as v3) ──────────────────────────────────────────────────
_HOLIDAYS = {
    (1, 1), (1, 2), (1, 15), (1, 16), (2, 19), (2, 20),
    (5, 29), (6, 19), (7, 4), (9, 4), (10, 9), (11, 10),
    (11, 23), (11, 24), (12, 25), (12, 31),
}

# ── Geometry helpers ──────────────────────────────────────────────────────────
def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    d = np.sin((lat2-lat1)*0.5)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lng2-lng1)*0.5)**2
    return 2 * 6371 * np.arcsin(np.sqrt(d))

def manhattan_dist(lat1, lng1, lat2, lng2):
    return haversine_array(lat1, lng1, lat1, lng2) + haversine_array(lat1, lng1, lat2, lng1)

def bearing_array(lat1, lng1, lat2, lng2):
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))

# ── Aggregate arrays ──────────────────────────────────────────────────────────
def build_aggregates(train):
    global_med = float(train["duration_seconds"].median())
    pair_median = np.full((_MAX_ZONE, _MAX_ZONE), np.nan, dtype=np.float32)
    pair_cnt = np.zeros((_MAX_ZONE, _MAX_ZONE), dtype=np.int32)

    grp = train.groupby(["pickup_zone", "dropoff_zone"], sort=False)["duration_seconds"]
    cnt, med = grp.size(), grp.median()
    pu_idx = cnt.index.get_level_values(0).astype(np.int32).to_numpy()
    do_idx = cnt.index.get_level_values(1).astype(np.int32).to_numpy()
    pair_cnt[pu_idx, do_idx] = cnt.to_numpy().astype(np.int32)
    pair_median[pu_idx, do_idx] = med.to_numpy().astype(np.float32)

    pickup_median = np.full(_MAX_ZONE, np.nan, dtype=np.float32)
    ps = train.groupby("pickup_zone")["duration_seconds"].median()
    pickup_median[ps.index.astype(np.int32)] = ps.to_numpy().astype(np.float32)
    pickup_median = np.where(np.isnan(pickup_median), global_med, pickup_median).astype(np.float32)

    dropoff_median = np.full(_MAX_ZONE, np.nan, dtype=np.float32)
    ds = train.groupby("dropoff_zone")["duration_seconds"].median()
    dropoff_median[ds.index.astype(np.int32)] = ds.to_numpy().astype(np.float32)
    dropoff_median = np.where(np.isnan(dropoff_median), global_med, dropoff_median).astype(np.float32)

    ts = pd.to_datetime(train["requested_at"])
    train_tmp = train.assign(hour_bin=(ts.dt.hour // 3).astype(np.int8))
    pair_hour_med = np.full((_MAX_ZONE, _MAX_ZONE, 8), np.nan, dtype=np.float32)
    grp_h = train_tmp.groupby(["pickup_zone", "dropoff_zone", "hour_bin"], sort=False)["duration_seconds"].median()
    ph_pu = grp_h.index.get_level_values(0).astype(np.int32).to_numpy()
    ph_do = grp_h.index.get_level_values(1).astype(np.int32).to_numpy()
    ph_hb = grp_h.index.get_level_values(2).astype(np.int32).to_numpy()
    pair_hour_med[ph_pu, ph_do, ph_hb] = grp_h.to_numpy().astype(np.float32)

    return pair_median, pair_cnt, pickup_median, dropoff_median, pair_hour_med, global_med

def build_weather_lookup(weather_path):
    wdf = pd.read_csv(weather_path)
    wdf["datetime"] = pd.to_datetime(wdf["datetime"])
    wdf = wdf.set_index("datetime").sort_index()
    full_range = pd.date_range(wdf.index.min(), wdf.index.max(), freq="h")
    wdf = wdf.reindex(full_range).ffill().bfill()
    wdf.index.name = "datetime"
    return wdf

# ── Feature extraction (returns categorical ints + continuous floats) ─────────
def extract_features(df, pair_median, pair_cnt, pickup_median, dropoff_median,
                     pair_hour_med, global_med, zone_lat, zone_lon, weather_df=None):
    ts = pd.to_datetime(df["requested_at"])
    hour = ts.dt.hour.to_numpy(np.int32)
    minute = ts.dt.minute.to_numpy(np.int32)
    dow = ts.dt.dayofweek.to_numpy(np.int32)
    month = ts.dt.month.to_numpy(np.int32)
    day = ts.dt.day.to_numpy(np.int32)
    doy = ts.dt.dayofyear.to_numpy(np.int32)
    pu = df["pickup_zone"].to_numpy(np.int32)
    do = df["dropoff_zone"].to_numpy(np.int32)

    # Pair priors
    pm = pair_median[pu, do]
    pc = pair_cnt[pu, do]
    has_pair = pc > 0
    pup = pickup_median[pu].astype(np.float64)
    dop = dropoff_median[do].astype(np.float64)
    pair_prior = np.where(has_pair, np.nan_to_num(pm, nan=pup), pup)
    pair_smoothed = np.where(has_pair, (pc * np.nan_to_num(pm, nan=pup) + _SMOOTHING_WEIGHT * pup) / (pc + _SMOOTHING_WEIGHT), pup)
    log1p_pc = np.where(has_pair, np.log1p(pc.astype(np.float64)), 0.0)
    hbin = hour // 3
    phm = pair_hour_med[pu, do, hbin]
    pair_hour_prior = np.where(~np.isnan(phm), phm, pair_prior)

    # Spatial
    pu_lat, pu_lon = zone_lat[pu], zone_lon[pu]
    do_lat, do_lon = zone_lat[do], zone_lon[do]
    h_dist = haversine_array(pu_lat, pu_lon, do_lat, do_lon)
    m_dist = manhattan_dist(pu_lat, pu_lon, do_lat, do_lon)
    bear = bearing_array(pu_lat, pu_lon, do_lat, do_lon)

    # Holiday
    is_hol = np.array([(m, d) in _HOLIDAYS for m, d in zip(month, day)], dtype=np.float32)
    is_xmas = (((month == 12) & (day >= 22)) | ((month == 1) & (day <= 2))).astype(np.float32)

    # Time
    is_wd = dow < 5
    is_mr = (is_wd & (hour >= 7) & (hour <= 10)).astype(np.float32)
    is_er = (is_wd & (hour >= 16) & (hour <= 19)).astype(np.float32)
    min_of_day = (hour * 60 + minute).astype(np.float32)
    two_pi_h = 2.0 * np.pi * hour / 24.0
    two_pi_d = 2.0 * np.pi * dow / 7.0

    # Weather
    if weather_df is not None:
        rounded = ts.dt.floor("h")
        wvals = weather_df.reindex(rounded).values
        temp_c = np.nan_to_num(wvals[:, 0], nan=15.0).astype(np.float32)
        precip = np.nan_to_num(wvals[:, 1], nan=0.0).astype(np.float32)
        wind = np.nan_to_num(wvals[:, 3], nan=10.0).astype(np.float32)
        vis = np.nan_to_num(wvals[:, 4], nan=16.0).astype(np.float32)
        is_rain = (precip > 0.1).astype(np.float32)
    else:
        n = len(df)
        temp_c = np.full(n, 15.0, np.float32)
        precip = np.zeros(n, np.float32)
        wind = np.full(n, 10.0, np.float32)
        vis = np.full(n, 16.0, np.float32)
        is_rain = np.zeros(n, np.float32)

    # Categorical tensors (for embedding lookup)
    cats = np.stack([pu, do, hour, dow], axis=1).astype(np.int32)

    # Continuous features
    conts = np.column_stack([
        pair_prior, pair_smoothed, log1p_pc,
        pup, dop, pair_hour_prior,
        h_dist, m_dist, bear,
        is_hol, is_xmas, is_mr, is_er,
        doy.astype(np.float32), min_of_day,
        np.sin(two_pi_h), np.cos(two_pi_h),
        np.sin(two_pi_d), np.cos(two_pi_d),
        temp_c, precip, wind, vis, is_rain,
        df["passenger_count"].to_numpy(np.float32),
        (dow >= 5).astype(np.float32),
        month.astype(np.float32),
    ]).astype(np.float32)

    return cats, conts

N_CONT = 27  # number of continuous features (must match extract_features)

# ── Neural Network Architecture ───────────────────────────────────────────────
class ETANet(nn.Module):
    """
    Entity Embedding MLP for ETA prediction.
    
    Categorical variables (zone, hour, dow) get learned dense embeddings.
    These embeddings allow the model to discover that spatially/temporally
    similar zones/hours behave similarly — something trees cannot do.
    """

    def __init__(self, n_continuous: int = N_CONT):
        super().__init__()
        # Embeddings — zone_id up to 265, hour 0-23, dow 0-6
        self.pu_emb = nn.Embedding(266, 16)
        self.do_emb = nn.Embedding(266, 16)
        self.hour_emb = nn.Embedding(24, 8)
        self.dow_emb = nn.Embedding(7, 4)

        emb_dim = 16 + 16 + 8 + 4  # = 44
        in_dim = emb_dim + n_continuous  # 44 + 27 = 71

        self.bn_input = nn.BatchNorm1d(n_continuous)
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, cats, conts):
        pu_e = self.pu_emb(cats[:, 0])
        do_e = self.do_emb(cats[:, 1])
        hr_e = self.hour_emb(cats[:, 2])
        dw_e = self.dow_emb(cats[:, 3])
        conts_n = self.bn_input(conts)
        x = torch.cat([pu_e, do_e, hr_e, dw_e, conts_n], dim=1)
        return self.net(x).squeeze(1)

    def get_zone_embeddings(self):
        """Extract pickup/dropoff zone embedding matrices (266 × 16)."""
        pu_w = self.pu_emb.weight.detach().cpu().numpy()  # (266, 16)
        do_w = self.do_emb.weight.detach().cpu().numpy()  # (266, 16)
        return pu_w, do_w


def train_nn(cats_tr, conts_tr, y_tr, cats_dev, conts_dev, y_dev,
             epochs=40, batch_size=8192, lr=1e-3):
    """Train ETANet, return trained model and training history."""
    model = ETANet(n_continuous=conts_tr.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
    )

    # Keep data on CPU initially to avoid massive VRAM consumption for large datasets
    cats_tr_t = torch.tensor(cats_tr, dtype=torch.long)
    conts_tr_t = torch.tensor(conts_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    cats_dev_t = torch.tensor(cats_dev, dtype=torch.long).to(DEVICE)
    conts_dev_t = torch.tensor(conts_dev, dtype=torch.float32).to(DEVICE)
    y_dev_t = torch.tensor(y_dev, dtype=torch.float32).to(DEVICE)

    ds = TensorDataset(cats_tr_t, conts_tr_t, y_tr_t)
    nw = min(os.cpu_count() or 1, 8)  # parallel workers
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, 
                        num_workers=nw, pin_memory=(DEVICE == "cuda"))

    best_mae, best_state, patience_count = float("inf"), None, 0
    patience = 7

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for bc, bx, by in loader:
            # Move batches to GPU dynamically
            bc, bx, by = bc.to(DEVICE), bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            pred = model(bc, bx)
            loss = torch.mean(torch.abs(pred - by))  # MAE loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(by)

        model.eval()
        with torch.no_grad():
            dev_pred = model(cats_dev_t, conts_dev_t)
            dev_mae = float(torch.mean(torch.abs(dev_pred - y_dev_t)))

        scheduler.step(dev_mae)
        train_mae = total_loss / len(y_tr)
        print(f"  epoch {epoch:02d}/{epochs} | train MAE: {train_mae:.1f}s | dev MAE: {dev_mae:.1f}s | lr: {optimizer.param_groups[0]['lr']:.2e}")

        if dev_mae < best_mae:
            best_mae = dev_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model.to("cpu")
    return model, best_mae

# ── Training patience ─────────────────────────────────────────────────────────
# patience = 7 (set inside train_nn)


# ── XGBoost with embedded features ───────────────────────────────────────────
XGB_CONT_FEATURES = [
    "pair_prior", "pair_smoothed", "log1p_pc",
    "pickup_med", "dropoff_med", "pair_hour_prior",
    "h_dist", "m_dist", "bearing",
    "is_hol", "is_xmas", "is_mr", "is_er",
    "doy", "min_of_day",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "temp_c", "precip", "wind", "vis", "is_rain",
    "pax", "is_wknd", "month",
]


def build_xgb_features(cats, conts, pu_emb, do_emb):
    """Concat raw features with zone embedding vectors for XGBoost."""
    pu_idx = cats[:, 0]
    do_idx = cats[:, 1]
    pu_vecs = pu_emb[pu_idx]  # (N, 16)
    do_vecs = do_emb[do_idx]  # (N, 16)
    return np.hstack([conts, pu_vecs, do_vecs]).astype(np.float64)


def train_xgb(X_tr, y_tr, X_dev, y_dev):
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
    
    # Enable GPU if DEVICE is cuda
    if DEVICE == "cuda":
        xgb_params.update({"tree_method": "gpu_hist", "predictor": "gpu_predictor"})
        print("  [XGB] Using GPU for training.")
    else:
        xgb_params.update({"tree_method": "hist", "n_jobs": -1})
        print("  [XGB] Using CPU for training.")

    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_tr, y_tr, eval_set=[(X_dev, y_dev)], verbose=False)
    preds = model.predict(X_dev)
    mae = float(np.mean(np.abs(preds - y_dev)))
    return model, mae


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
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
        print(f"  ETA_SAMPLE_FRAC={f}")
    print(f"  train: {len(train):,} | dev: {len(dev):,}")

    # Zone coordinates
    print("Loading zone coords...")
    coords_df = pd.read_csv(DATA_DIR / "zone_coords.csv")
    zone_lat = np.full(_MAX_ZONE, 40.75, dtype=np.float32)  # default NYC center
    zone_lon = np.full(_MAX_ZONE, -73.98, dtype=np.float32)
    ids = coords_df["zone_id"].to_numpy(np.int32)
    mask = ids < _MAX_ZONE
    zone_lat[ids[mask]] = coords_df["latitude"].to_numpy(np.float32)[mask]
    zone_lon[ids[mask]] = coords_df["longitude"].to_numpy(np.float32)[mask]

    # Weather
    weather_df = None
    wp = DATA_DIR / "weather_hourly.csv"
    if wp.exists():
        print("Loading weather...")
        weather_df = build_weather_lookup(wp)

    # Aggregates
    print("Computing aggregates...")
    t0 = time.time()
    pair_median, pair_cnt, pickup_median, dropoff_median, pair_hour_med, global_med = \
        build_aggregates(train)
    print(f"  done in {time.time()-t0:.1f}s")

    # Feature extraction
    print("Extracting features...")
    args = (pair_median, pair_cnt, pickup_median, dropoff_median, pair_hour_med, global_med, zone_lat, zone_lon, weather_df)
    cats_tr, conts_tr = extract_features(train, *args)
    y_tr = train["duration_seconds"].to_numpy(np.float32)
    cats_dev, conts_dev = extract_features(dev, *args)
    y_dev = dev["duration_seconds"].to_numpy(np.float32)
    print(f"  cats: {cats_tr.shape} | conts: {conts_tr.shape}")

    # ── Stage 1: Train Neural Network ──────────────────────────────────────
    print("\n══ Stage 1: Training Neural Network ══")
    nn_model, nn_mae = train_nn(
        cats_tr, conts_tr, y_tr,
        cats_dev, conts_dev, y_dev,
        epochs=40,
        batch_size=8192,
        lr=1e-3,
    )
    print(f"\n  NN Best Dev MAE: {nn_mae:.1f}s")

    # Get NN predictions
    nn_model.eval()
    with torch.no_grad():
        nn_preds_tr = nn_model(
            torch.tensor(cats_tr, dtype=torch.long),
            torch.tensor(conts_tr)
        ).numpy()
        nn_preds_dev = nn_model(
            torch.tensor(cats_dev, dtype=torch.long),
            torch.tensor(conts_dev)
        ).numpy()

    # Extract zone embeddings
    pu_emb, do_emb = nn_model.get_zone_embeddings()
    print(f"  Zone embeddings extracted: pickup {pu_emb.shape}, dropoff {do_emb.shape}")

    # ── Stage 2: Train XGBoost with zone embeddings ────────────────────────
    print("\n══ Stage 2: Training XGBoost with zone embeddings ══")
    X_tr_xgb = build_xgb_features(cats_tr, conts_tr, pu_emb, do_emb)
    X_dev_xgb = build_xgb_features(cats_dev, conts_dev, pu_emb, do_emb)
    t1 = time.time()
    xgb_model, xgb_mae = train_xgb(X_tr_xgb, y_tr.astype(np.float64), X_dev_xgb, y_dev.astype(np.float64))
    print(f"  XGBoost Dev MAE: {xgb_mae:.1f}s  (trained in {time.time()-t1:.0f}s, best_iter={xgb_model.best_iteration})")

    if hasattr(xgb_model, "get_booster"):
        xgb_model.get_booster().feature_names = None

    # ── Stage 3: Ensemble ──────────────────────────────────────────────────
    print("\n══ Stage 3: Ensemble ══")
    xgb_preds_dev = xgb_model.predict(X_dev_xgb)

    # Grid search ensemble weight on dev set
    best_w, best_ens_mae = 0.5, float("inf")
    for w in np.arange(0.1, 1.0, 0.1):
        ens = w * xgb_preds_dev + (1 - w) * nn_preds_dev
        mae = float(np.mean(np.abs(ens - y_dev)))
        if mae < best_ens_mae:
            best_ens_mae = mae
            best_w = w

    print(f"  Best ensemble weight: {best_w:.1f} XGB + {1-best_w:.1f} NN → Dev MAE: {best_ens_mae:.1f}s")
    print(f"\n  Score summary:")
    print(f"    NN alone:       {nn_mae:.1f}s")
    print(f"    XGBoost alone:  {xgb_mae:.1f}s")
    print(f"    Ensemble:       {best_ens_mae:.1f}s")

    # ── Save bundle ────────────────────────────────────────────────────────
    bundle = {
        "version": 4,
        "nn_model": nn_model,
        "xgb_model": xgb_model,
        "ensemble_weight_xgb": float(best_w),
        "pu_emb": pu_emb,
        "do_emb": do_emb,
        "pair_median": pair_median,
        "pair_cnt": pair_cnt,
        "pickup_median": pickup_median,
        "dropoff_median": dropoff_median,
        "pair_hour_median": pair_hour_med,
        "global_median": global_med,
        "zone_lat": zone_lat,
        "zone_lon": zone_lon,
        "n_cont": conts_tr.shape[1],
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nSaved bundle to {MODEL_PATH}")


if __name__ == "__main__":
    main()
