"""Submission interface — this is what Gobblecube's grader imports.

The grader will call `predict` once per held-out request. The signature below
is fixed; everything else (model type, preprocessing, etc.) is yours to change.
"""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path

import numpy as np

_MODEL_PATH = Path(__file__).parent / "model.pkl"

with open(_MODEL_PATH, "rb") as _f:
    _RAW = pickle.load(_f)

# --- US Federal Holidays + key NYC dates for 2023-2024 ---
_HOLIDAYS = {
    (1, 1), (1, 2), (1, 15), (1, 16),
    (2, 19), (2, 20),
    (5, 29),
    (6, 19),
    (7, 4),
    (9, 4),
    (10, 9),
    (11, 10),
    (11, 23), (11, 24),
    (12, 25),
    (12, 31),
}

def _haversine(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(d))

def _bearing(lat1, lng1, lat2, lng2):
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))

# --- Bundle from `baseline.py` (version 6: Refined Single-Model XGBoost) -------
if isinstance(_RAW, dict) and _RAW.get("version") == 6:
    _MODEL = _RAW["xgb"]
    _PAIR_MED = _RAW["pair_median"]
    _PAIR_CNT = _RAW["pair_cnt"]
    _PICKUP_MED = _RAW["pickup_median"]
    _DROPOFF_MED = _RAW["dropoff_median"]
    _PAIR_HOUR_MED = _RAW["pair_hour_median"]
    _ZONE_LAT = _RAW["zone_lat"]
    _ZONE_LON = _RAW["zone_lon"]
    _GLOBAL_MED = _RAW["global_median"]
    _SMOOTHING_W = 50

    if hasattr(_MODEL, "get_booster"):
        _MODEL.get_booster().feature_names = None

    _BOR = {}
    try:
        from pathlib import Path as _P
        _zl = __import__("pandas").read_csv(_P(__file__).parent / "data" / "zone_lookup.csv")
        for _, _r in _zl.iterrows():
            _BOR[int(_r["LocationID"])] = {"Manhattan":0,"Brooklyn":1,"Queens":2,
                "Bronx":3,"Staten Island":4,"EWR":5}.get(str(_r["Borough"]).strip(), 6)
    except Exception: pass

    def predict(request: dict) -> float:
        ts = datetime.fromisoformat(request["requested_at"])
        pu, do = int(request["pickup_zone"]), int(request["dropoff_zone"])
        hour, minute, dow = ts.hour, ts.minute, ts.weekday()
        month, day = ts.month, ts.day
        pax = int(request["passenger_count"])
        doy = float(ts.timetuple().tm_yday)
        mod = float(hour*60 + minute)

        pm = float(_PAIR_MED[pu, do]); pc = int(_PAIR_CNT[pu, do])
        pup = float(_PICKUP_MED[pu]); dop = float(_DROPOFF_MED[do])
        has = pc > 0 and not np.isnan(pm)
        pair_prior = pm if has else pup
        pair_smoothed = (pc*pm + _SMOOTHING_W*pup)/(pc+_SMOOTHING_W) if has else pup
        hbin = hour
        phm = float(_PAIR_HOUR_MED[pu, do, hbin])
        pair_hour_prior = phm if not np.isnan(phm) else pair_prior

        pu_lat, pu_lon = float(_ZONE_LAT[pu]), float(_ZONE_LON[pu])
        do_lat, do_lon = float(_ZONE_LAT[do]), float(_ZONE_LON[do])
        h_dist = _haversine(pu_lat, pu_lon, do_lat, do_lon)
        
        # NYC rotated grid Manhattan distance
        theta = np.radians(29.0)
        c, s = np.cos(theta), np.sin(theta)
        r_lat1 = pu_lat * c - pu_lon * s
        r_lng1 = pu_lat * s + pu_lon * c
        r_lat2 = do_lat * c - do_lon * s
        r_lng2 = do_lat * s + do_lon * c
        m_dist = np.abs(r_lat2 - r_lat1) * 111.1 + np.abs(r_lng2 - r_lng1) * 111.1 * np.cos(np.radians((pu_lat + do_lat) / 2.0))
        
        bear = _bearing(pu_lat, pu_lon, do_lat, do_lon)

        pu_bor, do_bor = _BOR.get(pu, 6), _BOR.get(do, 6)
        is_hol = 1.0 if (month, day) in _HOLIDAYS else 0.0
        is_hol_p = 1.0 if (month==12 and day>=22) or (month==1 and day<=2) else 0.0
        days_xmas = abs(day-25) if month==12 else (day+6 if month==1 else 180)
        days_ny = (31-day) if month==12 else (day if month==1 else 180)
        is_mr = 1.0 if dow<5 and 7<=hour<=10 else 0.0
        is_er = 1.0 if dow<5 and 16<=hour<=19 else 0.0
        speed = h_dist / (pair_prior/3600.0 + 1e-6) if pair_prior > 0 else 0.0

        x = np.array([[
            pu, do, hour, dow, month, pax, 
            1.0 if dow>=5 else 0.0,
            np.sin(2*np.pi*hour/24.0), np.cos(2*np.pi*hour/24.0),
            np.sin(2*np.pi*dow/7.0), np.cos(2*np.pi*dow/7.0),
            pair_prior, pair_smoothed, 
            np.log1p(pc) if has else 0.0,
            pup, dop, pair_hour_prior,
            h_dist, m_dist, bear,
            is_hol, is_hol_p, doy, mod,
            is_mr, is_er,
            float(pu_bor), float(do_bor), 1.0 if pu_bor!=do_bor else 0.0,
            1.0 if pu in (1,132,138) or do in (1,132,138) else 0.0,
            float(days_xmas), float(days_ny),
            float(speed),
            15.0, 0.0, 10.0, 16.0, 0.0, 0.0, 0.0
        ]], dtype=np.float32)
        
        return float(_MODEL.predict(x)[0])

else:
    # Legacy / Baseline compatibility
    _MODEL = _RAW
    if hasattr(_MODEL, "get_booster"):
        _MODEL.get_booster().feature_names = None

    def predict(request: dict) -> float:
        ts = datetime.fromisoformat(request["requested_at"])
        x = np.array([[
            int(request["pickup_zone"]),
            int(request["dropoff_zone"]),
            ts.hour,
            ts.weekday(),
            ts.month,
            int(request["passenger_count"]),
        ]], dtype=np.int32)
        return float(_MODEL.predict(x)[0])
