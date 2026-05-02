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

# --- Bundle from `baseline-copy.py` (version 2) --------------------------------
if isinstance(_RAW, dict) and _RAW.get("version") == 2:
    _MODEL = _RAW["xgb"]
    _PAIR_MED = _RAW["pair_median"]
    _PAIR_CNT = _RAW["pair_cnt"]
    _PICKUP_MED = _RAW["pickup_median"]
    _ZONE_LAT = _RAW.get("zone_lat")
    _ZONE_LON = _RAW.get("zone_lon")

    def haversine_scalar(lat1, lng1, lat2, lng2):
        lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
        lat = lat2 - lat1
        lng = lng2 - lng1
        d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
        return 2 * 6371 * np.arcsin(np.sqrt(d))

    def manhattan_scalar(lat1, lng1, lat2, lng2):
        return haversine_scalar(lat1, lng1, lat1, lng2) + haversine_scalar(lat1, lng1, lat2, lng1)

    def bearing_scalar(lat1, lng1, lat2, lng2):
        lng_delta_rad = np.radians(lng2 - lng1)
        lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
        y = np.sin(lng_delta_rad) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
        return np.degrees(np.arctan2(y, x))

    if hasattr(_MODEL, "get_booster"):
        _MODEL.get_booster().feature_names = None

    def predict(request: dict) -> float:
        ts = datetime.fromisoformat(request["requested_at"])
        pu = int(request["pickup_zone"])
        do = int(request["dropoff_zone"])
        hour = ts.hour
        dow = ts.weekday()
        month = ts.month
        pax = int(request["passenger_count"])

        two_pi_h = 2.0 * np.pi * hour / 24.0
        pm = float(_PAIR_MED[pu, do])
        pc = int(_PAIR_CNT[pu, do])
        pup = float(_PICKUP_MED[pu])
        if pc > 0 and not np.isnan(pm):
            pair_prior = pm
            log1p_pc = float(np.log1p(pc))
        else:
            pair_prior = pup
            log1p_pc = 0.0

        if _ZONE_LAT is not None and _ZONE_LON is not None:
            pu_lat, pu_lon = float(_ZONE_LAT[pu]), float(_ZONE_LON[pu])
            do_lat, do_lon = float(_ZONE_LAT[do]), float(_ZONE_LON[do])
            h_dist = haversine_scalar(pu_lat, pu_lon, do_lat, do_lon)
            m_dist = manhattan_scalar(pu_lat, pu_lon, do_lat, do_lon)
            bearing = bearing_scalar(pu_lat, pu_lon, do_lat, do_lon)
        else:
            h_dist, m_dist, bearing = 0.0, 0.0, 0.0

        x = np.array(
            [[
                pu,
                do,
                hour,
                dow,
                month,
                pax,
                1 if dow >= 5 else 0,
                np.sin(two_pi_h),
                np.cos(two_pi_h),
                pair_prior,
                log1p_pc,
                pup,
                h_dist,
                m_dist,
                bearing,
            ]],
            dtype=np.float32,
        )
        return float(_MODEL.predict(x)[0])

else:
    # Legacy: plain XGBoost from `baseline.py` (6 integer features)
    _MODEL = _RAW
    if hasattr(_MODEL, "get_booster"):
        _MODEL.get_booster().feature_names = None

    def predict(request: dict) -> float:
        ts = datetime.fromisoformat(request["requested_at"])
        x = np.array(
            [[
                int(request["pickup_zone"]),
                int(request["dropoff_zone"]),
                ts.hour,
                ts.weekday(),
                ts.month,
                int(request["passenger_count"]),
            ]],
            dtype=np.int32,
        )
        return float(_MODEL.predict(x)[0])
