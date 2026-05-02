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

def _manhattan(lat1, lng1, lat2, lng2):
    return _haversine(lat1, lng1, lat1, lng2) + _haversine(lat1, lng1, lat2, lng1)

def _bearing(lat1, lng1, lat2, lng2):
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))


# --- Bundle from `baseline-copy.py` (version 3) --------------------------------
if isinstance(_RAW, dict) and _RAW.get("version") == 3:
    _MODEL = _RAW["xgb"]
    _PAIR_MED = _RAW["pair_median"]
    _PAIR_CNT = _RAW["pair_cnt"]
    _PICKUP_MED = _RAW["pickup_median"]
    _DROPOFF_MED = _RAW["dropoff_median"]
    _PAIR_HOUR_MED = _RAW["pair_hour_median"]
    _GLOBAL_MED = _RAW["global_median"]
    _ZONE_LAT = _RAW.get("zone_lat")
    _ZONE_LON = _RAW.get("zone_lon")
    _SMOOTHING_W = 50

    if hasattr(_MODEL, "get_booster"):
        _MODEL.get_booster().feature_names = None

    def predict(request: dict) -> float:
        ts = datetime.fromisoformat(request["requested_at"])
        pu = int(request["pickup_zone"])
        do = int(request["dropoff_zone"])
        hour = ts.hour
        minute = ts.minute
        dow = ts.weekday()
        month = ts.month
        day = ts.day
        pax = int(request["passenger_count"])

        two_pi_h = 2.0 * np.pi * hour / 24.0
        two_pi_d = 2.0 * np.pi * dow / 7.0

        # Pair priors
        pm = float(_PAIR_MED[pu, do])
        pc = int(_PAIR_CNT[pu, do])
        pup = float(_PICKUP_MED[pu])
        dop = float(_DROPOFF_MED[do])

        if pc > 0 and not np.isnan(pm):
            pair_prior = pm
            log1p_pc = float(np.log1p(pc))
            pair_smoothed = (pc * pm + _SMOOTHING_W * pup) / (pc + _SMOOTHING_W)
        else:
            pair_prior = pup
            log1p_pc = 0.0
            pair_smoothed = pup

        # Time-conditioned pair prior
        hour_bin = hour // 3
        phm = float(_PAIR_HOUR_MED[pu, do, hour_bin])
        if np.isnan(phm):
            pair_hour_prior = pair_prior
        else:
            pair_hour_prior = phm

        # Spatial
        if _ZONE_LAT is not None and _ZONE_LON is not None:
            pu_lat, pu_lon = float(_ZONE_LAT[pu]), float(_ZONE_LON[pu])
            do_lat, do_lon = float(_ZONE_LAT[do]), float(_ZONE_LON[do])
            h_dist = _haversine(pu_lat, pu_lon, do_lat, do_lon)
            m_dist = _manhattan(pu_lat, pu_lon, do_lat, do_lon)
            bear = _bearing(pu_lat, pu_lon, do_lat, do_lon)
        else:
            h_dist, m_dist, bear = 0.0, 0.0, 0.0

        # Holiday features
        is_holiday = 1 if (month, day) in _HOLIDAYS else 0
        is_hol_period = 1 if ((month == 12 and day >= 22) or (month == 1 and day <= 2)) else 0

        # Rush hour
        is_weekday = dow < 5
        is_morning_rush = 1 if (is_weekday and 7 <= hour <= 10) else 0
        is_evening_rush = 1 if (is_weekday and 16 <= hour <= 19) else 0

        # Time features
        day_of_year = ts.timetuple().tm_yday
        minute_of_day = hour * 60 + minute

        # Weather defaults (no weather at inference — use neutral values)
        temp_c = 15.0
        precip_mm = 0.0
        wind_speed_kmh = 10.0
        visibility_km = 16.0
        is_raining = 0

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
                np.sin(two_pi_d),
                np.cos(two_pi_d),
                pair_prior,
                pair_smoothed,
                log1p_pc,
                pup,
                dop,
                pair_hour_prior,
                h_dist,
                m_dist,
                bear,
                is_holiday,
                is_hol_period,
                day_of_year,
                minute_of_day,
                is_morning_rush,
                is_evening_rush,
                temp_c,
                precip_mm,
                wind_speed_kmh,
                visibility_km,
                is_raining,
            ]],
            dtype=np.float32,
        )
        return float(_MODEL.predict(x)[0])

# --- Bundle from `baseline-copy.py` (version 2) --------------------------------
elif isinstance(_RAW, dict) and _RAW.get("version") == 2:
    _MODEL = _RAW["xgb"]
    _PAIR_MED = _RAW["pair_median"]
    _PAIR_CNT = _RAW["pair_cnt"]
    _PICKUP_MED = _RAW["pickup_median"]
    _ZONE_LAT = _RAW.get("zone_lat")
    _ZONE_LON = _RAW.get("zone_lon")

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
            h_dist = _haversine(pu_lat, pu_lon, do_lat, do_lon)
            m_dist = _manhattan(pu_lat, pu_lon, do_lat, do_lon)
            bearing = _bearing(pu_lat, pu_lon, do_lat, do_lon)
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
