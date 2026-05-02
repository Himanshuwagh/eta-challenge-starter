"""Submission interface — this is what Gobblecube's grader imports.

The grader will call `predict` once per held-out request. The signature below
is fixed; everything else (model type, preprocessing, etc.) is yours to change.
"""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path

import numpy as np

# Custom unpickler: models saved from __main__ (train_deep.py / Colab)
# store classes as __main__.ETANet. Redirect to train_deep module.
class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "ETANet":
            try:
                import train_deep
                return getattr(train_deep, name)
            except (ImportError, AttributeError):
                pass
        return super().find_class(module, name)

_MODEL_PATH = Path(__file__).parent / "model.pkl"

with open(_MODEL_PATH, "rb") as _f:
    _RAW = _Unpickler(_f).load()

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

# --- Bundle from `baseline-copy.py` (version 6: Refined Single-Model XGBoost) -
if isinstance(_RAW, dict) and _RAW.get("version") == 6:
    _MODEL = _RAW["xgb"]
    _PAIR_MED = _RAW["pair_median"]
    _PAIR_CNT = _RAW["pair_cnt"]
    _PICKUP_MED = _RAW["pickup_median"]
    _DROPOFF_MED = _RAW["dropoff_median"]
    _PAIR_HOUR_MED = _RAW["pair_hour_median"]
    _ZONE_LAT = _RAW["zone_lat"]
    _ZONE_LON = _RAW["zone_lon"]
    _FEAT_ORDER = _RAW["feature_order"]
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
        hbin = hour  # Changed to 1-hour bin (v6)
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

        import pandas as _pd
        x = _pd.DataFrame([{
            "pickup_zone": pu, "dropoff_zone": do, "hour": hour, "dow": dow, "month": month,
            "passenger_count": pax, "is_weekend": 1.0 if dow>=5 else 0.0,
            "hour_sin": np.sin(2*np.pi*hour/24.0), "hour_cos": np.cos(2*np.pi*hour/24.0),
            "dow_sin": np.sin(2*np.pi*dow/7.0), "dow_cos": np.cos(2*np.pi*dow/7.0),
            "pair_prior_sec": pair_prior, "pair_prior_smoothed": pair_smoothed,
            "log1p_pair_count": np.log1p(pc) if has else 0.0,
            "pickup_prior_sec": pup, "dropoff_prior_sec": dop,
            "pair_hour_prior_sec": pair_hour_prior,
            "haversine_dist": h_dist, "manhattan_dist": m_dist, "bearing": bear,
            "is_holiday": is_hol, "is_holiday_period": is_hol_p,
            "day_of_year": doy, "minute_of_day": mod,
            "is_morning_rush": is_mr, "is_evening_rush": is_er,
            "pu_bor": float(pu_bor), "do_bor": float(do_bor), "is_cross_borough": 1.0 if pu_bor!=do_bor else 0.0,
            "is_airport": 1.0 if pu in (1,132,138) or do in (1,132,138) else 0.0,
            "days_to_christmas": float(days_xmas), "days_to_newyear": float(days_ny),
            "speed_proxy": float(speed),
            "temp_c": 15.0, "precip_mm": 0.0, "wind_speed_kmh": 10.0, "visibility_km": 16.0, "is_raining": 0.0,
            "rain_x_airport": 0.0, "rain_x_dist": 0.0,
        }])
        return float(_MODEL.predict(x[_FEAT_ORDER])[0])

# --- Bundle from `train_v5_stack.py` (version 5: XGB+LGB+CAT Stack) ----------
elif isinstance(_RAW, dict) and _RAW.get("version") == 5:
    _XGB = _RAW["xgb_model"]
    _LGB = _RAW["lgb_model"]
    _CAT = _RAW["cat_model"]
    _W = _RAW["ensemble_weights"]  # (w_xgb, w_lgb, w_cat)
    _PAIR_MED = _RAW["pair_median"]
    _PAIR_CNT = _RAW["pair_cnt"]
    _PICKUP_MED = _RAW["pickup_median"]
    _DROPOFF_MED = _RAW["dropoff_median"]
    _PAIR_HOUR_MED = _RAW["pair_hour_median"]
    _PAIR_DOW_MED = _RAW["pair_dow_median"]
    _ZONE_LAT = _RAW["zone_lat"]
    _ZONE_LON = _RAW["zone_lon"]
    _SMOOTHING_W = 50
    _FEAT_NAMES = _RAW.get("feature_names", [])
    _CAT_COLS = [_FEAT_NAMES[i] for i in _RAW.get("cat_features", [])] if _FEAT_NAMES else []

    if hasattr(_XGB, "get_booster"):
        _XGB.get_booster().feature_names = None

    # Borough lookup
    _BOR = {}
    try:
        from pathlib import Path as _P
        _zl = __import__("pandas").read_csv(_P(__file__).parent / "data" / "zone_lookup.csv")
        for _, _r in _zl.iterrows():
            _BOR[int(_r["LocationID"])] = {"Manhattan":0,"Brooklyn":1,"Queens":2,
                "Bronx":3,"Staten Island":4,"EWR":5}.get(str(_r["Borough"]).strip(), 6)
    except Exception:
        pass

    def predict(request: dict) -> float:
        ts = datetime.fromisoformat(request["requested_at"])
        pu = int(request["pickup_zone"])
        do = int(request["dropoff_zone"])
        hour, minute, dow = ts.hour, ts.minute, ts.weekday()
        month, day, pax = ts.month, ts.day, int(request["passenger_count"])
        doy = float(ts.timetuple().tm_yday)
        woy = ts.isocalendar()[1]

        pm = float(_PAIR_MED[pu, do]); pc = int(_PAIR_CNT[pu, do])
        pup = float(_PICKUP_MED[pu]); dop = float(_DROPOFF_MED[do])
        has = pc > 0 and not np.isnan(pm)
        pair_prior = pm if has else pup
        pair_smoothed = (pc*pm + _SMOOTHING_W*pup)/(pc+_SMOOTHING_W) if has else pup
        log1p_pc = float(np.log1p(pc)) if has else 0.0
        hbin = hour // 3
        phm = float(_PAIR_HOUR_MED[pu, do, hbin])
        pair_hour_prior = phm if not np.isnan(phm) else pair_prior
        pdm = float(_PAIR_DOW_MED[pu, do, dow])
        pair_dow_prior = pdm if not np.isnan(pdm) else pair_prior

        pu_lat, pu_lon = float(_ZONE_LAT[pu]), float(_ZONE_LON[pu])
        do_lat, do_lon = float(_ZONE_LAT[do]), float(_ZONE_LON[do])
        h_dist = _haversine(pu_lat, pu_lon, do_lat, do_lon)
        m_dist = _manhattan(pu_lat, pu_lon, do_lat, do_lon)
        bear = _bearing(pu_lat, pu_lon, do_lat, do_lon)

        pu_bor = _BOR.get(pu, 6); do_bor = _BOR.get(do, 6)
        is_cross = 1.0 if pu_bor != do_bor else 0.0
        is_airport = 1.0 if pu in (1,132,138) or do in (1,132,138) else 0.0
        is_hol = 1.0 if (month, day) in _HOLIDAYS else 0.0
        is_xmas = 1.0 if (month==12 and day>=22) or (month==1 and day<=2) else 0.0
        days_xmas = abs(day-25) if month==12 else (day+6 if month==1 else 180)
        days_ny = (31-day) if month==12 else (day if month==1 else 180)
        is_wd = dow < 5
        is_mr = 1.0 if is_wd and 7<=hour<=10 else 0.0
        is_er = 1.0 if is_wd and 16<=hour<=19 else 0.0
        min_of_day = float(hour*60+minute)
        two_pi_h = 2*np.pi*hour/24.0
        two_pi_d = 2*np.pi*dow/7.0
        two_pi_w = 2*np.pi*woy/52.0
        speed_proxy = h_dist / (pair_prior/3600.0 + 1e-6) if pair_prior > 0 else 0.0

        x = np.array([[
            pu, do, hour, dow, month, float(pax), 1.0 if dow>=5 else 0.0,
            np.sin(two_pi_h), np.cos(two_pi_h),
            np.sin(two_pi_d), np.cos(two_pi_d),
            np.sin(two_pi_w), np.cos(two_pi_w),
            pair_prior, pair_smoothed, log1p_pc, pup, dop,
            pair_hour_prior, pair_dow_prior,
            h_dist, m_dist, bear,
            is_hol, is_xmas, float(days_xmas), float(days_ny),
            doy, min_of_day, is_mr, is_er,
            float(pu_bor), float(do_bor), is_cross, is_airport,
            float(speed_proxy),
            15.0, 0.0, 60.0, 10.0, 16.0, 0.0,  # weather defaults
            0.0, 0.0,  # rain interactions
            float(woy),
        ]], dtype=np.float32)

        p1 = float(_XGB.predict(x)[0])
        p2 = float(_LGB.predict(x)[0])
        # CatBoost needs DataFrame with int cat columns
        import pandas as _pd
        x_df = _pd.DataFrame(x, columns=_FEAT_NAMES)
        for _c in _CAT_COLS:
            x_df[_c] = x_df[_c].astype(int)
        p3 = float(_CAT.predict(x_df)[0])
        return float(_W[0]*p1 + _W[1]*p2 + _W[2]*p3)

# --- Bundle from `train_deep.py` (version 4: NN + XGBoost Ensemble) ----------
elif isinstance(_RAW, dict) and _RAW.get("version") == 4:

    import torch

    _NN_MODEL = _RAW["nn_model"]
    _NN_MODEL.eval()
    _XGB_MODEL = _RAW["xgb_model"]
    _W_XGB = _RAW["ensemble_weight_xgb"]
    _PU_EMB = _RAW["pu_emb"]   # (266, 16)
    _DO_EMB = _RAW["do_emb"]   # (266, 16)
    _PAIR_MED = _RAW["pair_median"]
    _PAIR_CNT = _RAW["pair_cnt"]
    _PICKUP_MED = _RAW["pickup_median"]
    _DROPOFF_MED = _RAW["dropoff_median"]
    _PAIR_HOUR_MED = _RAW["pair_hour_median"]
    _ZONE_LAT = _RAW["zone_lat"]
    _ZONE_LON = _RAW["zone_lon"]
    _SMOOTHING_W = 50

    if hasattr(_XGB_MODEL, "get_booster"):
        _XGB_MODEL.get_booster().feature_names = None

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

        # Pair priors
        pm = float(_PAIR_MED[pu, do])
        pc = int(_PAIR_CNT[pu, do])
        pup = float(_PICKUP_MED[pu])
        dop = float(_DROPOFF_MED[do])
        has_pair = pc > 0 and not np.isnan(pm)
        pair_prior = pm if has_pair else pup
        pair_smoothed = (pc * pm + _SMOOTHING_W * pup) / (pc + _SMOOTHING_W) if has_pair else pup
        log1p_pc = float(np.log1p(pc)) if has_pair else 0.0
        hbin = hour // 3
        phm = float(_PAIR_HOUR_MED[pu, do, hbin])
        pair_hour_prior = phm if not np.isnan(phm) else pair_prior

        # Spatial
        pu_lat, pu_lon = float(_ZONE_LAT[pu]), float(_ZONE_LON[pu])
        do_lat, do_lon = float(_ZONE_LAT[do]), float(_ZONE_LON[do])
        h_dist = _haversine(pu_lat, pu_lon, do_lat, do_lon)
        m_dist = _manhattan(pu_lat, pu_lon, do_lat, do_lon)
        bear = _bearing(pu_lat, pu_lon, do_lat, do_lon)

        # Holiday
        is_hol = 1.0 if (month, day) in _HOLIDAYS else 0.0
        is_xmas = 1.0 if ((month == 12 and day >= 22) or (month == 1 and day <= 2)) else 0.0
        is_wd = dow < 5
        is_mr = 1.0 if is_wd and 7 <= hour <= 10 else 0.0
        is_er = 1.0 if is_wd and 16 <= hour <= 19 else 0.0
        doy = float(ts.timetuple().tm_yday)
        min_of_day = float(hour * 60 + minute)
        two_pi_h = 2.0 * np.pi * hour / 24.0
        two_pi_d = 2.0 * np.pi * dow / 7.0

        # Weather defaults (neutral values at inference)
        temp_c, precip, wind, vis, is_rain = 15.0, 0.0, 10.0, 16.0, 0.0

        conts = np.array([[
            pair_prior, pair_smoothed, log1p_pc,
            pup, dop, pair_hour_prior,
            h_dist, m_dist, bear,
            is_hol, is_xmas, is_mr, is_er,
            doy, min_of_day,
            np.sin(two_pi_h), np.cos(two_pi_h),
            np.sin(two_pi_d), np.cos(two_pi_d),
            temp_c, precip, wind, vis, is_rain,
            float(pax), 1.0 if dow >= 5 else 0.0, float(month),
        ]], dtype=np.float32)

        cats = np.array([[pu, do, hour, dow]], dtype=np.int32)

        # NN prediction
        with torch.no_grad():
            cats_t = torch.tensor(cats, dtype=torch.long)
            conts_t = torch.tensor(conts)
            nn_pred = float(_NN_MODEL(cats_t, conts_t).item())

        # XGBoost prediction (conts + zone embeddings)
        pu_vec = _PU_EMB[pu]  # (16,)
        do_vec = _DO_EMB[do]  # (16,)
        x_xgb = np.hstack([conts[0], pu_vec, do_vec]).reshape(1, -1).astype(np.float64)
        xgb_pred = float(_XGB_MODEL.predict(x_xgb)[0])

        # Ensemble
        return float(_W_XGB * xgb_pred + (1.0 - _W_XGB) * nn_pred)

# --- Bundle from `baseline-copy.py` (version 3) --------------------------------
elif isinstance(_RAW, dict) and _RAW.get("version") == 3:

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
