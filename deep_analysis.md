# Deep Analysis: Breaking the 250s MAE Barrier

## Current State
- **Current MAE**: ~250s (on dev set, 50k sample)
- **Architecture**: Entity Embedding NN + XGBoost Ensemble (v4)
- **Train set**: 36.7M rows (Jan 1 – Dec 17, 2023)
- **Dev set**: 1.23M rows (Dec 18 – Dec 31, 2023 — **ENTIRE holiday season**)

## 🔴 Critical Finding: Domain Shift Problem

The dev set is **exclusively** the last 2 weeks of December (Christmas/New Year). The training set has **ZERO** rows from Dec 18+. This is the single biggest source of error:

| Factor | Impact |
|--------|--------|
| Holiday traffic patterns differ dramatically | HIGH |
| Train has only 5.8% December data total | HIGH |
| Christmas Eve/Day → 50% fewer trips, very different speeds | HIGH |
| New Year's Eve has extreme late-night activity | MEDIUM |

## 📊 Error Source Breakdown

### 1. Within-Pair Variability (LARGEST source)
- **Mean pair std: 714.8s** — even knowing the exact OD pair, there's massive variance
- This is the theoretical floor for any pair-median approach
- Must capture **what causes within-pair variance**: time-of-day, weather, holidays, congestion

### 2. Airport/Special Zone Trips (HIGH variance)
| Zone | Trips | Avg Duration | Std |
|------|-------|-------------|-----|
| 132 (JFK) | 2.25M | 2620s | 1173s |
| 138 (LGA) | 1.71M | 1822s | 809s |
| 1 (EWR) | 110K | 2538s | 972s |
| 265 (Unknown) | 155K | 2480s | 1452s |

Airport trips have std of 800-1450s — needs special handling.

### 3. Outlier Contamination
- 1.63% of trips exceed 3×IQR (>3586s)
- 8,592 trips exceed 2 hours (7200s)
- These skew pair medians and training signal

### 4. Rare Pair Problem
- 51.9% of zone pairs have <10 trips
- 37.4% have <5 trips
- Current smoothing helps but is too simple

---

## 🚀 Improvement Strategies (Ranked by Expected Impact)

### Strategy 1: **Multi-GBDT Stacking** (Expected: -15 to -30s MAE)
**Why**: Different GBDT implementations capture different interaction patterns.
- Add **LightGBM** (leaf-wise, captures different splits than XGBoost's level-wise)
- Add **CatBoost** (native ordered target encoding prevents leakage, handles categoricals better)
- Stack via Ridge regression meta-learner on out-of-fold predictions

### Strategy 2: **Huber Loss Instead of Pure MAE** (Expected: -5 to -15s MAE)
**Why**: Pure MAE (L1) loss has non-smooth gradient at zero, causing training instability.
- Use Huber loss (δ=500s) for NN: smooth near zero, linear for outliers
- Use `reg:pseudohubererror` for XGBoost
- Better convergence, more robust to outliers

### Strategy 3: **Borough-Level Features** (Expected: -5 to -10s MAE)
**Why**: Cross-borough trips (Manhattan→Brooklyn, Manhattan→JFK) have fundamentally different dynamics.
- `is_cross_borough`: binary flag
- `pickup_borough`, `dropoff_borough`: categorical embeddings
- `is_airport_trip`: special flag for zones 1, 132, 138
- Borough-pair median duration

### Strategy 4: **Enhanced Temporal Features for Holiday Period** (Expected: -5 to -10s MAE)
**Why**: Dev set is 100% holiday period but model doesn't capture fine-grained holiday effects.
- `days_to_christmas`: continuous distance to Dec 25
- `days_to_newyear`: continuous distance to Jan 1
- `is_christmas_week`: Dec 22-26
- `is_newyear_week`: Dec 27-Jan 2
- **Week-of-year cyclical encoding**
- Train on last 2 weeks of preceding months to capture late-month patterns

### Strategy 5: **Target Transformation** (Expected: -5 to -10s MAE)
**Why**: Duration has heavy right tail. Log transform can help the model.
- Train on `log(duration)`, predict in original scale
- Or use `sqrt(duration)` for less aggressive compression
- Compensate bias with `exp(pred + 0.5 * var)` correction

### Strategy 6: **Outlier Clipping/Winsorization** (Expected: -3 to -8s MAE)
**Why**: 0.27% trips >5000s contaminate aggregate stats.
- Clip training durations at 99th percentile (3990s) or apply Winsorization
- Rebuild aggregates on cleaned data
- Keep clipping only for aggregate computation, not training target

### Strategy 7: **Finer-Grained Temporal Aggregates** (Expected: -3 to -5s MAE)
**Why**: 3-hour bins are too coarse; rush hour effects change within 30-60 minutes.
- Add DOW-conditioned pair priors: `(PU, DO, dow)` median
- Add weekend vs weekday pair priors
- Add monthly pair priors to capture seasonal drift

### Strategy 8: **Speed-Based Features** (Expected: -2 to -5s MAE)
**Why**: Distance alone doesn't capture speed variation.
- `estimated_speed = haversine_dist / pair_prior_sec` — captures congestion
- `dist_per_pair_duration_ratio`: normalizes by expected travel time
- Historical average speed per zone per hour

### Strategy 9: **Residual Modeling** (Expected: -3 to -8s MAE)
**Why**: Train a second model to correct the first model's systematic errors.
- Train base model → compute residuals on train (via OOF predictions)
- Train residual model on `(features, residuals)` 
- Final prediction = base + residual correction

### Strategy 10: **Weather Interaction Features** (Expected: -2 to -5s MAE)
**Why**: Rain impacts airport trips more than local Manhattan trips.
- `rain_x_airport`: interaction of is_rain × is_airport
- `rain_x_distance`: rain impact scales with distance
- `low_vis_x_rush`: poor visibility during rush hour
- `wind_x_bridge`: wind impact on bridge-crossing routes (cross-borough)

---

## 📋 Implementation Priority

| Priority | Strategy | Expected Gain | Effort |
|----------|----------|---------------|--------|
| 1 | Multi-GBDT Stacking (LightGBM + CatBoost) | -15 to -30s | Medium |
| 2 | Huber Loss + Target Transform | -10 to -20s | Low |
| 3 | Borough + Airport Features | -5 to -10s | Low |
| 4 | Enhanced Holiday Features | -5 to -10s | Low |
| 5 | Outlier Clipping for Aggregates | -3 to -8s | Low |
| 6 | Finer Temporal Aggregates (DOW-cond) | -3 to -5s | Medium |
| 7 | Speed-Based Features | -2 to -5s | Low |
| 8 | Residual Modeling | -3 to -8s | Medium |
| 9 | Weather Interactions | -2 to -5s | Low |

**Realistic combined target: 200-220s MAE** (from current ~250s)

---

## ⚠️ What Won't Help Much

1. **More NN epochs / bigger network**: Already diminishing returns at 512→256→128→64
2. **Higher embedding dims**: 16-dim is already generous for 265 zones
3. **More training data**: Already using 36.7M rows; the bottleneck is feature quality
4. **Raw passenger_count**: Very weak predictor (most are 1)
