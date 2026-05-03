# 🚕 The ETA Challenge: Refined XGBoost Submission

*A production-grade ride-hailing ETA engine achieving **258.2s MAE** on NYC Yellow Taxi data.*

---

## Performance Summary

| Model | Architecture | Dev MAE | Improvement |
|-------|--------------|---------|-------------|
| **Baseline** | XGBoost (6 raw features) | ~351.0s | - |
| **Final (v6)** | **XGBoost (40 engineered features)** | **258.2s** | **+92.8s (26%)** |

---

## Core Innovations

Instead of over-complicating the model architecture, this submission focuses on high-signal feature engineering and data quality.

### 1. Spatial Intelligence
*   **Rotated Manhattan Distance**: The NYC street grid is rotated ~29° from true North. We rotate coordinates before calculating distance, which aligns the metric with actual driving paths far better than raw Euclidean or standard Manhattan distance.
*   **Borough & Airport Context**: Added borough-aware flags (JFK/LGA/EWR) to capture the unique traffic volatility of bridge/tunnel crossings.

### 2. High-Resolution Temporal Priors
*   **1-Hour Bayesian Medians**: We compute historical medians for every (Pickup, Dropoff) pair at 1-hour intervals. 
*   **Bayesian Smoothing**: For rare zone pairs with low sample sizes, we shrink the prediction toward the pickup-zone's median to prevent noise from outliers.
*   **Holiday Corridor Logic**: Replaced simple holiday flags with continuous "days-to-holiday" metrics to capture the unique traffic patterns during the Christmas-New Year winter corridor.

### 3. Data Hygiene & Model Tuning
*   **Target Clipping**: Removed extreme outliers (meter left running, long-haul errors) before computing aggregates to ensure priors represent "clean" trips.
*   **MAE-Optimized Training**: Switched the XGBoost objective to `reg:absoluteerror` to align the training process directly with the evaluation metric.

---

## What Didn't Work (Dead Ends)

1.  **Deep Learning (Entity Embeddings)**: While technically interesting (v4), the NN architecture required significantly more memory and didn't outperform the GBDT on tabular data.
2.  **Model Stacking**: Ensembling XGBoost with LightGBM and CatBoost (v5) added ~500MB to the image size and 2x inference latency for only a negligible (<1s) gain in MAE.
3.  **Raw Euclidean Distance**: Manhattan distance on a rotated grid gave an immediate 8s MAE drop over Euclidean.

---

## AI Tooling 

Used **Gemini 3-Flash & Claude 3.5 Sonnet** to:
- Generate vectorized NumPy implementations for geospatial rotations.
- Debug shared-memory deadlocks in high-memory environments.
- Automate distribution analysis for the holiday domain-shift.

---

## Reproduction

```bash
# 1. Setup & Data
python data/download_data.py
python extract_coords.py

# 2. Train (Uses GPU if available)
python baseline.py

# Optional: Fast iteration (1% sample)
# ETA_SAMPLE_FRAC=0.01 python baseline.py

# 3. Local Evaluation
python grade.py
```

## Deployment

The submission is fully self-contained in a Docker image:
- **Image Size**: ~2.1 GB (Python 3.11-slim + XGBoost)
- **Inference**: < 1ms per request on CPU
- **Portability**: Uses a versioned pickle bundle that manages all lookups and feature mappings automatically.

---
*Total Development Time: ~8 hours*
