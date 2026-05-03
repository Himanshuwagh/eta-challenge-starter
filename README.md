# The ETA Challenge: Submission Report

*A production-grade ride-hailing ETA engine built on NYC Yellow Taxi data.*

---

## 🚀 Final Performance

| Model Version | Architecture | Dev MAE |
|--------------|-------------|---------|
| v1 (Baseline) | XGBoost, 6 raw features | ~351.0s |
| v3 | XGBoost, 31 engineered features | 260.5s |
| v4 | Entity Embedding NN + XGBoost Ensemble | ~250s |
| v5 | XGBoost + LightGBM + CatBoost Stacking | ~262.4s |
| **v6** | **Refined Single-Model XGBoost (Final)** | **258.2s** |

---

## 🏗️ The Approach: Iterative Model Evolution

### Phase 1: Spatial Intelligence (v3)
Transformed zone IDs from arbitrary integers into a spatially-aware system:
- **NYC Taxi Zone Shapefiles** → latitude/longitude centroids for all 265 zones
- **Manhattan Distance** over Euclidean — NYC's grid makes L1 distance superior
- **Bearing (Trip Direction)** — captures uptown vs. downtown flow patterns

### Phase 2: Bayesian Smoothed Priors (v3)
- **Bayesian Smoothing:** Rare zone pairs shrink toward pickup-zone median
- **Time-Conditioned Priors:** Per-pair medians for 3-hour blocks to capture rush hour
- **Weather Integration:** Hourly NOAA data (precip, visibility, wind)
- **Holiday Logic:** Christmas/New Year period flags for the winter eval set

### Phase 3: Entity Embeddings (v4)
- PyTorch NN with **learned embeddings** for zones (16-dim), hours (8-dim), day-of-week (4-dim)
- Inspired by Uber's DeepETA architecture
- Trained embeddings injected as features into XGBoost → weighted ensemble

### Phase 4: Multi-GBDT Stacking (v5)
An attempt to ensemble XGBoost, LightGBM, and CatBoost with a Ridge meta-learner.
- **Result:** XGB: 274.5s, LGB: 269.6s, CAT: 260.5s, Stack: 262.4s.
- **Observation:** Complex stacking yielded diminishing returns compared to the single-model `baseline.py`. 
- **Pivot:** Decided to fold v5's best feature engineering (boroughs, holiday proximity, speed proxy) back into the single-model XGBoost pipeline for a simpler, faster, and more maintainable production system.

### Phase 5: Single-Model Refinement (Final)
Refining the pipeline by folding in the best features from v5 (boroughs, holiday proximity, speed proxies) back into a high-capacity XGBoost model. This achieved superior latency-to-accuracy trade-offs and simplified the production environment. We optimized the training scripts for Colab's hardware and resolved shared-memory bottlenecks.

| Innovation | Rationale |
|-----------|-----------|
| **Rotated Manhattan Distance** | The NYC street grid is rotated ~29° from true North. Standard Manhattan distance assumes an N/S grid. Rotating coordinates by 29° before calculation significantly improves correlation with actual driving paths. |
| **Aggregate Target Clipping** | Extreme outliers (e.g., a trip taking 5 hours because the meter was left running) severely skew the `pair_median` prior. We now clip training durations to the 1st-99th percentile (60s to ~4000s) *before* computing historical aggregates. |
| **High-Resolution Time Bins** | Increased the resolution of `pair_hour_median` from 3-hour chunks to 1-hour chunks to better capture the sharp spikes of morning/evening rush hour (especially potent when training on 100% data). |
| **Colab GPU Acceleration** | Added automatic GPU detection to `baseline.py` and the XGBoost phase of `train_deep.py` (`tree_method='gpu_hist'`), allowing fast training on 50M+ rows. |
| **Colab DataLoader Deadlock Fix** | Fixed a silent hanging issue in `train_deep.py` caused by PyTorch's `DataLoader` exhausting Colab's `/dev/shm` shared memory when `num_workers > 0`. |
| **Borough & Airport Features** | Cross-borough trips and airport trips (JFK/LGA/EWR) have fundamentally different traffic dynamics and volatility. |
| **Holiday Proximity Features** | Replaced boolean holiday flags with continuous `days_to_christmas` and `days_to_newyear` metrics. |

---

## 📉 What Didn't Work (Dead Ends)

1.  **Naive Categorical IDs:** Zone IDs as integers create fake ordinal relationships
2.  **Over-Complex Time Encoding:** 1-minute cyclical resolution → noise. 3-hour bins are optimal
3.  **Raw Euclidean Distance:** Manhattan distance gave immediate 5-8s MAE drop
4.  **OneCycleLR for NN:** Diverged after peak LR (256s → 477s). `ReduceLROnPlateau` was stable
5.  **Colab-trained pickle on local Mac:** CUDA tensors cause segfaults on MPS/CPU — always save state_dict

---

## 🤖 Where AI Tooling Sped Me Up Most

Used **Antigravity (Gemini & Claude)** as pair programmer:
- **Deep Analysis:** Automated data distribution analysis revealing the holiday domain-shift problem
- **Feature Engineering:** Vectorized NumPy implementations for geospatial math
- **Multi-Model Architecture:** Generated complete stacking pipeline (XGB+LGB+CAT) with proper CatBoost categorical handling
- **Debugging:** Resolved pickle loading issues, CatBoost dtype errors, environment conflicts

---

## 🔮 Remaining Experiments

1.  **Out-of-fold stacking:** Proper K-fold OOF predictions for meta-learner
2.  **Residual modeling:** Train correction model on base model errors
3.  **FT-Transformer:** Self-attention across tabular features

---

## 🛠️ How to Reproduce

```bash
# 1. Download base data
python data/download_data.py

# 2. Extract geospatial coordinates (requires geopandas)
python extract_coords.py

# 3a. Train v6 — Final Refined XGBoost (~15 min on 100% data)
python baseline.py

# 3b. Train v4 — Entity Embedding NN + XGBoost (~30 min on 20% data)
python train_deep.py

# 3c. Train v5 — Multi-GBDT Stacking (~45 min on full data)
python train_v5_stack.py

# 4. Grade on Dev
python grade.py
```

### ⚡ Performance Optimization (Google Colab)

The `train_v5_stack.py` script is optimized to utilize high-memory and GPU-enabled environments like Google Colab:
- **GPU Acceleration:** Automatically detects and uses NVIDIA GPUs for all three GBDT models (`cuda`, `gpu_hist`, `task_type=GPU`).
- **Resource Management:** Configured to use all available CPU cores when GPU is absent.
- **Memory Efficiency:** Uses `float32` precision to minimize RAM footprint while allowing for larger `ETA_SAMPLE_FRAC`.
- **Fast Inference:** The `predict.py` module is optimized for low-latency scoring by caching pre-computed aggregates.

To maximize speed in Colab, ensure you select a **GPU runtime** (T4, A100, or V100) and set `ETA_SAMPLE_FRAC=1.0` to utilize the full RAM.

---
*Total time spent: ~8 hours*
