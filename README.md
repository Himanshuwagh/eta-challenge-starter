# The ETA Challenge: Submission Report

*A production-grade ride-hailing ETA engine built on NYC Yellow Taxi data.*

---

## 🚀 Final Performance

| Model Version | Architecture | Dev MAE |
|--------------|-------------|---------|
| v1 (Baseline) | XGBoost, 6 raw features | ~351.0s |
| v3 | XGBoost, 31 engineered features | 260.5s |
| v4 | Entity Embedding NN + XGBoost Ensemble | ~250s |
| **v5** | **XGBoost + LightGBM + CatBoost Stacking** | **Training...** |

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

### Phase 4: Multi-GBDT Stacking (v5) ⬅️ Current
Deep analysis revealed key bottlenecks:
- **Dev set is 100% holiday period** (Dec 18-31) with zero matching training dates
- **Within-pair variability** averaging 714.8s std — the theoretical floor for lookup approaches
- **Airport zones** (JFK, LGA, EWR) have 800-1450s std

**v5 Strategy — 3-model stacking with enhanced features:**

| Innovation | Rationale |
|-----------|-----------|
| **XGBoost + LightGBM + CatBoost ensemble** | Different splitting strategies capture complementary patterns |
| **Huber loss (δ=500)** | Robust to outliers, smoother than MAE for training |
| **Borough features** | Cross-borough trips have fundamentally different dynamics |
| **DOW-conditioned pair priors** | Day-specific historical medians per route |
| **Holiday proximity features** | `days_to_christmas`, `days_to_newyear` — continuous distance |
| **Speed proxy** | `haversine / pair_duration` captures congestion level |
| **Weather interactions** | `rain × airport`, `rain × distance` |
| **Outlier clipping for aggregates** | 99th percentile cap prevents skewed priors |
| **Week-of-year encoding** | Captures seasonal patterns the model can extrapolate |

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

1.  **Full data training:** All experiments used 5-20% of 37M rows
2.  **Out-of-fold stacking:** Proper K-fold OOF predictions for meta-learner
3.  **Residual modeling:** Train correction model on base model errors
4.  **FT-Transformer:** Self-attention across tabular features

---

## 🛠️ How to Reproduce

```bash
# 1. Download base data
python data/download_data.py

# 2. Extract geospatial coordinates (requires geopandas)
python extract_coords.py

# 3a. Train v3 — XGBoost baseline (~5 min on 20% data)
ETA_SAMPLE_FRAC=0.2 python baseline-copy.py

# 3b. Train v4 — Entity Embedding NN + XGBoost (~30 min on 20% data)
ETA_SAMPLE_FRAC=0.2 python train_deep.py

# 3c. Train v5 — Multi-GBDT Stacking (~45 min on full data)
python train_v5_stack.py

# 4. Grade on Dev
python grade.py
```

---
*Total time spent: ~8 hours*
