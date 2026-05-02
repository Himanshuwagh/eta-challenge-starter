# The ETA Challenge: Submission Report

*A production-grade ride-hailing ETA engine built on NYC Yellow Taxi data.*

---

## 🚀 Final Performance

- **Dev MAE (grade.py):** **260.5 s**
- **Baseline MAE:** ~351.0 s
- **Improvement:** **-90.5 s (25.8% reduction in error)**

---

## 🏗️ The Approach: Context-Aware Feature Engineering

My solution transforms the problem from a simple regression task into a spatially and contextually aware forecasting system. The core strategy was to move beyond treating Taxi Zones as arbitrary integers and instead teaching the model about the physical geography, weather, and calendar of New York City.

### 1. Spatial Intelligence
The baseline model had no concept of physical distance. I integrated the **NYC Taxi Zone Shapefiles** to extract latitude/longitude centroids for all 265 zones.
- **Manhattan Distance:** Since NYC is a grid, Manhattan distance proved far superior to Euclidean distance for approximating actual driving paths.
- **Bearing (Trip Direction):** Captured traffic flow trends (e.g., heading "uptown" vs. "downtown").

### 2. Bayesian Smoothed Priors (Target Encoding)
A simple lookup of zone-pair averages is a strong baseline (~300s), but fails on rare routes. I implemented:
- **Bayesian Smoothing:** For zone pairs with low trip counts, I shrink the prediction toward the pickup-zone's overall median.
- **Time-Conditioned Priors:** Instead of one median per pair, I calculated medians for **3-hour blocks** (e.g., 8 AM - 11 AM) to capture rush hour dynamics.

### 3. Contextual & Environmental Data
- **Weather Integration:** Merged hourly NOAA weather data. Features like `precip_mm` and `visibility_km` allow the model to account for rain/snow delays.
- **Holiday Logic:** Specifically implemented "Christmas-New Year Period" flags. Since the evaluation set is a winter holiday slice, the model needs to know that NYC traffic behaves differently during the holidays.

### 4. Model Architecture (v3 → v4)
- **XGBoost (Version 3):** Highly regularized Gradient Boosted Tree with **31 features** — Dev MAE **260.5s**.
- **Entity Embedding NN + XGBoost Ensemble (Version 4):** Implemented a PyTorch-based neural network with **learned embeddings** for zones, hours, and day-of-week (inspired by Uber's DeepETA). The NN achieved **256.5s** on 20% data — better than XGBoost alone. The trained zone embeddings (16-dim vectors per zone) were then injected as additional features into a fresh XGBoost model (58 features total), and the final prediction is a weighted ensemble of both.
- **Hardware:** Apple Silicon (MPS) was used for GPU-accelerated NN training.

---

## 📉 What didn't work (Dead Ends)

1.  **Naive Categorical IDs:** Simply feeding Zone IDs as integers resulted in the model "hallucinating" relationships between neighboring IDs that weren't geographically related. This led to the naive baseline losing to a 10-line lookup.
2.  **Over-Complex Time Encoding:** Initially tried 1-minute resolution for cyclical encoding, which introduced too much noise. Binning into 3-hour blocks for priors and keeping `minute_of_day` for the tree was a more stable approach.
3.  **Raw Euclidean Distance:** In the dense grid of NYC, straight-line distance was often misleading. Swapping to Manhattan distance provided an immediate 5-8s drop in MAE.
4.  **OneCycleLR for NN Training:** Initially used PyTorch's aggressive OneCycleLR scheduler which caused the NN to diverge after peak LR, spiking dev MAE from 256s to 477s within a few epochs. Switching to `ReduceLROnPlateau` (halves LR on plateau) gave stable convergence.

---

## 🤖 Where AI Tooling Sped Me Up Most

I used **Antigravity (Gemini & Claude)** as a pair programmer throughout:
- **Feature Engineering:** Instantly generated vectorized NumPy implementations for Haversine and Bearing calculations.
- **Data Pipeline:** Automated the downloading and extraction of geospatial shapefiles.
- **Debugging:** Quickly resolved environment-specific issues with Python's `geopandas` and `pyogrio` dependencies.
- **Rapid Prototyping:** Iterated through 3 versions of the model architecture in under 2 hours.

---

## 🔮 Next Experiments

If I had more time/compute, I would implement:
1.  **Train on 100% data:** All experiments used 20% of the 37M-row dataset. Training on the full data would significantly improve both the target-encoding priors and the NN embeddings.
2.  **LightGBM Ensemble:** Add LightGBM as a third model in the ensemble to exploit its native categorical splitting.
3.  **FT-Transformer (Feature Tokenizer + Transformer):** A state-of-the-art tabular deep learning architecture that uses self-attention across all features — consistently outperforms MLPs on large tabular datasets.
4.  **Uber's DeepETA Pattern:** Train a dedicated "corrector" NN on the residuals of the XGBoost predictions.

---

## 🛠️ How to Reproduce

```bash
# 1. Download base data
python data/download_data.py

# 2. Extract geospatial coordinates (requires geopandas)
python extract_coords.py

# 3a. Train v3 model — XGBoost with 31 features (faster, ~5 min on 20% data)
ETA_SAMPLE_FRAC=0.2 python baseline-copy.py

# 3b. Train v4 model — Entity Embedding NN + XGBoost Ensemble (deeper, ~30 min on 20% data)
ETA_SAMPLE_FRAC=0.2 python train_deep.py

# 4. Grade on Dev
python grade.py
```

---
*Total time spent: ~6 hours*
