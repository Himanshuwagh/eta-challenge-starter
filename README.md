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

### 4. Model Architecture
- **XGBoost (Version 3):** Used a highly regularized Gradient Boosted Tree with **31 features**.
- **Objective:** Optimized directly for **MAE** (`reg:absoluteerror`) to align with the challenge metric.

---

## 📉 What didn't work (Dead Ends)

1.  **Naive Categorical IDs:** Simply feeding Zone IDs as integers resulted in the model "hallucinating" relationships between neighboring IDs that weren't geographically related. This led to the naive baseline losing to a 10-line lookup.
2.  **Over-Complex Time Encoding:** Initially tried 1-minute resolution for cyclical encoding, which introduced too much noise. Binning into 3-hour blocks for priors and keeping `minute_of_day` for the tree was a more stable approach.
3.  **Raw Euclidean Distance:** In the dense grid of NYC, straight-line distance was often misleading. Swapping to Manhattan distance provided an immediate 5-8s drop in MAE.

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
1.  **LightGBM Ensemble:** Switch to LightGBM to leverage its native categorical splitting and ensemble it with XGBoost for a ~3-5s error reduction.
2.  **Entity Embeddings (Deep Learning):** Train a neural network to learn latent 16-dimensional vectors for each zone, capturing semantic relationships between neighborhoods.
3.  **Uber's DeepETA Pattern:** Train a secondary model to predict the *residuals* (errors) of the current XGBoost model, acting as a post-processing "corrector."

---

## 🛠️ How to Reproduce

```bash
# 1. Download base data
python data/download_data.py

# 2. Extract geospatial coordinates (requires geopandas)
python extract_coords.py

# 3. Train the v3 model (takes ~5-10 min on 20% sample)
ETA_SAMPLE_FRAC=0.2 python baseline-copy.py

# 4. Grade on Dev
python grade.py
```

---
*Total time spent: ~4 hours*
