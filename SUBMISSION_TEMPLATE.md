# Your Submission: Writeup

---

## Your final score

Dev MAE: **260.5 s**

---

## Your approach, in one paragraph

I built a context-aware XGBoost model (v3) using 31 features. Key features include Manhattan and Haversine distances derived from NYC taxi zone centroids, US holiday/Christmas-period logic, and integrated hourly weather data (precip, visibility, wind). I implemented a tiered prior system using Bayesian smoothed target encoding for (pickup, dropoff) pairs and time-conditioned priors (3-hour bins) to capture rush-hour dynamics. The model was trained on a 20% sample of the 2023 dataset (7.3M rows) with heavy regularization to handle the high feature count.

## What you tried that didn't work

1.  **Naive Categorical IDs:** Passing raw zone integers to the tree model failed to capture physical proximity, initially losing to a simple lookup table.
2.  **Euclidean Distance:** Proved less accurate than Manhattan distance in NYC's grid layout, which better approximates actual driving paths.
3.  **High-Resolution Cyclical Encoding:** Using 1-minute cyclical features introduced noise; 3rd-party priors binned by 3-hour blocks were more robust.

## Where AI tooling sped you up most

Antigravity (Gemini 3.1 Pro and Claude 3.5 Sonnet) was instrumental in accelerating the development. Specifically:
- **Feature Engineering:** Instantly generated vectorized NumPy implementations for geospatial math (bearing, haversine).
- **Data Infrastructure:** Automated the script to download, unzip, and extract centroids from NYC shapefiles using `geopandas`.
- **Debugging:** Rapidly resolved dependency conflicts and git configuration issues.
The tooling fell short in handling extremely long-running terminal commands, which required manual monitoring and restart.

## Next experiments

1.  **LightGBM/CatBoost Ensemble:** Blending multiple GBDT models to capture different splitting patterns and native categorical optimizations.
2.  **Entity Embeddings:** Training a neural network to learn latent representations of zones to capture neighborhood "personality" beyond just physical distance.
3.  **Residual Post-Processing:** Implementing a "correction" layer that predicts the error of the current model based on live driver telemetry (if available).

## How to reproduce

```bash
# 1. Download base data
python data/download_data.py

# 2. Extract geospatial coordinates
python extract_coords.py

# 3. Train the model (20% sample)
ETA_SAMPLE_FRAC=0.2 python baseline-copy.py

# 4. Grade locally
python grade.py
```

---

*Total time spent on this challenge: ~4 hours.*
