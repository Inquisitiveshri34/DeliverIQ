# DeliverIQ — Urban Delivery Route Optimiser

An end-to-end supervised machine learning project that predicts urban delivery travel times from multi-source transportation data. The pipeline spans raw data ingestion, cleaning, exploratory analysis, feature engineering, model training and tuning, evaluation, and a live Streamlit prediction interface.

**Stack:** Python 3.13 · scikit-learn 1.4.2 · Streamlit 1.35.0

---

## Table of Contents

1. [Business Objective](#business-objective)
2. [Key Results](#key-results)
3. [Project Structure](#project-structure)
4. [Pipeline Architecture](#pipeline-architecture)
5. [Data Sources](#data-sources)
6. [Data Quality Issues](#data-quality-issues)
7. [Feature Engineering](#feature-engineering)
8. [Model Training](#model-training)
9. [Evaluation](#evaluation)
10. [Streamlit App](#streamlit-app)
11. [Setup and Launch](#setup-and-launch)
12. [Common Failure Modes](#common-failure-modes)

---

## Business Objective

Predict how long a delivery trip will take (in minutes) given route geometry, road conditions, traffic state, and weather — enabling logistics planners and dispatch systems to set realistic delivery windows.

**Target variable:** `travel_time_min` (raw) / `log_travel_time` (log1p-transformed, used for training)  
**Primary metric:** MAE (mean absolute error in minutes)

---

## Key Results

| Metric | Value | Status |
|---|---|---|
| **MAE (test set)** | **7.83 min** | Primary metric |
| RMSE (test set) | 35.72 min | Inflated by outlier trips |
| R² | 0.2054 | ❌ Below 0.70 target (expected — see note below) |
| MAPE | 37.78% | Unstable on very short trips |
| **MAE vs Baseline** | **−55.18%** | ✅ Target ≥ 30% improvement met |
| Baseline MAE | 17.48 min | Mean predictor floor |
| Best model | Random Forest (200 trees) | Fixed hyperparameters |
| Training rows | 191,935 | 80% of cleaned master |
| Test rows | 47,984 | 20% held out |

> **Why R² is low (expected behaviour):** The dataset contains ~3–7% invalid or missing foreign key references across the trips → roads, traffic, and weather joins, Gaussian noise injected on `travel_time_min`, and ~13.5% of trips with no valid road reference. These factors set a hard noise floor on achievable variance explanation. MAE (7.83 min) is the correct primary metric.

---
```text
DeliverIQ/
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_evaluation_summary.ipynb
│
├── src/
│   ├── __init__.py
│   ├── features.py
│   ├── train.py
│   └── predict.py
│
├── data/
│   ├── raw/
│   │   ├── roads.csv
│   │   ├── traffic.csv
│   │   ├── weather.csv
│   │   └── trips.csv
│   │
│   └── processed/
│       ├── cleaned/
│       │   ├── roads_clean.csv
│       │   ├── traffic_clean.csv
│       │   ├── weather_clean.csv
│       │   └── trips_clean.csv
│       │
│       ├── splits/
│       │   ├── X_train.csv
│       │   ├── X_test.csv
│       │   ├── y_train.csv
│       │   └── y_test.csv
│       │
│       └── merged/
│           └── master.csv
│           # 239,919 × 19 — modelling-ready flat table
│
├── models/
│   ├── best_model.pkl
│   ├── metrics.json
│   └── feature_importance.csv
│
├── output/
│   └── plots/
│       └── # 10 EDA plots (PNG, DPI 120)
│
├── app.py
└── requirements.txt
```

---

## Pipeline Architecture

Run notebooks in strict order. Each notebook depends on outputs from the previous one.

| Notebook / Module | Purpose | Output |
|---|---|---|
| `01_data_cleaning.ipynb` | Load, clean, and merge four raw CSVs into `master.csv` | `master.csv` (239,919 × 19, 23.8 MB) |
| `02_eda.ipynb` | Exploratory analysis — 10 diagnostic plots, 5 computed insights | 10 plots in `output/plots/` |
| `03_feature_engineering.ipynb` | Decode, impute, split, leakage demonstration | `X_train`, `X_test`, `y_train`, `y_test` CSVs |
| `04_model_training.ipynb` | Baseline → Linear → DT → RF; tune; save artifacts | `best_model.pkl`, `metrics.json`, `feature_importance.csv` |
| `05_evaluation_summary.ipynb` | Verify `.pkl`, reproduce metrics, smoke-test inference | Inline evaluation plots |
| `app.py` | Streamlit prediction interface | Live at `http://localhost:8501` |

---

## Data Sources

Four CSV files sourced from a multi-table urban transportation system. Each contains deliberate real-world messiness: typos in categorical labels, out-of-range numeric values, mixed timestamp formats, duplicate primary keys, and invalid foreign key references.

| File | Rows | Columns | Role |
|---|---|---|---|
| `roads.csv` | 25,000 | 5 | Dimension table — road geometry and zone classification |
| `traffic.csv` | 100,000 | 6 | Dimension table — traffic conditions by hour/day/zone |
| `weather.csv` | 75,000 | 5 | Dimension table — weather events (one event ID, multiple readings) |
| `trips.csv` | 250,000 | 11 | Fact table — individual delivery trips; target variable source |

### Column Schemas

**roads.csv**

| Column | Type | Notes |
|---|---|---|
| `road_id` | object | PK — 1,000 nulls dropped; 2,000 duplicates removed |
| `road_type` | object | `highway / urban / residential` — typo variants standardised |
| `num_lanes` | Int64 | Valid range 1–8; out-of-range → NaN |
| `num_signals` | Int64 | Valid range 0–20; outliers → NaN |
| `zone_type` | object | `commercial / industrial / residential` — 1,785 NaN remaining |

**traffic.csv**

| Column | Type | Notes |
|---|---|---|
| `traffic_id` | object | PK — 4,000 duplicate rows removed |
| `hour` | Int64 | Valid range 0–23; 2,850 invalid values → NaN |
| `day_of_week` | Int64 | 0–6 (Mon=0) — mixed integer/string formats normalised |
| `zone_type` | object | 23,187 NaN post-clean |
| `traffic_level` | object | `high / medium / low` — typo variants standardised |
| `avg_speed_kmph` | float64 | Valid range 1–120 km/h; injected outliers (0, 150, 200, 999) → NaN |

**weather.csv**

| Column | Type | Notes |
|---|---|---|
| `weather_id` | object | Event ID — intentional duplicates; 37,941 unique IDs aggregated to one row each |
| `timestamp` | datetime | Mixed formats — 5-format parser; 0 unparseable rows |
| `weather_type` | object | `clear / rain / fog` — typo variants standardised; 1,109 NaN |
| `temperature_c` | float64 | Valid range −3–55 °C; injected outliers → NaN |
| `visibility_km` | float64 | Range-clipped per weather type (fog: 0.05–4.0, rain: 1.0–8.0, clear: 4.0–12.0) |

**trips.csv (fact table)**

| Column | Type | Notes |
|---|---|---|
| `trip_id` | object | Dropped — arbitrary key, no predictive signal |
| `road_id` | object | FK → roads; 33,793 null or unresolvable |
| `traffic_id` | object | FK → traffic; 25,024 null or unresolvable |
| `weather_id` | object | FK → weather; 24,953 null or unresolvable |
| `timestamp` | datetime | Trip departure — 3,942 unparseable → NaT |
| `start_lat / start_lon` | float64 | India bbox (6–37°N, 68–98°E); sentinel 999/−999 → NaN |
| `end_lat / end_lon` | float64 | Same bounding box and sentinel nulling |
| `distance_km` | float64 | Valid range 0.01–100 km; negatives and extremes → NaN |
| `travel_time_min` | float64 | **TARGET** — valid range 0.1–600 min; 10,081 null rows dropped |

### FK Integrity Audit (post-cleaning)

| Join Path | Valid Refs | Null / Unresolvable | Orphaned |
|---|---|---|---|
| trips → roads | 216,207 (86.5%) | 33,793 (13.5%) | 0 (0.0%) |
| trips → traffic | 224,976 (90.0%) | 25,024 (10.0%) | 0 (0.0%) |
| trips → weather | 225,047 (90.0%) | 24,953 (10.0%) | 0 (0.0%) |

---

## Data Quality Issues

| Issue | Table | Magnitude | Handling |
|---|---|---|---|
| Null primary keys | roads | 1,000 rows | Dropped |
| Duplicate primary keys | roads | 2,000 rows | Deduplicated (keep first) |
| Duplicate primary keys | traffic | 4,000 rows | Deduplicated (keep first) |
| Intentional duplicate event IDs | weather | ~37% of rows | Aggregated to one row per `weather_id` |
| Typo variants in categoricals | roads, traffic, weather | ~3–5% per column | Strip / lowercase / map to canonical labels; unmapped → NaN |
| Out-of-range numeric values | roads, traffic, weather | Varies | Clipped via `.where()`; invalid → NaN |
| Mixed timestamp formats | weather, trips | 5 known formats | 5-format ordered fallback parser; 3,942 trip failures → NaT |
| Sentinel coordinates (999, −999) | trips | ~2,750 per coordinate column | India bbox clip; sentinels → NaN |
| Invalid FK references | trips | 3–7% per FK column | Validated against cleaned dimension sets; unresolvable → NaN |
| Extreme outlier travel times | trips | Injected | Clipped to 0.1–600 min |
| Missing target (`travel_time_min`) | trips | 10,081 rows | Dropped |
| Missing road FK (no join match) | master | 33,793 (13.5%) | `road_type` imputed with mode `'residential'` in NB-03 |
| Missing traffic FK (no join match) | master | 25,024 (10.0%) | `traffic_level` imputed with `'medium'` in NB-03 |
| Missing weather FK (no join match) | master | 24,953 (10.0%) | `weather_type` imputed with mode `'clear'` in NB-03 |

---

## Feature Engineering

**Input:** `master.csv` (239,919 × 19)  
**Output:** train/test splits in `data/processed/splits/`  
**Target:** `log_travel_time = np.log1p(travel_time_min)` (skewness = 8.47; `USE_LOG_TRANSFORM = True`)

### Final Feature Matrix — 13 Columns

| Feature | dtype | Pipeline Branch | Notes |
|---|---|---|---|
| `distance_km` | float64 | `num` → StandardScaler | 1,074 nulls filled with median 4.33 km |
| `num_lanes` | Int64 | `num` → StandardScaler | Imputed in NB-01 |
| `num_signals` | Int64 | `num` → StandardScaler | Imputed in NB-01 |
| `avg_speed_kmph` | float64 | `num` → StandardScaler | Imputed in NB-01 |
| `temperature_c` | float64 | `num` → StandardScaler | Imputed in NB-01 |
| `visibility_km` | float64 | `num` → StandardScaler | Imputed in NB-01 |
| `hour` | float64 | `num` → StandardScaler | 0–23; trip timestamp takes precedence over traffic hour |
| `day_of_week` | Int64 | `num` → StandardScaler | 0–6 (Mon=0) |
| `is_rush_hour` | int | `bool` → passthrough | 1 if `hour ∈ {7, 8, 17, 18}` |
| `is_weekend` | int | `bool` → passthrough | 1 if `day_of_week >= 5` |
| `traffic_level` | str | `ord` → OrdinalEncoder | `low=0, medium=1, high=2`; nulls filled with `'medium'` |
| `road_type` | str | `cat` → OneHotEncoder | `highway / urban / residential`; nulls filled with `'residential'` |
| `weather_type` | str | `cat` → OneHotEncoder | `clear / rain / fog`; nulls filled with `'clear'` |

### Train / Test Split

| Parameter | Value |
|---|---|
| `test_size` | 0.2 (20%) |
| `random_state` | 42 — **never change** |
| `X_train` shape | 191,935 × 13 |
| `X_test` shape | 47,984 × 13 |
| Nulls in splits | 0 |

---

## Model Training

### Pipeline Architecture

All preprocessing happens inside the sklearn `Pipeline` to prevent data leakage. Encoding and scaling are refitted on each CV fold's training slice.

Pipeline
```
└── preprocessor (ColumnTransformer, remainder='drop')
├── num   → StandardScaler       → 8 numeric features
├── bool  → passthrough          → is_rush_hour, is_weekend
├── ord   → OrdinalEncoder       → traffic_level
└── cat   → OneHotEncoder        → road_type, weather_type
└── model → estimator (swapped per stage)
```

### Model Progression

| Stage | Model | Purpose |
|---|---|---|
| 1 | Mean Baseline | Performance floor — Baseline MAE: 17.48 min |
| 2 | Linear Regression | Test linearity assumption; residual heteroscedasticity confirms non-linearity |
| 3 | Decision Tree (default) | Intentional overfitting demonstration |
| 4 | Decision Tree (GridSearchCV) | 60 combinations (5 `max_depth` × 4 `min_split` × 3 `min_leaf`), 5-fold CV |
| 5 | **Random Forest (final)** | Fixed hyperparameters; `n_jobs=1` (Windows MemoryError workaround) |

### Random Forest — Final Hyperparameters

| Parameter | Value |
|---|---|
| `n_estimators` | 200 |
| `max_depth` | 20 |
| `max_features` | `'sqrt'` |
| `min_samples_split` | 5 |
| `min_samples_leaf` | 3 |
| `bootstrap` | `True` |
| `random_state` | 42 |
| `n_jobs` | 1 |

---

## Evaluation

All metrics computed on the held-out test set (47,984 rows) after `np.expm1()` back-transformation to original minutes.

### Final Metrics

| Metric | Value | Interpretation |
|---|---|---|
| **MAE** | **7.83 min** | Off by ~7.8 minutes on average |
| RMSE | 35.72 min | Much higher than MAE — extreme outlier trips dominating squared error |
| R² | 0.2054 | 20.5% of variance explained |
| MAPE | 37.78% | Percentage error; unstable on very short trips |
| Mean Residual | +4.08 min | Slight positive bias — model under-predicts extreme trips |

### Feature Importance (Top 10)

| Rank | Feature | Importance |
|---|---|---|
| 1 | `distance_km` | 65.04% |
| 2 | `num_signals` | 9.40% |
| 3 | `avg_speed_kmph` | 3.72% |
| 4 | `road_type_highway` | 3.58% |
| 5 | `num_lanes` | 3.52% |
| 6 | `visibility_km` | 2.69% |
| 7 | `traffic_level` | 2.45% |
| 8 | `temperature_c` | 2.35% |
| 9 | `road_type_residential` | 2.06% |
| 10 | `hour` | 1.57% |

`distance_km` dominates at 65% — consistent with the synthetic data generation logic where trip distance is the primary driver of `travel_time_min`.

---

## Streamlit App

A self-contained real-time prediction interface. `build_input_df()` and `predict()` are implemented directly in `app.py` for portability (no `src/` import required).

### Artifact Dependencies

All three files must exist in `models/` before launching:

| File | Used by |
|---|---|
| `models/best_model.pkl` | `load_model()` via `@st.cache_resource` |
| `models/metrics.json` | Sidebar metrics panel (nothing hardcoded) |
| `models/feature_importance.csv` | Feature importance Plotly chart (expander) |

### Inference Flow

User fills form → build_input_df() → pipeline.predict() → np.expm1() → st.success()

Derived features (not collected from the user directly):

| Feature | Derivation |
|---|---|
| `avg_speed_kmph` | From `traffic_level`: low → 20.0, medium → 35.0, high → 60.0 km/h |
| `visibility_km` | From `weather_type`: clear → 10.0, rain → 4.0, fog → 1.5 km |
| `is_rush_hour` | `int(hour ∈ {7, 8, 17, 18})` |
| `is_weekend` | `int(day_of_week >= 5)` |
| `temperature_c` | Fixed at 28.0 °C (typical Delhi ambient) |

**Smoke test result:** 12.5 km, urban road, medium traffic, clear weather, 08:00 Monday → **27.45 min** (±7.83 min MAE band: [19.61, 35.28])

---

## Setup and Launch

### Requirements

pandas
numpy
scikit-learn
scipy
matplotlib
seaborn
plotly
streamlit
python-dateutil
joblib

## Project Structure

### Steps

```bash
# 1. Activate virtual environment
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify installation
python -c "import sklearn, pandas, streamlit, plotly; print('All packages OK')"

# 4. Run notebooks in order (from project root)
jupyter notebook

# 5. Launch the app
streamlit run app.py
```

**Required notebook execution order:**
`01_data_cleaning` → `02_eda` → `03_feature_engineering` → `04_model_training` → `05_evaluation_summary`

### Pre-launch Checklist

- [ ] `models/best_model.pkl` exists
- [ ] `models/metrics.json` exists
- [ ] `models/feature_importance.csv` exists
- [ ] `app.py` is at the project root
- [ ] Virtual environment is active with `scikit-learn==1.4.2`

---

## Common Failure Modes

| Error | Likely Cause | Fix |
|---|---|---|
| `ValueError: could not convert string to float` | `traffic_level` contains NaN — `OrdinalEncoder` cannot handle it | Impute nulls before fitting, or add `handle_unknown` to encoder |
| `KeyError: 'model__max_depth'` | GridSearchCV param key missing `model__` prefix | Prefix all estimator params with `model__` |
| `AssertionError` in `load_model()` | Wrong file loaded — not a Pipeline object | Confirm `save_model()` completed; check `.pkl` path |
| `ColumnTransformer` silently drops a column | Column name mismatch vs `NUM_FEATURES`/`CAT_FEATURES` constants | Print `X_train.columns.tolist()` and compare character-by-character |
| Metrics in NB-05 differ from NB-04 | Inconsistent `random_state` in `split_data()` | Always use `random_state=42` |
| `FileNotFoundError: best_model.pkl` | NB-04 not run to completion | Run all notebooks in order |
| Double-scaling (near-zero variance) | `scale_numerics()` called before passing data to pipeline | Never manually scale before the pipeline — `ColumnTransformer` handles it |
| `sparse_output` AttributeError | scikit-learn < 1.2 using old `sparse=False` parameter | Use `sparse_output=False`; verify `scikit-learn==1.4.2` in venv |
| `y_test.csv` loads as 2-column DataFrame | pandas saves row index as extra column | Extract by name: `y_test_raw['log_travel_time']` |
| `BrokenProcessPool` / `MemoryError` on Windows | `RandomizedSearchCV` with `n_jobs=-1` on a large forest | Set `n_jobs=1`; use fixed hyperparameters as in NB-04 Stage 5 |







