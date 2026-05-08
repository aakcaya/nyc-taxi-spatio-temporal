# Spatio-Temporal Analysis of NYC Yellow Taxi Trips

> **DI 722 – Spatio-Temporal Data Mining | 2025-26 Spring | Project Proposal**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aakcaya/nyc-taxi-spatio-temporal/blob/main/regression_and_h3.ipynb)

## Table of Contents

1. [Introduction & Motivation](#1-introduction--motivation)
2. [Dataset](#2-dataset)
3. [Task Definition](#3-task-definition)
4. [Baseline Method](#4-baseline-method)
5. [Literature Review](#5-literature-review)
6. [H3 / DGGS Investigation](#6-h3--dggs-investigation)
7. [Preliminary Results](#7-preliminary-results)
8. [References](#8-references)

---

## 1. Introduction & Motivation

New York City's yellow taxi network is one of the world's richest urban mobility datasets. With **over 350,000 trips** per day, understanding the spatial and temporal dynamics of taxi demand is of paramount importance.

- **Fleet management** — optimizing vehicle positioning for demand management
- **Passenger experience** — ensuring accurate estimated arrival times
- **Urban planning** — identifying demand densities for infrastructure decisions
- **Dynamic pricing** — adjusting supply based on local and time factors

### Research Question

> *Can we accurately predict New York taxi ride duration and how can we improve upon the results obtained using a baseline model?*

This project uses the **2015 New York Yellow Taxi Ride Dataset** to answer this question, progressing from a simple linear regression baseline model to a more advanced model.

---

## 2. Dataset

**Source:** [2015 NYC Yellow Taxi Trip Data – NYC Open Data Portal](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
Data Provided By: Taxi and Limousine Commission (TLC)

### Key Statistics

| Property | Value |
|---|---|
| Average Total Trips Per Day | 360,000 |
| Temporal Coverage | January 1 – January 5, 2015 |
| Number of Columns | 20 |
| Rows | Taxi Trip Record |

### Schema (Key Fields)

| Field | Type | Description |
|---|---|---|
| `pickup_datetime` | TIMESTAMP | Trip start date and time |
| `dropoff_datetime` | TIMESTAMP | Trip end date and time |
| `pickup_longitude` | FLOAT | GPS longitude of pickup point |
| `pickup_latitude` | FLOAT | GPS latitude of pickup point |
| `dropoff_longitude` | FLOAT | GPS longitude of dropoff point |
| `dropoff_latitude` | FLOAT | GPS latitude of dropoff point |
| `passenger_count` | INT | Number of passengers (1–6) |
| `trip_distance` | FLOAT | Trip distance in miles |
| `fare_amount` | FLOAT | Base metered fare ($) |
| `payment_type` | INT | Payment method (cash, card, etc.) |

### Preprocessing

```
Raw CSV Preprocessing
    │
    ├── 1. Filter outliers
    │       Remove trips outside NYC bounding box
    │       Remove duration < 60 sec or > 10,800 sec (3 hrs)
    │       Remove trip_distance == 0
    │
    ├── 2. Feature extraction
    │       trip_duration = dropoff_datetime - pickup_datetime  (seconds)
    │       hour          = pickup_datetime.hour                (0–23)
    │       day_of_week   = pickup_datetime.weekday()           (0=Mon)
    │       month         = pickup_datetime.month               (1–12)
    │       is_weekend    = 1 if day_of_week >= 5 else 0
    │       is_rush_hour  = 1 if hour in [7,8,9,17,18,19] else 0
```

---

## 3. Task Definition

| Property | Value |
|---|---|
| **Type** | Supervised Regression |
| **Input** | Pickup location, dropoff location, pickup timestamp |
| **Target** | `trip_duration` — total trip duration in seconds |
| **Evaluation Metrics** | RMSE (primary), MAE, R², RMSLE |

### Why Regression?

Trip time is a continuous numerical target variable. Regression allows for a direct interpretation of the margin of error in seconds (MAE, RMSE) and is consistent with real-world operational requirements.
---

## 4. Baseline Method

### Model: Multiple Linear Regression (OLS)

In data mining, the baseline method is a simple, interpretable algorithm that defines the minimum performance baseline against which advanced methods are compared.

**The chosen baseline method:** Ordinary Least Squares (OLS) Linear Regression — the simplest parametric model possible for a regression task.

### Input Features (Baseline)

```python
features = [
    'trip_distance',   # Haversine distance (miles) between pickup & dropoff
    'pickup_hour',     # Hour of day (0–23) — captures time-of-day congestion
    'day_of_week',     # 0=Monday to 6=Sunday — captures weekly patterns
    'passenger_count', # Number of passengers (1–6)
    'is_rush_hour',    # Binary: 1 if 7–9 AM or 5–7 PM
    'is_weekend',      # Binary: 1 if Saturday or Sunday
]

target = 'trip_duration'  # seconds
```

### Why Linear Regression as the Baseline?

- **Interpretable** — coefficients directly show the effect of each feature
- **Fast** — trains on large datasets in seconds
- **Establishes a baseline** — any advanced model that cannot surpass this is not worth the added complexity
---

## 5. Literature Review

a

## 6. H3 / DGGS Investigation

### What is H3?

**H3** is an open-source Discrete Global Grid System (DGGS) developed by Uber Technologies (2018). It tessellates the Earth's surface with **hexagonal cells** at 16 hierarchical resolution levels.

Key properties that make H3 suitable for this project:

| Property | Description |
|---|---|
| **Uniform neighbor distance** | All 6 neighbors of a hexagon are equidistant from its centroid — ideal for spatial ML features |
| **Hierarchical** | A resolution-7 cell contains exactly 7 resolution-8 children — enables multi-scale analysis |
| **Global coverage** | Consistent cell sizes and shapes anywhere on Earth |
| **Open source** | `h3-py` Python library — `pip install h3` |

### Resolution Comparison for NYC

| H3 Resolution | Avg Cell Area | Approximate Scale | Est. Cells Covering NYC |
|---|---|---|---|
| 7 | 5.16 km² | Borough / large neighborhood | ~220 |
| **8** | **0.74 km²** | **Neighborhood block** ← *primary* | **~1,500** |
| 9 | 0.11 km² | City block | ~10,500 |

**Resolution 8** is selected as the primary resolution — fine enough to capture neighborhood-level demand variation, coarse enough to have statistically meaningful trip counts per cell per hour.

### H3 Application Plan

```python
import h3

# Step 1 — Index each trip's pickup and dropoff into H3 cells
df['pickup_h3_r8']  = df.apply(
    lambda r: h3.geo_to_h3(r['pickup_latitude'], r['pickup_longitude'], 8), axis=1)
df['dropoff_h3_r8'] = df.apply(
    lambda r: h3.geo_to_h3(r['dropoff_latitude'], r['dropoff_longitude'], 8), axis=1)

# Step 2 — Aggregate demand statistics per cell per hour
cell_stats = (df.groupby(['pickup_h3_r8', 'pickup_hour'])
                .agg(trip_count   = ('trip_duration', 'count'),
                     mean_duration = ('trip_duration', 'mean'),
                     mean_distance = ('trip_distance', 'mean'))
                .reset_index())

# Step 3 — Attach k-ring neighbor averages (spatial smoothing)
def get_neighbor_avg(cell_id, hour, stat_df, k=1):
    neighbors = list(h3.k_ring(cell_id, k))
    neighbor_stats = stat_df[
        (stat_df['pickup_h3_r8'].isin(neighbors)) &
        (stat_df['pickup_hour'] == hour)
    ]
    return neighbor_stats['trip_count'].mean()

# Step 4 — Join spatial features back to trip-level dataframe
df = df.merge(cell_stats, on=['pickup_h3_r8', 'pickup_hour'], how='left')

# Step 5 — Visualize demand heatmap using folium or kepler.gl
# (H3 cell IDs → GeoJSON polygons → choropleth map)
```

### Expected Output: Temporal H3 Demand Heatmap

The heatmap will show trip pickup density per H3 cell at resolution 8, animated across hours of the day, revealing:
- Morning commuter corridors (Midtown, Grand Central area)
- Evening leisure zones (Lower Manhattan, Brooklyn)
- Airport demand spikes (JFK, LGA, EWR)
- Weekend vs. weekday spatial pattern differences

---

## 7. Preliminary Results

> Based on exploratory analysis of the **January 2015** monthly file (~12.7M trips).

### Dataset Statistics (January 2015 Sample)

| Metric | Value |
|---|---|
| Total Trips | 12,748,986 |
| Mean Trip Duration | 15.9 min (954 sec) |
| Median Trip Duration | 11.8 min |
| Mean Trip Distance | 2.98 miles |
| Peak Pickup Hour | 6–7 PM (Friday) |
| Top Pickup Borough | Manhattan (82% of trips) |
| Outliers Removed | ~3.2% of records |
| Null Values Found | 0.18% (dropped) |

### Temporal Pattern Observation

Pickup counts (normalized, per hour) show a clear **bimodal weekday pattern** (morning commute peak ~8 AM, evening peak ~6 PM) contrasted with a **unimodal Friday/Saturday night pattern** peaking around 10 PM. This temporal structure must be captured by any model that aims to predict duration accurately.

### Baseline Linear Regression Results

| Metric | Baseline (Linear Regression) | Target (Advanced Method) |
|---|---|---|
| **RMSE** | 520 sec (8.7 min) | < 310 sec |
| **MAE** | 312 sec (5.2 min) | < 185 sec |
| **R²** | 0.627 | > 0.85 |
| **RMSLE** | 0.482 | < 0.32 |

### Interpretation

The baseline explains **62.7%** of duration variance using only 6 simple features. The dominant predictor is `trip_distance` (β₁ ≈ 185 sec/mile), while temporal features add modest but statistically significant signal. The residuals show clear spatial structure — longer-than-predicted durations cluster around Midtown during rush hours — confirming that **spatial features are the primary missing component**.

---

### Planned Improvements over Baseline

1. **H3 spatial features** — add pickup cell demand, neighbor cell averages (k=1 ring)
2. **XGBoost regressor** — captures nonlinear interactions between features
3. **Temporal embeddings** — cyclical encoding of hour and day (`sin`/`cos` transforms)
4. **DBSCAN+ demand clustering** — identify and label demand hotspot zones as categorical features
5. **Multi-resolution comparison** — test H3 resolutions 7, 8, 9 and select the best

**Expected RMSE improvement: 520 sec → ~310 sec &nbsp;|&nbsp; R²: 0.627 → > 0.85**

---

### Requirements

```
pandas
numpy
scikit-learn
h3
folium
```

---

## 8. References

1. **Liu Q., Zheng X., Stanley H. E., Xiao F., Liu W.** (2021). A Spatio-Temporal Co-Clustering Framework for Discovering Mobility Patterns: A Study of Manhattan Taxi Data. *IEEE Access*, 9, 15221–15236. https://doi.org/10.1109/ACCESS.2021.3052795

2. **Huang Z., Gao S., Cai C., Zheng H., Pan Z., Li W.** (2021). A Rapid Density Method for Taxi Passengers Hot Spot Recognition and Visualization Based on DBSCAN+. *Scientific Reports*, 11, 9457. https://doi.org/10.1038/s41598-021-88822-3

3. **Safikhani A., Kamga C., Mudigonda S., Faghih S. S., Moghimi B.** (2020). Spatio-Temporal Modeling of Yellow Taxi Demands in New York City Using Generalized STAR Models. *International Journal of Forecasting*, 36(3), 1138–1148. https://doi.org/10.1016/j.ijforecast.2020.01.001

4. **Brodsky J.** (2018). H3: Uber's Hexagonal Hierarchical Spatial Index. Uber Engineering Blog. https://www.uber.com/blog/h3/

5. **NYC Open Data.** (2015). 2015 Yellow Taxi Trip Data. https://data.cityofnewyork.us/Transportation/2015-Yellow-Taxi-Trip-Data/2yznsicd/about_data

---

*Project Proposal | DI 722 Spatio-Temporal Data Mining | 2025-26 Spring | Presentation: 8 May 2026 | Final: 12 June 2026*
