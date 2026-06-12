# Spatio-Temporal Analysis of NYC Yellow Taxi Trips

> **DI 722 – Spatio-Temporal Data Mining | 2025-26 Spring | Final Project**

[[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/umutcemkartal/advanced-method)]
## Table of Contents

1. [Introduction & Motivation](#1-introduction--motivation)
2. [Dataset](#2-dataset)
3. [Task Definition](#3-task-definition)
4. [Baseline Method](#4-baseline-method)
5. [Literature Review](#5-literature-review)
6. [H3 / DGGS Investigation](#6-h3--dggs-investigation)
7. [Preliminary Results — Baseline](#7-preliminary-results--baseline)
8. [Advanced Method: XGBoost + Spatial Grid Features](#8-advanced-method-xgboost--spatial-grid-features)
9. [Cell Topologies & Scale Analysis](#9-cell-topologies--scale-analysis)
10. [Development Journey](#10-development-journey)
11. [References](#11-references)

---

## 1. Introduction & Motivation

New York City's yellow taxi network is one of the world's richest urban mobility datasets. With **over 350,000 trips** per day, understanding the spatial and temporal dynamics of taxi demand is of paramount importance.

- **Fleet Management** — optimizing vehicle positioning for demand management
- **Passenger Experience** — ensuring accurate estimated arrival times
- **Urban Planning** — identifying demand densities for infrastructure decisions
- **Dynamic Pricing** — adjusting supply based on local and time factors

### Research Question

> *Can we accurately predict NYC taxi trip duration, and does applied spatial grid features improve upon a non-spatial baseline — and if so, does cell granularity or cell type (hexagonal vs. square) matter more?*

This project approaches the problem in two stages:

- **Stage 1:** Linear regression baseline with using only trip-level features.
- **Stage 2:** XGBoost method, enriched with spatial features which derived from square grids and H3 hexagonal at three different resolution levels.

---

## 2. Dataset

**Source:** [2015 NYC Yellow Taxi Trip Data – NYC Open Data Portal](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)  
Data Provided By: Taxi and Limousine Commission (TLC)

### Key Statistics

| Property | Value |
|---|---|
| Average Total Trips Per Day | 365,000 |
| Temporal Coverage | January 1 – January 5, 2015 |
| Number of Columns | 20 |

### Schema (Key Fields)

| Field | Type | Description |
|---|---|---|
| `tpep_pickup_datetime` | TIMESTAMP | Trip start date and time |
| `tpep_dropoff_datetime` | TIMESTAMP | Trip end date and time |
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
    │       Remove duration < 60 sec or > 10,800 sec (3 hrs)
    │       Remove trip_distance == 0 or > 100 miles
    │       Remove passenger_count outside [1, 6]
    │       Drop null values
    │
    ├── 2. Spatial filtering via NYC boundary polygon
    │       GeoDataFrame spatial join against official
    │       NYC outer boundary (all 5 boroughs)
    │       Replaces bounding box approach — no coordinate gaps
    │
    ├── 3. Feature extraction
    │       trip_duration = dropoff_datetime - pickup_datetime  (seconds)
    │       pickup_hour   = pickup_datetime.hour                (0–23)
    │       day_of_week   = pickup_datetime.weekday()           (0=Mon)
    │       is_weekend    = 1 if day_of_week >= 5 else 0
    │       is_rush_hour  = 1 if hour in [7,8,9,17,18,19] else 0
    │
    └── 4. Spatial indexing
            Each trip matched to H3 and square grid cells
            via spatial join against QGIS-generated grids
            at resolutions 8, 9, and 10
```

---

## 3. Task Definition

| Property | Value |
|---|---|
| **Type** | Supervised Regression |
| **Input** | Pickup location, dropoff location, pickup timestamp |
| **Target** | `trip_duration` — total trip duration in seconds |
| **Evaluation Metrics** | MAE (primary), RMSE, RMSLE, R² |

Travel time is a continuous numerical variable. Regression allows for a direct interpretation of the true margin of error (MAE, RMSE) in seconds.
MAE was choosen as the primary metric. It measures its prediction error without being widely inflated by outliers.

---

## 4. Baseline Method

### Model: Multiple Linear Regression (OLS)

In data mining, the baseline method is a simple, interpretable algorithm that defines the minimum performance baseline against which advanced methods are compared.

**The chosen baseline method:** Ordinary Least Squares (OLS) Linear Regression — the simplest parametric model possible for a regression task.

### Input Features

```python
features = [
    'trip_distance',   # Distance between pickup & dropoff (miles)
    'pickup_hour',     # Hour of day (0–23)
    'day_of_week',     # 0=Monday to 6=Sunday
    'passenger_count', # Number of passengers (1–6)
    'is_rush_hour',    # Binary: 1 if 7–9 AM or 5–7 PM
    'is_weekend',      # Binary: 1 if Saturday or Sunday
]
target = 'trip_duration'  # seconds
```

> No spatial features needed for this stage. Coordinates used only to find out `trip_distance`.

### Why Linear Regression as the Baseline?

- **Interpretable** — coefficients directly show the effect of each feature
- **Fast** — trains on large datasets in seconds
- **Establishes a floor** — any advanced model that cannot surpass this is not worth the added complexity

---

## 5. Literature Review

> Scopus literature review

---

### Article 1 — Roy & Rout (2022)

**Title:** Predicting Taxi Journey Time Using Machine Learning Techniques
Considering Weekends and Holidays

This study uses January 2015 NYC Yellow Taxi records for journey time estimation. Additionally, Uber data was used to expand the modeling regions. Chi-Square scores were applied for feature selection; variables such as pick-up/drop-off coordinates, time, day of the week, and number of passengers were determined.
Three machine learning models were compared: Decision Tree Regression (DTR), Random Forest Regression (RFR), and K-Nearest Neighbor Regression (KNNR). The study also hierarchically evaluated the models used, explaining the improvement achieved by other advanced models based on the decision tree structure. Furthermore, the effects of weekends and holidays on journey time were evaluated, demonstrating the importance of temporal relationships.

**Relevance to This Project:**
This article presents methodological parallels with my work — the same forecasting task and the same dataset. Validating the superiority of tree-based methods over linear models on this dataset was the primary motivation for choosing XGBoost as the enhanced method. The article also validates our use of the `is_weekend` and `is_rush_hour` variables and demonstrates that the performance difference between linear and tree-based models stems from the difference in their ability to capture nonlinear interactions.

---

### Article 2 — Wang et al. (2019)

**Title:** A Simple Baseline for Travel Time Estimation Using Large-Scale
Trip Data

This article highlights how efficient complex algorithms based on big data can be compared to more basic methods. Data from the NYC Taxi and Limousine Commission was used for this study. The similarity-based estimation model used for travel time surpasses the estimations of the Bing Maps and Baidu Maps APIs.
This article directly addresses how to define and construct a "baseline." The main argument of the article is that "a simple but well-designed approach can be powerful." On the other hand, the neighborhood-based method in the article (considering the source-destination region pair) uses region-based statistics instead of raw coordinates, which increases estimation accuracy. This work demonstrates why the transition between baseline and advanced methods is meaningful.

**Relevance to This Project:** 
By assigning each trip to a square grid or H3 cell and adding the historical statistics of that cell, consistent with the application in this study, we are replacing coordinate-level specificity with information on the distribution at the region level. The paper also explains why the use of both H3 and square grid cells provides an improvement over the baseline. The system, which groups similar trips and consistently records spatial division, is superior to a model that works only with raw coordinates.

---

### Article 3 — Safikhani et al. (2020)

**Title:** Spatio-Temporal Modeling of Yellow Taxi Demands in New York City
Using Generalized STAR Models

This study estimates taxi demand using 2015 NYC Yellow Taxi data. The model used is the Generalized Spatio-Temporal Autoregressive (STAR) model. Taxi demand in a region varies not only according to the historical trend of that region but also according to the historical trend of neighboring regions. To model this spatial dependence, a weight matrix and the LASSO method for high-dimensional parameter estimation are used. The STAR model is superior to ARMA and VAR, which are purely time series models, in terms of accuracy and interpretability. The main finding of the article is that the inclusion of spatial neighborhood information in the model significantly improves the quality of the estimate.

**Relevance to This Project:**
This article provides the theoretical rationale for the neighbor-based spatial features (`neighbor_avg_demand`, k-ring average) that I use. The STAR model reveals that behavior in a region is shaped by its neighbors. Considering this spatial dependence improves prediction quality. My application calculates the average historical demand of k=1 cells surrounding each cell from the training set and adds it as model input.

---

### Article 4 — Poongodi et al. (2022)

**Title:** New York City Taxi Trip Duration Prediction Using MLP and XGBoost

This research focuses on predicting taxi ride times in New York City using machine learning methods, focusing on key factors such as departure and arrival points, travel distance, start time, and number of passengers. The performance of two different advanced models was examined. These are Extreme Gradient Boosting (XGBoost), a gradient boosting method that generates ordered decision trees and corrects residual errors, and Multilayer Perceptron (MLP), a backpropagation-optimized feedforward artificial neural network. Experimental findings show that both models are significantly more efficient than the standard linear regression method, and that XGBoost requires less computational cost and shorter training time than the MLP model, achieving a competitive Root Mean Square Logarithmic Error (RMSLE) score.

**Relevance to This Project:**
This study provides a suitable methodological benchmark for the advanced modeling phase. A major advantage is that the study offers benefits such as lower computational training costs and higher model comprehensibility. Furthermore, the widespread use of RMSLE as an evaluation criterion allows for direct comparison of my observations across different studies.

---

### Article 5 — Liu et al. (2022)

This study, using data from Chengdu, China, reveals the effects of topological structures and spatial partitioning accuracy on vehicle summoning request predictions. The researchers developed the model with a system that captures spatial structures using hexagonal units and addresses temporal dependencies using LSTM units. Performing tests on 36 different combinations of cell sizes and time periods, the study systematically examined the effects of square and hexagonal grids with the same surface area on the measurements. The findings observed that hexagonal partitions generally provided better performance than square partitions due to their equally spaced neighborhood structures, and that prediction accuracy decreased at over-sensitivities due to local variation loss.

**Relevance to This Project:**
This paper provides a solid foundation for my multilayer spatial topology experiments. The authors' key finding is that the size of the spatial analysis unit affects prediction accuracy beyond the choice of grid shape. This finding is consistent with both H3 and square grids exhibiting the most successful results at resolution 9. Furthermore, the performance degradation at over-sensitivities is consistent with the problems I encountered at resolution 10. The slight advantage observed for H3 at Resolution 8 compared to square grids is consistent with the finding that hexagonal grids reduce directional bias more effectively at small scales—findings that directly guided the scales used in the study.

---

## 6. H3 / DGGS Investigation

In theory, because all six neighbors are equidistant from the center, a hexagon is advantageous as a spatial analysis unit. This overcome the directional bias of square grids, where diagonal neighbors are ~1.41× farther than edge neighbors.

**Three resolution levels** were investigated:

| Resolution | Avg Cell Area | Scale | Cells Covering NYC |
|---|---|---|---|
| 8 | ~0.74 km² | Neighborhood | ~1,500 |
| **9** | **~0.11 km²** | **City block ← optimal** | **~8,000** |
| 10 | ~0.015 km² | Sub-block | ~50,000+ |

All spatial reference grid systems used in the project — both square and H3 hexagonal — were generated using the **QGIS tools and H3 Toolkit plugin.** Official NYC borough boundary polygon used to select the cells. This approach ensures that all five boroughs are covered, including Staten Island, which  was excluded by a simple bounding box filter.

Each trip was matched to its corresponding square grid and H3 cell via **spatial join** (`gpd.sjoin`), enabling a fair comparison for topologies at identical scales.

**Resolution 9** was chosen as the primary resolution — a scale fine enough to capture neighborhood-level demand variation, yet coarse enough to have statistically significant travel counts per cell per hour.
A heat map was created showing the travel density per cell H3 at resolution 9, animated throughout the hours of the day.

[Interactive Heatmaps](https://aakcaya.github.io/nyc-taxi-spatio-temporal/comparison_map.html) 

---

## 7. Preliminary Results — Baseline

> Analysis is based on 5 CSV files covering the first 5 days of January 2015 (January 1–5, 2015).

### Dataset Statistics

| Property | Value |
|---|---|
| Raw Record Count | 1,824,104 |
| Records After Cleaning & Spatial Filter | ~1,772,000 |
| Removed Record Ratio | ~2.9% |
| Training Set (80%) | ~1,417,600 records |
| Test Set (20%) | ~354,400 records |

### Baseline Model Results

**Model:** Multiple Linear Regression (OLS)  
*Consistent across all scales — spatial filtering has negligible effect on baseline.*

| Metric | Value |
|---|---|
| **RMSE** | 5.120 min (307.2 sec) | Error magnitude — penalizes large errors more heavily |
| **MAE** | 3.572 min (214.3 sec) | Typical absolute error |
| **R²** | 0.705 | Model explains 70.5% of the variance in trip duration |
| **RMSLE** | 0.474 | Log-scale error — evaluates short and long trips more equally |

### Model Coefficients

| Feature | Coefficient | Interpretation |
|---|---|---|
| `trip_distance` | +111.008 | Each mile added adds approximately 111 seconds. It's the strongest predictive factor. |
| `pickup_hour` | +1.815 | Travel duration increases slightly as the hour advances. |
| `day_of_week` | −8.132 | Later days of the week show marginally shorter trips. |
| `passenger_count` | +3.821 | Passenger count has negligible effect. |
| `is_rush_hour` | +1.387 | Weak independent effect due to overlap with `pickup_hour`. |
| `is_weekend` | −0.753 | Weekend effect is statistically negligible. |

### Interpretation

The basic model explains **70.5%** of the journey time variance using only 6 non-spatial features. The dominant predictor is `trip_distance`, and the main limitation of the model is that it is **completely devoid of spatial context**. The near-zero contribution of the `is_rush_hour` feature reveals that linear regression fails to capture the interaction effects between features; for example, the combined effect of peak hours and high-density passenger pickup areas such as Midtown Manhattan. This finding directly encourages the use of H3-based spatial features and advanced models in the later phases of the project.

---

## 8. Advanced Method: XGBoost + Spatial Grid Features

### Model: XGBoost Regressor

XGBoost builds decision trees **sequentially**. New tree corrects the residual errors of the previous ones. XGBoost learns different rules for different parts of the feature space, unlike linear regression, which applies a single global formula. Advanced method includes spatial context.

### Pipeline

#### Step 1 — Spatial Matching via QGIS Grids

Each trip is matched to both its square grid cell and H3 cell via spatial join.

This replaces coordinate-based cell assignment and ensures geometric correctness — particularly at official borough boundaries.

#### Step 2 — Cell-Level Statistics (Training Set Only)

Statistics are computed from the training set to prevent data leakage.

Unmatched test cells receive the global training mean duration and zero trip count.

#### Step 3 — Training

```python
xgb_model = xgb.XGBRegressor(
    n_estimators  = 300,
    learning_rate = 0.07,
    max_depth     = 6,
    random_state  = 42,
    n_jobs        = -1
)
```

The same parameters are applied identically to both square grid and H3 variants, ensuring the fairness of comparison.

---

## 9. Cell Topologies & Scale Analysis

### Full Results Table

| Scale | Model | MAE (min) | RMSE (min) | RMSLE | R² |
|---|---|---|---|---|---|
| **Large Cells (Res 8)** | Baseline (Linear) | 3.573 | 5.120 | 0.474 | 0.705 |
| **Large Cells (Res 8)** | XGBoost + H3 | 3.166 | 5.365 | 0.370 | 0.676 |
| **Large Cells (Res 8)** | XGBoost + Square | 3.172 | 5.386 | 0.368 | 0.674 |
| **Medium Cells (Res 9)** | Baseline (Linear) | 3.572 | 5.119 | 0.474 | 0.705 |
| **Medium Cells (Res 9)** | XGBoost + H3 | **3.165** | 5.363 | 0.370 | 0.676 |
| **Medium Cells (Res 9)** | XGBoost + Square | **3.160** | 5.359 | **0.367** | **0.677** |
| **Small Cells (Res 10)** | Baseline (Linear) | 3.573 | 5.119 | 0.474 | 0.705 |
| **Small Cells (Res 10)** | XGBoost + H3 | 3.189 | 5.404 | 0.371 | 0.672 |
| **Small Cells (Res 10)** | XGBoost + Square | 3.186 | 5.405 | 0.370 | 0.671 |

### Key Findings

**1. Granularity matters more than topology.**

The most important factor is cell size. The best performance values ​​for both H3 and square grids were observed at resolution 9. Switching to smaller cells (resolution 10) reduced performance. This is due to the noiseier values ​​caused by fewer travels within the cells. This finding is consistent with Liu et al. (2022).

**2. H3 and square grids are essentially equivalent at the optimal scale.**

At Resolution 9, the difference in MAE between H3 (3,165 min) and Square (3,160 min) is **0.005 min = 0.3 seconds** – within the margin of noise. Practically, results at this scale are negligible for both topologies.

**3. At coarser scale (Res 8), H3 has a slight edge.**

H3 performed better than squared at Resolution 8 in terms of both RMSE (5.365 vs. 5.386) and MAE (3.166 vs. 3.172). This small performance improvement is related to the fact that the hexagonal cells have equivalent neighbor distances, which allows for more efficient operation against the high variance between neighboring units that occurs in larger units.

**4. All XGBoost variants consistently improve MAE and RMSLE over baseline.**

In all six configurations where the advanced method was applied, the MAE value improved by approximately **11-12%**, while the RMSLE improved by approximately **22-24%**. This is true for both topologies and three resolutions.

**5. RMSE remains above baseline across all configurations.**

RMSE is significantly affected by outliers due to the quadratic contribution. In this context, modeling cells containing measurements close to outliers presents a challenge in the model developed using 5 days of data. This is more related to data volume than a weakness of the model.

### Improvement Summary (vs. Baseline at Res 9)

| Metric | Baseline | XGBoost + H3 (Res 9) | XGBoost + Square (Res 9) |
|---|---|---|---|
| **MAE** | 3.572 min | 3.165 min (**−11.4%**) | 3.160 min (**−11.5%**) |
| **RMSE** | 5.119 min | 5.363 min (+4.8%) | 5.359 min (+4.7%) |
| **RMSLE** | 0.474 | 0.370 (**−21.9%**) | 0.367 (**−22.6%**) |
| **R²** | 0.705 | 0.676 | 0.677 |

### Conclusion

In topology analysis conducted at different scales, measurements taken at the same resolutions to observe the relationship between the use of H3 and square grid cells revealed that cell topology selection did not create a significant difference, but resolution change was critical. The most ideal resolution observed in the measurements for this dataset was determined to be 9. This resolution level both captures demand variation at a sufficient scale and provides geographic specificity. Increasing the dataset size will also improve RMSE values, leading to a more stable result.

---

## 10. Development Journey

This section documents the challenges encountered, corrections, and methodological improvements made across the development process.

---

### Phase 1 — Baseline

| Property | Value |
|---|---|
| Model | Multiple Linear Regression (OLS) |
| Features | 6 non-spatial features |
| MAE | **3.573 min** |
| RMSE | **5.120 min** |
| R² | 0.705 |

**Key findings:**
- `trip_distance` dominates — all other features contribute marginally
- `is_rush_hour` coefficient ≈ +1.4 sec — near zero, overlaps with `pickup_hour`
- Model assigns identical prediction to "2 miles at 8 AM in Midtown" and "2 miles at 8 AM in Staten Island" — **lacking spatial depth**

---

### Phase 2a — XGBoost, No Early Stopping 

**Change:** H3 cell statistics added as features. XGBoost trained for 300 iterations.

**Training log revealed overfitting:**

| Iteration | Validation RMSE |
|---|---|
| 0 | 541.5 sec |
| 100 | **319.3 sec** ← best point |
| 200 | 322.8 sec ← degrading |
| 299 | 324.1 sec ← final model |

**Result vs. baseline:**

| Metric | Baseline | XGBoost v1 | Δ |
|---|---|---|---|
| MAE | 3.573 min | 3.200 min |  −10.4% |
| RMSE | 5.120 min | 5.400 min |  +5.5% |

- MAE improved — typical predictions better
- RMSE degraded — outlier trips inflated after overfitting past iter 100
- **Root cause:** no `early_stopping_rounds` → model continued 200 iterations past its optimum

---

### Phase 2b — Early Stopping Added 

**Change:** `early_stopping_rounds=20` added to XGBoost.

**Training log:**

| Iteration | Validation RMSE |
|---|---|
| 0 | 541.5 sec |
| **69** | **313.9 sec** ← stopped automatically |

**Result vs. baseline:**

| Metric | Baseline | XGBoost v1 | XGBoost v2 | Δ (v1→v2) |
|---|---|---|---|---|
| MAE | 3.573 min | 3.200 min | **2.970 min** | −7.2% |
| RMSE | 5.120 min | 5.400 min | **5.160 min** | −4.4% |
| RMSLE | 0.4741 | 0.3758 | **0.3359** | −10.6% |

- Overfitting resolved — 230 unnecessary iterations eliminated
- MAE improvement over baseline jumped from 10.4% → **17.0%**
- RMSLE improvement: **29.1%** over baseline

---

### Phase 3 — Spatial Coverage Fix 

**Problem discovered:** Bounding box filter `pickup_longitude >= −74.25` cut off western Staten Island (coordinates extend to ≈ −74.26°).
**Fix applied:**
- Replaced bounding box with `gpd.sjoin(gdf_taxi, nyc_boundary, predicate="within")`
- All 5 boroughs now fully covered
- Minimal effect on record count (~120 additional records retained)

---

### Phase 4 — QGIS Grid Integration 

**Problem discovered:** Generating H3 cells from trip coordinates produced cells only where trips existed — creating an irregular, gap-filled grid in low-density areas.
**Fix applied:**
- H3 grids (res 8, 9, 10) generated in **QGIS H3 Toolkit** from official NYC boundary polygon
- Square grids (res 8, 9, 10) generated in **QGIS Grid Tool** at equivalent cell areas
- All trips matched via spatial join — not coordinate-based assignment
- Result: 6 complete, gap-free reference grids for fair cross-topology comparison (3 for square grid and 3 for H3 cells)

---

### Phase 5 — Multi-Scale Topology Analysis 

**Setup:** 3 resolutions × 2 topologies × 1 baseline = **9 model configurations** tested.

**Full results:**

| Scale | Topology | MAE (min) | RMSE (min) | RMSLE | R² |
|---|---|---|---|---|---|
| Res 8 — Large | Baseline | 3.573 | 5.120 | 0.474 | 0.705 |
| Res 8 — Large | H3 | 3.166 | 5.365 | 0.370 | 0.676 |
| Res 8 — Large | Square | 3.172 | 5.386 | 0.368 | 0.674 |
| Res 9 — Medium | Baseline | 3.572 | 5.119 | 0.474 | 0.705 |
| Res 9 — Medium | **H3** | **3.165** | 5.363 | 0.370 | 0.676 |
| Res 9 — Medium | **Square** | **3.160** | 5.359 | **0.367** | **0.677** |
| Res 10 — Small | Baseline | 3.573 | 5.119 | 0.474 | 0.705 |
| Res 10 — Small | H3 | 3.189 | 5.404 | 0.371 | 0.672 |
| Res 10 — Small | Square | 3.186 | 5.405 | 0.370 | 0.671 |

**Key findings:**
- **Granularity > Topology** — cell size is the dominant factor, not cell shape
- **Res 9 is optimal** — best MAE at both topologies; Res 10 degrades due to data sparsity
- **H3 ≈ Square at Res 9** — difference is 0.005 min = 0.3 seconds (noise level)
- **H3 slightly better at Res 8** — marginal advantage at coarser scale where equidistant neighbors matter more
- **RMSE above baseline across all configs** — consistent with 5-day data limitation; outlier cells remain data-sparse

---

### Summary of Iterations

| Phase | Key Change | MAE Result | RMSE Result |
|---|---|---|---|
| 1 — Baseline | Linear Regression, no spatial features | 3.573 min | 5.120 min |
| 2a — First XGBoost | H3 features, no early stopping | 3.200 min | 5.400 min  overfit |
| 2b — Early stopping added | `early_stopping_rounds=20` | 2.970 min | 5.160 min |
| 3 — Boundary fix | Spatial join replaces bounding box | — | Full coverage |
| 4 — QGIS grids | Complete city coverage, both topologies | — | Grid integrity |
| 5 — Multi-scale analysis | Res 8 / 9 / 10, H3 + Square | **3.165 min** (Res 9 H3) | 5.363 min |

---

### Requirements

```
pandas
numpy
scikit-learn
xgboost
h3
geopandas
folium
shapely
```

---

## 11. References

1. **Roy, B., & Rout, D.** (2022). Predicting Taxi Travel Time Using Machine Learning Techniques Considering Weekend and Holidays. *Lecture Notes in Networks and Systems*, 417, 258–267. https://doi.org/10.1007/978-3-030-96302-6_24

2. **Wang, H., Tang, X., Kuo, Y.-H., Kifer, D., & Li, Z.** (2019). A Simple Baseline for Travel Time Estimation using Large-scale Trip Data. *ACM Transactions on Intelligent Systems and Technology*, 10(2), 1–22. https://doi.org/10.1145/3293317

3. **Safikhani, A., Kamga, C., Mudigonda, S., Faghih, S. S., & Moghimi, B.** (2020). Spatio-temporal modeling of yellow taxi demands in New York City using generalized STAR models. *International Journal of Forecasting*, 36(3), 1138–1148. https://doi.org/10.1016/j.ijforecast.2018.10.001

4. **Poongodi, M., Malviya, M., Kumar, C., Hamdi, M., Vijayakumar, V., Nebhen, J., & Alyamani, H.** (2022). New York City taxi trip duration prediction using MLP and XGBoost. *International Journal of System Assurance Engineering and Management*, 13(Suppl 1), 16–27. https://doi.org/10.1007/s13198-021-01130-x

5. **Liu, K., Chen, Z., Yamamoto, T., & Tuo, L.** (2022). Exploring the Impact of Spatiotemporal Granularity on the Demand Prediction of Dynamic Ride-Hailing. *IEEE Transactions on Intelligent Transportation Systems*, 24, 104–114. https://doi.org/10.1109/TITS.2022.3216016

---

*Final Project | DI 722 Spatio-Temporal Data Mining | 2025-26 Spring | Final Presentation: 12 June 2026*
