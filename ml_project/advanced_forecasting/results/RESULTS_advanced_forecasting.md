# Advanced Wind Power Forecasting Methods - Comparison Report

Three advanced forecasting approaches applied to BSMI offshore met mast data (2016-2021),
using NREL 5MW reference turbine power simulation, split by year (Train: 2016-2018, Test: 2020-2021).

---

## 1. Method Overview

| Method | Algorithm | Output | Use Case |
|---|---|---|---|
| **Interval** | Conformal Prediction + LightGBM | Center + 90% CI | Reserve capacity planning |
| **Quantile** | LightGBM Quantile Regression | P10 / P50 / P90 | Risk management |
| **Trajectory** | Recursive Multi-Step LightGBM | 144-step power trajectory | Grid scheduling |

---

## 2. RMSE Comparison

| Horizon | Interval RMSE | Quantile P50 RMSE | Recursive RMSE | Direct RMSE |
|---|---|---|---|---|
| 10min | 0.464 MW | 0.440 MW | 0.429 MW | 0.426 MW |
| 1h | 0.849 MW | 0.818 MW | 0.935 MW | 0.813 MW |
| 6h | 1.535 MW | 1.442 MW | 1.813 MW | 1.433 MW |
| 24h | 2.264 MW | 2.178 MW | 2.266 MW | 2.076 MW |

---

## 3. Interval Forecast Results

| Horizon | RMSE | R2 | Coverage | Avg Width | Winkler | Skill |
|---|---|---|---|---|---|---|
| 10min | 0.464 | 0.952 | 88.6% | 0.541 | 2.140 | +2.05% |
| 1h | 0.849 | 0.839 | 88.6% | 1.704 | 3.818 | +3.65% |
| 6h | 1.535 | 0.482 | 85.3% | 3.310 | 5.764 | +9.96% |
| 24h | 2.264 | -0.128 | 83.5% | 4.517 | 6.295 | -3.96% |

---

## 4. Quantile Forecast Results

| Horizon | P50 RMSE | P50 R2 | CRPS | Sharpness | Skill |
|---|---|---|---|---|---|
| 10min | 0.440 | 0.957 | 0.0807 | 1.524 | +6.94% |
| 1h | 0.818 | 0.851 | 0.1343 | 1.525 | +7.19% |
| 6h | 1.442 | 0.542 | 0.2946 | 2.379 | +15.39% |
| 24h | 2.178 | -0.044 | 0.4711 | 4.296 | -0.03% |

---

## 5. Trajectory Forecast Results

| Horizon | Recursive RMSE | Recursive R2 | Direct RMSE | Direct R2 |
|---|---|---|---|---|
| 10min | 0.429 | 0.955 | 0.426 | 0.959 |
| 1h | 0.935 | 0.786 | 0.813 | 0.853 |
| 6h | 1.813 | 0.149 | 1.433 | 0.548 |
| 24h | 2.266 | -0.275 | 2.076 | 0.051 |

---

## 6. Figure Gallery

### Interval Forecast
1. `interval_timeseries_ci.png` - Time series with 90% CI band
2. `interval_coverage_width.png` - Coverage and width by horizon
3. `interval_width_boxplot.png` - Width distribution boxplot
4. `interval_power_vs_energy.png` - Power vs cumulative energy intervals

### Quantile Forecast
5. `quantile_fan_chart.png` - P10/P50/P90 fan chart
6. `quantile_reliability.png` - Reliability diagram
7. `quantile_pinball_loss.png` - Pinball loss by horizon
8. `quantile_crps.png` - CRPS by horizon
9. `quantile_sharpness.png` - Sharpness (P90-P10) boxplot

### Trajectory Forecast
10. `trajectory_timeseries.png` - 24h trajectory examples
11. `trajectory_rmse_decay.png` - RMSE decay vs Direct anchors
12. `trajectory_r2_decay.png` - R2 decay curve
13. `trajectory_smoothness.png` - Step jump distribution

### Comparison
14. `compare_radar.png` - Multi-dimension radar chart
15. `compare_rmse_bars.png` - RMSE bar comparison
16. `compare_uncertainty.png` - Uncertainty quantification comparison
