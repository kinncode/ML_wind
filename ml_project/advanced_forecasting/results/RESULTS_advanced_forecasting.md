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
| 10min | 0.350 MW | 0.376 MW | 0.339 MW | 0.350 MW |
| 1h | 0.650 MW | 0.684 MW | 0.775 MW | 0.648 MW |
| 6h | 1.145 MW | 1.200 MW | 1.719 MW | 1.141 MW |
| 24h | 1.600 MW | 1.674 MW | 1.996 MW | 1.593 MW |

---

## 3. Interval Forecast Results

| Horizon | RMSE | R2 | Coverage | Avg Width | Winkler | Skill |
|---|---|---|---|---|---|---|
| 10min | 0.350 | 0.972 | 91.6% | 0.554 | 1.519 | +13.97% |
| 1h | 0.650 | 0.904 | 92.9% | 1.666 | 2.815 | +14.90% |
| 6h | 1.145 | 0.705 | 92.5% | 3.159 | 4.425 | +27.14% |
| 24h | 1.600 | 0.421 | 92.5% | 4.206 | 5.078 | +18.73% |

---

## 4. Quantile Forecast Results

| Horizon | P50 RMSE | P50 R2 | CRPS | Sharpness | Skill |
|---|---|---|---|---|---|
| 10min | 0.376 | 0.968 | 0.0740 | 1.479 | +7.50% |
| 1h | 0.684 | 0.893 | 0.1130 | 1.581 | +10.48% |
| 6h | 1.200 | 0.676 | 0.2104 | 2.343 | +23.64% |
| 24h | 1.674 | 0.366 | 0.3434 | 4.707 | +14.98% |

---

## 5. Trajectory Forecast Results

| Horizon | Recursive RMSE | Recursive R2 | Direct RMSE | Direct R2 |
|---|---|---|---|---|
| 10min | 0.339 | 0.975 | 0.350 | 0.972 |
| 1h | 0.775 | 0.864 | 0.648 | 0.904 |
| 6h | 1.719 | 0.317 | 1.141 | 0.707 |
| 24h | 1.996 | 0.101 | 1.593 | 0.426 |

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
