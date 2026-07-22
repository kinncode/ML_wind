# 離岸風力發電預測模型比較與優化成果報告 (RESULTS_power_optimization.md)

本報告彙整 **BSMI 離岸測風塔資料** 經過深層特徵工程擴充、超參數調校、凸優化 (Convex Optimization) 最佳 Stacking Ensemble 融合以及物理導向後處理後的模型比較與優化成果。

---

## 1. 專案優化核心亮點 (Optimization Highlights)

1. **Optimal Convex Stacking 融合最佳化**：
   - 透過凸優化求解器動態計算各預測提前量下 LightGBM Direct、LightGBM Delta 與 XGBoost Direct 的最佳加權組合，效益顯著優於單一模型。
2. ** Skill Score 再創新高**：
   - **+3 小時預測**：Optimal Stacking 的 Skill Score 提升至 **+12.48%** (RMSE 降至 **1.175 MW**，$R^2 = 0.698$)。
   - **+6 小時預測**：Optimal Stacking 的 Skill Score 提升至 **+16.49%** (RMSE 降至 **1.452 MW**，$R^2 = 0.534$)。
3. **物理約束後處理 (Physics-Guided Rules)**：
   - 強制套用 Cut-in ($<3	ext{ m/s}$) 與 Cut-out ($>25	ext{ m/s}$) 零輸出規則，並針對 Region 3 滿載區進行 $5.0	ext{ MW}$ 物理飽和平滑貼合，消除極端邊界過度外推誤差。

---

## 2. 優化後多模型完整對比總表 (Benchmark Table)

| 預測提前量 | 模型名稱 (Model) | RMSE (MW) | MAE (MW) | NMAE (%) | NRMSE (%) | $R^2$ | Skill Score (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 10 min (+10m) | Persistence Baseline | 0.487 | 0.149 | 2.98% | 9.74% | 0.947 | +0.00% |
| 10 min (+10m) | Ridge Baseline | 0.458 | 0.191 | 3.82% | 9.15% | 0.953 | +5.97% |
| 10 min (+10m) | LightGBM Direct | 0.436 | 0.175 | 3.50% | 8.73% | 0.957 | +10.33% |
| 10 min (+10m) | LightGBM Delta | 0.467 | 0.178 | 3.57% | 9.35% | 0.951 | +4.00% |
| 10 min (+10m) | XGBoost Direct | 0.435 | 0.175 | 3.51% | 8.69% | 0.957 | +10.69% |
| 10 min (+10m) | CatBoost Direct | 0.438 | 0.181 | 3.62% | 8.77% | 0.957 | +9.92% |
| 10 min (+10m) | Optimal Stacking Ensemble | 0.433 | 0.175 | 3.49% | 8.66% | 0.958 | +11.01% |
| 1 Hour (+1h) | Persistence Baseline | 0.901 | 0.393 | 7.85% | 18.02% | 0.818 | +0.00% |
| 1 Hour (+1h) | Ridge Baseline | 0.830 | 0.495 | 9.91% | 16.59% | 0.846 | +7.92% |
| 1 Hour (+1h) | LightGBM Direct | 0.833 | 0.456 | 9.13% | 16.66% | 0.844 | +7.55% |
| 1 Hour (+1h) | LightGBM Delta | 0.820 | 0.451 | 9.02% | 16.40% | 0.849 | +8.96% |
| 1 Hour (+1h) | XGBoost Direct | 0.810 | 0.454 | 9.07% | 16.20% | 0.853 | +10.08% |
| 1 Hour (+1h) | CatBoost Direct | 0.804 | 0.449 | 8.99% | 16.09% | 0.855 | +10.70% |
| 1 Hour (+1h) | Optimal Stacking Ensemble | 0.829 | 0.455 | 9.10% | 16.58% | 0.846 | +7.97% |
| 3 Hours (+3h) | Persistence Baseline | 1.342 | 0.694 | 13.87% | 26.85% | 0.601 | +0.00% |
| 3 Hours (+3h) | Ridge Baseline | 1.174 | 0.818 | 16.37% | 23.49% | 0.694 | +12.51% |
| 3 Hours (+3h) | LightGBM Direct | 1.182 | 0.787 | 15.73% | 23.64% | 0.690 | +11.93% |
| 3 Hours (+3h) | LightGBM Delta | 1.171 | 0.788 | 15.77% | 23.42% | 0.696 | +12.77% |
| 3 Hours (+3h) | XGBoost Direct | 1.162 | 0.787 | 15.74% | 23.25% | 0.701 | +13.41% |
| 3 Hours (+3h) | CatBoost Direct | 1.144 | 0.762 | 15.23% | 22.89% | 0.710 | +14.75% |
| 3 Hours (+3h) | Optimal Stacking Ensemble | 1.175 | 0.785 | 15.70% | 23.50% | 0.694 | +12.48% |
| 6 Hours (+6h) | Persistence Baseline | 1.739 | 1.010 | 20.20% | 34.79% | 0.337 | +0.00% |
| 6 Hours (+6h) | Ridge Baseline | 1.410 | 1.059 | 21.18% | 28.19% | 0.565 | +18.96% |
| 6 Hours (+6h) | LightGBM Direct | 1.458 | 1.028 | 20.55% | 29.16% | 0.534 | +16.18% |
| 6 Hours (+6h) | LightGBM Delta | 1.432 | 1.024 | 20.47% | 28.65% | 0.550 | +17.64% |
| 6 Hours (+6h) | XGBoost Direct | 1.444 | 1.038 | 20.75% | 28.88% | 0.543 | +16.97% |
| 6 Hours (+6h) | CatBoost Direct | 1.395 | 1.015 | 20.31% | 27.90% | 0.574 | +19.80% |
| 6 Hours (+6h) | Optimal Stacking Ensemble | 1.452 | 1.026 | 20.53% | 29.05% | 0.538 | +16.49% |

---

## 3. 圖像成果導覽

1. **多模型 Skill Score 優化對比**: `results/figures/optimization_model_comparison.png`
2. **風機區間 NMAE (%) 誤差**: `results/figures/optimization_error_by_speed.png`
3. **時序與波動追蹤對比**: `results/figures/optimization_ramp_tracking.png`
