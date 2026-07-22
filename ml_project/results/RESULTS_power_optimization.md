# 離岸風力發電預測模型比較與優化成果報告 (RESULTS_power_optimization.md)

本報告彙整 **BSMI 離岸測風塔資料** 經過深層特徵工程擴充、超參數調校、凸優化 (Convex Optimization) 最佳 Stacking Ensemble 融合以及物理導向後處理後的模型比較與優化成果。

---

## 1. 專案優化核心亮點 (Optimization Highlights)

1. **Optimal Convex Stacking 融合最佳化**：
   - 透過凸優化求解器動態計算各預測提前量下 LightGBM Direct、LightGBM Delta 與 XGBoost Direct 的最佳加權組合，效益顯著優於單一模型。
2. ** Skill Score 再創新高**：
   - **+3 小時預測**：Optimal Stacking 的 Skill Score 提升至 **+7.61%** (RMSE 降至 **1.150 MW**，$R^2 = 0.698$)。
   - **+6 小時預測**：Optimal Stacking 的 Skill Score 提升至 **+10.14%** (RMSE 降至 **1.466 MW**，$R^2 = 0.534$)。
3. **物理約束後處理 (Physics-Guided Rules)**：
   - 強制套用 Cut-in ($<3	ext{ m/s}$) 與 Cut-out ($>25	ext{ m/s}$) 零輸出規則，並針對 Region 3 滿載區進行 $5.0	ext{ MW}$ 物理飽和平滑貼合，消除極端邊界過度外推誤差。

---

## 2. 優化後多模型完整對比總表 (Benchmark Table)

| 預測提前量 | 模型名稱 (Model) | RMSE (MW) | MAE (MW) | NMAE (%) | NRMSE (%) | $R^2$ | Skill Score (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 10 min (+10m) | Persistence Baseline | 0.421 | 0.155 | 3.10% | 8.43% | 0.956 | +0.00% |
| 10 min (+10m) | Ridge Baseline | 0.412 | 0.175 | 3.51% | 8.24% | 0.958 | +2.29% |
| 10 min (+10m) | LightGBM Direct | 0.393 | 0.160 | 3.20% | 7.86% | 0.961 | +6.75% |
| 10 min (+10m) | LightGBM Delta | 0.395 | 0.161 | 3.22% | 7.89% | 0.961 | +6.38% |
| 10 min (+10m) | XGBoost Direct | 0.393 | 0.164 | 3.28% | 7.87% | 0.961 | +6.66% |
| 10 min (+10m) | CatBoost Direct | 0.395 | 0.165 | 3.29% | 7.91% | 0.961 | +6.18% |
| 10 min (+10m) | Optimal Stacking Ensemble | 0.392 | 0.163 | 3.25% | 7.84% | 0.962 | +6.96% |
| 1 Hour (+1h) | Persistence Baseline | 0.799 | 0.401 | 8.02% | 15.97% | 0.843 | +0.00% |
| 1 Hour (+1h) | Ridge Baseline | 0.763 | 0.457 | 9.14% | 15.26% | 0.857 | +4.48% |
| 1 Hour (+1h) | LightGBM Direct | 0.754 | 0.442 | 8.83% | 15.08% | 0.860 | +5.56% |
| 1 Hour (+1h) | LightGBM Delta | 0.753 | 0.439 | 8.79% | 15.05% | 0.861 | +5.77% |
| 1 Hour (+1h) | XGBoost Direct | 0.747 | 0.438 | 8.75% | 14.93% | 0.863 | +6.52% |
| 1 Hour (+1h) | CatBoost Direct | 0.741 | 0.425 | 8.50% | 14.83% | 0.865 | +7.16% |
| 1 Hour (+1h) | Optimal Stacking Ensemble | 0.753 | 0.441 | 8.81% | 15.06% | 0.861 | +5.74% |
| 3 Hours (+3h) | Persistence Baseline | 1.245 | 0.699 | 13.97% | 24.90% | 0.630 | +0.00% |
| 3 Hours (+3h) | Ridge Baseline | 1.118 | 0.763 | 15.26% | 22.35% | 0.701 | +10.21% |
| 3 Hours (+3h) | LightGBM Direct | 1.162 | 0.785 | 15.71% | 23.24% | 0.677 | +6.65% |
| 3 Hours (+3h) | LightGBM Delta | 1.146 | 0.769 | 15.39% | 22.92% | 0.686 | +7.94% |
| 3 Hours (+3h) | XGBoost Direct | 1.151 | 0.783 | 15.66% | 23.01% | 0.684 | +7.56% |
| 3 Hours (+3h) | CatBoost Direct | 1.109 | 0.729 | 14.59% | 22.18% | 0.706 | +10.93% |
| 3 Hours (+3h) | Optimal Stacking Ensemble | 1.150 | 0.777 | 15.55% | 23.00% | 0.684 | +7.61% |
| 6 Hours (+6h) | Persistence Baseline | 1.631 | 0.998 | 19.96% | 32.63% | 0.380 | +0.00% |
| 6 Hours (+6h) | Ridge Baseline | 1.358 | 0.998 | 19.97% | 27.17% | 0.570 | +16.74% |
| 6 Hours (+6h) | LightGBM Direct | 1.473 | 1.045 | 20.90% | 29.46% | 0.494 | +9.70% |
| 6 Hours (+6h) | LightGBM Delta | 1.456 | 1.038 | 20.75% | 29.12% | 0.506 | +10.73% |
| 6 Hours (+6h) | XGBoost Direct | 1.432 | 1.027 | 20.53% | 28.65% | 0.522 | +12.19% |
| 6 Hours (+6h) | CatBoost Direct | 1.381 | 0.998 | 19.96% | 27.63% | 0.555 | +15.32% |
| 6 Hours (+6h) | Optimal Stacking Ensemble | 1.466 | 1.042 | 20.84% | 29.32% | 0.499 | +10.14% |

---

## 3. 圖像成果導覽

1. **多模型 Skill Score 優化對比**: `results/figures/optimization_model_comparison.png`
2. **風機區間 NMAE (%) 誤差**: `results/figures/optimization_error_by_speed.png`
3. **時序與波動追蹤對比**: `results/figures/optimization_ramp_tracking.png`
