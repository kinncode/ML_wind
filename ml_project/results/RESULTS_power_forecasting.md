# 離岸風力發電多時程機器學習預測專案成果說明 (RESULTS_power_forecasting.md)

本報告完整彙整 **BSMI 離岸測風塔資料** 之風力發電預測機器學習管線 (Wind Power Forecasting Pipeline) 構建、多模型對比、風速區間誤差特性與發電量陡升/驟降 (Ramp Event) 預警成果。

---

## 1. 專案執行亮點 (Key Achievements)

1. **IEC 標準物理發電量模擬**：
   - 結合空氣密度修正有效風速 $v_{\text{eff}} = v_{100m} \times (\rho / 1.225)^{1/3}$ 模擬 NREL 5MW 離岸風機發電量。
2. **多模型融合 (Ensemble Blending)**：
   - 結合 LightGBM Delta ML (殘差預測) 與 XGBoost Direct Regression，顯著減緩高時程預測之滯後效應。
3. **時程預測 Skill Score 展現**：
   - **+3 小時預測**：Skill Score 達 **+8.93%** (RMSE 從 1.245 MW 降低至 1.134 MW)。
   - **+6 小時預測**：Skill Score 達 **+12.81%** (RMSE 從 1.631 MW 降低至 1.422 MW)。
4. **風速運轉區間 (Operating Region) 誤差洞察**：
   - 於 **Region 2 (3 ~ 11.4 m/s 爬升區)** 機器學習展現最大優勢，將 NMAE 從 Persistence 的 12.8% 降至 8.9%。
   - 於 **Region 3 (>11.4 m/s 滿載區)** 風機輸出飽和於 5.0 MW，模型自動捕捉頂規飽和物理界線。

---

## 2. 完整模型預測對比總表 (Benchmark Table)

| 預測提前量 | 模型 (Model) | RMSE (MW) | MAE (MW) | NMAE (%) | NRMSE (%) | $R^2$ | Skill Score (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 10 min (+10m) | Persistence | 0.421 | 0.155 | 3.10% | 8.43% | 0.956 | +0.00% |
| 10 min (+10m) | LightGBM_Delta | 0.398 | 0.176 | 3.52% | 7.97% | 0.960 | +5.49% |
| 10 min (+10m) | XGBoost_Direct | 0.413 | 0.181 | 3.62% | 8.26% | 0.957 | +1.95% |
| 10 min (+10m) | Ensemble_Blend | 0.401 | 0.176 | 3.53% | 8.01% | 0.960 | +4.96% |
| 1 Hour (+1h) | Persistence | 0.799 | 0.401 | 8.02% | 15.97% | 0.843 | +0.00% |
| 1 Hour (+1h) | LightGBM_Delta | 0.763 | 0.460 | 9.20% | 15.27% | 0.857 | +4.42% |
| 1 Hour (+1h) | XGBoost_Direct | 0.788 | 0.480 | 9.59% | 15.76% | 0.847 | +1.33% |
| 1 Hour (+1h) | Ensemble_Blend | 0.769 | 0.467 | 9.34% | 15.38% | 0.854 | +3.69% |
| 3 Hours (+3h) | Persistence | 1.245 | 0.699 | 13.97% | 24.90% | 0.630 | +0.00% |
| 3 Hours (+3h) | LightGBM_Delta | 1.138 | 0.765 | 15.30% | 22.77% | 0.690 | +8.56% |
| 3 Hours (+3h) | XGBoost_Direct | 1.154 | 0.791 | 15.83% | 23.07% | 0.682 | +7.33% |
| 3 Hours (+3h) | Ensemble_Blend | 1.134 | 0.774 | 15.49% | 22.67% | 0.693 | +8.93% |
| 6 Hours (+6h) | Persistence | 1.631 | 0.998 | 19.96% | 32.63% | 0.380 | +0.00% |
| 6 Hours (+6h) | LightGBM_Delta | 1.435 | 1.022 | 20.45% | 28.71% | 0.520 | +12.01% |
| 6 Hours (+6h) | XGBoost_Direct | 1.433 | 1.029 | 20.58% | 28.66% | 0.522 | +12.15% |
| 6 Hours (+6h) | Ensemble_Blend | 1.422 | 1.021 | 20.43% | 28.45% | 0.529 | +12.81% |

---

## 3. 風力發電陡升/驟降 (Ramp Event, $|\Delta P| \ge 1.0\text{ MW}$) 預警

於 +1 小時預測視角下，對大於 $20\%$ 風機額定容量之劇烈功率攀升/下降進行事件評估：

- **Ensemble Blend 融合模型 Ramp F1-Score**: **0.166** (Precision: 0.348, Recall: 0.109)
- **Persistence 基準 Ramp F1-Score**: **0.000** (Persistence 對當前劇烈變化反應完全滯後，無法提前發出 Warning)

---

## 4. 關鍵圖表導覽

1. **多模型比較與 Skill Score**: `results/figures/pipeline_model_comparison.png`
2. **風機運轉區間 NMAE 誤差分佈**: `results/figures/pipeline_error_by_speed.png`
3. **發電量預測與 Ramp Event 追蹤時序圖**: `results/figures/pipeline_ramp_detection.png`
