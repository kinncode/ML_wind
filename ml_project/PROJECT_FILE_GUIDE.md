# BSMI 離岸風場機器學習與發電量預測專案 — 檔案與程式指南 (Project File Guide)

本文件完整說明 **BSMI 離岸測風塔機器學習專案** 的程式碼位置、資料集結構、模型管線檔案與產出報告，幫助您快速定位所有程式與成果。

---

## 📁 1. 核心 Python 程式碼 (Python Scripts)

所有機器學習與資料處理程式皆位於 `d:/wind_d/ML_wind/ml_project/` 目錄下：

| 程式檔名 (Script Path) | 核心功能與說明 |
| :--- | :--- |
| 🚀 **[train_power_forecast_pipeline.py](file:///d:/wind_d/ML_wind/ml_project/train_power_forecast_pipeline.py)** | **風力發電預測主管線 (Main ML Pipeline)**<br>• 一鍵執行 NREL 5MW 風機物理發電量模擬與空氣密度修正<br>• 建立多時程標的 ($t+10\text{min}, +1\text{h}, +3\text{h}, +6\text{h}$)<br>• 訓練 LightGBM Delta ML、XGBoost Direct 與 Ensemble 融合模型<br>• 評估發電量陡升/驟降 (Ramp Event) 預警與風速區間誤差 |
| 📊 **[explore_power_suitability.py](file:///d:/wind_d/ML_wind/ml_project/explore_power_suitability.py)** | **風力發電特徵可行性與相關性驗證**<br>• 計算 Pearson 線性相關與 Spearman 秩相關（三次方功率曲線關聯）<br>• 分析 LightGBM 特徵 Gain % 重要性隨預測時程動態轉移 |
| ⚙️ **[preprocess.py](file:///d:/wind_d/ML_wind/ml_project/preprocess.py)** | **原始資料預處理與 IEC 降採樣器**<br>• 將 14 GB、1.6 億筆 1Hz 逐秒原始觀測檔清洗並降採樣為 10 分鐘統計格<br>• 計算 100m/69m/38m 風速平均/標準差/陣風/湍流強度/風切 $\alpha$/空氣密度 |
| 🌪️ **[extract_turbulence.py](file:///d:/wind_d/ML_wind/ml_project/extract_turbulence.py)** | **湍流與流場特徵萃取器**<br>• 從逐秒資料計算 10 分鐘湍流積分長度尺度 ($L_u$)、功率譜斜率與極端陣風特徵 |
| 📈 **[explore_feature_suitability.py](file:///d:/wind_d/ML_wind/ml_project/explore_feature_suitability.py)** | **風速預測與垂直外推可行性驗證**<br>• 驗證 38m $\to$ 100m 垂直風速外推 (Power Law vs LightGBM)<br>• 評估多時程風速預測 Skill Score |
| 🎨 **[make_figures.py](file:///d:/wind_d/ML_wind/ml_project/make_figures.py)** | **成果統計與圖表繪製腳本** |
| 🎨 **[make_figures_spectrum.py](file:///d:/wind_d/ML_wind/ml_project/make_figures_spectrum.py)** | **湍流功率譜分析與視覺化腳本** |

---

## 💾 2. 資料集目錄 (Data Files)

資料集位於 `d:/wind_d/ML_wind/ml_project/data/` 目錄：

| 資料檔名 (Data Path) | 格式與大小 | 說明 |
| :--- | :--- | :--- |
| 📦 **[BSMI_10min.parquet](file:///d:/wind_d/ML_wind/ml_project/data/BSMI_10min.parquet)** | Parquet (~14.7 MB) | 71,261 列 × 48 欄，包含 5.5 年完整 10 分鐘氣象與品管標記。 |
| 📦 **[BSMI_turb.parquet](file:///d:/wind_d/ML_wind/ml_project/data/BSMI_turb.parquet)** | Parquet (~8.3 MB) | 71,261 列，包含高頻湍流特徵 (積分長度尺度、陣風因子、極端百分位數)。 |
| 📄 **[qc_report.csv](file:///d:/wind_d/ML_wind/ml_project/data/qc_report.csv)** | CSV (239 Bytes) | 品管抽樣報告。 |

---

## 📈 3. 預測結果與評估數據 (Results & Benchmarks)

模型評估數據位於 `d:/wind_d/ML_wind/ml_project/results/`：

| 結果檔名 (File Path) | 類型 | 說明 |
| :--- | :--- | :--- |
| 📝 **[RESULTS_power_forecasting.md](file:///d:/wind_d/ML_wind/ml_project/results/RESULTS_power_forecasting.md)** | MD 報告 | **風力發電預測專案成果說明總報告** |
| 📊 **[power_model_benchmark.csv](file:///d:/wind_d/ML_wind/ml_project/results/power_model_benchmark.csv)** | CSV 數據 | 多模型 (LightGBM / XGBoost / Ensemble / Persistence) 多時程評估表 (RMSE, MAE, NMAE %, Skill Score) |
| 📊 **[power_ramp_events.csv](file:///d:/wind_d/ML_wind/ml_project/results/power_ramp_events.csv)** | CSV 數據 | 1 小時劇烈發電量陡升/驟降 (Ramp Event, $|\Delta P| \ge 1\text{MW}$) 預警 Precision / Recall / F1-Score |
| 📊 **[power_feature_correlation.csv](file:///d:/wind_d/ML_wind/ml_project/results/power_feature_correlation.csv)** | CSV 數據 | 各氣象特徵與未來自發電量之 Pearson 與 Spearman 相關係數表 |
| 📊 **[power_feature_importance.csv](file:///d:/wind_d/ML_wind/ml_project/results/power_feature_importance.csv)** | CSV 數據 | LightGBM 特徵 Gain % 重要性隨預測時程變化表 |
| 📘 **[PLAN.md](file:///d:/wind_d/ML_wind/ml_project/PLAN.md)** | MD 規劃 | 測風塔專案全景規劃與問題定義文件 |

---

## 🖼️ 4. 高解析度圖表目錄 (Figures Directory)

視覺化圖表儲存於 `d:/wind_d/ML_wind/ml_project/results/figures/`：

| 圖表檔名 (Figure Path) | 內容說明 |
| :--- | :--- |
| 🖼️ **[pipeline_model_comparison.png](file:///d:/wind_d/ML_wind/ml_project/results/figures/pipeline_model_comparison.png)** | 各模型 (LightGBM, XGBoost, Ensemble) 之 Skill Score (%) 與 NMAE (%) 柱狀圖 |
| 🖼️ **[pipeline_error_by_speed.png](file:///d:/wind_d/ML_wind/ml_project/results/figures/pipeline_error_by_speed.png)** | 風機運轉區間 (Region 1 / 2 / 3) 之發電量預測誤差 NMAE 分佈 |
| 🖼️ **[pipeline_ramp_detection.png](file:///d:/wind_d/ML_wind/ml_project/results/figures/pipeline_ramp_detection.png)** | 測試集連續時序預測發電量與 Ramp Event 陡升/驟降預警追蹤圖 |
| 🖼️ **[power_correlation_heatmap.png](file:///d:/wind_d/ML_wind/ml_project/results/figures/power_correlation_heatmap.png)** | 氣象/物理特徵與發電量標的 Spearman 相關係數熱圖 |
| 🖼️ **[power_feature_importance.png](file:///d:/wind_d/ML_wind/ml_project/results/figures/power_feature_importance.png)** | LightGBM 特徵 Gain % 貢獻度長條圖 |
| 🖼️ **[power_forecast_feasibility.png](file:///d:/wind_d/ML_wind/ml_project/results/figures/power_forecast_feasibility.png)** | Skill Score 隨預測提前量 (10m $\to$ 6h) 成長曲線圖 |
| 🖼️ **[power_prediction_timeseries.png](file:///d:/wind_d/ML_wind/ml_project/results/figures/power_prediction_timeseries.png)** | 1 小時發電量預測時序對比圖 |

---

## 🚀 5. 快速執行指令 (How to Run)

如果您想重新執行發電量機器學習預測管線並更新所有數據與圖表：

```bash
# 進入專案目錄
cd d:/wind_d/ML_wind/ml_project

# 執行風力發電機器學習預測管線 (一鍵訓練、評估與繪圖)
python train_power_forecast_pipeline.py

# 執行特徵相關性與可行性分析
python explore_power_suitability.py
```
