# power_forecast_interval — 離岸風場資源評估與區間發電能量預測專案

基於 `ml_project/power_forecast` 專案架構進行**標的維度革新**：將單點瞬時功率預測改為未來區間 $[t+1 .. t+k]$ 的 **累積總發電能量 $E_{[t, t+H]}$ (kWh/MW)** 與區間平均風速預測。

---

## 快速開始

### 1. 安裝套件
```bash
pip install pandas pyarrow lightgbm xgboost scikit-learn matplotlib numpy
```

### 2. 執行 5 段式獨立管線
```bash
cd ml_project/power_forecast_interval
python 01_load_validate.py          # Stage 1: 獨立 4 重 QC 驗證
python resource_assessment.py       # Stage 2: 風資源評估 (CF=45.05%)
python forecast_interval_features.py# Stage 3: 44 特徵與區間能量標的建置
python forecast_interval_train.py   # Stage 4: 區間能量多模型訓練 (LGBM, XGB, Ridge)
python forecast_interval_figures.py # Stage 5: 自動產出圖表與技術報告
```

---

## 核心成果與 R² 擬合度

1. **區間累積總發電能量預測 R² 擬合度**：
   - 1 小時區間能量預測 ($E_{1h}$)：LightGBM **$R^2 = 0.971$**
   - 3 小時區間能量預測 ($E_{3h}$)：LightGBM **$R^2 = 0.925$**
   - 6 小時區間能量預測 ($E_{6h}$)：LightGBM **$R^2 = 0.874$**
   - 24 小時日前能量預測 ($E_{24h}$): LightGBM **$R^2 = 0.710$**

2. **檔案結構**：
   - `config.py` — 設定檔
   - `virtual_power.py` — IEC 功率曲線與空氣密度修正
   - `01_load_validate.py` — QC 數據清洗
   - `resource_assessment.py` — 風資源評估
   - `forecast_interval_features.py` — 特徵與標的工程
   - `forecast_interval_train.py` — 多模型區間訓練
   - `forecast_interval_figures.py` — 圖表報告產出
   - `README.md` & `REPORT_INTERVAL.md` — 專案說明與報告
