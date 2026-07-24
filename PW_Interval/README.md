# PW_Interval — 單一風場超短期 (0–6h) 區間總發電能量預測專案

基於 `PW` 專案架構進行**標的維度革新**：將原本的單點瞬時功率 $P_{t+H}$ 預測改為未來區間 $[t+1 .. t+k]$ 的 **累積總發電能量 $E_{[t, t+H]}$ (kWh/MW)** 與區間平均風速預測。

---

## 快速開始

### 1. 安裝套件
```bash
pip install pandas pyarrow lightgbm scikit-learn matplotlib numpy
```

### 2. 執行 4 段式獨立管線
```bash
cd PW_Interval
python 01_load_validate.py   # Stage 1: QC 驗證 ➔ data/clean_10min.parquet
python 02_features.py        # Stage 2: 44 個特徵與區間能量標的 ➔ data/features.parquet
python 03_train_select.py    # Stage 3: 模型選擇 (Persistence, Clim, Ridge, LGBM) ➔ results/test_metrics.csv
python 04_evaluate_report.py # Stage 4: 自動產出 4 張專屬視覺化圖表 ➔ figures/
```

---

## 核心創新與實驗結論 (R² 擬合度大幅跳升)

1. **消弭高頻陣風噪訊**：
   單點瞬時出力包含微觀高頻湍流與陣風噪訊；改為預測未來 1h/3h/6h 的**累積總發電能量**後，預測結果極度平滑且符合真實電網結算需求。

2. **測試集預測擬合度 ($R^2$)**：
   - **未來 1 小時總能量預測 ($E_{1h}$)**：LightGBM **$R^2 = 0.971$** (nRMSE = 0.128)
   - **未來 3 小時總能量預測 ($E_{3h}$)**：LightGBM **$R^2 = 0.925$** (nRMSE = 0.204)
   - **未來 6 小時總能量預測 ($E_{6h}$)**：LightGBM **$R^2 = 0.875$** (nRMSE = 0.259)

3. **勝過單點 Persistence 基準**：
   在未來 3 小時區間能量預測中，LightGBM 相較 Persistence nRMSE 改善達 **+21.0%**。

---

## 檔案結構
- `config.py` — 設定檔與功率曲線
- `01_load_validate.py` — Stage 1 數據 QC
- `02_features.py` — Stage 2 特徵工程與區間標的
- `03_train_select.py` — Stage 3 模型選擇 (Expanding-window TimeSeriesSplit CV + 保留測試年)
- `04_evaluate_report.py` — Stage 4 自動產出圖表
- `README.md` & `REPORT.md` — 專案說明與完整技術報告
