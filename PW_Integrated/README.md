# PW_Integrated — 離岸風場資源評估與超短期 (0–6h) 機器學習發電預測整合專案

本專案整合了 **風場資源評估** 與 **高嚴謹度四段式機器學習預測管線**，僅採用 BSMI 100m 測風塔長序時數據，提供獨立、可重現的風資源分析、點預測與 $p10/p50/p90$ 機率預測。
整合了D:\ML_wind\ml_project\power_forecast與 D:\ML_wind\PW
---

## 快速開始

### 1. 安裝套件
```bash
pip install pandas pyarrow lightgbm scikit-learn matplotlib numpy
```

### 2. 執行完整管線
```bash
cd PW_Integrated
python 01_load_validate.py        # Stage 1: 獨立四重 QC 驗證 ➔ data/clean_10min.parquet
python 02_resource_assessment.py  # Stage 2: 風資源評估 (CF、滿載時數) ➔ results/resource_stats.json
python 03_features.py             # Stage 3: 44 個特徵工程與無洩漏遮罩 ➔ data/features.parquet
python 04_train_select.py        # Stage 4: 點預測 (Persistence, Clim, Ridge, LGBM) ➔ results/test_metrics.csv
python 05_quantile_forecast.py   # Stage 5: p10/p50/p90 機率預測 ➔ results/quantile_metrics.csv
python 06_evaluate_report.py     # Stage 6: 自動生成 5 張視覺化圖表與系統摘要 ➔ figures/
```

---

## 核心亮點與結果

1. **風資源評估**：
   - **容量因數 (CF)**：**45.1%** (代表性 8 MW 離岸機型)
   - **等效滿載時數**：**3,949 小時/年**
   - **季節形態**：冬季東北季風 CF 達 60–71%，夏季 16–21%

2. **超短期 (0–6h) 發電預測**：
   - **+1h**: LightGBM nRMSE = 0.248 (相對 Persistence **+12.6%**)
   - **+3h**: LightGBM nRMSE = 0.362 (相對 Persistence **+20.6%**)
   - **+6h**: LightGBM nRMSE = 0.451 (相對 Persistence **+27.7%**)

3. **機率預測 ($p10/p50/p90$)**：
   - $p90$ 測試集實質涵蓋率達 **90.2%**，校準優良，提供可靠的備轉容量信賴區間。

---

## 檔案結構
- `config.py` — 全局設定與功率曲線
- `01–06_*.py` — 獨立分段執行腳本
- `PLAN_48h_NWP_MOS.md` — 日前 0–48h NWP 降尺度規劃
- `REPORT.md` — 完整技術報告
- `data/ models/ results/ figures/` — 產出檔
