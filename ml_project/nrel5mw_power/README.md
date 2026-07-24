# NREL 5MW 虛擬風場（絕對 MW 版）— 獨立模組

把 BSMI 測風塔的風速，用 **NREL 5MW Reference Offshore Wind Turbine** 官方功率曲線
換算成「虛擬風機出力」（**絕對 MW，額定 5.0 MW**），並做資源評估與多時程發電預測。

## 為什麼有這個資料夾

專案原本散落**兩套不一致的功率曲線**：主管線用 NREL 5MW 立方近似（絕對 kW），
其他子專案（interval / advanced / PW*）用一條正規化 8MW 代表曲線（0–1）。
本資料夾把 **官方 NREL 5MW 查表曲線** 抽成單一可信來源（`nrel_5mw.py`），
任何子專案 `from nrel_5mw import nrel_5mw_power_mw` 就能得到一致的絕對 MW 出力。

## 檔案

| 檔案 | 說明 |
|---|---|
| `nrel_5mw.py` | 可重用模組：官方功率曲線、IEC 密度修正、輪轂外推、資料載入、CJK 字型 |
| `01_resource_assessment.py` | 資源評估（容量因數、月/時形態）＋ NREL 5MW vs 8MW 曲線對照 |
| `02_forecast_pipeline.py` | 多時程發電預測（Persistence / LightGBM / XGBoost / Ensemble） |
| `run_all.py` | 一鍵跑完上面兩支 |
| `REPORT.md` | 成果報告（含實際數字與圖表） |
| `results/` | CSV 指標 + `figures/` 圖表 |

## 執行

```bash
cd ml_project/nrel5mw_power
python run_all.py
# 或分開跑
python 01_resource_assessment.py
python 02_forecast_pipeline.py
```

需要 `../data/BSMI_10min.parquet`（由 `preprocess.py` 產生）。
套件：`pandas numpy pyarrow lightgbm xgboost scikit-learn matplotlib`。

## 當別的專案要用

```python
import nrel_5mw as N
p_mw = N.nrel_5mw_power_mw(df["WS_100_mean"], df["air_density"])   # 絕對 MW，0–5
# 換到更高輪轂（用實測風切外推）
p_mw = N.nrel_5mw_power_mw(df["WS_100_mean"], df["air_density"], hub=120, alpha=df["shear_alpha"])
```
