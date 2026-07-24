# PW — 風場超短期風力發電預測（獨立管線）

只用 BSMI 100 m 測風塔資料，做 **0–6 小時**的 100 m 風速與正規化發電量預測。
四段管線：**驗證 → 特徵提取/轉換 → 模型選擇 → 評估**。完整說明見 [REPORT.md](REPORT.md)。

## 快速開始
```bash
pip install pandas pyarrow lightgbm scikit-learn matplotlib
python3 01_load_validate.py          # 驗證 → data/clean_10min.parquet
python3 02_features.py               # 特徵/目標 → data/features.parquet
for t in ws100 power; do             # 模型選擇（每 target×時程各一次）
  for h in 1 3 6; do python3 03_train_select.py $t $h; done
done
python3 04_evaluate_report.py        # 圖表 + summary
```

## 結果（保留測試期 2020-06 ~ 2021-10，最佳模型皆 LightGBM）
| 目標 | +1h | +3h | +6h |
|---|---|---|---|
| 100 m 風速 nRMSE | 0.136 | 0.218 | 0.294 |
| 發電量 nRMSE | 0.248 | 0.362 | 0.451 |
| 發電量 勝 persistence | +12.6% | +20.6% | +27.7% |

## 檔案
- `config.py` — 路徑、時程、目標、功率曲線
- `01–04_*.py` — 四段管線腳本
- `data/ models/ results/ figures/` — 產出
