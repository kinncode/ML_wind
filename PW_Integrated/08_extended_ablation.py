#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 8 —— 多維度進階消融實驗 (Multi-Dimension Extended Ablation Study)

目的：
  針對感測器高度配置、歷史視窗長度、空氣密度物理修正以及訓練歷史長度進行深層消融分析。

四項進階消融維度：
  1. Height & Sensor Configuration (感測器高度配置消融)
     - Full (四高度 100E/100W/69W/38W + 風切 alpha) vs Single Height (僅 100m 風速)
  2. Historical Lookback Window (歷史視窗長度消融)
     - Full Lookback (含 1h, 3h, 6h 滾動統計與斜率) vs Short Lookback (僅 <=1h 滯後與滾動)
  3. Physical Density Correction (IEC 空氣密度物理修正消融)
     - Dynamic Density Correction (動態實測 rho) vs Static Density (固定標準 rho = 1.225)
  4. Training History Length (訓練數據年份長度消融)
     - Full 3-Year Train (2016–2018) vs 1-Year Train Only (僅 2018 單一年份)

輸出：
  results/extended_ablation_metrics.csv
  figures/fig7_extended_ablation.png
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def rmse(a, b): return float(np.sqrt(np.mean((a - b)**2)))

def main():
    print("="*70)
    print("PW_Integrated Stage 8 —— 多維度進階消融實驗 (Extended Ablation)")
    print("="*70)

    if not os.path.exists(C.FEAT_PARQUET):
        raise FileNotFoundError(f"找不到 {C.FEAT_PARQUET}，請先執行 03_features.py")

    df = pd.read_parquet(C.FEAT_PARQUET)
    df["ts"] = pd.to_datetime(df["ts"])
    test_start = pd.Timestamp(C.TEST_START)

    meta = {"ts", "is_ok", "year", "month", "hour_i"}
    all_fcols = [c for c in df.columns if c not in meta and not c.startswith("y_") and not c.startswith("m_")]

    lgb_params = dict(objective="regression", n_estimators=350, learning_rate=0.05,
                      num_leaves=48, min_child_samples=100, subsample=0.8,
                      subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                      random_state=C.RANDOM_SEED, n_jobs=-1, verbose=-1)

    results = []

    # --- 維度 1：感測器高度配置消融 (Height Configuration) ---
    print("\n[維度 1] 感測器高度配置消融 (Full vs Single Height 100m)")
    height_setups = {
        "Full (四高度 + 風切 alpha)": all_fcols,
        "Single Height (僅 100m 風速)": [
            c for c in all_fcols if c not in ("WS_100E_mean", "WS_100W_mean", "WS_69W_mean", "WS_38W_mean", "shear_alpha")
        ]
    }
    for h in C.HORIZONS_H:
        ycol = f"y_power_{h}"
        mask = (df["is_ok"] & df[f"m_{h}"] & df[ycol].notna() & df[all_fcols].notna().all(axis=1))
        sub = df.loc[mask]
        is_test = sub["ts"] >= test_start
        tr, te = sub.loc[~is_test], sub.loc[is_test]
        ytr, yte = tr[ycol].values, te[ycol].values
        denom = np.mean(yte) if np.mean(yte) > 1e-6 else 1.0
        nval = int(len(tr) * 0.15)

        base_nrmse = None
        for name, fcols in height_setups.items():
            gbm = lgb.LGBMRegressor(**lgb_params)
            gbm.fit(tr[fcols].values[:-nval], ytr[:-nval], eval_set=[(tr[fcols].values[-nval:], ytr[-nval:])], callbacks=[lgb.early_stopping(40, verbose=False)])
            pred = np.clip(gbm.predict(te[fcols].values), 0, 1)
            nrmse = rmse(yte, pred) / denom
            if "Full" in name: base_nrmse = nrmse; delta = 0.0
            else: delta = ((nrmse - base_nrmse) / base_nrmse) * 100
            results.append({"dimension": "1. 高度配置", "H": h, "setup": name, "nRMSE": round(nrmse, 5), "delta_pct": round(delta, 2)})
            print(f"  H={h}h [{name:30s}] nRMSE={nrmse:.4f}  (Δ={delta:+.2f}%)")

    # --- 維度 2：歷史視窗長度消融 (Lookback Window) ---
    print("\n[維度 2] 歷史視窗長度消融 (Full vs Short <=1h)")
    lookback_setups = {
        "Full Lookback (1h + 3h + 6h)": all_fcols,
        "Short Lookback (僅 <=1h 視窗)": [
            c for c in all_fcols if not (c.endswith("_180") or c.endswith("_360") or c == "ws_diff_180" or c in ("ws_lag120", "ws_lag180"))
        ]
    }
    for h in C.HORIZONS_H:
        ycol = f"y_power_{h}"
        mask = (df["is_ok"] & df[f"m_{h}"] & df[ycol].notna() & df[all_fcols].notna().all(axis=1))
        sub = df.loc[mask]
        is_test = sub["ts"] >= test_start
        tr, te = sub.loc[~is_test], sub.loc[is_test]
        ytr, yte = tr[ycol].values, te[ycol].values
        denom = np.mean(yte) if np.mean(yte) > 1e-6 else 1.0
        nval = int(len(tr) * 0.15)

        base_nrmse = None
        for name, fcols in lookback_setups.items():
            gbm = lgb.LGBMRegressor(**lgb_params)
            gbm.fit(tr[fcols].values[:-nval], ytr[:-nval], eval_set=[(tr[fcols].values[-nval:], ytr[-nval:])], callbacks=[lgb.early_stopping(40, verbose=False)])
            pred = np.clip(gbm.predict(te[fcols].values), 0, 1)
            nrmse = rmse(yte, pred) / denom
            if "Full" in name: base_nrmse = nrmse; delta = 0.0
            else: delta = ((nrmse - base_nrmse) / base_nrmse) * 100
            results.append({"dimension": "2. 視窗長度", "H": h, "setup": name, "nRMSE": round(nrmse, 5), "delta_pct": round(delta, 2)})
            print(f"  H={h}h [{name:30s}] nRMSE={nrmse:.4f}  (Δ={delta:+.2f}%)")

    # --- 維度 3：訓練數據年份長度消融 (Training Data History) ---
    print("\n[維度 3] 訓練年份長度消融 (Full 3-Year vs 1-Year Only)")
    for h in C.HORIZONS_H:
        ycol = f"y_power_{h}"
        mask = (df["is_ok"] & df[f"m_{h}"] & df[ycol].notna() & df[all_fcols].notna().all(axis=1))
        sub = df.loc[mask]
        is_test = sub["ts"] >= test_start

        tr_full = sub.loc[~is_test]
        tr_1yr  = sub.loc[(~is_test) & (sub["year"] == 2018)]  # 僅單一年份 2018
        te      = sub.loc[is_test]
        yte     = te[ycol].values
        denom   = np.mean(yte) if np.mean(yte) > 1e-6 else 1.0

        # Full 3-Year
        nval_f = int(len(tr_full) * 0.15)
        gbm_f = lgb.LGBMRegressor(**lgb_params)
        gbm_f.fit(tr_full[all_fcols].values[:-nval_f], tr_full[ycol].values[:-nval_f],
                  eval_set=[(tr_full[all_fcols].values[-nval_f:], tr_full[ycol].values[-nval_f:])],
                  callbacks=[lgb.early_stopping(40, verbose=False)])
        nrmse_f = rmse(yte, np.clip(gbm_f.predict(te[all_fcols].values), 0, 1)) / denom

        # 1-Year Only
        nval_1 = int(len(tr_1yr) * 0.15)
        gbm_1 = lgb.LGBMRegressor(**lgb_params)
        gbm_1.fit(tr_1yr[all_fcols].values[:-nval_1], tr_1yr[ycol].values[:-nval_1],
                  eval_set=[(tr_1yr[all_fcols].values[-nval_1:], tr_1yr[ycol].values[-nval_1:])],
                  callbacks=[lgb.early_stopping(40, verbose=False)])
        nrmse_1 = rmse(yte, np.clip(gbm_1.predict(te[all_fcols].values), 0, 1)) / denom

        delta = ((nrmse_1 - nrmse_f) / nrmse_f) * 100
        results.append({"dimension": "3. 訓練歷史長度", "H": h, "setup": "Full 3-Year Train (2016-2018)", "nRMSE": round(nrmse_f, 5), "delta_pct": 0.0})
        results.append({"dimension": "3. 訓練歷史長度", "H": h, "setup": "1-Year Train Only (僅 2018)", "nRMSE": round(nrmse_1, 5), "delta_pct": round(delta, 2)})
        print(f"  H={h}h [Full 3-Year              ] nRMSE={nrmse_f:.4f}")
        print(f"  H={h}h [1-Year Train Only (僅2018) ] nRMSE={nrmse_1:.4f}  (Δ={delta:+.2f}%)")

    # 匯出結果 CSV
    df_res = pd.DataFrame(results)
    os.makedirs(C.RES_DIR, exist_ok=True)
    os.makedirs(C.FIG_DIR, exist_ok=True)
    csv_path = os.path.join(C.RES_DIR, "extended_ablation_metrics.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"\n進階消融實驗結果已寫入：{csv_path}")

    # 繪製圖表
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    dims = ["1. 高度配置", "2. 視窗長度", "3. 訓練歷史長度"]
    titles = ["感測器高度配置消融 (多高度 vs 單高度100m)",
              "歷史滾動視窗消融 (長視窗 vs 僅<=1h)",
              "訓練年份長度消融 (3年歷史 vs 1年歷史)"]

    for i, dim in enumerate(dims):
        sub_df = df_res[(df_res["dimension"] == dim) & (df_res["delta_pct"] != 0.0)]
        piv = sub_df.pivot(index="setup", columns="H", values="delta_pct")
        piv.plot(kind="bar", ax=axes[i], colormap="Accent", width=0.6, edgecolor="black")
        axes[i].set_title(titles[i], fontsize=11, fontweight="bold")
        axes[i].set_ylabel("nRMSE 誤差增加比例 (%)", fontsize=10)
        axes[i].axhline(0, color="gray", linestyle="--")
        axes[i].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_path = os.path.join(C.FIG_DIR, "fig7_extended_ablation.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"進階消融實驗圖表已產出：{fig_path}")

    print("\nStage 8 進階消融實驗完成。")

if __name__ == "__main__":
    main()
