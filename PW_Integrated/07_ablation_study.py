#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 7 —— 特徵群組消融實驗 (Ablation Study)

目的：
  系統性移除特定特徵群組，定量評估各氣象/時序特徵對 0–6h 發電預測精度的貢獻程度 (ΔnRMSE %)。

消融實驗組 (Ablation Setups)：
  1. Full Model (全特徵基準 - 44 特徵)
  2. w/o Lags & Rolling (移除 歷史滯後 lag10..180 與 過去 1h/3h/6h 滾動統計/趨勢斜率)
  3. w/o Turbulence (移除 湍流強度 TI, 陣風因子, 風向標準差 WD_sigma)
  4. w/o Wind Direction (移除 風向正餘弦向量 WD_sin, WD_cos)
  5. w/o Air Density & Atmosphere (移除 空氣密度, 氣溫, 氣壓, 濕度)
  6. w/o Cyclical Time Encodings (移除 小時 sin/cos 與 年內日 sin/cos 季節週期)

輸出：
  results/ablation_metrics.csv
  figures/fig6_ablation_study.png
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
def mae(a, b):  return float(np.mean(np.abs(a - b)))

def main():
    print("="*70)
    print("PW_Integrated Stage 7 —— 特徵群組消融實驗 (Ablation Study)")
    print("="*70)

    if not os.path.exists(C.FEAT_PARQUET):
        raise FileNotFoundError(f"找不到 {C.FEAT_PARQUET}，請先執行 03_features.py")

    df = pd.read_parquet(C.FEAT_PARQUET)
    df["ts"] = pd.to_datetime(df["ts"])
    test_start = pd.Timestamp(C.TEST_START)

    meta = {"ts", "is_ok", "year", "month", "hour_i"}
    all_fcols = [c for c in df.columns if c not in meta and not c.startswith("y_") and not c.startswith("m_")]

    # 定義消融特徵群組過濾字典
    ablation_setups = {
        "Full Model (全特徵)": lambda cols: cols,

        "w/o Lags & Rolling (無滯後/滾動趨勢)": lambda cols: [
            c for c in cols if not (c.startswith("ws_lag") or c.startswith("ws_rmean") or
                                   c.startswith("ws_rstd") or c.startswith("ws_rmin") or
                                   c.startswith("ws_rmax") or c.startswith("ws_slope") or
                                   c.startswith("ws_diff"))
        ],

        "w/o Turbulence (無湍流/陣風)": lambda cols: [
            c for c in cols if c not in ("WS_100E_ti", "WS_100E_gust_factor", "WD_97_sigma")
        ],

        "w/o Wind Direction (無風向向量)": lambda cols: [
            c for c in cols if c not in ("WD_97_sin", "WD_97_cos")
        ],

        "w/o Density & Atmosphere (無空氣密度/大氣)": lambda cols: [
            c for c in cols if c not in ("air_density", "AT_95_mean", "RH_95_mean", "BP_93_mean")
        ],

        "w/o Cyclical Time (無日夜/季節週期)": lambda cols: [
            c for c in cols if c not in ("hour_sin", "hour_cos", "doy_sin", "doy_cos")
        ]
    }

    lgb_params = dict(objective="regression", n_estimators=350, learning_rate=0.05,
                      num_leaves=48, min_child_samples=100, subsample=0.8,
                      subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                      random_state=C.RANDOM_SEED, n_jobs=-1, verbose=-1)

    results = []

    for h in C.HORIZONS_H:
        ycol = f"y_power_{h}"
        mask = (df["is_ok"] & df[f"m_{h}"] & df[ycol].notna() & df[all_fcols].notna().all(axis=1))
        sub = df.loc[mask]
        is_test = sub["ts"] >= test_start

        tr, te = sub.loc[~is_test], sub.loc[is_test]
        ytr, yte = tr[ycol].values, te[ycol].values
        nval = int(len(tr) * 0.15)

        full_nrmse = None

        print(f"\n=== 發電預測 Power H={h}h 消融實驗 ===  Train 樣本 {len(tr):,} / Test 樣本 {len(te):,}")

        for setup_name, filter_fn in ablation_setups.items():
            fcols = filter_fn(all_fcols)
            Xtr, Xte = tr[fcols].values, te[fcols].values

            gbm = lgb.LGBMRegressor(**lgb_params)
            gbm.fit(Xtr[:-nval], ytr[:-nval],
                    eval_set=[(Xtr[-nval:], ytr[-nval:])],
                    callbacks=[lgb.early_stopping(40, verbose=False)])

            pred = np.clip(gbm.predict(Xte), 0.0, 1.0)
            r = rmse(yte, pred)
            m = mae(yte, pred)
            denom = np.mean(yte) if np.mean(yte) > 1e-6 else 1.0
            nrmse = r / denom

            if setup_name == "Full Model (全特徵)":
                full_nrmse = nrmse
                delta_pct = 0.0
            else:
                delta_pct = ((nrmse - full_nrmse) / full_nrmse) * 100.0

            print(f"  [{setup_name:35s}] 剩餘特徵 {len(fcols):2d} 個 ｜ nRMSE={nrmse:.4f} ｜ 誤差增加={delta_pct:+.2f}%")

            results.append({
                "H": h,
                "setup": setup_name,
                "num_features": len(fcols),
                "nRMSE": round(nrmse, 5),
                "nMAE": round(m / denom, 5),
                "delta_nrmse_pct": round(delta_pct, 2)
            })

    # 匯出 CSV 檔
    df_res = pd.DataFrame(results)
    os.makedirs(C.RES_DIR, exist_ok=True)
    os.makedirs(C.FIG_DIR, exist_ok=True)
    csv_path = os.path.join(C.RES_DIR, "ablation_metrics.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"\n消融實驗結果已寫入：{csv_path}")

    # 繪製消融實驗視覺化圖表
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot_df = df_res[df_res["setup"] != "Full Model (全特徵)"].pivot(index="setup", columns="H", values="delta_nrmse_pct")
    pivot_df.plot(kind="barh", ax=ax, width=0.75, colormap="tab10", edgecolor="black")

    ax.set_title("特徵群組消融實驗：移除特定特徵後發電預測誤差增加比例 (%)", fontsize=13, fontweight="bold")
    ax.set_xlabel("nRMSE 誤差增加比例 (%) — 越高代表該特徵群組越關鍵", fontsize=11)
    ax.set_ylabel("移除之特徵群組", fontsize=11)
    ax.axvline(0, color="gray", linestyle="--")
    ax.legend(title="預測提前量 H", labels=["+1h", "+3h", "+6h"])
    plt.tight_layout()

    fig_path = os.path.join(C.FIG_DIR, "fig6_ablation_study.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"消融實驗圖表已產出：{fig_path}")

    print("\nStage 7 消融實驗完成。")

if __name__ == "__main__":
    main()
