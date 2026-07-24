#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 5 —— 單點預測 (Point Forecast) vs 區間發電能量預測 (Interval Forecast) 差異實驗對比

目的：
  同數據集、同測試時間段 (2020-06 至 2021-10) 嚴格量化比較「單點瞬時預測 P(t+H)」與「區間累積能量預測 E_[t, t+H]」的精確度差異。

輸出：
  results/point_vs_interval_metrics.csv
  figures/fig5_point_vs_interval_comparison.png
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import config as C

# 全局載入微軟正黑體字型檔
msjh_font = "C:/Windows/Fonts/msjh.ttc"
if os.path.exists(msjh_font):
    plt.rcParams['font.family'] = FontProperties(fname=msjh_font).get_name()
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'MingLiU', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def rmse(a, b): return float(np.sqrt(np.mean((a - b)**2)))

def main():
    print("="*70)
    print("PW_Interval Stage 5 —— 單點預測 vs 區間能量預測 差異實驗對比")
    print("="*70)

    if not os.path.exists(C.FEAT_PARQUET):
        raise FileNotFoundError(f"找不到 {C.FEAT_PARQUET}，請先執行 02_features.py")

    df = pd.read_parquet(C.FEAT_PARQUET)
    df["ts"] = pd.to_datetime(df["ts"])
    test_start = pd.Timestamp(C.TEST_START)

    meta = {"ts", "is_ok", "year", "month", "hour_i"}
    fcols = [c for c in df.columns if c not in meta and not c.startswith("y_") and not c.startswith("m_")]

    lgb_params = dict(objective="regression", n_estimators=350, learning_rate=0.05,
                      num_leaves=48, min_child_samples=100, subsample=0.8,
                      colsample_bytree=0.8, reg_lambda=1.0, random_state=C.RANDOM_SEED,
                      n_jobs=-1, verbose=-1)

    P_now_full = C.virtual_power(df["WS_100_mean"], df["air_density"])
    P_series = pd.Series(P_now_full, index=df.index)

    rows = []

    for h in C.HORIZONS_H:
        k = C.HORIZON_STEPS[h]
        tag = f"H{h}h"

        # -----------------------------------------------------------
        # 標的 A: 單點瞬時預測 y_point = P(t+k)
        # -----------------------------------------------------------
        y_point = P_series.shift(-k)
        mask_point = df["is_ok"] & df[f"m_{h}"] & y_point.notna() & df[fcols].notna().all(axis=1)
        sub_p = df.loc[mask_point]
        is_test_p = sub_p["ts"] >= test_start

        tr_p, te_p = sub_p.loc[~is_test_p], sub_p.loc[is_test_p]
        ytr_p, yte_p = y_point.loc[tr_p.index].values, y_point.loc[te_p.index].values

        nval_p = int(len(tr_p) * 0.15)
        gbm_p = lgb.LGBMRegressor(**lgb_params)
        gbm_p.fit(tr_p[fcols].values[:-nval_p], ytr_p[:-nval_p],
                  eval_set=[(tr_p[fcols].values[-nval_p:], ytr_p[-nval_p:])],
                  callbacks=[lgb.early_stopping(40, verbose=False)])

        pred_point = np.clip(gbm_p.predict(te_p[fcols].values), 0, 1)
        denom_p = np.mean(yte_p) if np.mean(yte_p) > 1e-6 else 1.0
        nrmse_point = rmse(yte_p, pred_point) / denom_p
        r2_point = r2_score(yte_p, pred_point)

        # -----------------------------------------------------------
        # 標的 B: 區間能量預測 y_interval = E_[t, t+k]
        # -----------------------------------------------------------
        y_interval = P_series.iloc[::-1].rolling(k).mean().iloc[::-1]
        mask_intv = df["is_ok"] & df[f"m_{h}"] & y_interval.notna() & df[fcols].notna().all(axis=1)
        sub_i = df.loc[mask_intv]
        is_test_i = sub_i["ts"] >= test_start

        tr_i, te_i = sub_i.loc[~is_test_i], sub_i.loc[is_test_i]
        ytr_i, yte_i = y_interval.loc[tr_i.index].values, y_interval.loc[te_i.index].values

        nval_i = int(len(tr_i) * 0.15)
        gbm_i = lgb.LGBMRegressor(**lgb_params)
        gbm_i.fit(tr_i[fcols].values[:-nval_i], ytr_i[:-nval_i],
                  eval_set=[(tr_i[fcols].values[-nval_i:], ytr_i[-nval_i:])],
                  callbacks=[lgb.early_stopping(40, verbose=False)])

        pred_intv = np.clip(gbm_i.predict(te_i[fcols].values), 0, 1)
        denom_i = np.mean(yte_i) if np.mean(yte_i) > 1e-6 else 1.0
        nrmse_intv = rmse(yte_i, pred_intv) / denom_i
        r2_intv = r2_score(yte_i, pred_intv)

        # 比較差異
        nrmse_diff_pct = ((nrmse_intv - nrmse_point) / nrmse_point) * 100.0
        r2_gain = r2_intv - r2_point

        print(f"\n=== 預測提前量 H={h}h 差異比對 ===")
        print(f"  單點預測 (Point P_t+H)      : nRMSE = {nrmse_point:.4f} ｜ R² = {r2_point:.4f}")
        print(f"  區間預測 (Interval E_t..t+H): nRMSE = {nrmse_intv:.4f} ｜ R² = {r2_intv:.4f}")
        print(f"  ★ 誤差降幅 (nRMSE Reduction) : {nrmse_diff_pct:+.2f}%")
        print(f"  ★ 擬合度增益 (R² Gain)       : {r2_gain:+.4f}")

        rows.append({"H": h, "mode": "單點瞬時預測 (Point)", "nRMSE": round(nrmse_point, 5), "R2": round(r2_point, 5)})
        rows.append({"H": h, "mode": "區間總能量預測 (Interval)", "nRMSE": round(nrmse_intv, 5), "R2": round(r2_intv, 5)})

    df_res = pd.DataFrame(rows)
    os.makedirs(C.RES_DIR, exist_ok=True)
    os.makedirs(C.FIG_DIR, exist_ok=True)

    csv_path = os.path.join(C.RES_DIR, "point_vs_interval_metrics.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"\n對比結果已寫入：{csv_path}")

    # 繪製視覺化圖表
    fp = FontProperties(fname="C:/Windows/Fonts/msjh.ttc", size=11) if os.path.exists("C:/Windows/Fonts/msjh.ttc") else None
    fp_title = FontProperties(fname="C:/Windows/Fonts/msjh.ttc", size=12, weight="bold") if os.path.exists("C:/Windows/Fonts/msjh.ttc") else None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    piv_nrmse = df_res.pivot(index="H", columns="mode", values="nRMSE")
    piv_r2    = df_res.pivot(index="H", columns="mode", values="R2")

    # nRMSE 對比長條圖
    piv_nrmse.plot(kind="bar", ax=axes[0], color=["#d62728", "#2ca02c"], width=0.6, edgecolor="black")
    axes[0].set_title("單點預測 vs 區間能量預測 nRMSE 誤差比較 (越低越好)", fontproperties=fp_title)
    axes[0].set_xlabel("預測時間區間 H (小時)", fontproperties=fp)
    axes[0].set_ylabel("nRMSE", fontproperties=fp)
    axes[0].legend(prop=fp)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Annotate delta %
    for idx, row in enumerate(piv_nrmse.iterrows()):
        h_val, vals = row
        p_val, i_val = vals["單點瞬時預測 (Point)"], vals["區間總能量預測 (Interval)"]
        pct = ((i_val - p_val) / p_val) * 100
        axes[0].annotate(f"{pct:.1f}%", xy=(idx, i_val + 0.01), ha="center", va="bottom", fontweight="bold", color="#2ca02c")

    # R² 對比長條圖
    piv_r2.plot(kind="bar", ax=axes[1], color=["#d62728", "#2ca02c"], width=0.6, edgecolor="black")
    axes[1].set_title("單點預測 vs 區間能量預測 R² 擬合度比較 (越高越精準)", fontproperties=fp_title)
    axes[1].set_xlabel("預測時間區間 H (小時)", fontproperties=fp)
    axes[1].set_ylabel("R² 得分", fontproperties=fp)
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(prop=fp)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    for idx, row in enumerate(piv_r2.iterrows()):
        h_val, vals = row
        i_val = vals["區間總能量預測 (Interval)"]
        axes[1].annotate(f"R²={i_val:.3f}", xy=(idx + 0.15, i_val + 0.02), ha="center", va="bottom", fontweight="bold", color="#2ca02c", fontsize=9)

    plt.tight_layout()
    fig_path = os.path.join(C.FIG_DIR, "fig5_point_vs_interval_comparison.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"差異對比圖表已產出：{fig_path}")
    print(f"差異對比圖表已產出：{fig_path}")

    print("\nStage 5 差異比對實驗完成。")

if __name__ == "__main__":
    main()
