#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 8 —— 極限最長區間發電能量預測 (Max-Horizon Interval Forecast: 12h ~ 168h / 7日)

目的：
  探討在無數值天氣預報 (NWP) 輔助下，純依據測風塔觀測與氣候時間特徵能達到的「最長區間發電能量預測視窗」。

評估極限區間：
  - 12 小時 (12h / 半日)
  - 24 小時 (24h / 1日)
  - 48 小時 (48h / 2日)
  - 72 小時 (72h / 3日)
  - 168 小時 (168h / 7日 週累積發電能量 —— 極限預測視窗)

輸出：
  results/max_horizon_metrics.json
  results/max_horizon_metrics.csv
  figures/fig15_max_horizon_r2_nrmse.png
  figures/fig16_weekly_168h_energy_forecast.png
"""
from __future__ import annotations
import os, json, time
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
    print("PW_Interval Stage 8 —— 極限最長區間發電能量預測 (12h ~ 168h / 7日)")
    print("="*70)

    if not os.path.exists(C.FEAT_PARQUET):
        raise FileNotFoundError(f"找不到 {C.FEAT_PARQUET}，請先執行 02_features.py")

    df = pd.read_parquet(C.FEAT_PARQUET)
    df["ts"] = pd.to_datetime(df["ts"])
    test_start = pd.Timestamp(C.TEST_START)

    meta = {"ts", "is_ok", "year", "month", "hour_i"}
    fcols = [c for c in df.columns if c not in meta and not c.startswith("y_") and not c.startswith("m_")]

    P_now_full = C.virtual_power(df["WS_100_mean"], df["air_density"])
    P_series = pd.Series(P_now_full, index=df.index)

    # 定義最長時間視窗 (小時)
    max_horizons_h = [1, 3, 6, 12, 24, 48, 72, 168]

    lgb_params = dict(objective="regression", n_estimators=450, learning_rate=0.03,
                      num_leaves=63, min_child_samples=100, subsample=0.8,
                      colsample_bytree=0.8, reg_lambda=1.0, random_state=C.RANDOM_SEED,
                      n_jobs=-1, verbose=-1)

    results = []
    plot_series_168h = {}

    for h in max_horizons_h:
        k = int(h * 60 // C.STEP_MIN)  # 步階數 (10min 解析度)

        # 未來 h 小時區間累積總能量 (等效滿載小時數)
        y_intv = P_series.iloc[::-1].rolling(k).mean().iloc[::-1]

        # 無洩漏遮罩：要求未來 k 步無缺漏
        ok = df["is_ok"].values
        notok = (~ok).astype(np.int64)
        cs = np.concatenate([[0], np.cumsum(notok)])
        N = len(ok)
        m = np.zeros(N, dtype=bool)
        valid_end = N - k
        idx = np.arange(valid_end)
        bad = cs[idx + k + 1] - cs[idx + 1]
        m[idx] = (bad == 0)

        mask = df["is_ok"] & m & y_intv.notna() & df[fcols].notna().all(axis=1)
        sub = df.loc[mask]
        sub_y = y_intv.loc[mask]

        is_test = sub["ts"] >= test_start
        tr_X, te_X = sub.loc[~is_test][fcols], sub.loc[is_test][fcols]
        tr_y, te_y = sub_y.loc[~is_test].values, sub_y.loc[is_test].values
        persist_te = te_X["P_now"].values

        nval = int(len(tr_X) * 0.15)
        gbm = lgb.LGBMRegressor(**lgb_params)
        gbm.fit(tr_X.values[:-nval], tr_y[:-nval], eval_set=[(tr_X.values[-nval:], tr_y[-nval:])],
                callbacks=[lgb.early_stopping(40, verbose=False)])

        pred = np.clip(gbm.predict(te_X.values), 0, 1)

        denom = np.mean(te_y) if np.mean(te_y) > 1e-6 else 1.0
        nrmse_ml = rmse(te_y, pred) / denom
        nrmse_per = rmse(te_y, persist_te) / denom
        r2_ml = float(r2_score(te_y, pred))
        r2_per = float(r2_score(te_y, persist_te))
        skill = (1 - nrmse_ml / nrmse_per) * 100.0

        label_h = f"{h}h ({h//24}日)" if h >= 24 else f"{h}h"
        print(f"  區間 H={label_h:12s} ｜ ML R²={r2_ml:.4f} (Persist R²={r2_per:.4f}) ｜ ML nRMSE={nrmse_ml:.4f} vs Persist {nrmse_per:.4f} (提升 {skill:+.1f}%)")

        results.append({
            "H_hours": h,
            "horizon_label": label_h,
            "k_steps": k,
            "test_samples": len(te_y),
            "nrmse_ml": round(nrmse_ml, 5),
            "nrmse_persist": round(nrmse_per, 5),
            "r2_ml": round(r2_ml, 5),
            "r2_persist": round(r2_per, 5),
            "skill_pct": round(skill, 2)
        })

        if h == 168: # 保存 7 日週區間能量比對資料
            plot_series_168h["ts"] = sub.loc[is_test]["ts"]
            plot_series_168h["y_true"] = te_y
            plot_series_168h["pred_ml"] = pred
            plot_series_168h["persist"] = persist_te

    df_res = pd.DataFrame(results)
    os.makedirs(C.RES_DIR, exist_ok=True)
    os.makedirs(C.FIG_DIR, exist_ok=True)

    csv_out = os.path.join(C.RES_DIR, "max_horizon_metrics.csv")
    df_res.to_csv(csv_out, index=False)
    print(f"\n極限區間預測結果已寫入：{csv_out}")

    fp = FontProperties(fname="C:/Windows/Fonts/msjh.ttc", size=10) if os.path.exists("C:/Windows/Fonts/msjh.ttc") else None
    fp_title = FontProperties(fname="C:/Windows/Fonts/msjh.ttc", size=12, weight="bold") if os.path.exists("C:/Windows/Fonts/msjh.ttc") else None

    # ------------------------------------------------------------------
    # 圖表 1: fig15_max_horizon_r2_nrmse.png (1h ~ 168h 極限區間性能衰減與擬合曲線)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    x_labels = [r["horizon_label"] for r in results]
    r2_ml_list = [r["r2_ml"] for r in results]
    r2_per_list = [r["r2_persist"] for r in results]
    nrmse_ml_list = [r["nrmse_ml"] for r in results]
    nrmse_per_list = [r["nrmse_persist"] for r in results]

    x_idx = np.arange(len(results))

    # 子圖 1: R² 擬合度衰減與宏觀氣候回升折線圖
    axes[0].plot(x_idx, r2_ml_list, color="#2ca02c", marker="o", linewidth=2.5, label="ML 區間總能量預測 R²")
    axes[0].plot(x_idx, r2_per_list, color="#d62728", marker="x", linewidth=2.0, linestyle="--", label="Persistence R²")
    axes[0].set_title("1h ~ 168h (7日) 極限區間發電能量預測 R² 擬合曲線", fontproperties=fp_title)
    axes[0].set_xlabel("預測時間區間 H", fontproperties=fp)
    axes[0].set_ylabel("R² 得分", fontproperties=fp)
    axes[0].set_xticks(x_idx)
    axes[0].set_xticklabels(x_labels, fontproperties=fp)
    axes[0].set_ylim(-0.1, 1.05)
    axes[0].legend(prop=fp)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    for i, r2 in enumerate(r2_ml_list):
        axes[0].annotate(f"{r2:.3f}", xy=(i, r2), xytext=(0, 8), textcoords="offset points", ha="center", fontweight="bold", color="#2ca02c", fontsize=9)

    # 子圖 2: nRMSE 誤差比較與勝過 Persistence 比例
    w = 0.35
    axes[1].bar(x_idx - w/2, nrmse_ml_list, w, label="ML 區間預測 nRMSE", color="#1f77b4", edgecolor="black")
    axes[1].bar(x_idx + w/2, nrmse_per_list, w, label="Persistence nRMSE", color="#d62728", alpha=0.6, edgecolor="black")
    axes[1].set_title("1h ~ 168h 區間預測 nRMSE 誤差比較 (越低越好)", fontproperties=fp_title)
    axes[1].set_xlabel("預測時間區間 H", fontproperties=fp)
    axes[1].set_ylabel("nRMSE", fontproperties=fp)
    axes[1].set_xticks(x_idx)
    axes[1].set_xticklabels(x_labels, fontproperties=fp)
    axes[1].legend(prop=fp)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig15_path = os.path.join(C.FIG_DIR, "fig15_max_horizon_r2_nrmse.png")
    plt.savefig(fig15_path, dpi=200)
    plt.close()
    print(f"極限區間擬合曲線圖已產出：{fig15_path}")

    # ------------------------------------------------------------------
    # 圖表 2: fig16_weekly_168h_energy_forecast.png (168h / 7日週累積總能量預測時序圖)
    # ------------------------------------------------------------------
    if plot_series_168h:
        fig, ax = plt.subplots(figsize=(15, 5))
        ts_168 = plot_series_168h["ts"]
        y_168 = plot_series_168h["y_true"]
        p_168 = plot_series_168h["pred_ml"]

        # 取測試集約 4 個月時序展現 7 日週能量趨勢
        step_sub = 6 * 24  # 每天取 1 點繪製滑順趨勢
        ax.plot(ts_168[::step_sub], y_168[::step_sub], color="black", label="真實 168h (7日) 週累積發電總能量 (Real E_168h)", linewidth=2.0)
        ax.plot(ts_168[::step_sub], p_168[::step_sub], color="#2ca02c", label=f"ML 7日週區間能量預測 (R²={results[-1]['r2_ml']:.4f})", linewidth=2.0, linestyle="--")

        ax.set_title("測試集 7 日 (168h) 週累積發電能量預測時序比對", fontproperties=fp_title)
        ax.set_xlabel("預測發起時間點 t (預測從 t 開始未來 7 天 [t ~ t+7日] 之累積總發電能量)", fontproperties=fp)
        ax.set_ylabel("7日區間累積發電能量 (等效滿載小時數 h)", fontproperties=fp)
        ax.legend(prop=fp)
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        fig16_path = os.path.join(C.FIG_DIR, "fig16_weekly_168h_energy_forecast.png")
        plt.savefig(fig16_path, dpi=200)
        plt.close()
        print(f"7日週極限預測圖表已產出：{fig16_path}")

    print("\nStage 8 完成。")

if __name__ == "__main__":
    main()
