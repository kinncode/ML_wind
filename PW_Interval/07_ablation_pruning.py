#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 7 —— 特徵剪枝與消融實驗 (Feature Pruning & Ablation Study)

目的：
  驗證剔除低重要度特徵對區間發電能量預測精確度、R² 擬合度與訓練效率的影響。

比較實驗配置：
  1. Full Baseline (完整 44 特徵)
  2. Pruned Top-22 (剪枝剔除後 50% 低重要度特徵，保留 Top 22)
  3. Pruned Top-10 (極簡模式：僅保留前 10 大核心特徵)
  4. No-Cyclical (剔除時間週期特徵 hour/doy sin/cos)
  5. No-Turbulence (剔除湍流與風切特徵 ti, shear_alpha)
  6. No-Atmosphere (剔除氣壓/溫度/濕度/密度)
  7. No-Lags (剔除滯後風速 lag10..180)

輸出：
  results/ablation_pruning_metrics.csv
  figures/fig8_ablation_pruning_results.png
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
    print("PW_Interval Stage 7 —— 特徵剪枝與消融實驗 (Ablation Study)")
    print("="*70)

    if not os.path.exists(C.FEAT_PARQUET):
        raise FileNotFoundError(f"找不到 {C.FEAT_PARQUET}，請先執行 02_features.py")

    df = pd.read_parquet(C.FEAT_PARQUET)
    df["ts"] = pd.to_datetime(df["ts"])
    test_start = pd.Timestamp(C.TEST_START)

    meta = {"ts", "is_ok", "year", "month", "hour_i"}
    all_fcols = [c for c in df.columns if c not in meta and not c.startswith("y_") and not c.startswith("m_")]

    # 取得特徵重要度 Gain 排名
    h_ref = 3
    imp_csv = os.path.join(C.RES_DIR, f"importance_power_H{h_ref}.csv")
    if os.path.exists(imp_csv):
        df_imp = pd.read_csv(imp_csv)
        ranked_fcols = df_imp["feature"].tolist()
    else:
        ranked_fcols = list(all_fcols)

    top_22_fcols = ranked_fcols[:22]
    top_10_fcols = ranked_fcols[:10]

    # 定義消融組合
    configs = {
        "Full Baseline (全 44 特徵)": all_fcols,
        "Pruned Top-22 (保留前 50% 核心特徵)": top_22_fcols,
        "Pruned Top-10 (極簡模式 Top-10)": top_10_fcols,
        "無時間週期 (No Cyclical)": [c for c in all_fcols if not (c.startswith("hour_") or c.startswith("doy_"))],
        "無湍流與風切 (No Turbulence)": [c for c in all_fcols if not ("shear" in c or "ti" in c or "gust" in c or "std" in c)],
        "無氣象密度 (No Atmosphere)": [c for c in all_fcols if not (c.startswith("air_density") or c.startswith("AT_") or c.startswith("RH_") or c.startswith("BP_"))],
        "無風速滯後 (No Lags)": [c for c in all_fcols if not c.startswith("ws_lag")]
    }

    lgb_params = dict(objective="regression", n_estimators=350, learning_rate=0.05,
                      num_leaves=48, min_child_samples=100, subsample=0.8,
                      colsample_bytree=0.8, reg_lambda=1.0, random_state=C.RANDOM_SEED,
                      n_jobs=-1, verbose=-1)

    P_now_full = C.virtual_power(df["WS_100_mean"], df["air_density"])
    P_series = pd.Series(P_now_full, index=df.index)

    results = []

    for h in [1, 3, 6]:
        k = C.HORIZON_STEPS[h]
        y_intv = P_series.iloc[::-1].rolling(k).mean().iloc[::-1]

        mask = df["is_ok"] & df[f"m_{h}"] & y_intv.notna() & df[all_fcols].notna().all(axis=1)
        sub = df.loc[mask]
        sub_y = y_intv.loc[mask]

        is_test = sub["ts"] >= test_start
        tr_sub, te_sub = sub.loc[~is_test], sub.loc[is_test]
        yte = sub_y.loc[te_sub.index].values
        nval = int(len(tr_sub) * 0.15)
        denom = np.mean(yte)

        print(f"\n--- 測試時間視窗 H={h}h 區間能量預測 ---")

        base_nrmse = None

        for cfg_name, fcols in configs.items():
            t0 = time.time()
            tr_X = tr_sub[fcols].values
            te_X = te_sub[fcols].values
            ytr  = sub_y.loc[tr_sub.index].values

            gbm = lgb.LGBMRegressor(**lgb_params)
            gbm.fit(tr_X[:-nval], ytr[:-nval], eval_set=[(tr_X[-nval:], ytr[-nval:])],
                    callbacks=[lgb.early_stopping(40, verbose=False)])

            pred = np.clip(gbm.predict(te_X), 0, 1)
            t_cost = time.time() - t0

            nrmse_val = rmse(yte, pred) / denom
            r2_val    = float(r2_score(yte, pred))

            if cfg_name.startswith("Full Baseline"):
                base_nrmse = nrmse_val
                delta_pct = 0.0
            else:
                delta_pct = ((nrmse_val - base_nrmse) / base_nrmse) * 100.0

            print(f"  [{cfg_name:32s}] 特徵數={len(fcols):2d} ｜ nRMSE={nrmse_val:.4f} ({delta_pct:+.2f}%) ｜ R²={r2_val:.4f} ｜ 耗時={t_cost:.2f}s")

            results.append({
                "H": f"{h}h",
                "config": cfg_name,
                "n_features": len(fcols),
                "nRMSE": round(nrmse_val, 5),
                "R2": round(r2_val, 5),
                "delta_nrmse_pct": round(delta_pct, 2),
                "time_sec": round(t_cost, 2)
            })

    df_res = pd.DataFrame(results)
    csv_out = os.path.join(C.RES_DIR, "ablation_pruning_metrics.csv")
    df_res.to_csv(csv_out, index=False)
    print(f"\n消融實驗結果已寫入：{csv_out}")

    # 繪製視覺化圖表
    fp = FontProperties(fname="C:/Windows/Fonts/msjh.ttc", size=10) if os.path.exists("C:/Windows/Fonts/msjh.ttc") else None
    fp_title = FontProperties(fname="C:/Windows/Fonts/msjh.ttc", size=12, weight="bold") if os.path.exists("C:/Windows/Fonts/msjh.ttc") else None

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 子圖 1: 3h 區間下特徵剪枝與消融對 nRMSE 變化的影響 (%)
    df_h3 = df_res[df_res["H"] == "3h"]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b", "#e377c2"]

    bars = axes[0].barh(df_h3["config"], df_h3["delta_nrmse_pct"], color=colors, edgecolor="black", alpha=0.85)
    axes[0].set_title("3 小時區間能量預測 特徵消融/剪枝相對誤差變化率 (ΔnRMSE %)", fontproperties=fp_title)
    axes[0].set_xlabel("相對全特徵基準誤差變化率 (%) (越低越好，>0 表示性能下降)", fontproperties=fp)
    axes[0].set_yticks(range(len(df_h3)))
    axes[0].set_yticklabels(df_h3["config"], fontproperties=fp)
    axes[0].axvline(0, color="black", linestyle="--", linewidth=1.5)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    for bar in bars:
        w_val = bar.get_width()
        axes[0].annotate(f"{w_val:+.2f}%", xy=(w_val, bar.get_y() + bar.get_height() / 2),
                         xytext=(5 if w_val >= 0 else -30, 0), textcoords="offset points",
                         ha="left", va="center", fontweight="bold", fontsize=9)

    # 子圖 2: 1h, 3h, 6h 區間下前 22 特徵剪枝與全特徵 R² 對比
    piv_r2 = df_res.pivot(index="H", columns="config", values="R2")
    cols_to_plot = ["Full Baseline (全 44 特徵)", "Pruned Top-22 (保留前 50% 核心特徵)", "Pruned Top-10 (極簡模式 Top-10)"]
    piv_r2[cols_to_plot].plot(kind="bar", ax=axes[1], color=["#1f77b4", "#2ca02c", "#ff7f0e"], width=0.6, edgecolor="black")

    axes[1].set_title("特徵剪枝前後 R² 擬合度對比", fontproperties=fp_title)
    axes[1].set_xlabel("預測時間區間 H", fontproperties=fp)
    axes[1].set_ylabel("R² 得分", fontproperties=fp)
    axes[1].set_xticks([0, 1, 2])
    axes[1].set_xticklabels(["1 小時 (1h)", "3 小時 (3h)", "6 小時 (6h)"], fontproperties=fp)
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(prop=fp)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_path = os.path.join(C.FIG_DIR, "fig8_ablation_pruning_results.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print(f"消融視覺化圖表已產出：{fig_path}")
    print("Stage 7 完成。")

if __name__ == "__main__":
    main()
