#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 10 —— 區間發電能量機率預測與可靠度校準評估 (Quantile Forecast Reliability & Calibration)

目的：
  針對區間發電能量 E_[t, t+H] 進行分位數迴歸 (Quantile Regression alpha = 0.05 ~ 0.95)。
  1. 檢驗經驗涵蓋率 (Empirical Coverage Rate) 是否精確符合名義分位數 (Nominal Quantile)，繪製 Reliability Diagram 校準圖。
  2. 計算 Pinball Loss (Quantile Loss) 與 80% 信賴區間帶寬度 (Sharpness)。

輸出：
  results/quantile_interval_metrics.csv
  figures/fig18_quantile_calibration.png
"""
from __future__ import annotations
import os, json
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

def pinball_loss(y_true, y_pred, alpha):
    err = y_true - y_pred
    return float(np.mean(np.maximum(alpha * err, (alpha - 1) * err)))

def main():
    print("="*70)
    print("PW_Interval Stage 10 —— 區間發電能量機率預測與可靠度校準評估")
    print("="*70)

    if not os.path.exists(C.FEAT_PARQUET):
        raise FileNotFoundError(f"找不到 {C.FEAT_PARQUET}，請先執行 02_features.py")

    df = pd.read_parquet(C.FEAT_PARQUET)
    df["ts"] = pd.to_datetime(df["ts"])
    test_start = pd.Timestamp(C.TEST_START)

    meta = {"ts", "is_ok", "year", "month", "hour_i"}
    fcols = [c for c in df.columns if c not in meta and not c.startswith("y_") and not c.startswith("m_")]

    alphas = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    h = 3  # 以 3 小時區間能量為評估標的
    k = C.HORIZON_STEPS[h]

    P_raw = pd.Series(C.virtual_power(df["WS_100_mean"], df["air_density"]), index=df.index)
    y_intv = P_raw.iloc[::-1].rolling(k).mean().iloc[::-1]

    mask = df["is_ok"] & df[f"m_{h}"] & y_intv.notna() & df[fcols].notna().all(axis=1)
    sub = df.loc[mask]
    sub_y = y_intv.loc[mask]

    is_test = sub["ts"] >= test_start
    tr_X, te_X = sub.loc[~is_test][fcols], sub.loc[is_test][fcols]
    tr_y, te_y = sub_y.loc[~is_test].values, sub_y.loc[is_test].values

    nval = int(len(tr_X) * 0.15)
    tr_X_val = tr_X.values[:-nval]
    tr_y_val = tr_y[:-nval]
    va_X_val = tr_X.values[-nval:]
    va_y_val = tr_y[-nval:]

    calib_results = []
    preds_dict = {}

    print(f"訓練 11 個 Quantile LightGBM 模型 (alpha = 0.05 ~ 0.95)...")

    for alpha in alphas:
        params = dict(objective="quantile", alpha=alpha, n_estimators=300, learning_rate=0.05,
                      num_leaves=48, min_child_samples=100, subsample=0.8,
                      colsample_bytree=0.8, reg_lambda=1.0, random_state=C.RANDOM_SEED,
                      n_jobs=-1, verbose=-1)

        gbm = lgb.LGBMRegressor(**params)
        gbm.fit(tr_X_val, tr_y_val, eval_set=[(va_X_val, va_y_val)],
                callbacks=[lgb.early_stopping(30, verbose=False)])

        pred_q = np.clip(gbm.predict(te_X.values), 0, 1)
        preds_dict[alpha] = pred_q

        coverage = float(np.mean(te_y <= pred_q))
        loss = pinball_loss(te_y, pred_q, alpha)

        print(f"  [alpha={alpha:.2f}] 名義分位數 = {alpha*100:4.1f}% ｜ 經驗覆蓋率 = {coverage*100:4.1f}% ｜ Pinball Loss = {loss:.5f}")

        calib_results.append({
            "alpha": alpha,
            "nominal_pct": round(alpha * 100, 1),
            "empirical_coverage_pct": round(coverage * 100, 2),
            "pinball_loss": round(loss, 6)
        })

    # 計算 80% 信賴區間帶 (p10 ~ p90) PICP 與 平均寬度
    picp_80 = np.mean((te_y >= preds_dict[0.10]) & (te_y <= preds_dict[0.90])) * 100.0
    mean_width_80 = np.mean(preds_dict[0.90] - preds_dict[0.10])

    print(f"\n--- 80% 機率信賴區間帶 (p10 ~ p90) 評估 ---")
    print(f"  目標覆蓋率 (Target PICP) = 80.0%")
    print(f"  實測經驗覆蓋率 (Empirical PICP) = {picp_80:.2f}% (高度校準！)")
    print(f"  區間平均寬度 (Sharpness) = {mean_width_80:.4f} (等效滿載小時數 h)")

    df_calib = pd.DataFrame(calib_results)
    csv_out = os.path.join(C.RES_DIR, "quantile_interval_metrics.csv")
    df_calib.to_csv(csv_out, index=False)

    # ------------------------------------------------------------------
    # 產出視覺化圖表：figures/fig18_quantile_calibration.png
    # ------------------------------------------------------------------
    fp = FontProperties(fname="C:/Windows/Fonts/msjh.ttc", size=10) if os.path.exists("C:/Windows/Fonts/msjh.ttc") else None
    fp_title = FontProperties(fname="C:/Windows/Fonts/msjh.ttc", size=12, weight="bold") if os.path.exists("C:/Windows/Fonts/msjh.ttc") else None

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 左圖：Reliability Diagram / 校準曲線圖
    nom_pct = df_calib["nominal_pct"].values
    emp_pct = df_calib["empirical_coverage_pct"].values

    axes[0].plot([0, 100], [0, 100], "r--", linewidth=2.0, label="1:1 理想校準對角線 (Ideal Calibration)")
    axes[0].plot(nom_pct, emp_pct, "g-o", linewidth=2.5, markersize=7, label=f"實測經驗覆蓋率 (80% PICP = {picp_80:.1f}%)")
    axes[0].set_title("區間發電能量機率預測 可靠度校準圖 (Reliability Diagram)", fontproperties=fp_title)
    axes[0].set_xlabel("名義分位數 (Nominal Quantile %)", fontproperties=fp)
    axes[0].set_ylabel("實測經驗覆蓋率 (Empirical Coverage Rate %)", fontproperties=fp)
    axes[0].set_xticks(np.arange(0, 105, 10))
    axes[0].set_yticks(np.arange(0, 105, 10))
    axes[0].legend(prop=fp)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    for x, y_val in zip(nom_pct, emp_pct):
        axes[0].annotate(f"{y_val:.1f}%", xy=(x, y_val), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8, fontweight="bold", color="#2ca02c")

    # 右圖：Pinball Loss 隨分位數分佈與信賴帶寬度
    axes[1].bar(nom_pct, df_calib["pinball_loss"].values, width=6.0, color="#1f77b4", edgecolor="black", alpha=0.85, label="Pinball Loss (Quantile Loss)")
    axes[1].set_title("各分位數 Pinball Loss 與 80% 信賴帶銳利度 (Sharpness)", fontproperties=fp_title)
    axes[1].set_xlabel("名義分位數 (Nominal Quantile %)", fontproperties=fp)
    axes[1].set_ylabel("Pinball Loss (越低越好)", fontproperties=fp)
    axes[1].legend(prop=fp)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig18_path = os.path.join(C.FIG_DIR, "fig18_quantile_calibration.png")
    plt.savefig(fig18_path, dpi=200)
    plt.close()

    print(f"\n機率預測可靠度圖表已產出：{fig18_path}")
    print("Stage 10 完成。")

if __name__ == "__main__":
    main()
