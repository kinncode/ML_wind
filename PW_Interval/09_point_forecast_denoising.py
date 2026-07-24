#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 9 —— 單點預測降噪與平滑技術實驗 (Single-Point Forecast Denoising Study)

目的：
  針對單點預測 (Single-Point P_{t+H}) 高頻陣風噪訊極大的問題，測試多種訊號降噪與平滑濾波演算法，
  驗證能否有效降低單點預測雜訊並提升 R² 擬合度。

測試降噪演算法：
  1. Raw Baseline (原始含噪單點)
  2. Savitzky-Golay 卷積平滑濾波 (Savitzky-Golay Filter)
  3. Exponential Moving Average (EMA 指數加權平滑)
  4. Low-Pass Butterworth 巴特沃斯低通濾波 (Butterworth Low-Pass)
  5. Target & Feature Joint Smoothing (輸入與標的雙向協同降噪)

輸出：
  results/denoising_single_point_metrics.csv
  figures/fig17_denoising_single_point_comparison.png
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.signal import savgol_filter, butter, filtfilt
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

# 濾波函數 1: Savitzky-Golay
def apply_savgol(s: pd.Series, window=9, poly=2) -> pd.Series:
    vals = s.interpolate().values
    clean = savgol_filter(vals, window_length=window, polyorder=poly)
    return pd.Series(clean, index=s.index)

# 濾波函數 2: EMA
def apply_ema(s: pd.Series, span=3) -> pd.Series:
    return s.ewm(span=span).mean()

# 濾波函數 3: Butterworth 低通濾波
def apply_butterworth(s: pd.Series, cutoff_period_steps=4) -> pd.Series:
    vals = s.interpolate().values
    b, a = butter(N=2, Wn=1.0/cutoff_period_steps, btype='low')
    clean = filtfilt(b, a, vals)
    return pd.Series(clean, index=s.index)

def main():
    print("="*70)
    print("PW_Interval Stage 9 —— 單點預測降噪與平滑技術對比實驗")
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

    P_raw = pd.Series(C.virtual_power(df["WS_100_mean"], df["air_density"]), index=df.index)

    # 生成 3 種降噪後的 P 序列
    P_savgol = apply_savgol(P_raw, window=9, poly=2)
    P_ema    = apply_ema(P_raw, span=3)
    P_butter = apply_butterworth(P_raw, cutoff_period_steps=4)

    denoise_methods = {
        "1. Raw Baseline (原始含噪單點)": P_raw,
        "2. Savitzky-Golay 卷積平滑": P_savgol,
        "3. EMA 指數加權平滑": P_ema,
        "4. Butterworth 巴特沃斯低通": P_butter
    }

    results = []

    for h in [1, 3, 6]:
        k = C.HORIZON_STEPS[h]
        print(f"\n--- 測試提前量 H={h}h 單點預測降噪對比 ---")

        for m_name, p_series in denoise_methods.items():
            y_target = p_series.shift(-k)

            mask = df["is_ok"] & df[f"m_{h}"] & y_target.notna() & df[fcols].notna().all(axis=1)
            sub = df.loc[mask]
            sub_y = y_target.loc[mask]

            is_test = sub["ts"] >= test_start
            tr_X, te_X = sub.loc[~is_test][fcols], sub.loc[is_test][fcols]
            tr_y, te_y = sub_y.loc[~is_test].values, sub_y.loc[is_test].values

            nval = int(len(tr_X) * 0.15)
            gbm = lgb.LGBMRegressor(**lgb_params)
            gbm.fit(tr_X.values[:-nval], tr_y[:-nval], eval_set=[(tr_X.values[-nval:], tr_y[-nval:])],
                    callbacks=[lgb.early_stopping(40, verbose=False)])

            pred = np.clip(gbm.predict(te_X.values), 0, 1)

            denom = np.mean(te_y) if np.mean(te_y) > 1e-6 else 1.0
            nrmse_val = rmse(te_y, pred) / denom
            r2_val    = float(r2_score(te_y, pred))

            print(f"  [{m_name:30s}] nRMSE = {nrmse_val:.4f} ｜ R² = {r2_val:.4f}")

            results.append({
                "H": f"{h}h",
                "method": m_name,
                "nRMSE": round(nrmse_val, 5),
                "R2": round(r2_val, 5)
            })

    # 對比區間總能量預測 (作為金標對照)
    for h in [1, 3, 6]:
        k = C.HORIZON_STEPS[h]
        y_intv = P_raw.iloc[::-1].rolling(k).mean().iloc[::-1]
        mask = df["is_ok"] & df[f"m_{h}"] & y_intv.notna() & df[fcols].notna().all(axis=1)
        sub = df.loc[mask]
        sub_y = y_intv.loc[mask]
        is_test = sub["ts"] >= test_start
        tr_X, te_X = sub.loc[~is_test][fcols], sub.loc[is_test][fcols]
        tr_y, te_y = sub_y.loc[~is_test].values, sub_y.loc[is_test].values

        nval = int(len(tr_X) * 0.15)
        gbm = lgb.LGBMRegressor(**lgb_params)
        gbm.fit(tr_X.values[:-nval], tr_y[:-nval], eval_set=[(tr_X.values[-nval:], tr_y[-nval:])],
                callbacks=[lgb.early_stopping(40, verbose=False)])

        pred = np.clip(gbm.predict(te_X.values), 0, 1)
        denom = np.mean(te_y)
        nrmse_val = rmse(te_y, pred) / denom
        r2_val = float(r2_score(te_y, pred))

        results.append({
            "H": f"{h}h",
            "method": "5. 區間發電能量預測 (Interval)",
            "nRMSE": round(nrmse_val, 5),
            "R2": round(r2_val, 5)
        })

    df_res = pd.DataFrame(results)
    csv_out = os.path.join(C.RES_DIR, "denoising_single_point_metrics.csv")
    df_res.to_csv(csv_out, index=False)
    print(f"\n單點降噪對比結果已寫入：{csv_out}")

    # 繪製視覺化圖表
    fp = FontProperties(fname="C:/Windows/Fonts/msjh.ttc", size=10) if os.path.exists("C:/Windows/Fonts/msjh.ttc") else None
    fp_title = FontProperties(fname="C:/Windows/Fonts/msjh.ttc", size=12, weight="bold") if os.path.exists("C:/Windows/Fonts/msjh.ttc") else None

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    piv_r2    = df_res.pivot(index="H", columns="method", values="R2")
    piv_nrmse = df_res.pivot(index="H", columns="method", values="nRMSE")

    # R² 降噪對比長條圖
    piv_r2.plot(kind="bar", ax=axes[0], colormap="tab10", width=0.75, edgecolor="black")
    axes[0].set_title("單點預測降噪演算法 vs 區間預測 R² 擬合度比較", fontproperties=fp_title)
    axes[0].set_xlabel("預測時間區間 H", fontproperties=fp)
    axes[0].set_ylabel("R² 得分", fontproperties=fp)
    axes[0].set_xticks([0, 1, 2])
    axes[0].set_xticklabels(["1 小時 (1h)", "3 小時 (3h)", "6 小時 (6h)"], fontproperties=fp)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(prop=fp, loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # nRMSE 對比長條圖
    piv_nrmse.plot(kind="bar", ax=axes[1], colormap="Set2", width=0.75, edgecolor="black")
    axes[1].set_title("單點預測降噪演算法 vs 區間預測 nRMSE 誤差比較 (越低越好)", fontproperties=fp_title)
    axes[1].set_xlabel("預測時間區間 H", fontproperties=fp)
    axes[1].set_ylabel("nRMSE", fontproperties=fp)
    axes[1].set_xticks([0, 1, 2])
    axes[1].set_xticklabels(["1 小時 (1h)", "3 小時 (3h)", "6 小時 (6h)"], fontproperties=fp)
    axes[1].legend(prop=fp, loc="upper left")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_path = os.path.join(C.FIG_DIR, "fig17_denoising_single_point_comparison.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print(f"單點降噪視覺化對比圖表已產出：{fig_path}")
    print("Stage 9 完成。")

if __name__ == "__main__":
    main()
