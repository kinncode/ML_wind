#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 12 —— 台電 15 分鐘 (15-min) 綠電轉供與電力交易平台 (ETP) 區間能量預測模擬模組

背景：
  台電與綠電交易市場 (CPPA / REC) 規範以每 15 分鐘 (00, 15, 30, 45 分鐘) 為基本電表計量與結算區間。
  本腳本將 BSMI 測風塔 10 分鐘觀測數據透過線性插值重採樣為「台電 15 分鐘智慧電表區間累積度數 (kWh)」，
  並評估 15-min、1-hr、3-hr 區間發電能量預測精度與台電偏差避險效果。

輸出：
  results/taipower_15min_metrics.json
  figures/fig13_taipower_15min_settlement.png
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

def rmse(a, b): return float(np.sqrt(np.mean((a - b)**2)))

def main():
    print("="*70)
    print("PW_Interval Stage 12 —— 台電 15 分鐘 (15-min) 綠電轉供區間能量預測對接")
    print("="*70)

    if not os.path.exists(C.CLEAN_PARQUET):
        raise FileNotFoundError(f"找不到 {C.CLEAN_PARQUET}，請先執行 01_load_validate.py")

    df_10min = pd.read_parquet(C.CLEAN_PARQUET).set_index("ts").sort_index()

    # 1. 測風塔 10-min 轉換為台電 15-min 智慧電表時間軸
    print("\n[步驟 1] 10 分鐘測風塔觀測數據 ➔ 台電 15 分鐘 (15-min) 智慧電表區間重採樣...")

    # 計算 10-min 出力 P_10min (0-1)
    df_10min["P_10min"] = C.virtual_power(df_10min["WS_100_mean"], df_10min["air_density"])

    # 透過高精度線性插值將 10min 重採樣為 15min (00, 15, 30, 45 分鐘)
    df_15min = df_10min[["WS_100_mean", "P_10min", "air_density", "is_ok"]].resample("15min").interpolate(method="time")
    df_15min["is_ok"] = df_10min["is_ok"].resample("15min").min().fillna(False).astype(bool)

    print(f"  重採樣後台電 15 分鐘電表網格筆數：{len(df_15min):,} 筆 (起訖：{df_15min.index.min()} ~ {df_15min.index.max()})")

    # 2. 建立 15-min 區間預測特徵與標的 (1 個 15min 步階、4 個步階=1hr、12 個步階=3hr)
    print("\n[步驟 2] 建立台電 15-min 結算規範之區間累積發電能量標的...")

    ws_15 = df_15min["WS_100_mean"]
    P_15 = df_15min["P_10min"]

    # 滯後特徵 (15min, 30min, 45min, 60min, 120min, 180min)
    feat_15 = pd.DataFrame(index=df_15min.index)
    feat_15["WS_100_mean"] = ws_15
    feat_15["P_now"] = P_15
    feat_15["air_density"] = df_15min["air_density"]

    for m in [15, 30, 45, 60, 120, 180]:
        feat_15[f"ws_lag_{m}m"] = ws_15.shift(m // 15)

    # 滾動統計 (60min=4, 180min=12)
    for win in [4, 12]:
        r = ws_15.rolling(win, min_periods=win//2)
        feat_15[f"ws_rmean_{win*15}m"] = r.mean()
        feat_15[f"ws_rstd_{win*15}m"]  = r.std()

    hour = df_15min.index.hour + df_15min.index.minute / 60.0
    doy  = df_15min.index.dayofyear
    feat_15["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    feat_15["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    feat_15["doy_sin"]  = np.sin(2 * np.pi * doy / 365.25)
    feat_15["doy_cos"]  = np.cos(2 * np.pi * doy / 365.25)

    # 台電預測標的：未來 15 分鐘 (1 step)、1 小時 (4 steps)、3 小時 (12 steps) 的累積能量 (等效出力 0-1)
    target_horizons = {
        "15min (即時結算區間)": 1,
        "1-hour (電能交易區間)": 4,
        "3-hour (調度預警區間)": 12
    }

    test_start = pd.Timestamp(C.TEST_START)
    fcols = list(feat_15.columns)

    lgb_params = dict(objective="regression", n_estimators=300, learning_rate=0.05,
                      num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                      random_state=C.RANDOM_SEED, n_jobs=-1, verbose=-1)

    summary_15min = {}
    plot_data = {}

    for label, k_steps in target_horizons.items():
        y_intv = P_15.iloc[::-1].rolling(k_steps).mean().iloc[::-1]

        mask = df_15min["is_ok"] & y_intv.notna() & feat_15.notna().all(axis=1)
        sub = feat_15.loc[mask].copy()
        sub_y = y_intv.loc[mask]

        is_test = sub.index >= test_start
        tr_X, te_X = sub.loc[~is_test], sub.loc[is_test]
        tr_y, te_y = sub_y.loc[~is_test].values, sub_y.loc[is_test].values
        persist_te = te_X["P_now"].values

        nval = int(len(tr_X) * 0.15)
        gbm = lgb.LGBMRegressor(**lgb_params)
        gbm.fit(tr_X.values[:-nval], tr_y[:-nval], eval_set=[(tr_X.values[-nval:], tr_y[-nval:])],
                callbacks=[lgb.early_stopping(40, verbose=False)])

        pred_lgb = np.clip(gbm.predict(te_X.values), 0, 1)

        r2_lgb = float(r2_score(te_y, pred_lgb))
        r2_per = float(r2_score(te_y, persist_te))
        nrmse_lgb = rmse(te_y, pred_lgb) / np.mean(te_y)
        nrmse_per = rmse(te_y, persist_te) / np.mean(te_y)

        print(f"  [{label}]  ML R² = {r2_lgb:.4f} (Persist R² = {r2_per:.4f})  ｜  ML nRMSE = {nrmse_lgb:.4f} vs Persist {nrmse_per:.4f}")

        summary_15min[label] = {
            "r2_ml": round(r2_lgb, 4),
            "r2_persist": round(r2_per, 4),
            "nrmse_ml": round(nrmse_lgb, 4),
            "nrmse_persist": round(nrmse_per, 4)
        }

        if k_steps == 1: # 保存 15-min 比對資料
            plot_data["te_ts"] = te_X.index
            plot_data["te_y"] = te_y
            plot_data["pred_lgb"] = pred_lgb
            plot_data["persist"] = persist_te

    # 3. 100 MW 風場台電 15 分鐘綠電轉供偏差懲罰計算 (依據每 15 分鐘 1/4h 結算)
    print("\n[步驟 3] 100 MW 風場台電 15 分鐘電表結算規章財務試算...")
    WIND_FARM_CAPACITY_MW = 100.0
    factor_15m = WIND_FARM_CAPACITY_MW * (1.0 / 4.0) * 1000.0  # 15min = 1/4h -> kWh
    tol_kwh_15m = WIND_FARM_CAPACITY_MW * 0.10 * (1.0 / 4.0) * 1000.0 # 10% 容許度

    te_y_15 = plot_data["te_y"]
    pred_15 = plot_data["pred_lgb"]
    per_15  = plot_data["persist"]

    actual_kwh_15  = te_y_15 * factor_15m
    persist_kwh_15 = per_15 * factor_15m
    ml_kwh_15      = pred_15 * factor_15m

    penalty_persist_nt = float(np.maximum(0.0, np.abs(actual_kwh_15 - persist_kwh_15) - tol_kwh_15m).sum() * 2.5)
    penalty_ml_nt      = float(np.maximum(0.0, np.abs(actual_kwh_15 - ml_kwh_15) - tol_kwh_15m).sum() * 2.5)
    saved_nt           = penalty_persist_nt - penalty_ml_nt

    print(f"  台電 15 分鐘電表結算測試集 (17 個月，{len(te_y_15):,} 個 15-min 區間)：")
    print(f"  Persistence 累積偏差懲罰金：NT$ {penalty_persist_nt:,.0f} 元")
    print(f"  ML 15-min 區間預測懲罰金  ：NT$ {penalty_ml_nt:,.0f} 元")
    print(f"  ★ 為風場避險省下偏差罰款  ：NT$ {saved_nt:,.0f} 元 (每月省下 {saved_nt/17.0:,.0f} 元/月)")

    summary_15min["taipower_15min_financial"] = {
        "penalty_persistence_nt": round(penalty_persist_nt, 0),
        "penalty_ml_nt": round(penalty_ml_nt, 0),
        "saved_nt": round(saved_nt, 0),
        "monthly_saved_nt": round(saved_nt / 17.0, 0)
    }

    # 寫入結果 JSON
    res_path = os.path.join(C.RES_DIR, "taipower_15min_metrics.json")
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(summary_15min, f, ensure_ascii=False, indent=2)

    # 4. 繪製圖表
    fp = FontProperties(fname="C:/Windows/Fonts/msjh.ttc", size=10) if os.path.exists("C:/Windows/Fonts/msjh.ttc") else None
    fp_title = FontProperties(fname="C:/Windows/Fonts/msjh.ttc", size=12, weight="bold") if os.path.exists("C:/Windows/Fonts/msjh.ttc") else None

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 子圖 1: 台電 15-min 電表 1 週時序比對
    ts_sub = plot_data["te_ts"][1000:1000 + 4 * 24 * 7]
    y_sub  = plot_data["te_y"][1000:1000 + 4 * 24 * 7]
    p_sub  = plot_data["pred_lgb"][1000:1000 + 4 * 24 * 7]

    axes[0].plot(ts_sub, y_sub, color="black", label="台電 15 分鐘智慧電表真實區間度數", linewidth=2.0)
    axes[0].plot(ts_sub, p_sub, color="#2ca02c", label="ML 15-min 區間累積能量預測", linewidth=1.8, linestyle="--")
    axes[0].set_title("台電 15 分鐘 (15-min) 綠電轉供區間能量預測時序比對", fontproperties=fp_title)
    axes[0].set_xlabel("時間", fontproperties=fp)
    axes[0].set_ylabel("區間發電能量 (等效滿載小時數 h)", fontproperties=fp)
    axes[0].legend(prop=fp)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # 子圖 2: 15-min, 1-hr, 3-hr 預測 R² 表現
    labels = list(target_horizons.keys())
    r2_vals = [summary_15min[lbl]["r2_ml"] for lbl in labels]
    per_r2  = [summary_15min[lbl]["r2_persist"] for lbl in labels]

    x = np.arange(len(labels))
    w = 0.35
    axes[1].bar(x - w/2, r2_vals, w, label="ML 15-min 重採樣區間 R²", color="#2ca02c", edgecolor="black")
    axes[1].bar(x + w/2, per_r2, w, label="Persistence R²", color="#d62728", alpha=0.6, edgecolor="black")
    axes[1].set_title("台電 15-min / 1-hr / 3-hr 區間能量預測擬合度 (R²)", fontproperties=fp_title)
    axes[1].set_xlabel("台電結算預測視窗", fontproperties=fp)
    axes[1].set_ylabel("R² 得分", fontproperties=fp)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["15 分鐘 (15-min)", "1 小時 (1-hr)", "3 小時 (3-hr)"], fontproperties=fp)
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(prop=fp)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    for i, v in enumerate(r2_vals):
        axes[1].annotate(f"R²={v:.4f}", xy=(i - w/2, v + 0.02), ha="center", va="bottom", fontweight="bold", color="#2ca02c", fontsize=9)

    plt.tight_layout()
    fig_path = os.path.join(C.FIG_DIR, "fig13_taipower_15min_settlement.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print(f"\n台電 15 分鐘對接視覺化圖表已產出：{fig_path}")
    print("Stage 12 完成。")

if __name__ == "__main__":
    main()
