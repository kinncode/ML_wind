#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 13 —— 日前 24 小時 (24-hour / 96 個 15min 區間) 綠電與台電電力交易平台 (ETP) 能量預測

背景：
  台電日前市場 (Day-Ahead Market) 與企業綠電購售 (CPPA) 規範以 24 小時為排程規劃與日結算週期。
  本腳本評估未來的 24 小時 (24-hr) 區間發電能量 $E_{[t, t+24h]}$ 預測精度、24 小時日發電曲線擬合度與 100 MW 風場 24h 偏差避險台幣金額！

輸出：
  results/day_ahead_24h_metrics.json
  figures/fig14_day_ahead_24h_settlement.png
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
    print("PW_Interval Stage 13 —— 日前 24 小時 (24-hr) 綠電與台電市場區間能量預測對接")
    print("="*70)

    if not os.path.exists(C.CLEAN_PARQUET):
        raise FileNotFoundError(f"找不到 {C.CLEAN_PARQUET}，請先執行 01_load_validate.py")

    df_10min = pd.read_parquet(C.CLEAN_PARQUET).set_index("ts").sort_index()

    print("\n[步驟 1] 10 分鐘測風塔數據 ➔ 台電 15 分鐘智慧電表重採樣...")
    df_10min["P_10min"] = C.virtual_power(df_10min["WS_100_mean"], df_10min["air_density"])
    df_15min = df_10min[["WS_100_mean", "P_10min", "air_density", "is_ok"]].resample("15min").interpolate(method="time")
    df_15min["is_ok"] = df_10min["is_ok"].resample("15min").min().fillna(False).astype(bool)

    ws_15 = df_15min["WS_100_mean"]
    P_15 = df_15min["P_10min"]

    # 建立 24h (96 個 15min 步阶) 特徵
    feat_15 = pd.DataFrame(index=df_15min.index)
    feat_15["WS_100_mean"] = ws_15
    feat_15["P_now"] = P_15
    feat_15["air_density"] = df_15min["air_density"]

    for m in [15, 30, 45, 60, 120, 180, 360, 720, 1440]:
        feat_15[f"ws_lag_{m}m"] = ws_15.shift(m // 15)

    for win in [4, 12, 24, 48, 96]:
        r = ws_15.rolling(win, min_periods=win//2)
        feat_15[f"ws_rmean_{win*15}m"] = r.mean()
        feat_15[f"ws_rstd_{win*15}m"]  = r.std()

    hour = df_15min.index.hour + df_15min.index.minute / 60.0
    doy  = df_15min.index.dayofyear
    feat_15["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    feat_15["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    feat_15["doy_sin"]  = np.sin(2 * np.pi * doy / 365.25)
    feat_15["doy_cos"]  = np.cos(2 * np.pi * doy / 365.25)

    test_start = pd.Timestamp(C.TEST_START)

    # 標的 1: 未來 24 小時區間累積總能量 E_[t, t+24h] (96 個 15-min 步階)
    k_24h = 96
    y_24h = P_15.iloc[::-1].rolling(k_24h).mean().iloc[::-1]

    mask = df_15min["is_ok"] & y_24h.notna() & feat_15.notna().all(axis=1)
    sub = feat_15.loc[mask].copy()
    sub_y = y_24h.loc[mask]

    is_test = sub.index >= test_start
    tr_X, te_X = sub.loc[~is_test], sub.loc[is_test]
    tr_y, te_y = sub_y.loc[~is_test].values, sub_y.loc[is_test].values
    persist_te = te_X["P_now"].values

    print("\n[步驟 2] 訓練 24 小時 (24-hr / 96 個 15min 區間) 能量預測模型...")
    lgb_params = dict(objective="regression", n_estimators=450, learning_rate=0.03,
                      num_leaves=63, min_child_samples=100, subsample=0.8,
                      colsample_bytree=0.8, reg_lambda=1.0,
                      random_state=C.RANDOM_SEED, n_jobs=-1, verbose=-1)

    nval = int(len(tr_X) * 0.15)
    gbm = lgb.LGBMRegressor(**lgb_params)
    gbm.fit(tr_X.values[:-nval], tr_y[:-nval], eval_set=[(tr_X.values[-nval:], tr_y[-nval:])],
            callbacks=[lgb.early_stopping(40, verbose=False)])

    pred_24h = np.clip(gbm.predict(te_X.values), 0, 1)

    # 儲存 H24 模型與特徵重要度
    os.makedirs(C.MODEL_DIR, exist_ok=True)
    os.makedirs(C.RES_DIR, exist_ok=True)
    gbm.booster_.save_model(os.path.join(C.MODEL_DIR, "lgbm_power_H24.txt"))
    df_imp_24 = pd.DataFrame({"feature": list(tr_X.columns), "gain": gbm.booster_.feature_importance("gain")})
    df_imp_24 = df_imp_24.sort_values("gain", ascending=False)
    df_imp_24.to_csv(os.path.join(C.RES_DIR, "importance_power_H24.csv"), index=False)

    r2_ml = float(r2_score(te_y, pred_24h))
    r2_per = float(r2_score(te_y, persist_te))
    nrmse_ml = rmse(te_y, pred_24h) / np.mean(te_y)
    nrmse_per = rmse(te_y, persist_te) / np.mean(te_y)

    print(f"  日前 24 小時總發電能量預測 E_[t, t+24h] 性能：")
    print(f"  ML R² = {r2_ml:.4f}  (Persistence R² = {r2_per:.4f})")
    print(f"  ML nRMSE = {nrmse_ml:.4f}  vs  Persist {nrmse_per:.4f}  (精確度提升 +{(1 - nrmse_ml/nrmse_per)*100:.1f}%)")

    # 標的 2: 100 MW 風場日前 24 小時市場結算試算
    print("\n[步驟 3] 100 MW 風場台電日前 24 小時市場結算財務試算 (17 個月測試期)...")
    WIND_FARM_CAPACITY_MW = 100.0
    factor_24h = WIND_FARM_CAPACITY_MW * 24.0 * 1000.0 # 24h = 2400 MWh
    tol_kwh_24h = WIND_FARM_CAPACITY_MW * 0.10 * 24.0 * 1000.0 # 10% 容許度

    actual_kwh_24  = te_y * factor_24h
    persist_kwh_24 = persist_te * factor_24h
    ml_kwh_24      = pred_24h * factor_24h

    penalty_persist_nt = float(np.maximum(0.0, np.abs(actual_kwh_24 - persist_kwh_24) - tol_kwh_24h).sum() * 2.5)
    penalty_ml_nt      = float(np.maximum(0.0, np.abs(actual_kwh_24 - ml_kwh_24) - tol_kwh_24h).sum() * 2.5)
    saved_nt           = penalty_persist_nt - penalty_ml_nt

    print(f"  Persistence 24h 累積偏差懲罰金：NT$ {penalty_persist_nt:,.0f} 元")
    print(f"  ML 24h 區間能量預測懲罰金      ：NT$ {penalty_ml_nt:,.0f} 元")
    print(f"  -----------------------------------------------------------------")
    print(f"  ★ 日前 24h ML 為風場避險省下罰款：NT$ {saved_nt:,.0f} 元 (節省 {(saved_nt/penalty_persist_nt)*100:.1f}% 懲罰款)")
    print(f"  ★ 平均每月節省金額             ：NT$ {saved_nt/17.0:,.0f} 元/月")

    summary_24h = {
        "horizon_label": "24-hour (日前 24 小時區間能量)",
        "r2_ml": round(r2_ml, 4),
        "r2_persist": round(r2_per, 4),
        "nrmse_ml": round(nrmse_ml, 4),
        "nrmse_persist": round(nrmse_per, 4),
        "financial_model": {
            "penalty_persistence_nt": round(penalty_persist_nt, 0),
            "penalty_ml_nt": round(penalty_ml_nt, 0),
            "saved_nt": round(saved_nt, 0),
            "monthly_saved_nt": round(saved_nt / 17.0, 0)
        }
    }

    res_path = os.path.join(C.RES_DIR, "day_ahead_24h_metrics.json")
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(summary_24h, f, ensure_ascii=False, indent=2)

    # 4. 繪製圖表
    fp = FontProperties(fname="C:/Windows/Fonts/msjh.ttc", size=10) if os.path.exists("C:/Windows/Fonts/msjh.ttc") else None
    fp_title = FontProperties(fname="C:/Windows/Fonts/msjh.ttc", size=12, weight="bold") if os.path.exists("C:/Windows/Fonts/msjh.ttc") else None

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 子圖 1: 測試集 2 週日前 24h 累積能量時序比對
    ts_sub = te_X.index[1000:1000 + 4 * 24 * 14]
    y_sub  = te_y[1000:1000 + 4 * 24 * 14]
    p_sub  = pred_24h[1000:1000 + 4 * 24 * 14]

    axes[0].plot(ts_sub, y_sub, color="black", label="真實 24h 區間累積發電能量 (Real E_24h)", linewidth=2.0)
    axes[0].plot(ts_sub, p_sub, color="#2ca02c", label=f"ML 日前 24h 區間能量預測 (R²={r2_ml:.4f})", linewidth=1.8, linestyle="--")
    axes[0].set_title("日前 24 小時區間發電能量預測時序比對 (2 週片段)", fontproperties=fp_title)
    axes[0].set_xlabel("時間", fontproperties=fp)
    axes[0].set_ylabel("24h 累積能量 (等效滿載小時數 h)", fontproperties=fp)
    axes[0].legend(prop=fp)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # 子圖 2: 100 MW 風場 24h 日前結算懲罰避險長條圖
    bars = axes[1].bar(["Persistence 24h 懲罰款", "ML 24h 預測懲罰款", "日前 24h 省下/避險金額"],
                       [penalty_persist_nt / 1e4, penalty_ml_nt / 1e4, saved_nt / 1e4],
                       color=["#d62728", "#1f77b4", "#2ca02c"], edgecolor="black", alpha=0.85)
    axes[1].set_title("100 MW 風場 日前 24h 電力市場偏差懲罰避險金額 (萬元 NT$)", fontproperties=fp_title)
    axes[1].set_ylabel("金額 (萬元台幣)", fontproperties=fp)
    axes[1].set_xticks([0, 1, 2])
    axes[1].set_xticklabels(["Persistence 24h 懲罰款", "ML 24h 預測懲罰款", "日前 24h 省下/避險金額"], fontproperties=fp)
    for bar in bars:
        h_val = bar.get_height()
        axes[1].annotate(f'{h_val:,.0f} 萬', xy=(bar.get_x() + bar.get_width() / 2, h_val),
                         xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontproperties=fp)

    plt.tight_layout()
    fig_path = os.path.join(C.FIG_DIR, "fig14_day_ahead_24h_settlement.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print(f"\n日前 24h 區間對接視覺化圖表已產出：{fig_path}")
    print("Stage 13 完成。")

if __name__ == "__main__":
    main()
