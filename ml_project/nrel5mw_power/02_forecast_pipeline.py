#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_forecast_pipeline.py — 以 NREL 5MW 出力（絕對 MW）為標的的多時程發電預測
==========================================================================

把「其他專案」的發電預測方法搬過來，改用 NREL 5MW 絕對 MW 當預測標的，
重新訓練與評估，看預測指標長什麼樣。

方法學（與專案紅線一致，無洩漏）：
  * 時間切分：訓練 2016–2018、測試 2020–2021（2019 當緩衝，不用）。
  * 基準線：Persistence（下一刻 = 現在出力）。
  * 主模型：LightGBM Delta（預測出力變化量）、XGBoost Direct（直接預測）、
            以及兩者 50/50 Ensemble。
  * 時程：t+10min / +1h / +3h / +6h。
  * 指標：RMSE(MW)、MAE(MW)、NMAE(%)、R²、Skill Score(% vs persistence)。

輸出
  results/forecast_benchmark.csv     多時程多模型評估總表
  results/figures/fig5_skill_nmae.png
  results/figures/fig6_timeseries_1h.png
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

import nrel_5mw as N

N.setup_cjk_font()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """特徵工程：出力/風速的 lag、rolling、diff，氣壓趨勢，時間週期編碼。"""
    d = df.sort_values("ts").reset_index(drop=True).copy()

    # 週期時間特徵
    h = d.ts.dt.hour + d.ts.dt.minute / 60.0
    doy = d.ts.dt.dayofyear
    d["hour_sin"] = np.sin(2 * np.pi * h / 24.0)
    d["hour_cos"] = np.cos(2 * np.pi * h / 24.0)
    d["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    d["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # 風速 lag / rolling / diff
    d["WS100_lag1"] = d["WS_100_mean"].shift(1)
    d["WS100_lag3"] = d["WS_100_mean"].shift(3)
    d["WS100_lag6"] = d["WS_100_mean"].shift(6)
    d["WS100_diff1"] = d["WS_100_mean"] - d["WS100_lag1"]
    d["WS100_diff6"] = d["WS_100_mean"] - d["WS100_lag6"]
    d["WS100_roll_mean_1h"] = d["WS_100_mean"].shift(1).rolling(6).mean()
    d["WS100_roll_std_1h"] = d["WS_100_mean"].shift(1).rolling(6).std()
    d["WS100_roll_mean_3h"] = d["WS_100_mean"].shift(1).rolling(18).mean()

    # 出力 lag / rolling / diff
    d["P_lag1"] = d["P_mw"].shift(1)
    d["P_lag3"] = d["P_mw"].shift(3)
    d["P_lag6"] = d["P_mw"].shift(6)
    d["P_diff1"] = d["P_mw"] - d["P_lag1"]
    d["P_diff6"] = d["P_mw"] - d["P_lag6"]
    d["P_roll_mean_1h"] = d["P_mw"].shift(1).rolling(6).mean()
    d["P_roll_std_1h"] = d["P_mw"].shift(1).rolling(6).std()
    d["P_roll_mean_3h"] = d["P_mw"].shift(1).rolling(18).mean()

    # 氣壓趨勢（天氣系統移動指標）
    d["delta_BP_1h"] = d["BP_93_mean"] - d["BP_93_mean"].shift(6)
    d["delta_BP_3h"] = d["BP_93_mean"] - d["BP_93_mean"].shift(18)
    d["delta_BP_6h"] = d["BP_93_mean"] - d["BP_93_mean"].shift(36)

    return d


FEATURES = [
    "P_mw", "P_lag1", "P_lag3", "P_lag6", "P_diff1", "P_diff6",
    "P_roll_mean_1h", "P_roll_std_1h", "P_roll_mean_3h",
    "WS_100_mean", "WS100_lag1", "WS100_lag3", "WS100_lag6",
    "WS100_diff1", "WS100_diff6", "WS100_roll_mean_1h", "WS100_roll_std_1h", "WS100_roll_mean_3h",
    "shear_alpha", "WS_100E_ti", "WS_100E_gust_factor",
    "WD_97_sin", "WD_97_cos",
    "BP_93_mean", "delta_BP_1h", "delta_BP_3h", "delta_BP_6h",
    "AT_95_mean", "RH_95_mean", "air_density",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
]

HORIZONS = [
    ("+10min", 1),
    ("+1h", 6),
    ("+3h", 18),
    ("+6h", 36),
]


def main():
    print("[1/3] 載入資料、以 NREL 5MW 換算出力並建立特徵...")
    d = N.load_power_table()
    d = build_features(d)

    # 預測標的（直接值與變化量）
    for name, step in HORIZONS:
        d[f"y_{name}"] = d["P_mw"].shift(-step)
        d[f"dy_{name}"] = d[f"y_{name}"] - d["P_mw"]

    target_cols = [f"y_{n}" for n, _ in HORIZONS] + [f"dy_{n}" for n, _ in HORIZONS]
    d_clean = d.dropna(subset=FEATURES + target_cols).copy()

    train = d_clean[d_clean["year"] <= 2018].copy()
    test = d_clean[d_clean["year"] >= 2020].copy()
    print(f"  有效樣本 {len(d_clean):,}｜訓練(2016–2018) {len(train):,}｜測試(2020–2021) {len(test):,}")

    print("\n[2/3] 逐時程訓練 LightGBM Delta / XGBoost Direct / Ensemble...")
    rows = []
    preds_1h = None
    for name, step in HORIZONS:
        X_tr, X_te = train[FEATURES], test[FEATURES]
        y_tr_dir = train[f"y_{name}"]
        y_tr_del = train[f"dy_{name}"]
        y_te = test[f"y_{name}"].to_numpy()
        base = test["P_mw"].to_numpy()      # persistence

        # LightGBM Delta
        lgbm = lgb.LGBMRegressor(objective="regression", learning_rate=0.05,
                                 num_leaves=31, n_estimators=150, random_state=42, verbose=-1)
        lgbm.fit(X_tr, y_tr_del)
        pred_lgb = np.clip(base + lgbm.predict(X_te), 0.0, N.RATED_MW)

        # XGBoost Direct
        xgbm = xgb.XGBRegressor(objective="reg:squarederror", learning_rate=0.05,
                                max_depth=6, n_estimators=150, n_jobs=4, random_state=42)
        xgbm.fit(X_tr, y_tr_dir)
        pred_xgb = np.clip(xgbm.predict(X_te), 0.0, N.RATED_MW)

        # Ensemble
        pred_ens = 0.5 * pred_lgb + 0.5 * pred_xgb

        rmse_p = np.sqrt(mean_squared_error(y_te, base))
        for m_name, pred in [("Persistence", base), ("LightGBM_Delta", pred_lgb),
                             ("XGBoost_Direct", pred_xgb), ("Ensemble", pred_ens)]:
            rmse = np.sqrt(mean_squared_error(y_te, pred))
            mae = mean_absolute_error(y_te, pred)
            rows.append({
                "Horizon": name, "Model": m_name,
                "RMSE_MW": round(rmse, 4), "MAE_MW": round(mae, 4),
                "NMAE_%": round(100 * mae / N.RATED_MW, 2),
                "R2": round(r2_score(y_te, pred), 4),
                "Skill_%": round(100 * (1 - rmse / rmse_p), 2),
            })
        if name == "+1h":
            preds_1h = {"ts": test["ts"].to_numpy(), "true": y_te,
                        "persist": base, "ensemble": pred_ens}
        print(f"  {name:7s} 完成")

    res = pd.DataFrame(rows)
    res.to_csv(N.RESULTS_DIR / "forecast_benchmark.csv", index=False, encoding="utf-8-sig")
    print("\n--- NREL 5MW 多時程發電預測評估總表 ---")
    print(res.to_string(index=False))

    print("\n[3/3] 繪圖...")
    # 圖5：Skill 與 NMAE
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    order = [n for n, _ in HORIZONS]
    for ax, col, title in [(ax1, "Skill_%", "Skill Score (%) vs Persistence"),
                           (ax2, "NMAE_%", "NMAE (%) 相對 5MW 額定")]:
        sub = res if col == "NMAE_%" else res[res["Model"] != "Persistence"]
        for m in sub["Model"].unique():
            m_df = sub[sub["Model"] == m].set_index("Horizon").reindex(order)
            ax.plot(order, m_df[col], "o-", lw=2, label=m)
        ax.set_title(title)
        ax.set_xlabel("預測提前量")
        ax.grid(alpha=0.3)
        ax.legend()
    ax1.set_ylabel("Skill Score (%)")
    ax2.set_ylabel("NMAE (%)")
    fig.tight_layout()
    fig.savefig(N.FIG_DIR / "fig5_skill_nmae.png", dpi=140)
    plt.close(fig)

    # 圖6：+1h 時序（測試集一段）
    fig, ax = plt.subplots(figsize=(13, 5))
    ts = pd.to_datetime(preds_1h["ts"])
    sl = slice(0, 600)   # 前約 100 小時
    ax.plot(ts[sl], preds_1h["true"][sl], color="black", lw=1.6, label="實際 NREL 5MW 出力")
    ax.plot(ts[sl], preds_1h["ensemble"][sl], color="#d62728", lw=1.3, alpha=0.85, label="Ensemble 預測 (+1h)")
    ax.plot(ts[sl], preds_1h["persist"][sl], color="#1f77b4", lw=1.0, alpha=0.55, ls="--", label="Persistence")
    ax.set_ylabel("出力 (MW)")
    ax.set_title("測試集 +1h 發電預測時序（前約 100 小時）")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(N.FIG_DIR / "fig6_timeseries_1h.png", dpi=140)
    plt.close(fig)

    print("完成。輸出於 results/forecast_benchmark.csv 與 results/figures/")


if __name__ == "__main__":
    main()
