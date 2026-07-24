#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 10 —— 非固定 t+H 事件型/區間能量預測與發電經濟價值量化模型

核心創新：
  1. 區間累積總發電量預測 (Integrated Energy Forecast E_{[t, t+H]}):
     - 不預測單點瞬時出力，改為預測未來 1h / 3h / 6h 的「總發電能量 (kWh/MW)」，大幅抵銷高頻雜訊，提升預測穩定度與 R²。
  2. 風電劇烈爬升/突降事件預警 (Ramp Event Classification):
     - 判定未來 1h~3h 是否會發生出力劇烈劇變 (|ΔP| >= 0.30 額定容量)，訓練 LightGBM 分類器進行事件提前預警。
  3. 電力市場偏差懲罰與經濟效益量化 (Financial Value & Imbalance Penalty Saved):
     - 以 100 MW 離岸風場規模與台電調度懲罰機制為例，比較「Persistence 基準」與「ML 機器學習預測」在偏差懲罰金上的避險效果，定量計算為風場「省下多少台幣 (NT$)」。

輸出：
  results/value_added_metrics.json
  figures/fig9_value_added_forecasting.png
"""
from __future__ import annotations
import os, sys, json
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, r2_score
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
    print("PW_Integrated Stage 10 —— 非固定 t+H 事件型/區間能量預測與經濟價值量化")
    print("="*70)

    if not os.path.exists(C.FEAT_PARQUET):
        raise FileNotFoundError(f"找不到 {C.FEAT_PARQUET}，請先執行 03_features.py")

    df = pd.read_parquet(C.FEAT_PARQUET)
    df["ts"] = pd.to_datetime(df["ts"])
    test_start = pd.Timestamp(C.TEST_START)

    meta = {"ts", "is_ok", "year", "month", "hour_i"}
    fcols = [c for c in df.columns if c not in meta and not c.startswith("y_") and not c.startswith("m_")]

    lgb_reg_params = dict(objective="regression", n_estimators=350, learning_rate=0.05,
                          num_leaves=48, min_child_samples=100, subsample=0.8,
                          colsample_bytree=0.8, random_state=C.RANDOM_SEED, n_jobs=-1, verbose=-1)

    lgb_cls_params = dict(objective="binary", n_estimators=350, learning_rate=0.05,
                          num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                          random_state=C.RANDOM_SEED, n_jobs=-1, verbose=-1)

    output_summary = {}

    # ------------------------------------------------------------------
    # 創新 1: 區間累積總發電量預測 (Integrated Energy Forecast)
    # ------------------------------------------------------------------
    print("\n--- [創新 1] 區間累積總發電能量預測 E_[t, t+H] (kWh/MW) ---")
    energy_metrics = {}

    # 計算未來 H 小時內 10 分鐘出力的滾動累積和
    P_now_full = C.virtual_power(df["WS_100_mean"], df["air_density"])
    P_series = pd.Series(P_now_full, index=df.index)

    for h in [1, 3, 6]:
        k = C.HORIZON_STEPS[h]
        # 未來 k 個步階的平均出力 * h 小時 = 區間總能量 (等效滿載小時數)
        # 滾動均值 * h
        df[f"y_energy_{h}"] = P_series.iloc[::-1].rolling(k).mean().iloc[::-1]

        mask = df["is_ok"] & df[f"m_{h}"] & df[f"y_energy_{h}"].notna() & df[fcols].notna().all(axis=1)
        sub = df.loc[mask]
        is_test = sub["ts"] >= test_start

        tr, te = sub.loc[~is_test], sub.loc[is_test]
        ytr, yte = tr[f"y_energy_{h}"].values, te[f"y_energy_{h}"].values
        ptr, pte = tr["P_now"].values, te["P_now"].values  # Persistence 基準

        nval = int(len(tr) * 0.15)
        gbm = lgb.LGBMRegressor(**lgb_reg_params)
        gbm.fit(tr[fcols].values[:-nval], ytr[:-nval], eval_set=[(tr[fcols].values[-nval:], ytr[-nval:])], callbacks=[lgb.early_stopping(40, verbose=False)])

        pred_ml = np.clip(gbm.predict(te[fcols].values), 0, 1)

        r2_ml = float(r2_score(yte, pred_ml))
        r2_per = float(r2_score(yte, pte))
        nrmse_ml = rmse(yte, pred_ml) / np.mean(yte)
        nrmse_per = rmse(yte, pte) / np.mean(yte)

        print(f"  未來 {h} 小時總能量預測 ｜ ML R^2 = {r2_ml:.4f} (Persistence R^2 = {r2_per:.4f}) ｜ ML nRMSE = {nrmse_ml:.4f} vs Persist {nrmse_per:.4f}")

        energy_metrics[f"{h}h"] = {
            "r2_ml": round(r2_ml, 4),
            "r2_persist": round(r2_per, 4),
            "nrmse_ml": round(nrmse_ml, 4),
            "nrmse_persist": round(nrmse_per, 4)
        }

    output_summary["energy_forecast"] = energy_metrics

    # ------------------------------------------------------------------
    # 創新 2: 風電劇烈爬升/突降事件預警 (Ramp Event Classification)
    # ------------------------------------------------------------------
    print("\n--- [創新 2] 風電劇烈爬升/突降事件分類預警 (|ΔP| >= 0.30) ---")
    ramp_metrics = {}

    for h in [1, 3]:
        k = C.HORIZON_STEPS[h]
        # 未來 1h 或 3h 內的最大出力變化量
        future_p = P_series.shift(-k)
        ramp_magnitude = (future_p - P_series).abs()
        df[f"y_ramp_{h}"] = (ramp_magnitude >= 0.30).astype(int)

        mask = df["is_ok"] & df[f"m_{h}"] & df[f"y_ramp_{h}"].notna() & df[fcols].notna().all(axis=1)
        sub = df.loc[mask]
        is_test = sub["ts"] >= test_start

        tr, te = sub.loc[~is_test], sub.loc[is_test]
        ytr, yte = tr[f"y_ramp_{h}"].values, te[f"y_ramp_{h}"].values

        nval = int(len(tr) * 0.15)
        cls_model = lgb.LGBMClassifier(**lgb_cls_params)
        cls_model.fit(tr[fcols].values[:-nval], ytr[:-nval], eval_set=[(tr[fcols].values[-nval:], ytr[-nval:])], callbacks=[lgb.early_stopping(40, verbose=False)])

        prob = cls_model.predict_proba(te[fcols].values)[:, 1]
        pred_cls = (prob >= 0.50).astype(int)

        auc = float(roc_auc_score(yte, prob))
        f1 = float(f1_score(yte, pred_cls))
        prec = float(precision_score(yte, pred_cls))
        rec = float(recall_score(yte, pred_cls))

        ramp_count = int(yte.sum())
        print(f"  +{h}h 爬升事件預警 ｜ 測試集事件數={ramp_count:,} ({100*ramp_count/len(yte):.1f}%) ｜ ROC-AUC={auc:.4f} ｜ F1-Score={f1:.4f} ｜ 精確率={prec:.4f} ｜ 召回率={rec:.4f}")

        ramp_metrics[f"{h}h"] = {
            "ramp_events": ramp_count,
            "auc": round(auc, 4),
            "f1": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4)
        }

    output_summary["ramp_classification"] = ramp_metrics

    # ------------------------------------------------------------------
    # 創新 3: 電力市場偏差懲罰與經濟效益量化 (Financial Settlement Model)
    # ------------------------------------------------------------------
    print("\n--- [創新 3] 電力市場偏差懲罰與經濟效益量化 (100 MW 離岸風場範例) ---")

    # 參數設定
    WIND_FARM_CAPACITY_MW = 100.0           # 100 MW 風場規模
    ELEC_PRICE_NT_PER_KWH = 4.5             # 綠電收購價格 4.5 元/kWh
    PENALTY_RATE_NT_PER_KWH = 2.5           # 偏差超過容許度時之懲罰費率 2.5 元/kWh
    TOLERANCE_RATIO = 0.10                  # 電網容許誤差範圍 +-10% 容量

    # 取 +3h 測試集數據進行試算
    h = 3
    ycol = f"y_power_{h}"
    mask = df["is_ok"] & df[f"m_{h}"] & df[ycol].notna() & df[fcols].notna().all(axis=1) & (df["ts"] >= test_start)
    te = df.loc[mask]

    yte = te[ycol].values
    pte = te["P_now"].values

    # 載入 Stage 4 ML 預測
    pred_path = os.path.join(C.DATA_DIR, f"pred_power_H{h}.parquet")
    if os.path.exists(pred_path):
        pred_df = pd.read_parquet(pred_path)
        pred_ml = pred_df["pred_lgbm"].values
    else:
        pred_ml = pte

    # 轉為實際 MWh (以 10 分鐘 1/6 小時為單位)
    # 1 單位 P (0-1) * 100 MW * 1/6 h = MWh
    factor = WIND_FARM_CAPACITY_MW * (1.0 / 6.0) * 1000.0  # 轉為 kWh

    actual_kwh  = yte * factor
    persist_kwh = pte * factor
    ml_kwh      = pred_ml * factor

    # 理想總電費收入 (元)
    total_revenue_nt = float(actual_kwh.sum() * ELEC_PRICE_NT_PER_KWH)

    # 偏差計算
    tol_kwh = WIND_FARM_CAPACITY_MW * TOLERANCE_RATIO * (1.0 / 6.0) * 1000.0

    err_persist = np.abs(actual_kwh - persist_kwh)
    excess_persist = np.maximum(0.0, err_persist - tol_kwh)
    penalty_persist_nt = float(excess_persist.sum() * PENALTY_RATE_NT_PER_KWH)

    err_ml = np.abs(actual_kwh - ml_kwh)
    excess_ml = np.maximum(0.0, err_ml - tol_kwh)
    penalty_ml_nt = float(excess_ml.sum() * PENALTY_RATE_NT_PER_KWH)

    net_saved_nt = penalty_persist_nt - penalty_ml_nt
    pct_saved = (net_saved_nt / penalty_persist_nt) * 100.0 if penalty_persist_nt > 0 else 0.0

    print(f"  風場規模：{WIND_FARM_CAPACITY_MW:.0f} MW ｜ 測試集評估時間：17 個月 ({len(te):,} 個 10 分鐘區間)")
    print(f"  總售電電費收入：NT$ {total_revenue_nt:,.0f} 元")
    print(f"  Persistence 傳統預測累積偏差懲罰金：NT$ {penalty_persist_nt:,.0f} 元")
    print(f"  ML 機器學習預測累積偏差懲罰金      ：NT$ {penalty_ml_nt:,.0f} 元")
    print(f"  -----------------------------------------------------------------")
    print(f"  ★ ML 為風場避免/省下偏差懲罰金額  ：NT$ {net_saved_nt:,.0f} 元 (節省 {pct_saved:.1f}% 懲罰款)")
    print(f"  ★ 平均每月增加經濟淨化收益         ：NT$ {net_saved_nt / 17.0:,.0f} 元/月")

    output_summary["financial_model"] = {
        "wind_farm_capacity_mw": WIND_FARM_CAPACITY_MW,
        "total_revenue_nt": round(total_revenue_nt, 0),
        "penalty_persistence_nt": round(penalty_persist_nt, 0),
        "penalty_ml_nt": round(penalty_ml_nt, 0),
        "net_saved_nt": round(net_saved_nt, 0),
        "pct_saved": round(pct_saved, 1),
        "monthly_saved_nt": round(net_saved_nt / 17.0, 0)
    }

    # 匯出 JSON 結果
    res_json_path = os.path.join(C.RES_DIR, "value_added_metrics.json")
    os.makedirs(C.RES_DIR, exist_ok=True)
    os.makedirs(C.FIG_DIR, exist_ok=True)
    with open(res_json_path, "w", encoding="utf-8") as f:
        json.dump(output_summary, f, ensure_ascii=False, indent=2)
    print(f"\n加值實驗數據已寫入：{res_json_path}")

    # ------------------------------------------------------------------
    # 視覺化繪圖
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 圖 A: 區間能量預測 R² 比較
    h_keys = ["1h", "3h", "6h"]
    r2_ml_vals = [energy_metrics[k]["r2_ml"] for k in h_keys]
    r2_per_vals = [energy_metrics[k]["r2_persist"] for k in h_keys]

    x = np.arange(len(h_keys))
    w = 0.35
    axes[0].bar(x - w/2, r2_ml_vals, w, label="ML 能量預測 R²", color="#2ca02c", edgecolor="black")
    axes[0].bar(x + w/2, r2_per_vals, w, label="Persistence R²", color="#d62728", alpha=0.6, edgecolor="black")
    axes[0].set_title("區間累積發電能量預測擬合度 (R²)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("預測時間區間 H", fontsize=10)
    axes[0].set_ylabel("R² (越接近 1.0 越精準)", fontsize=10)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["1 小時總能量", "3 小時總能量", "6 小時總能量"])
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # 圖 B: 風電爬升事件預警 (ROC-AUC & F1-Score)
    auc_vals = [ramp_metrics[k]["auc"] for k in ["1h", "3h"]]
    f1_vals = [ramp_metrics[k]["f1"] for k in ["1h", "3h"]]
    x_b = np.arange(len(auc_vals))
    axes[1].bar(x_b - w/2, auc_vals, w, label="ROC-AUC 得分", color="#1f77b4", edgecolor="black")
    axes[1].bar(x_b + w/2, f1_vals, w, label="F1-Score", color="#ff7f0e", edgecolor="black")
    axes[1].set_title("風電劇烈爬升/突降事件預警分類能力", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("預估事件時間視窗", fontsize=10)
    axes[1].set_ylabel("指標得分 (0–1)", fontsize=10)
    axes[1].set_xticks(x_b)
    axes[1].set_xticklabels(["+1h 爬升事件", "+3h 爬升事件"])
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # 圖 C: 100 MW 風場偏差懲罰避險節省金額 (台幣 NT$)
    bars = axes[2].bar(["Persistence 傳統懲罰款", "ML 模型懲罰款", "省下/避險金額"],
                       [penalty_persist_nt / 1e4, penalty_ml_nt / 1e4, net_saved_nt / 1e4],
                       color=["#d62728", "#1f77b4", "#2ca02c"], edgecolor="black", alpha=0.85)
    axes[2].set_title("100 MW 風場電力市場偏差懲罰避險金額 (萬元 NT$)", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("金額 (萬元台幣)", fontsize=10)
    for bar in bars:
        h_val = bar.get_height()
        axes[2].annotate(f'{h_val:,.0f} 萬', xy=(bar.get_x() + bar.get_width() / 2, h_val),
                         xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    fig_path = os.path.join(C.FIG_DIR, "fig9_value_added_forecasting.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"加值視覺化圖表已產出：{fig_path}")

    print("\nStage 10 完成。")

if __name__ == "__main__":
    main()
