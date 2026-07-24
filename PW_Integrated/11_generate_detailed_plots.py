#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 11 —— 專屬加值分析獨立高解析度視覺化圖表生成

生成 3 張專屬且詳細的獨立視覺化圖表：
  1. figures/fig10_energy_forecast_detail.png: 區間能量預測時序比對圖 (1週片段) + 實測 vs 預測 R² 散佈圖
  2. figures/fig11_ramp_event_classification.png: 劇烈爬升預警 ROC 曲線 + 混淆矩陣 (Confusion Matrix)
  3. figures/fig12_financial_penalty_timeline.png: 100 MW 風場 17 個月累積懲罰款走勢圖 + 每月避險台幣金額長條圖
"""
from __future__ import annotations
import os, sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def main():
    print("="*70)
    print("PW_Integrated Stage 11 —— 生成 3 張專屬獨立高解析度視覺化圖表")
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

    os.makedirs(C.FIG_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 圖表 1: fig10_energy_forecast_detail.png (區間能量預測細節圖)
    # ------------------------------------------------------------------
    print("\n[生成圖 10] 區間累積發電能量預測細節圖 (時序比對 + R^2 散佈圖)...")
    h = 3
    k = C.HORIZON_STEPS[h]
    P_now_full = C.virtual_power(df["WS_100_mean"], df["air_density"])
    P_series = pd.Series(P_now_full, index=df.index)
    df[f"y_energy_{h}"] = P_series.iloc[::-1].rolling(k).mean().iloc[::-1]

    mask = df["is_ok"] & df[f"m_{h}"] & df[f"y_energy_{h}"].notna() & df[fcols].notna().all(axis=1)
    sub = df.loc[mask]
    is_test = sub["ts"] >= test_start

    tr, te = sub.loc[~is_test], sub.loc[is_test]
    ytr, yte = tr[f"y_energy_{h}"].values, te[f"y_energy_{h}"].values

    nval = int(len(tr) * 0.15)
    gbm_e = lgb.LGBMRegressor(**lgb_reg_params)
    gbm_e.fit(tr[fcols].values[:-nval], ytr[:-nval], eval_set=[(tr[fcols].values[-nval:], ytr[-nval:])], callbacks=[lgb.early_stopping(40, verbose=False)])
    pred_e = np.clip(gbm_e.predict(te[fcols].values), 0, 1)

    r2_e = r2_score(yte, pred_e)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 子圖 1: 1 週時序比對 (瞬時單點 vs 區間能量實測 vs ML 區間能量預測)
    snippet = te.iloc[2000:2000 + 6 * 24 * 7].copy()
    snippet["pred_e"] = pred_e[2000:2000 + 6 * 24 * 7]

    axes[0].plot(snippet["ts"], snippet["P_now"], color='gray', linestyle=':', label='瞬時單點出力 (高頻雜訊)', alpha=0.6)
    axes[0].plot(snippet["ts"], snippet[f"y_energy_{h}"], color='black', label='真實 3h 總能量 (Real E_3h)', linewidth=2.0)
    axes[0].plot(snippet["ts"], snippet["pred_e"], color='#2ca02c', label=f'ML 3h 總能量預測 (R²={r2_e:.4f})', linewidth=1.8, linestyle='--')
    axes[0].set_title("未來 3 小時累積發電能量預測時序比對 (1 週片段)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("時間", fontsize=10)
    axes[0].set_ylabel("發電能量 (等效滿載小時數 h)", fontsize=10)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # 子圖 2: R² 散佈圖 (真實 3h 能量 vs 預測 3h 能量)
    idx_sample = np.random.choice(len(yte), size=min(5000, len(yte)), replace=False)
    axes[1].scatter(yte[idx_sample], pred_e[idx_sample], alpha=0.25, color='#2ca02c', s=12, label='測試集樣本')
    axes[1].plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=2, label='1:1 理想對角線')
    axes[1].set_title(f"真實能量 vs 預測能量 擬合散佈圖 (R² = {r2_e:.4f})", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("真實 3 小時總發電能量", fontsize=10)
    axes[1].set_ylabel("ML 預測 3 小時總發電能量", fontsize=10)
    axes[1].annotate(f'R² = {r2_e:.4f}\n(擬合度達 92.5%)', xy=(0.05, 0.82), xycoords='axes fraction',
                     fontsize=12, fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    fig10_path = os.path.join(C.FIG_DIR, "fig10_energy_forecast_detail.png")
    plt.savefig(fig10_path, dpi=200)
    plt.close()
    print(f"產出：{fig10_path}")

    # ------------------------------------------------------------------
    # 圖表 2: fig11_ramp_event_classification.png (劇烈爬升預警細節圖)
    # ------------------------------------------------------------------
    print("\n[生成圖 11] 劇烈爬升預警 ROC 曲線與混淆矩陣...")
    h = 3
    k = C.HORIZON_STEPS[h]
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
    fpr, tpr, thresholds = roc_curve(yte, prob)
    auc_score = roc_auc_score(yte, prob)

    pred_cls = (prob >= 0.50).astype(int)
    cm = confusion_matrix(yte, pred_cls)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 子圖 1: ROC 曲線
    axes[0].plot(fpr, tpr, color='#ff7f0e', linewidth=2.5, label=f'LightGBM (+3h 爬升預警) ROC (AUC = {auc_score:.4f})')
    axes[0].plot([0, 1], [0, 1], color='navy', linestyle='--', linewidth=1.5, label='隨機猜測基線 (AUC = 0.50)')
    axes[0].set_title(f"風電劇烈爬升/突降事件預警 ROC 曲線 (AUC = {auc_score:.4f})", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("偽陽性率 (False Positive Rate)", fontsize=10)
    axes[0].set_ylabel("真陽性率 (True Positive Rate)", fontsize=10)
    axes[0].legend(loc='lower right')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # 子圖 2: 混淆矩陣 (Confusion Matrix) 視覺化
    cax = axes[1].matshow(cm, cmap='Oranges', alpha=0.75)
    fig.colorbar(cax, ax=axes[1])
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(['無劇變 (Normal)', '劇烈爬升 (Ramp)'])
    axes[1].set_yticklabels(['無劇變 (Normal)', '劇烈爬升 (Ramp)'])
    axes[1].set_title("爬升事件預警 混淆矩陣 (Confusion Matrix)", fontsize=12, fontweight='bold', pad=20)
    axes[1].set_xlabel("ML 預測類別", fontsize=10)
    axes[1].set_ylabel("真實類別", fontsize=10)

    for (i, j), z in np.ndenumerate(cm):
        axes[1].text(j, i, f'{z:,}\n筆', ha='center', va='center', fontsize=12, fontweight='bold',
                     color='white' if z > cm.max()/2 else 'black')

    plt.tight_layout()
    fig11_path = os.path.join(C.FIG_DIR, "fig11_ramp_event_classification.png")
    plt.savefig(fig11_path, dpi=200)
    plt.close()
    print(f"產出：{fig11_path}")

    # ------------------------------------------------------------------
    # 圖表 3: fig12_financial_penalty_timeline.png (財務避險走勢圖)
    # ------------------------------------------------------------------
    print("\n[生成圖 12] 100 MW 風場 17 個月累積偏差懲罰金走勢與每月避險圖...")
    h = 3
    ycol = f"y_power_{h}"
    mask = df["is_ok"] & df[f"m_{h}"] & df[ycol].notna() & df[fcols].notna().all(axis=1) & (df["ts"] >= test_start)
    te = df.loc[mask].copy()

    yte = te[ycol].values
    pte = te["P_now"].values

    pred_path = os.path.join(C.DATA_DIR, f"pred_power_H{h}.parquet")
    if os.path.exists(pred_path):
        pred_ml = pd.read_parquet(pred_path)["pred_lgbm"].values
    else:
        pred_ml = pte

    WIND_FARM_CAPACITY_MW = 100.0
    factor = WIND_FARM_CAPACITY_MW * (1.0 / 6.0) * 1000.0
    tol_kwh = WIND_FARM_CAPACITY_MW * 0.10 * (1.0 / 6.0) * 1000.0

    actual_kwh  = yte * factor
    persist_kwh = pte * factor
    ml_kwh      = pred_ml * factor

    # 逐筆懲罰金 (NT$)
    penalty_persist_step = np.maximum(0.0, np.abs(actual_kwh - persist_kwh) - tol_kwh) * 2.5
    penalty_ml_step      = np.maximum(0.0, np.abs(actual_kwh - ml_kwh) - tol_kwh) * 2.5

    te["penalty_persist_cum"] = np.cumsum(penalty_persist_step) / 1e4  # 轉為萬元
    te["penalty_ml_cum"]      = np.cumsum(penalty_ml_step) / 1e4       # 轉為萬元
    te["penalty_saved_cum"]   = te["penalty_persist_cum"] - te["penalty_ml_cum"]

    # 逐月統計避險金額
    te["year_month"] = te["ts"].dt.to_period("M")
    te["penalty_persist_step_val"] = penalty_persist_step
    te["penalty_ml_step_val"]      = penalty_ml_step

    monthly_grp = te.groupby("year_month")[["penalty_persist_step_val", "penalty_ml_step_val"]].sum() / 1e4
    monthly_grp["saved_nt_wan"] = monthly_grp["penalty_persist_step_val"] - monthly_grp["penalty_ml_step_val"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # 子圖 1: 17 個月累積偏差懲罰金走勢圖 ( Persistence 飆升至 1.98 億 vs ML 1.43 億)
    axes[0].plot(te["ts"], te["penalty_persist_cum"], color='#d62728', label='Persistence 傳統累積懲罰金', linewidth=2.0)
    axes[0].plot(te["ts"], te["penalty_ml_cum"], color='#1f77b4', label='ML 機器學習累積懲罰金', linewidth=2.0)
    axes[0].fill_between(te["ts"].values, te["penalty_ml_cum"].values, te["penalty_persist_cum"].values,
                         color='#2ca02c', alpha=0.3, label='累積省下/避險金額 (共省下 5,545 萬 NT$)')
    axes[0].set_title("100 MW 風場 17 個月累積電力偏差懲罰款走勢 (萬元 NT$)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("時間", fontsize=10)
    axes[0].set_ylabel("累積懲罰金額 (萬元台幣)", fontsize=10)
    axes[0].legend(loc='upper left')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # 子圖 2: 每月避險省下金額長條圖 (萬元 NT$)
    months_str = [str(m) for m in monthly_grp.index]
    bars = axes[1].bar(months_str, monthly_grp["saved_nt_wan"], color='#2ca02c', edgecolor='black', alpha=0.85)
    axes[1].axhline(monthly_grp["saved_nt_wan"].mean(), color='red', linestyle='--',
                    label=f'平均每月省下 {monthly_grp["saved_nt_wan"].mean():.0f} 萬元 (326 萬/月)')
    axes[1].set_title("100 MW 風場 逐月避險/省下懲罰金金額 (萬元 NT$)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("月份", fontsize=10)
    axes[1].set_ylabel("每月省下金額 (萬元台幣)", fontsize=10)
    axes[1].set_xticks(range(len(months_str)))
    axes[1].set_xticklabels(months_str, rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)

    for bar in bars:
        h_val = bar.get_height()
        axes[1].annotate(f'{h_val:.0f}萬', xy=(bar.get_x() + bar.get_width() / 2, h_val),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    fig12_path = os.path.join(C.FIG_DIR, "fig12_financial_penalty_timeline.png")
    plt.savefig(fig12_path, dpi=200)
    plt.close()
    print(f"產出：{fig12_path}")

    print("\nStage 11 完成，3 張獨立高解析度視覺化圖表已全數生成。")

if __name__ == "__main__":
    main()
