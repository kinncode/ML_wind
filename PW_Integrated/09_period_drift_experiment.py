#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 9 —— 設備數據跨期漂移與單一期內部驗證實驗 (Sensor Shift & Intra-Period Experiment)

目的：
  驗證三個原始數據資料夾 (2016–2017, 2018–2019, 2020–2021) 是否存在設備更換/感測器漂移 (Domain Shift)，
  並比較：
  1. 單一時期內部訓練與測試 (Intra-Period Validation)：
     - Period 1 (2016–2017)：Train 2016.03–2017.06 ➔ Test 2017.07–2017.12
     - Period 2 (2018–2019)：Train 2018.01–2019.06 ➔ Test 2019.07–2019.12
     - Period 3 (2020–2021)：Train 2020.01–2020.12 ➔ Test 2021.01–2021.10
  2. 跨時期評估 (Cross-Period Transfer Validation)：
     - 以 P1 模型 (2016–2017)、P2 模型 (2018–2019)、P1+P2 混合模型 (2016–2019) 分別測試於 P3 測試集 (2021)，
       並與 P3 本地模型 (2020 訓練) 進行比較，客觀驗證前後數據是否一致！

輸出：
  results/period_drift_metrics.csv
  figures/fig8_period_drift_experiment.png
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import lightgbm as lgb
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
    print("PW_Integrated Stage 9 —— 設備數據跨期漂移與單一期內部驗證實驗")
    print("="*70)

    if not os.path.exists(C.FEAT_PARQUET):
        raise FileNotFoundError(f"找不到 {C.FEAT_PARQUET}，請先執行 03_features.py")

    df = pd.read_parquet(C.FEAT_PARQUET)
    df["ts"] = pd.to_datetime(df["ts"])

    meta = {"ts", "is_ok", "year", "month", "hour_i"}
    fcols = [c for c in df.columns if c not in meta and not c.startswith("y_") and not c.startswith("m_")]

    lgb_params = dict(objective="regression", n_estimators=350, learning_rate=0.05,
                      num_leaves=48, min_child_samples=100, subsample=0.8,
                      subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                      random_state=C.RANDOM_SEED, n_jobs=-1, verbose=-1)

    results = []

    # ---------------------------------------------------------
    # PART 1: 三個時期各自獨立內部訓練與測試 (Intra-Period Experiments)
    # ---------------------------------------------------------
    print("\n--- [PART 1] 三個時期各自獨立內部訓練與測試 (Intra-Period) ---")

    periods_config = {
        "Period 1 (2016-2017)": {
            "tr": (df["ts"] >= "2016-03-01") & (df["ts"] <= "2017-06-30"),
            "te": (df["ts"] >= "2017-07-01") & (df["ts"] <= "2017-12-31")
        },
        "Period 2 (2018-2019)": {
            "tr": (df["ts"] >= "2018-01-01") & (df["ts"] <= "2019-06-30"),
            "te": (df["ts"] >= "2019-07-01") & (df["ts"] <= "2019-12-31")
        },
        "Period 3 (2020-2021)": {
            "tr": (df["ts"] >= "2020-01-01") & (df["ts"] <= "2020-12-31"),
            "te": (df["ts"] >= "2021-01-01") & (df["ts"] <= "2021-10-31")
        }
    }

    models = {}

    for name, cfg in periods_config.items():
        print(f"\n[{name}]")
        for h in [1, 3, 6]:
            ycol = f"y_power_{h}"
            mask_tr = df["is_ok"] & df[f"m_{h}"] & df[ycol].notna() & df[fcols].notna().all(axis=1) & cfg["tr"]
            mask_te = df["is_ok"] & df[f"m_{h}"] & df[ycol].notna() & df[fcols].notna().all(axis=1) & cfg["te"]

            tr_sub, te_sub = df.loc[mask_tr], df.loc[mask_te]
            Xtr, ytr = tr_sub[fcols].values, tr_sub[ycol].values
            Xte, yte = te_sub[fcols].values, te_sub[ycol].values

            nval = int(len(Xtr) * 0.15)
            gbm = lgb.LGBMRegressor(**lgb_params)
            gbm.fit(Xtr[:-nval], ytr[:-nval], eval_set=[(Xtr[-nval:], ytr[-nval:])], callbacks=[lgb.early_stopping(40, verbose=False)])

            # 儲存 H=3 的模型給 Part 2 使用
            if h == 3:
                models[name] = (gbm, fcols)

            pred = np.clip(gbm.predict(Xte), 0, 1)
            denom = np.mean(yte) if np.mean(yte) > 1e-6 else 1.0
            nrmse = rmse(yte, pred) / denom

            print(f"  H={h}h  Train={len(tr_sub):5,} / Test={len(te_sub):5,} ｜ nRMSE = {nrmse:.4f}")
            results.append({
                "experiment": "Part 1: 單一期內部",
                "setup": name,
                "H": h,
                "train_samples": len(tr_sub),
                "test_samples": len(te_sub),
                "test_period": "同時期尾端",
                "nRMSE": round(nrmse, 5)
            })

    # ---------------------------------------------------------
    # PART 2: 跨時期漂移與泛化測試 (Cross-Period Evaluation on P3 2021 Test Set)
    # ---------------------------------------------------------
    print("\n--- [PART 2] 跨時期評估 (統一對 P3 2021 測試集進行評估，測試是否存在設備/數據漂移) ---")

    # 訓練 P1+P2 混合模型 (2016-2019)
    h = 3
    ycol = f"y_power_{h}"
    p12_tr_mask = df["is_ok"] & df[f"m_{h}"] & df[ycol].notna() & df[fcols].notna().all(axis=1) & (df["ts"] <= "2019-12-31")
    p3_te_mask  = df["is_ok"] & df[f"m_{h}"] & df[ycol].notna() & df[fcols].notna().all(axis=1) & (df["ts"] >= "2021-01-01")

    tr_p12 = df.loc[p12_tr_mask]
    te_p3  = df.loc[p3_te_mask]

    Xtr12, ytr12 = tr_p12[fcols].values, tr_p12[ycol].values
    Xte3, yte3   = te_p3[fcols].values, te_p3[ycol].values

    nval12 = int(len(tr_p12) * 0.15)
    gbm_p12 = lgb.LGBMRegressor(**lgb_params)
    gbm_p12.fit(Xtr12[:-nval12], ytr12[:-nval12], eval_set=[(Xtr12[-nval12:], ytr12[-nval12:])], callbacks=[lgb.early_stopping(40, verbose=False)])

    models["Period 1+2 混合 (2016-2019)"] = (gbm_p12, fcols)

    denom3 = np.mean(yte3) if np.mean(yte3) > 1e-6 else 1.0

    # 先計算 P3 本地模型的基準 nRMSE
    p3_local_mdl = models["Period 3 (2020-2021)"][0]
    p3_local_pred = np.clip(p3_local_mdl.predict(Xte3), 0, 1)
    p3_base_nrmse = rmse(yte3, p3_local_pred) / denom3

    cross_setups = {
        "P3 本地模型 (2020 訓練) → 測 2021": models["Period 3 (2020-2021)"][0],
        "P1 模型 (僅 2016-2017 訓練) → 測 2021": models["Period 1 (2016-2017)"][0],
        "P2 模型 (僅 2018-2019 訓練) → 測 2021": models["Period 2 (2018-2019)"][0],
        "P1+P2 模型 (2016-2019 訓練) → 測 2021": models["Period 1+2 混合 (2016-2019)"][0],
    }

    for label, mdl in cross_setups.items():
        pred = np.clip(mdl.predict(Xte3), 0, 1)
        nrmse_val = rmse(yte3, pred) / denom3

        if "P3 本地模型" in label:
            diff_str = " (基準)"
            delta_pct = 0.0
        else:
            delta_pct = ((nrmse_val - p3_base_nrmse) / p3_base_nrmse) * 100
            diff_str = f" (Δ={delta_pct:+.2f}%)"

        print(f"  [{label:45s}] nRMSE = {nrmse_val:.4f}{diff_str}")

        results.append({
            "experiment": "Part 2: 跨期對照",
            "setup": label,
            "H": 3,
            "train_samples": -1,
            "test_samples": len(te_p3),
            "test_period": "Period 3 (2021)",
            "nRMSE": round(nrmse_val, 5),
            "delta_vs_local_pct": round(delta_pct, 2)
        })

    # 匯出 CSV 檔
    df_res = pd.DataFrame(results)
    os.makedirs(C.RES_DIR, exist_ok=True)
    os.makedirs(C.FIG_DIR, exist_ok=True)
    csv_path = os.path.join(C.RES_DIR, "period_drift_metrics.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"\n跨期驗證實驗結果已寫入：{csv_path}")

    # 繪製視覺化圖表
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 圖 A: 單一時期內部發電預測 nRMSE (H=1,3,6)
    df_p1 = df_res[df_res["experiment"] == "Part 1: 單一期內部"]
    piv1 = df_p1.pivot(index="setup", columns="H", values="nRMSE")
    piv1.plot(kind="bar", ax=axes[0], colormap="Set2", width=0.6, edgecolor="black")
    axes[0].set_title("三個時期各自單一內部預測 nRMSE (無跨期)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("資料時期與數據批次", fontsize=10)
    axes[0].set_ylabel("測試集 nRMSE", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # 圖 B: 跨時期測試至 P3 (2021) 評估資料一致性
    df_p2 = df_res[df_res["experiment"] == "Part 2: 跨期對照"]
    bars = axes[1].barh(df_p2["setup"], df_p2["nRMSE"], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], edgecolor='black', alpha=0.85)
    axes[1].set_title("跨時期測試至 2021 年 (驗證數據是否一致與漂移)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("nRMSE (越低越好)", fontsize=10)
    for bar in bars:
        w = bar.get_width()
        axes[1].annotate(f'{w:.4f}', xy=(w, bar.get_y() + bar.get_height() / 2),
                         xytext=(5, 0), textcoords="offset points", ha='left', va='center', fontweight='bold')

    plt.tight_layout()
    fig_path = os.path.join(C.FIG_DIR, "fig8_period_drift_experiment.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"跨期驗證圖表已產出：{fig_path}")

    print("\nStage 9 實驗完成。")

if __name__ == "__main__":
    main()
