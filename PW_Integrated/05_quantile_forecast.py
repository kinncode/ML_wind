#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 5 —— 不確定性分位數機率預測 (Quantile Probabilistic Forecast)

針對正規化發電量 P (0-1)，訓練 LightGBM Quantile Regressor 輸出不確定性區間 p10 / p50 / p90。
評估：
  - 實質涵蓋率 (Empirical Coverage): 測試集中實際值 <= 預測分位數的比例
  - Pinball Loss (Quantile Loss)
  - 匯出 results/quantile_metrics.csv 與 data/pred_quantile_H{h}.parquet
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import config as C

def pinball_loss(y_true, y_pred, q):
    err = y_true - y_pred
    return float(np.mean(np.maximum(q * err, (q - 1.0) * err)))

def main():
    print("="*70)
    print("PW_Integrated Stage 5 —— 不確定性分位數機率預測 (p10/p50/p90)")
    print("="*70)

    if not os.path.exists(C.FEAT_PARQUET):
        raise FileNotFoundError(f"找不到 {C.FEAT_PARQUET}，請先執行 03_features.py")

    df = pd.read_parquet(C.FEAT_PARQUET)
    df["ts"] = pd.to_datetime(df["ts"])
    test_start = pd.Timestamp(C.TEST_START)

    meta = {"ts", "is_ok", "year", "month", "hour_i"}
    fcols = [c for c in df.columns if c not in meta and not c.startswith("y_") and not c.startswith("m_")]

    rows = []

    for h in C.HORIZONS_H:
        ycol = f"y_power_{h}"
        mask = (df["is_ok"] & df[f"m_{h}"] & df[ycol].notna() & df[fcols].notna().all(axis=1))
        sub = df.loc[mask]
        is_test = sub["ts"] >= test_start

        tr, te = sub.loc[~is_test], sub.loc[is_test]
        ytr, yte = tr[ycol].values, te[ycol].values
        Xtr, Xte = tr[fcols].values, te[fcols].values

        nval = int(len(Xtr) * 0.15)
        pred_dict = {"ts": te["ts"].values, "y_true": yte}

        print(f"\n=== Power H={h}h 機率預測 === Train 樣本 {len(tr):,} / Test 樣本 {len(te):,}")

        for q in C.QUANTILES:
            q_params = dict(
                objective="quantile",
                alpha=q,
                learning_rate=0.05,
                num_leaves=63,
                min_child_samples=50,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=C.RANDOM_SEED,
                n_jobs=-1,
                verbose=-1
            )

            gbm = lgb.LGBMRegressor(**q_params)
            gbm.fit(Xtr[:-nval], ytr[:-nval],
                    eval_set=[(Xtr[-nval:], ytr[-nval:])],
                    callbacks=[lgb.early_stopping(50, verbose=False)])

            pred_q = np.clip(gbm.predict(Xte), 0.0, 1.0)
            coverage = float(np.mean(yte <= pred_q))
            ploss = pinball_loss(yte, pred_q, q)
            q_tag = f"q{int(q*100)}"
            pred_dict[q_tag] = pred_q

            model_path = os.path.join(C.MODEL_DIR, f"lgbm_quant_power_H{h}_q{int(q*100)}.txt")
            gbm.booster_.save_model(model_path)

            print(f"  [分位 p{int(q*100):02d}] 理想={q:.1f} ｜ 測試集實質涵蓋率={coverage:.4f} ｜ Pinball Loss={ploss:.5f}")

            rows.append({
                "H": h,
                "quantile": q,
                "empirical_coverage": round(coverage, 4),
                "pinball_loss": round(ploss, 5)
            })

        pd.DataFrame(pred_dict).to_parquet(os.path.join(C.DATA_DIR, f"pred_quantile_H{h}.parquet"), index=False)

    pd.DataFrame(rows).to_csv(os.path.join(C.RES_DIR, "quantile_metrics.csv"), index=False)
    print(f"\n分位數評估完成，結果已寫入 {os.path.join(C.RES_DIR, 'quantile_metrics.csv')}")
    print("Stage 5 完成。")

if __name__ == "__main__":
    main()
