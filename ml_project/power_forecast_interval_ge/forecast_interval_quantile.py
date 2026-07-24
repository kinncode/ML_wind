#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
forecast_interval_quantile.py —— power_forecast_interval 區間發電能量 p10/p50/p90 機率預測模組

功能：
  利用 LightGBM Quantile Regression 針對 1h, 3h, 6h, 24h 區間發電能量建立 p10 (下界), p50 (中位數), p90 (上界) 信賴帶模型。
  完全排除當前點洩漏 (0 洩漏標的)，評估 Pinball Loss、80% 信賴區間覆蓋率 (PICP) 與信賴帶寬度 (Sharpness)。

輸出：
  results/quantile_interval_metrics.csv
  data/pred_quantile_power_interval_H{h}.parquet
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score
import config as C

META = {"ts", "is_ok", "year", "month", "hour_i"}
def feature_cols(df):
    return [c for c in df.columns
            if c not in META and not c.startswith("y_") and not c.startswith("m_")]

def pinball_loss(y_true, y_pred, alpha):
    err = y_true - y_pred
    return float(np.mean(np.maximum(alpha * err, (alpha - 1) * err)))

def main():
    print("="*70)
    print("power_forecast_interval —— p10/p50/p90 區間機率預測模組")
    print("="*70)

    if not os.path.exists(C.FEAT_PARQUET):
        raise FileNotFoundError(f"找不到 {C.FEAT_PARQUET}，請先執行 forecast_interval_features.py")

    df = pd.read_parquet(C.FEAT_PARQUET)
    df["ts"] = pd.to_datetime(df["ts"])
    test_start = pd.Timestamp(C.TEST_START)
    fcols = feature_cols(df)

    alphas = [0.10, 0.50, 0.90]
    q_results = []

    os.makedirs(C.MODEL_DIR, exist_ok=True)
    os.makedirs(C.RES_DIR, exist_ok=True)

    for h in C.HORIZONS_H:
        ycol = f"y_power_{h}"
        mask = df["is_ok"] & df[f"m_{h}"] & df[ycol].notna() & df[fcols].notna().all(axis=1)
        sub = df.loc[mask]

        is_test = sub["ts"] >= test_start
        tr, te = sub.loc[~is_test], sub.loc[is_test]
        ytr, yte = sub.loc[~is_test][ycol].values, sub.loc[is_test][ycol].values
        Xtr, Xte = tr[fcols].values, te[fcols].values

        nval = int(len(Xtr) * 0.15)
        Xtr_val, ytr_val = Xtr[:-nval], ytr[:-nval]
        Xva_val, yva_val = Xtr[-nval:], ytr[-nval:]

        print(f"\n=== 區間 H={h}h 機率預測 (p10/p50/p90) === Train {len(tr):,} / Test {len(te):,}")

        preds = {}
        for alpha in alphas:
            q_name = f"p{int(alpha*100)}"
            params = dict(objective="quantile", alpha=alpha, n_estimators=350, learning_rate=0.05,
                          num_leaves=63, min_child_samples=100, subsample=0.8,
                          colsample_bytree=0.8, reg_lambda=1.0, random_state=C.RANDOM_SEED,
                          n_jobs=-1, verbose=-1)

            gbm = lgb.LGBMRegressor(**params)
            gbm.fit(Xtr_val, ytr_val, eval_set=[(Xva_val, yva_val)],
                    callbacks=[lgb.early_stopping(40, verbose=False)])

            pred_q = np.clip(gbm.predict(Xte), 0, 1)
            preds[q_name] = pred_q

            loss = pinball_loss(yte, pred_q, alpha)
            coverage = float(np.mean(yte <= pred_q))
            print(f"   [{q_name:4s}] Pinball Loss = {loss:.5f} ｜ 經驗覆蓋率 = {coverage*100:5.1f}%")

            gbm.booster_.save_model(os.path.join(C.MODEL_DIR, f"lgbm_quantile_{q_name}_H{h}.txt"))

        # 計算 80% 機率信賴帶 (p10 ~ p90)
        picp = float(np.mean((yte >= preds["p10"]) & (yte <= preds["p90"])) * 100.0)
        sharpness = float(np.mean(preds["p90"] - preds["p10"]))
        r2_p50 = float(r2_score(yte, preds["p50"]))

        print(f"   ★ 80% 信賴帶 (p10~p90) 實測經驗覆蓋率 (PICP) = {picp:.2f}% (目標 80.0%)")
        print(f"   ★ 信賴帶平均寬度 (Sharpness) = {sharpness:.4f} ｜ p50 R² = {r2_p50:.4f}")

        q_results.append({
            "H": h,
            "horizon_label": f"{h}h",
            "picp_80_pct": round(picp, 2),
            "sharpness": round(sharpness, 5),
            "r2_p50": round(r2_p50, 5),
            "pinball_loss_p10": round(pinball_loss(yte, preds["p10"], 0.10), 5),
            "pinball_loss_p50": round(pinball_loss(yte, preds["p50"], 0.50), 5),
            "pinball_loss_p90": round(pinball_loss(yte, preds["p90"], 0.90), 5),
        })

        # 儲存機率預測 parquet 檔案
        pd.DataFrame({
            "ts": te["ts"].values,
            "y_true": yte,
            "p10": preds["p10"],
            "p50": preds["p50"],
            "p90": preds["p90"]
        }).to_parquet(os.path.join(C.DATA_DIR, f"pred_quantile_power_interval_H{h}.parquet"), index=False)

    df_qres = pd.DataFrame(q_results)
    csv_out = os.path.join(C.RES_DIR, "quantile_interval_metrics.csv")
    df_qres.to_csv(csv_out, index=False)

    print(f"\n機率預測指標已寫入：{csv_out}")
    print("forecast_interval_quantile.py 執行完成。")

if __name__ == "__main__":
    main()
