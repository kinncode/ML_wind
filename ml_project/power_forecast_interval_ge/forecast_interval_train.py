#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
forecast_interval_train.py —— 修正版區間能量多模型訓練與選擇模組

修正內容：
  1. 正規化指標標準化：
     - nRMSE_cap  = RMSE / 1.0 (額定容量正規化，國際標準，與原版一對一對齊)
     - nRMSE_mean = RMSE / mean(y) (均值正規化，附註說明)
  2. XGBoost 補上 `early_stopping_rounds=40` 確保模型訓練公平度。
  3. 對稱式 Persistence 基準：採用過去 H 小時滞動均值作為區間對照 (P_rmean_{h} / ws_rmean)。
  4. 評估標的：區間平均功率 (Power Mean 0~1) 與 區間平均風速。
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import lightgbm as lgb
import xgboost as xgb
import config as C

np.random.seed(C.RANDOM_SEED)

META = {"ts", "is_ok", "year", "month", "hour_i"}
def feature_cols(df):
    return [c for c in df.columns
            if c not in META and not c.startswith("y_") and not c.startswith("m_")]

def rmse(a, b): return float(np.sqrt(np.mean((a - b)**2)))
def mae(a, b):  return float(np.mean(np.abs(a - b)))

def metrics(y, pred, persist, scale_factor=1.0):
    r = rmse(y, pred); m = mae(y, pred)
    r2 = float(r2_score(y, pred))
    denom_cap = scale_factor  # 額定容量 (1.0 或 H)
    denom_mean = np.mean(y) if np.mean(y) > 1e-6 else 1.0
    rp = rmse(y, persist)
    return {
        "RMSE": r,
        "MAE": m,
        "nRMSE_cap": r / denom_cap,       # 國際標準：額定容量正規化 nRMSE
        "nRMSE_mean": r / denom_mean,     # 均值正規化 nRMSE
        "R2": r2,
        "skill_vs_persist": 1 - r / rp if rp > 0 else np.nan
    }

def main():
    print("="*70)
    print("power_forecast_interval Stage 4 —— 修正版區間多模型訓練與選擇")
    print("="*70)

    if not os.path.exists(C.FEAT_PARQUET):
        raise FileNotFoundError(f"找不到 {C.FEAT_PARQUET}，請先執行 forecast_interval_features.py")

    df = pd.read_parquet(C.FEAT_PARQUET)
    df["ts"] = pd.to_datetime(df["ts"])
    test_start = pd.Timestamp(C.TEST_START)

    test_rows = []
    lgb_params = dict(objective="regression", n_estimators=350, learning_rate=0.05,
                      num_leaves=48, min_child_samples=100, subsample=0.8,
                      colsample_bytree=0.8, reg_lambda=1.0, random_state=C.RANDOM_SEED,
                      n_jobs=-1, verbose=-1)

    xgb_params = dict(objective="reg:squarederror", n_estimators=350, learning_rate=0.05,
                      max_depth=6, subsample=0.8, colsample_bytree=0.8,
                      reg_lambda=1.0, random_state=C.RANDOM_SEED, n_jobs=-1,
                      early_stopping_rounds=40)

    os.makedirs(C.MODEL_DIR, exist_ok=True)
    os.makedirs(C.RES_DIR, exist_ok=True)

    for target in C.TARGETS:
        for h in C.HORIZONS_H:
            ycol = f"y_{target}_{h}"
            fcols = feature_cols(df)
            mask = df["is_ok"] & df[f"m_{h}"] & df[ycol].notna() & df[fcols].notna().all(axis=1)
            sub = df.loc[mask]

            # 對稱式 Persistence 基準：用過去 H 小時滞動均值預測未來 H 小時區間均值
            rmean_col = f"P_rmean_{h}"
            if target == "ws100":
                # 風速用對應的滞動均值特徵（取最接近的 rolling window）
                win_min = h * 60
                candidates = [60, 180, 360]
                best_win = min(candidates, key=lambda w: abs(w - win_min))
                persist = sub[f"ws_rmean_{best_win}"].values
            elif rmean_col in sub.columns:
                persist = sub[rmean_col].values
            else:
                persist = sub["P_now"].values  # fallback
            y = sub[ycol].values
            is_test = sub["ts"] >= test_start

            tr, te = sub.loc[~is_test], sub.loc[is_test]
            ytr, yte = y[~is_test.values], y[is_test.values]
            ptr, pte = persist[~is_test.values], persist[is_test.values]
            Xtr, Xte = tr[fcols].values, te[fcols].values
            tag = f"{target}_interval_H{h}"

            scale_factor = 1.0  # 額定基準
            print(f"\n=== 區間 {tag} ===  Train {len(tr):,} / Test {len(te):,}")

            res = {}
            res["persistence"] = metrics(yte, pte, pte, scale_factor=scale_factor)

            # Climatology
            key = tr.groupby(["month", "hour_i"])[ycol].mean()
            glob = tr[ycol].mean()
            idx = list(zip(te["month"], te["hour_i"]))
            clim = np.array([key.get(k, glob) for k in idx])
            res["climatology"] = metrics(yte, clim, pte, scale_factor=scale_factor)

            # Ridge
            rid = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(Xtr, ytr)
            pred_rid = rid.predict(Xte)
            if target == "power": pred_rid = np.clip(pred_rid, 0, 1)
            res["ridge"] = metrics(yte, pred_rid, pte, scale_factor=scale_factor)

            # XGBoost (對齊 Early Stopping)
            nval = int(len(Xtr) * 0.15)
            xmdl = xgb.XGBRegressor(**xgb_params)
            xmdl.fit(Xtr[:-nval], ytr[:-nval], eval_set=[(Xtr[-nval:], ytr[-nval:])], verbose=False)
            pred_xgb = xmdl.predict(Xte)
            if target == "power": pred_xgb = np.clip(pred_xgb, 0, 1)
            res["xgboost"] = metrics(yte, pred_xgb, pte, scale_factor=scale_factor)

            # LightGBM (對齊 Early Stopping)
            gbm = lgb.LGBMRegressor(**lgb_params)
            gbm.fit(Xtr[:-nval], ytr[:-nval], eval_set=[(Xtr[-nval:], ytr[-nval:])],
                    callbacks=[lgb.early_stopping(40, verbose=False)])
            pred_lgb = gbm.predict(Xte)
            if target == "power": pred_lgb = np.clip(pred_lgb, 0, 1)
            res["lightgbm"] = metrics(yte, pred_lgb, pte, scale_factor=scale_factor)

            best = min(res, key=lambda k: res[k]["nRMSE_cap"])
            for name, mt in res.items():
                test_rows.append({"tag": tag, "target": target, "H": h, "model": name,
                                  **{k: round(v, 5) for k, v in mt.items()},
                                  "is_best": name == best})

            print(f"   TEST nRMSE(cap): " + "  ".join(f"{n}={res[n]['nRMSE_cap']:.4f} (R2={res[n]['R2']:.3f})" for n in res) + f"  -> 最佳={best}")

            gbm.booster_.save_model(os.path.join(C.MODEL_DIR, f"lgbm_{tag}.txt"))
            xmdl.save_model(os.path.join(C.MODEL_DIR, f"xgb_{tag}.json"))

            pd.DataFrame({
                "ts": te["ts"].values,
                "y_true": yte,
                "persist": pte,
                "pred_lgbm": pred_lgb,
                "pred_xgb": pred_xgb,
                "pred_ridge": pred_rid
            }).to_parquet(os.path.join(C.DATA_DIR, f"pred_{tag}.parquet"), index=False)

    pd.DataFrame(test_rows).to_csv(os.path.join(C.RES_DIR, "test_metrics_interval.csv"), index=False)
    print(f"\n修正版評估結果已寫入：{os.path.join(C.RES_DIR, 'test_metrics_interval.csv')}")
    print("Stage 4 完成。")

if __name__ == "__main__":
    main()
