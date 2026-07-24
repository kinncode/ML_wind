#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_train_select_interval.py —— PW 專案獨立區間發電能量模型選擇腳本 (不修改原始檔案)

功能：
  讀取 data/features_interval.parquet，針對區間累積發電能量 E_[t, t+H] 評估模型：
  1) persistence  ：ŷ = 當前值
  2) climatology  ：歷史區間均值
  3) ridge        ：標準化 L2 迴歸
  4) lightgbm     ：梯度提升樹

評估指標：
  - nRMSE, nMAE, R² 擬合度
  - 相對 Persistence 改善比例

獨立輸出至 results/test_metrics_interval.csv，完全不影響單點模型評估結果！
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
import lightgbm as lgb
import config as C

np.random.seed(C.RANDOM_SEED)

META = {"ts", "is_ok", "year", "month", "hour_i"}
def feature_cols(df):
    return [c for c in df.columns
            if c not in META and not c.startswith("y_") and not c.startswith("m_")]

def rmse(a, b): return float(np.sqrt(np.mean((a - b)**2)))
def mae(a, b):  return float(np.mean(np.abs(a - b)))

def metrics(y, pred, persist):
    r = rmse(y, pred); m = mae(y, pred)
    r2 = float(r2_score(y, pred))
    denom = np.mean(y) if np.mean(y) > 1e-6 else 1.0
    rp = rmse(y, persist)
    return {"RMSE": r, "MAE": m, "nRMSE": r / denom, "R2": r2,
            "skill_vs_persist": 1 - r / rp if rp > 0 else np.nan}

def build_xy(df, target, h):
    ycol = f"y_ws100_{h}" if target == "ws100" else f"y_power_{h}"
    fcols = feature_cols(df)
    mask = (df["is_ok"] & df[f"m_{h}"] & df[ycol].notna() & df[fcols].notna().all(axis=1))
    sub = df.loc[mask]
    persist = sub["WS_100_mean"].values if target == "ws100" else sub["P_now"].values
    return sub, fcols, sub[ycol].values, persist

def climatology_pred(train_df, eval_df, ycol):
    key = train_df.groupby(["month", "hour_i"])[ycol].mean()
    glob = train_df[ycol].mean()
    idx = list(zip(eval_df["month"], eval_df["hour_i"]))
    return np.array([key.get(k, glob) for k in idx])

def main():
    print("="*70)
    print("PW 專案 —— 獨立區間能量預測模型選擇 (03_train_select_interval.py)")
    print("="*70)

    feat_path = os.path.join(C.DATA_DIR, "features_interval.parquet")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"找不到 {feat_path}，請先執行 02_features_interval.py")

    df = pd.read_parquet(feat_path)
    df["ts"] = pd.to_datetime(df["ts"])
    test_start = pd.Timestamp(C.TEST_START)

    print(f"總筆數：{len(df):,}｜測試年起始點：{C.TEST_START}｜Targets={C.TARGETS}｜Horizons={C.HORIZONS_H}")

    cv_rows, test_rows = [], []
    lgb_params = dict(objective="regression", n_estimators=350, learning_rate=0.05,
                      num_leaves=48, min_child_samples=100, subsample=0.8,
                      subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                      random_state=C.RANDOM_SEED, n_jobs=-1, verbose=-1)

    os.makedirs(C.MODEL_DIR, exist_ok=True)
    os.makedirs(C.RES_DIR, exist_ok=True)
    os.makedirs(C.DATA_DIR, exist_ok=True)

    for target in C.TARGETS:
        for h in C.HORIZONS_H:
            sub, fcols, y, persist = build_xy(df, target, h)
            ycol = f"y_ws100_{h}" if target == "ws100" else f"y_power_{h}"
            is_test = sub["ts"] >= test_start

            tr, te = sub.loc[~is_test], sub.loc[is_test]
            ytr, yte = y[~is_test.values], y[is_test.values]
            ptr, pte = persist[~is_test.values], persist[is_test.values]
            Xtr, Xte = tr[fcols].values, te[fcols].values
            tag = f"{target}_interval_H{h}"

            print(f"\n=== 區間 {tag} ===  Train 樣本 {len(tr):,} / Test 樣本 {len(te):,}  (特徵數 {len(fcols)})")

            # 1) 時序 CV (訓練期)
            tscv = TimeSeriesSplit(n_splits=C.N_CV_SPLITS)
            for name, mk in [("ridge", lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0))),
                             ("lightgbm", lambda: lgb.LGBMRegressor(**lgb_params))]:
                fold_scores = []
                for fi, (a, b) in enumerate(tscv.split(Xtr)):
                    mdl = mk()
                    mdl.fit(Xtr[a], ytr[a])
                    pr = mdl.predict(Xtr[b])
                    denom = np.mean(ytr[b]) if np.mean(ytr[b]) > 1e-6 else 1.0
                    sc = rmse(ytr[b], pr) / denom
                    fold_scores.append(sc)
                    cv_rows.append({"tag": tag, "target": target, "H": h, "model": name,
                                    "fold": fi, "nRMSE": sc})
                print(f"   CV {name:9s} nRMSE mean = {np.mean(fold_scores):.4f}")

            # 2) 保留測試年評估 (4 個模型)
            results = {}
            results["persistence"] = metrics(yte, pte, pte)

            clim = climatology_pred(tr.assign(**{ycol: ytr}), te, ycol)
            results["climatology"] = metrics(yte, clim, pte)

            rid = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(Xtr, ytr)
            pred_rid = rid.predict(Xte)
            if target == "power": pred_rid = np.clip(pred_rid, 0, 1)
            results["ridge"] = metrics(yte, pred_rid, pte)

            nval = int(len(Xtr) * 0.15)
            gbm = lgb.LGBMRegressor(**lgb_params)
            gbm.fit(Xtr[:-nval], ytr[:-nval],
                    eval_set=[(Xtr[-nval:], ytr[-nval:])],
                    callbacks=[lgb.early_stopping(40, verbose=False)])
            pred_lgb = gbm.predict(Xte)
            if target == "power": pred_lgb = np.clip(pred_lgb, 0, 1)
            results["lightgbm"] = metrics(yte, pred_lgb, pte)

            best = min(results, key=lambda k: results[k]["nRMSE"])
            for name, mt in results.items():
                test_rows.append({"tag": tag, "target": target, "H": h, "model": name,
                                  **{k: round(v, 5) for k, v in mt.items()},
                                  "is_best": name == best})

            print(f"   TEST nRMSE: " + "  ".join(f"{n}={results[n]['nRMSE']:.4f} (R²={results[n]['R2']:.3f})" for n in results) + f"  → 最佳={best}")

            # 儲存區間獨立模型與預測結果
            gbm.booster_.save_model(os.path.join(C.MODEL_DIR, f"lgbm_{tag}.txt"))

            imp = pd.DataFrame({"feature": fcols, "gain": gbm.booster_.feature_importance("gain")})
            imp = imp.sort_values("gain", ascending=False)
            imp.to_csv(os.path.join(C.RES_DIR, f"importance_{tag}.csv"), index=False)

            pd.DataFrame({
                "ts": te["ts"].values,
                "y_true": yte,
                "persist": pte,
                "pred_lgbm": pred_lgb,
                "pred_ridge": pred_rid,
                "pred_clim": clim
            }).to_parquet(os.path.join(C.DATA_DIR, f"pred_{tag}.parquet"), index=False)

    pd.DataFrame(cv_rows).to_csv(os.path.join(C.RES_DIR, "cv_scores_interval.csv"), index=False)
    pd.DataFrame(test_rows).to_csv(os.path.join(C.RES_DIR, "test_metrics_interval.csv"), index=False)

    print(f"\n模型訓練與評估完成，指標已寫入 {os.path.join(C.RES_DIR, 'test_metrics_interval.csv')}")
    print("03_train_select_interval.py 執行完成。")

if __name__ == "__main__":
    main()
