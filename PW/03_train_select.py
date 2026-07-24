#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3 —— 模型選擇（時序前推驗證 + 保留測試年）

對每個 (目標 target ∈ {ws100, power}) × (時程 H ∈ {1,3,6}h) 比較：
  1) persistence  ：ŷ = 當前值（風速的 WS_100_mean / 出力的 P_now）
  2) climatology  ：訓練期 月×時 的目標平均
  3) ridge        ：標準化 + L2 線性回歸
  4) lightgbm     ：梯度提升樹

驗證：訓練期（< TEST_START）內用 expanding-window TimeSeriesSplit 做 CV，
      回報各折 nRMSE 平均；再用整個訓練期重訓，於保留測試年評估。
選模：以測試年 nRMSE 最低者為每個 (target,H) 的最佳模型。

輸出：
  results/cv_scores.csv        每折 CV 分數
  results/test_metrics.csv     測試年各模型指標 + 是否最佳
  results/importance_*.csv     最佳 LGBM 特徵重要度
  models/*.txt                 最佳 LGBM 模型
  data/pred_*.parquet          測試年預測（給 Stage 4 畫圖）
"""
from __future__ import annotations
import os, json, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import config as C

np.random.seed(C.RANDOM_SEED)

META = {"ts","is_ok","year","month","hour_i"}
def feature_cols(df):
    return [c for c in df.columns
            if c not in META and not c.startswith("y_") and not c.startswith("m_")]

def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))
def mae(a,b):  return float(np.mean(np.abs(a-b)))

def metrics(y, pred, persist):
    r = rmse(y,pred); m = mae(y,pred)
    denom = np.mean(y) if np.mean(y)>1e-6 else 1.0
    rp = rmse(y,persist)
    return {"RMSE":r, "MAE":m, "nRMSE":r/denom,
            "skill_vs_persist": 1 - r/rp if rp>0 else np.nan}

def build_xy(df, target, h):
    ycol = f"y_{target if target!='ws100' else 'ws100'}_{h}" if target=="power" else f"y_ws100_{h}"
    ycol = f"y_ws100_{h}" if target=="ws100" else f"y_power_{h}"
    fcols = feature_cols(df)
    mask = (df["is_ok"] & df[f"m_{h}"] & df[ycol].notna()
            & df[fcols].notna().all(axis=1))
    sub = df.loc[mask]
    persist = sub["WS_100_mean"].values if target=="ws100" else sub["P_now"].values
    return sub, fcols, sub[ycol].values, persist

def climatology_pred(train_df, eval_df, ycol):
    key = train_df.groupby(["month","hour_i"])[ycol].mean()
    glob = train_df[ycol].mean()
    idx = list(zip(eval_df["month"], eval_df["hour_i"]))
    return np.array([key.get(k, glob) for k in idx])

def main():
    # argv: [target] [H]  —— 可指定單一 target 與單一時程，以在執行時限內完成
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    only_h = int(sys.argv[2]) if len(sys.argv) > 2 else None
    targets = C.TARGETS if which == "all" else [which]
    horizons = C.HORIZONS_H if only_h is None else [only_h]

    df = pd.read_parquet(C.FEAT_PARQUET)
    df["ts"] = pd.to_datetime(df["ts"])
    test_start = pd.Timestamp(C.TEST_START)
    print(f"總筆數 {len(df):,}｜測試年起 {C.TEST_START}｜targets={targets}｜H={horizons}")

    cv_rows, test_rows = [], []
    lgb_params = dict(objective="regression", n_estimators=350, learning_rate=0.05,
                      num_leaves=48, min_child_samples=100, subsample=0.8,
                      subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                      random_state=C.RANDOM_SEED, n_jobs=-1, verbose=-1)

    for target in targets:
        for h in horizons:
            sub, fcols, y, persist = build_xy(df, target, h)
            ycol = f"y_ws100_{h}" if target=="ws100" else f"y_power_{h}"
            ts = sub["ts"].values
            is_test = sub["ts"] >= test_start
            tr, te = sub.loc[~is_test], sub.loc[is_test]
            ytr, yte = y[~is_test.values], y[is_test.values]
            ptr, pte = persist[~is_test.values], persist[is_test.values]
            Xtr, Xte = tr[fcols].values, te[fcols].values
            tag = f"{target}_H{h}"
            print(f"\n=== {tag} ===  train {len(tr):,} / test {len(te):,}  feats {len(fcols)}")

            # ---------- 時序 CV（只在訓練期，比較 ridge vs lgb） ----------
            tscv = TimeSeriesSplit(n_splits=C.N_CV_SPLITS)
            for name, mk in [("ridge", lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0))),
                             ("lightgbm", lambda: lgb.LGBMRegressor(**lgb_params))]:
                fold_scores=[]
                for fi,(a,b) in enumerate(tscv.split(Xtr)):
                    mdl = mk(); mdl.fit(Xtr[a], ytr[a])
                    pr = mdl.predict(Xtr[b])
                    denom = np.mean(ytr[b]) if np.mean(ytr[b])>1e-6 else 1.0
                    sc = rmse(ytr[b],pr)/denom
                    fold_scores.append(sc)
                    cv_rows.append({"tag":tag,"target":target,"H":h,"model":name,
                                    "fold":fi,"nRMSE":sc})
                print(f"   CV {name:9s} nRMSE mean={np.mean(fold_scores):.4f}")

            # ---------- 測試年評估（4 個模型）----------
            results={}
            # persistence
            results["persistence"] = metrics(yte, pte, pte)
            # climatology
            clim = climatology_pred(tr.assign(**{ycol:ytr}), te, ycol)
            results["climatology"] = metrics(yte, clim, pte)
            # ridge
            rid = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(Xtr,ytr)
            results["ridge"] = metrics(yte, rid.predict(Xte), pte)
            # lightgbm (early stopping 用訓練尾端 15% 當 val)
            nval=int(len(Xtr)*0.15)
            gbm = lgb.LGBMRegressor(**lgb_params)
            gbm.fit(Xtr[:-nval], ytr[:-nval],
                    eval_set=[(Xtr[-nval:], ytr[-nval:])],
                    callbacks=[lgb.early_stopping(40, verbose=False)])
            predL = gbm.predict(Xte)
            if target=="power": predL = np.clip(predL,0,1)
            results["lightgbm"] = metrics(yte, predL, pte)

            # 選最佳
            best = min(results, key=lambda k: results[k]["nRMSE"])
            for name,mt in results.items():
                test_rows.append({"tag":tag,"target":target,"H":h,"model":name,
                                  **{k:round(v,5) for k,v in mt.items()},
                                  "is_best": name==best})
            print(f"   TEST nRMSE: " + "  ".join(f"{n}={results[n]['nRMSE']:.4f}" for n in results)
                  + f"   → best={best}")

            # 存最佳 LGBM 模型 + 重要度 + 測試年預測
            gbm.booster_.save_model(os.path.join(C.MODEL_DIR, f"lgbm_{tag}.txt"))
            imp = pd.DataFrame({"feature":fcols,"gain":gbm.booster_.feature_importance("gain")}
                               ).sort_values("gain",ascending=False)
            imp.to_csv(os.path.join(C.RES_DIR,f"importance_{tag}.csv"),index=False)
            pd.DataFrame({"ts":te["ts"].values,"y_true":yte,"persist":pte,
                          "pred_lgbm":predL,"pred_ridge":rid.predict(Xte),
                          "pred_clim":clim}).to_parquet(
                          os.path.join(C.DATA_DIR,f"pred_{tag}.parquet"),index=False)

    os.makedirs(C.RES_DIR, exist_ok=True)
    if which == "all":
        suffix = ""
    else:
        suffix = f"_{which}" + (f"_H{only_h}" if only_h is not None else "")
    pd.DataFrame(cv_rows).to_csv(os.path.join(C.RES_DIR,f"cv_scores{suffix}.csv"),index=False)
    tm = pd.DataFrame(test_rows)
    tm.to_csv(os.path.join(C.RES_DIR,f"test_metrics{suffix}.csv"),index=False)
    print(f"\n輸出：results/cv_scores{suffix}.csv, results/test_metrics{suffix}.csv, importance_*.csv, models/*.txt")
    print("\n===== 測試年最佳模型摘要 =====")
    for _,r in tm[tm.is_best].iterrows():
        print(f"  {r.tag:12s} best={r.model:11s} nRMSE={r.nRMSE:.4f} "
              f"skill_vs_persist={r.skill_vs_persist:+.3f}")

if __name__ == "__main__":
    main()
