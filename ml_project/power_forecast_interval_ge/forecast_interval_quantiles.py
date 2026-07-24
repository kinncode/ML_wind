#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
forecast_interval_quantiles.py —— 為「窗口平均出力」補上機率區間 (p10/p50/p90)。（可續跑）

原本的 power_forecast_interval 只給窗口平均出力的「點預測」(一個數字)。
本模組加上分位數迴歸，讓它也能輸出「不確定區間」：

  例：未來 3 小時平均出力 p10=32% / p50=48% / p90=65%
      → 未來 3 小時總發電量有 80% 機率落在 [32%, 65%] × 額定 × 3h

沿用 forecast_interval_features.py 產生的無洩漏特徵與窗口平均標的 y_power_{h}。
三段時間切分（CQR 需獨立校正集）：
  訓練 < 2019-06 ｜ 校正 2019-06 ~ 2020-06 ｜ 測試 ≥ 2020-06（與原專案測試集一致）
"""
from __future__ import annotations
import json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import config as C

HERE = Path(__file__).resolve().parent
CACHE = HERE / "quant_cache.json"
CAL_START = "2019-06-01"
TEST_START = C.TEST_START            # 2020-06-01
QUANTILES = [0.10, 0.50, 0.90]       # 使用者指定 p10–p90（含中位數）
META = {"ts", "is_ok", "year", "month", "hour_i"}

LGB = dict(objective="quantile", n_estimators=800, learning_rate=0.05,
           num_leaves=63, min_child_samples=80, subsample=0.85,
           colsample_bytree=0.85, reg_lambda=1.0, random_state=C.RANDOM_SEED,
           n_jobs=2, verbose=-1)


def feature_cols(df):
    return [c for c in df.columns
            if c not in META and not c.startswith("y_") and not c.startswith("m_")
            and not c.startswith("P_rmean")]


def qc(q): return f"q{int(round(q*100)):02d}"


def load_cache():
    if CACHE.exists() and CACHE.stat().st_size:
        return json.loads(CACHE.read_text())
    return {"done": []}


def save_cache(c): CACHE.write_text(json.dumps(c, ensure_ascii=False, indent=2))


def main(budget=38.0):
    df = pd.read_parquet(C.FEAT_PARQUET)
    df["ts"] = pd.to_datetime(df["ts"])
    fcols = feature_cols(df)
    c = load_cache(); t0 = time.time()

    for h in C.HORIZONS_H:
        ycol = f"y_power_{h}"
        mask = df["is_ok"] & df[f"m_{h}"] & df[ycol].notna() & df[fcols].notna().all(axis=1)
        sub = df.loc[mask].copy()
        is_cal = (sub.ts >= CAL_START) & (sub.ts < TEST_START)
        is_test = sub.ts >= TEST_START
        tr = sub.loc[~is_cal & ~is_test]; cal = sub.loc[is_cal]; te = sub.loc[is_test]
        Xtr, ytr = tr[fcols].values, tr[ycol].values
        Xcal, Xte = cal[fcols].values, te[fcols].values

        for q in QUANTILES:
            key = f"H{h}:{q}"
            if key in c["done"]:
                continue
            if time.time() - t0 > budget:
                print("時間用完，請再執行一次"); save_cache(c); return
            p = dict(LGB); p["alpha"] = q
            nval = int(len(Xtr) * 0.15)
            mdl = lgb.LGBMRegressor(**p)
            mdl.fit(Xtr[:-nval], ytr[:-nval], eval_set=[(Xtr[-nval:], ytr[-nval:])],
                    callbacks=[lgb.early_stopping(40, verbose=False)])
            mdl.booster_.save_model(str(HERE / "models" / f"q_power_H{h}_{qc(q)}.txt"))
            pcal = np.clip(mdl.predict(Xcal), 0, 1)
            pte = np.clip(mdl.predict(Xte), 0, 1)
            for tag, base_df, pv in [("cal", cal, pcal), ("test", te, pte)]:
                f = HERE / "data" / f"pred_q{tag}_power_H{h}.parquet"
                b = pd.read_parquet(f) if f.exists() else \
                    pd.DataFrame({"ts": base_df.ts.values, "y": base_df[ycol].values})
                b[qc(q)] = pv
                b.to_parquet(f, index=False)
            c["done"].append(key); save_cache(c)
            print(f"  ✓ H{h} p{int(q*100):02d}")

    total = len(C.HORIZONS_H) * len(QUANTILES)
    if len(c["done"]) >= total:
        print(f"\n=== 分位數模型全部完成（{total} 個）===")
        evaluate()
    else:
        print(f"進度 {len(c['done'])}/{total}")


def cqr_Q(cal, nominal=0.80):
    """CQR 保形校正量（對 p10–p90 的 80% 區間）。"""
    y = cal.y.to_numpy(); lo = cal.q10.to_numpy(); hi = cal.q90.to_numpy()
    E = np.maximum(lo - y, y - hi)
    n = len(E); level = min(np.ceil((n + 1) * nominal) / n, 1.0)
    return float(np.quantile(E, level, method="higher"))


def evaluate():
    rows = []; store = {}
    for h in C.HORIZONS_H:
        cal = pd.read_parquet(HERE / "data" / f"pred_qcal_power_H{h}.parquet")
        te = pd.read_parquet(HERE / "data" / f"pred_qtest_power_H{h}.parquet")
        # 單調化
        for d in (cal, te):
            M = np.sort(d[["q10", "q50", "q90"]].to_numpy(), axis=1)
            d[["q10", "q50", "q90"]] = M
        y = te.y.to_numpy()
        r2 = 1 - ((y - te.q50) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        cov0 = np.mean((y >= te.q10) & (y <= te.q90)); w0 = np.mean(te.q90 - te.q10)
        Q = cqr_Q(cal, 0.80); store[f"H{h}"] = round(Q, 4)
        lo2 = np.clip(te.q10 - Q, 0, 1); hi2 = np.clip(te.q90 + Q, 0, 1)
        cov1 = np.mean((y >= lo2) & (y <= hi2)); w1 = np.mean(hi2 - lo2)
        pb = np.mean([np.mean(np.maximum(q * (y - te[qc(q)]), (q - 1) * (y - te[qc(q)])))
                      for q in QUANTILES])
        rows.append({"H": h, "p50_R2": round(float(r2), 4),
                     "raw_cov80_pct": round(100 * cov0, 1), "raw_width_pct": round(100 * w0, 1),
                     "cqr_cov80_pct": round(100 * cov1, 1), "cqr_width_pct": round(100 * w1, 1),
                     "pinball": round(float(pb), 5)})
    m = pd.DataFrame(rows)
    m.to_csv(HERE / "results" / "interval_prob_metrics.csv", index=False, encoding="utf-8-sig")
    (HERE / "results" / "cqr_adjust.json").write_text(json.dumps(store, ensure_ascii=False, indent=2))
    print(m.to_string(index=False))


if __name__ == "__main__":
    main()
