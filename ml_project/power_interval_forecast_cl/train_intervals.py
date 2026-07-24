#!/usr/bin/env python3
"""
發電區間（機率）預測 —— 訓練多分位數 LightGBM。（可續跑）

目標：不只給一個發電量數字，而是給「區間」——例如「3 小時後出力有 80%
機率落在 35%–72% 額定之間」。這對電網備轉容量規劃才是真正有用的資訊。

方法總覽
--------
  1. 分位數迴歸：對每個時程(1/3/6h)訓練 7 條分位數 (0.05–0.95)
  2. 保形校正 CQR（在 evaluate.py）：用獨立校正集，讓區間有「保證涵蓋率」
  3. 機率評分（在 evaluate.py）：涵蓋率、區間寬度、pinball、CRPS、Winkler

三段時間切分（CQR 需要獨立校正集）：
  訓練 2016–2018 ｜ 校正 2019 ｜ 測試 2020–2021

本程式只負責訓練分位數模型並存下「校正集」與「測試集」的預測；
CQR 與評估在 evaluate.py。以 cache 記錄進度，可分多次跑完。
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PF = HERE.parent / "power_forecast"
sys.path.insert(0, str(PF))
from forecast_train import build_frame, FEATURES   # 重用特徵工程與虛擬出力

MODELS = HERE / "models"; MODELS.mkdir(exist_ok=True)
RES = HERE / "results"; RES.mkdir(exist_ok=True)
CACHE = HERE / "cache.json"

HORIZONS = {"1h": 6, "3h": 18, "6h": 36}
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

LGB = dict(objective="quantile", learning_rate=0.05, num_leaves=127,
           min_data_in_leaf=40, feature_fraction=0.85, bagging_fraction=0.85,
           bagging_freq=1, verbose=-1, num_threads=2, seed=42, bagging_seed=42,
           feature_fraction_seed=42, data_random_seed=42)
ROUNDS, EARLY = 2500, 150


def load_cache():
    if CACHE.exists() and CACHE.stat().st_size:
        return json.loads(CACHE.read_text())
    return {"done": []}


def save_cache(c): CACHE.write_text(json.dumps(c, ensure_ascii=False, indent=2))


def qcol(q): return f"q{int(round(q*100)):02d}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--budget", type=float, default=38.0)
    args = ap.parse_args()
    import lightgbm as lgb

    X, grid = build_frame()
    Pfull = X.P
    tr = X[X.year <= 2018]; cal = X[X.year == 2019]; te = X[X.year >= 2020]

    def future(df, steps): return Pfull.shift(-steps).reindex(df.index)

    c = load_cache(); t0 = time.time()

    for hz, steps in HORIZONS.items():
        ytr = future(tr, steps); ycal = future(cal, steps); yte = future(te, steps)
        mtr = tr[FEATURES].notna().all(axis=1) & ytr.notna()
        mcal = cal[FEATURES].notna().all(axis=1) & ycal.notna()
        mte = te[FEATURES].notna().all(axis=1) & yte.notna()

        for q in QUANTILES:
            key = f"{hz}:{q}"
            if key in c["done"]:
                continue
            if time.time() - t0 > args.budget:
                print("時間用完，請再執行一次"); save_cache(c); return
            p = dict(LGB); p["alpha"] = q
            ds = lgb.Dataset(tr.loc[mtr, FEATURES], ytr[mtr])
            dv = lgb.Dataset(cal.loc[mcal, FEATURES], ycal[mcal], reference=ds)
            mdl = lgb.train(p, ds, ROUNDS, valid_sets=[dv],
                            callbacks=[lgb.early_stopping(EARLY, verbose=False)])
            mdl.save_model(str(MODELS / f"q_{hz}_{qcol(q)}.txt"))
            pc = np.clip(mdl.predict(cal.loc[mcal, FEATURES], num_iteration=mdl.best_iteration), 0, 1)
            pt = np.clip(mdl.predict(te.loc[mte, FEATURES], num_iteration=mdl.best_iteration), 0, 1)

            # 寫入/更新校正集與測試集預測表：檔案存在就加欄，否則新建
            # （同一時程各分位數的 ts/y 完全相同，故安全）
            for tag, mask, sub, yv, pv in [("cal", mcal, cal, ycal, pc), ("test", mte, te, yte, pt)]:
                f = RES / f"pred_{tag}_{hz}.parquet"
                base = pd.read_parquet(f) if f.exists() else \
                       pd.DataFrame({"ts": sub.index[mask], "y": yv[mask].to_numpy()})
                base[qcol(q)] = pv
                base.to_parquet(f, index=False)

            c["done"].append(key); save_cache(c)
            print(f"  ✓ {hz} p{int(q*100):02d}  ({mdl.best_iteration}輪)")

    total = len(HORIZONS) * len(QUANTILES)
    if len(c["done"]) >= total:
        print(f"\n=== 全部完成（{total} 個分位數模型）===")
    else:
        print(f"進度 {len(c['done'])}/{total}")


if __name__ == "__main__":
    raise SystemExit(main())
