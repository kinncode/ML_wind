#!/usr/bin/env python3
"""
超短期（0–6h）虛擬風場發電預測 —— 只用 BSMI 塔資料，不需外部 NWP。（可續跑）

為什麼只做 0–6h：塔的觀測外推在約 6 小時後就逼近「猜長期平均」而失效
（persistence 24h nRMSE ≈ 40%，見報告）。日前 48h 必須引入數值天氣預報。

做法
----
  目標：正規化虛擬出力 P（0–1）在 t+h 的值，h ∈ {30min,1h,2h,3h,6h}
  特徵：只用時間 t（含）以前的資訊 —— 出力/風速的近時 lag、風速趨勢、
        風向、亂流強度、空氣密度、時間週期
  基準：persistence（P 現值）、氣候平均（該月該時歷史均值）
  模型：LightGBM 點預測（每時程一個）＋ 3h 的分位數 p10/p50/p90
  切分：訓練 2016–2018、驗證 2019、測試 2020–2021
  評估：nRMSE / nMAE（正規化 by 額定）、技術得分 vs persistence 與氣候

以 cache.json 記錄進度，可分多次在 45 秒批次限制下跑完。
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd

from virtual_power import load_power_table

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "BSMI_10min.parquet"
MODELS = HERE / "models"; MODELS.mkdir(exist_ok=True)
RES = HERE / "results"; RES.mkdir(exist_ok=True)
CACHE = HERE / "fc_cache.json"

HORIZONS = {"30min": 3, "1h": 6, "2h": 12, "3h": 18, "6h": 36}  # 以 10 分鐘為步
QUANTILE_H = "3h"                                                # 分位數示範時程
QS = [0.1, 0.5, 0.9]

LGB = dict(objective="regression", metric="rmse", learning_rate=0.05, num_leaves=63,
           min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=2, seed=42,
           bagging_seed=42, feature_fraction_seed=42, data_random_seed=42)
ROUNDS, EARLY = 1200, 60

# 分位數需要更大容量：出力在額定處有大量點質量（~24% 為 1.0），
# 高分位（p90）容易underfit塌成常數，故加大葉數、降正則、拉長迭代。
QUANT_LGB = dict(objective="quantile", learning_rate=0.05, num_leaves=127,
                 min_data_in_leaf=40, feature_fraction=0.85, bagging_fraction=0.85,
                 bagging_freq=1, verbose=-1, num_threads=2, seed=42, bagging_seed=42,
                 feature_fraction_seed=42, data_random_seed=42)
QUANT_ROUNDS, QUANT_EARLY = 3000, 200


def build_frame():
    d = load_power_table(str(DATA))
    d["ti"] = (d["WS_100E_std"] / d["WS_100E_mean"]).clip(0, 1)
    s = d.set_index("ts").sort_index()
    grid = pd.date_range(s.index.min(), s.index.max(), freq="10min")
    s = s.reindex(grid)
    X = pd.DataFrame(index=grid)
    for L in [0, 1, 3, 6, 18, 36]:
        X[f"P_lag{L}"] = s.P.shift(L)
        X[f"WS_lag{L}"] = s.WS_100_mean.shift(L)
    X["WS_tr_1h"] = s.WS_100_mean - s.WS_100_mean.shift(6)
    X["WS_tr_3h"] = s.WS_100_mean - s.WS_100_mean.shift(18)
    X["P_tr_1h"] = s.P - s.P.shift(6)
    X["WD_sin"] = s.WD_97_sin; X["WD_cos"] = s.WD_97_cos
    X["ti"] = s.ti; X["rho"] = s.air_density
    h = grid.hour + grid.minute / 60; doy = grid.dayofyear
    X["hs"] = np.sin(2 * np.pi * h / 24); X["hc"] = np.cos(2 * np.pi * h / 24)
    X["ds"] = np.sin(2 * np.pi * doy / 365.25); X["dc"] = np.cos(2 * np.pi * doy / 365.25)
    X["P_now"] = s.P                       # persistence 基準用
    X["year"] = grid.year; X["month"] = grid.month; X["hour"] = grid.hour
    X["P"] = s.P
    return X, grid


FEATURES = [f"{p}_lag{L}" for L in [0, 1, 3, 6, 18, 36] for p in ("P", "WS")] + \
           ["WS_tr_1h", "WS_tr_3h", "P_tr_1h", "WD_sin", "WD_cos", "ti", "rho",
            "hs", "hc", "ds", "dc"]


def metrics(y, p):
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    e = y - p
    return {"nrmse": float(np.sqrt((e**2).mean())), "nmae": float(np.abs(e).mean()),
            "n": int(len(y))}


def load_cache():
    return json.loads(CACHE.read_text()) if CACHE.exists() and CACHE.stat().st_size else {"point": {}, "quant": {}, "clim_done": False}


def save_cache(c): CACHE.write_text(json.dumps(c, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--budget", type=float, default=38.0)
    args = ap.parse_args()
    import lightgbm as lgb

    X, grid = build_frame()
    tr = X[X.year <= 2018]; va = X[X.year == 2019]; te = X[X.year >= 2020]
    c = load_cache(); t0 = time.time()

    # 氣候基準（月×時 平均出力），只算一次
    if not c["clim_done"]:
        clim = tr.groupby(["month", "hour"]).P.mean()
        clim.to_frame("P_clim").to_csv(RES / "climatology.csv", encoding="utf-8-sig")
        c["clim_done"] = True; save_cache(c)
    clim = pd.read_csv(RES / "climatology.csv").set_index(["month", "hour"]).P_clim

    def add_targets(df, steps):
        y = df.P.shift(-steps).reindex(df.index)
        return y

    # 為了對齊「未來目標」，用整體 X 建立 shift 後對齊到各子集
    Pfull = X.P
    def future(df, steps):
        yf = Pfull.shift(-steps)
        return yf.reindex(df.index)

    # ---- 點預測 ----
    for hz, steps in HORIZONS.items():
        if hz in c["point"]:
            continue
        if time.time() - t0 > args.budget:
            print("時間用完，請再執行一次"); save_cache(c); return
        ytr = future(tr, steps); yva = future(va, steps); yte = future(te, steps)
        mtr = tr[FEATURES].notna().all(axis=1) & ytr.notna()
        mva = va[FEATURES].notna().all(axis=1) & yva.notna()
        mte = te[FEATURES].notna().all(axis=1) & yte.notna()
        ds = lgb.Dataset(tr.loc[mtr, FEATURES], ytr[mtr])
        dv = lgb.Dataset(va.loc[mva, FEATURES], yva[mva], reference=ds)
        mdl = lgb.train(LGB, ds, ROUNDS, valid_sets=[dv],
                        callbacks=[lgb.early_stopping(EARLY, verbose=False)])
        mdl.save_model(str(MODELS / f"point_{hz}.txt"))
        pred = mdl.predict(te.loc[mte, FEATURES], num_iteration=mdl.best_iteration)
        ml = metrics(yte[mte].to_numpy(), pred)
        per = metrics(yte[mte].to_numpy(), te.loc[mte, "P_now"].to_numpy())
        cl = metrics(yte[mte].to_numpy(),
                     te.loc[mte].set_index(["month", "hour"]).index.map(clim).to_numpy())
        c["point"][hz] = {
            "steps": steps, "best_iter": int(mdl.best_iteration),
            "ml_nrmse": ml["nrmse"], "ml_nmae": ml["nmae"],
            "persist_nrmse": per["nrmse"], "clim_nrmse": cl["nrmse"],
            "skill_vs_persist": round(1 - ml["nrmse"] / per["nrmse"], 4),
            "skill_vs_clim": round(1 - ml["nrmse"] / cl["nrmse"], 4), "n_test": ml["n"]}
        save_cache(c)
        # 存 3h 的測試預測供畫圖
        if hz == QUANTILE_H:
            out = te.loc[mte, ["month", "hour"]].copy()
            out["ts"] = te.index[mte]
            out["y"] = yte[mte].to_numpy(); out["pred"] = pred
            out["persist"] = te.loc[mte, "P_now"].to_numpy()
            out.to_parquet(RES / "pred_3h_test.parquet", index=False)
        print(f"  [點] {hz:5s} ML nRMSE={ml['nrmse']*100:.2f}%  "
              f"vs persist {per['nrmse']*100:.2f}%  技術得分 {100*(1-ml['nrmse']/per['nrmse']):.1f}%")

    # ---- 分位數（3h）----
    steps = HORIZONS[QUANTILE_H]
    ytr = future(tr, steps); yva = future(va, steps); yte = future(te, steps)
    mtr = tr[FEATURES].notna().all(axis=1) & ytr.notna()
    mva = va[FEATURES].notna().all(axis=1) & yva.notna()
    mte = te[FEATURES].notna().all(axis=1) & yte.notna()
    for q in QS:
        key = f"{q}"
        if key in c["quant"]:
            continue
        if time.time() - t0 > args.budget:
            print("時間用完，請再執行一次"); save_cache(c); return
        p = dict(QUANT_LGB); p["alpha"] = q
        ds = lgb.Dataset(tr.loc[mtr, FEATURES], ytr[mtr])
        dv = lgb.Dataset(va.loc[mva, FEATURES], yva[mva], reference=ds)
        mdl = lgb.train(p, ds, QUANT_ROUNDS, valid_sets=[dv],
                        callbacks=[lgb.early_stopping(QUANT_EARLY, verbose=False)])
        mdl.save_model(str(MODELS / f"quant_{q}.txt"))
        pred = np.clip(mdl.predict(te.loc[mte, FEATURES], num_iteration=mdl.best_iteration), 0, 1)
        cover = float((yte[mte].to_numpy() <= pred).mean())    # 實際落在此分位下的比例
        c["quant"][key] = {"alpha": q, "best_iter": int(mdl.best_iteration),
                           "empirical_coverage": round(cover, 4)}
        save_cache(c)
        # 存分位預測（第一個分位時重建檔案，避免殘留舊欄）
        qf = RES / "pred_3h_quantiles.parquet"
        base = pd.read_parquet(qf) if (qf.exists() and q != QS[0]) else \
               pd.DataFrame({"ts": te.index[mte], "y": yte[mte].to_numpy()})
        base[f"q{int(q*100)}"] = pred
        base.to_parquet(qf, index=False)
        print(f"  [分位] p{int(q*100)} 實際涵蓋率={cover:.3f}（理想={q}, {mdl.best_iteration}輪）")

    # ---- 分季節評估（用 3h 模型）----
    if "pred_3h_test.parquet" in [p.name for p in RES.iterdir()] and not c.get("season_done"):
        pr = pd.read_parquet(RES / "pred_3h_test.parquet")
        pr["season"] = pr.month.map(lambda m: "冬" if m in (12, 1, 2) else "春" if m in (3, 4, 5)
                                    else "夏" if m in (6, 7, 8) else "秋")
        rows = []
        for s, g in pr.groupby("season"):
            e = g.y - g.pred; ep = g.y - g.persist
            rows.append({"season": s, "n": len(g),
                         "ml_nrmse_pct": round(100 * np.sqrt((e**2).mean()), 2),
                         "persist_nrmse_pct": round(100 * np.sqrt((ep**2).mean()), 2)})
        pd.DataFrame(rows).to_csv(RES / "forecast_by_season.csv", index=False, encoding="utf-8-sig")
        c["season_done"] = True; save_cache(c)

    # ---- 完成：彙整指標 ----
    if len(c["point"]) == len(HORIZONS) and len(c["quant"]) == len(QS):
        rows = [{"horizon": h, **{k: v[k] for k in
                ("ml_nrmse", "persist_nrmse", "clim_nrmse", "skill_vs_persist", "skill_vs_clim", "best_iter")}}
                for h, v in c["point"].items()]
        df = pd.DataFrame(rows)
        for col in ("ml_nrmse", "persist_nrmse", "clim_nrmse"):
            df[col] = (df[col] * 100).round(2)
        df.to_csv(RES / "forecast_metrics.csv", index=False, encoding="utf-8-sig")
        print("\n=== 全部完成 ===")
        print(df.to_string(index=False))
        print("分位數涵蓋率:", {k: v["empirical_coverage"] for k, v in c["quant"].items()})


if __name__ == "__main__":
    raise SystemExit(main())
