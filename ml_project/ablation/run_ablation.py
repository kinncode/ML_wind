#!/usr/bin/env python3
"""
消融實驗主程式（可續跑）。

回答兩個問題：
  1. 哪個預測目標最值得做？（跨目標比較）
  2. 每個特徵／特徵群到底貢獻多少？（逐一消融）

實驗清單
--------
A. 跨目標比較：8 個目標 × {B0 常數, B2 線性, B3 只用風速, B4 完整}
B. 特徵群消融（主目標 tgt_ti）：
     - 只用風速
     - 風速 + 各群（add-one-in）
     - 完整
     - 完整 − 各群（leave-one-group-out）
C. 逐特徵消融（主目標 tgt_ti）：完整 − 每個單一特徵（leave-one-feature-out）

設計成「可續跑」：每個實驗算完就把一列寫進 results/ablation_raw.csv，
再次執行會跳過已完成的，直到全部跑完。這樣就能在 45 秒的批次限制下分多次跑完。

用法
----
    python run_ablation.py            # 跑到時間預算用完就停，重複執行直到印出「全部完成」
    python run_ablation.py --budget 38
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_data import FEATURES, GROUPS

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
RES.mkdir(exist_ok=True)
RAW = RES / "ablation_raw.csv"

# 時間切分：訓練 2016–2018 / 驗證 2019 / 測試 2020–2021
SPLIT = {"train": (2016, 2018), "val": (2019, 2019), "test": (2020, 2021)}

TARGETS = {
    "tgt_ti":        "湍流強度 TI = σ/U",
    "tgt_gustfac":   "3 秒陣風因子 gust/U",
    "tgt_p99n":      "p99 / U",
    "tgt_p01n":      "p01 / U",
    "tgt_specslope": "頻譜斜率",
    "tgt_intscale":  "積分時間尺度 T_u",
    "tgt_gust_raw":  "3 秒陣風（未正規化）",
    "tgt_p99_raw":   "p99（未正規化）",
}
PRIMARY = "tgt_ti"

# 消融用的快速參數（重點是相對比較，不是壓榨絕對分數）
LGB_PARAMS = dict(
    objective="regression", metric="rmse", learning_rate=0.1,
    num_leaves=31, min_data_in_leaf=100, feature_fraction=0.9,
    bagging_fraction=0.9, bagging_freq=1, verbose=-1, num_threads=2,
    seed=42, bagging_seed=42, feature_fraction_seed=42, data_random_seed=42,
)
NUM_ROUNDS = 800
EARLY_STOP = 40


def r2(y, p):
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    ss = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - p) ** 2).sum() / ss) if ss > 0 else np.nan


def rmse(y, p):
    m = np.isfinite(y) & np.isfinite(p)
    e = y[m] - p[m]
    return float(np.sqrt((e ** 2).mean()))


def build_experiments() -> list[dict]:
    """列出所有實驗，每個給一個唯一 id。"""
    exps = []

    # A. 跨目標比較
    for tgt in TARGETS:
        for kind in ["B0_const", "B2_linear", "B3_speed", "B4_full"]:
            exps.append({"id": f"A::{tgt}::{kind}", "block": "target_compare",
                         "target": tgt, "kind": kind, "features": None})

    # B. 特徵群消融（主目標）
    speed = GROUPS["風速剖面"]
    exps.append({"id": f"B::speed_only", "block": "group", "target": PRIMARY,
                 "kind": "lgbm", "features": list(speed), "label": "只用風速剖面"})
    for gname, gfeats in GROUPS.items():
        if gname == "風速剖面":
            continue
        feats = list(dict.fromkeys(speed + gfeats))
        exps.append({"id": f"B::add::{gname}", "block": "group", "target": PRIMARY,
                     "kind": "lgbm", "features": feats, "label": f"風速 + {gname}"})
    exps.append({"id": "B::full", "block": "group", "target": PRIMARY,
                 "kind": "lgbm", "features": list(FEATURES), "label": "完整（全部特徵）"})
    for gname, gfeats in GROUPS.items():
        feats = [f for f in FEATURES if f not in gfeats]
        exps.append({"id": f"B::loo_group::{gname}", "block": "group", "target": PRIMARY,
                     "kind": "lgbm", "features": feats, "label": f"完整 − {gname}"})

    # C. 逐特徵消融（主目標）：完整 − 每個單一特徵
    for f in FEATURES:
        feats = [x for x in FEATURES if x != f]
        exps.append({"id": f"C::loo_feat::{f}", "block": "feature_loo", "target": PRIMARY,
                     "kind": "lgbm", "features": feats, "label": f"完整 − {f}",
                     "dropped": f})
    return exps


def load_done() -> set:
    if RAW.exists() and RAW.stat().st_size > 0:
        try:
            return set(pd.read_csv(RAW)["id"])
        except Exception:                       # 空檔或損毀 → 當作從頭開始
            return set()
    return set()


# 固定欄位順序，確保每一列的欄位完全一致（否則續寫的 CSV 會欄數對不上）
COLUMNS = ["id", "block", "target", "target_label", "label", "dropped",
           "n_features", "best_iter", "n_test", "test_mean", "test_std", "r2", "rmse"]


def append_row(row: dict) -> None:
    df = pd.DataFrame([row])[COLUMNS]
    need_header = (not RAW.exists()) or RAW.stat().st_size == 0
    df.to_csv(RAW, mode="a", header=need_header, index=False, encoding="utf-8-sig")


def run_one(exp: dict, data: dict) -> dict:
    import lightgbm as lgb
    tr, va, te = data["tr"], data["va"], data["te"]
    tgt = exp["target"]
    y_tr = tr[tgt].to_numpy(); y_te = te[tgt].to_numpy()

    base = {"id": exp["id"], "block": exp["block"], "target": tgt,
            "target_label": TARGETS[tgt], "label": exp.get("label", exp["kind"]),
            "dropped": exp.get("dropped", ""), "n_features": 0, "best_iter": -1,
            "n_test": int(np.isfinite(y_te).sum()),
            "test_mean": float(np.nanmean(y_te)), "test_std": float(np.nanstd(y_te)),
            "r2": np.nan, "rmse": np.nan}

    kind = exp["kind"]
    if kind == "B0_const":
        p = np.full(len(y_te), np.nanmean(y_tr))
    elif kind == "B2_linear":
        x = tr["WS_100_mean"].to_numpy(); A = np.column_stack([x, np.ones_like(x)])
        m = np.isfinite(y_tr)
        coef, *_ = np.linalg.lstsq(A[m], y_tr[m], rcond=None)
        p = coef[0] * te["WS_100_mean"].to_numpy() + coef[1]
    else:
        feats = ["WS_100_mean"] if kind == "B3_speed" else \
                (exp["features"] if exp["features"] else list(FEATURES))
        base["n_features"] = len(feats)
        mtr = np.isfinite(y_tr)
        ds = lgb.Dataset(tr.loc[mtr, feats], y_tr[mtr])
        mva = np.isfinite(va[tgt].to_numpy())
        dv = lgb.Dataset(va.loc[mva, feats], va[tgt].to_numpy()[mva], reference=ds)
        model = lgb.train(LGB_PARAMS, ds, NUM_ROUNDS, valid_sets=[dv],
                          callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
        p = model.predict(te[feats], num_iteration=model.best_iteration)
        base["best_iter"] = int(model.best_iteration)

    base["r2"] = r2(y_te, p)
    base["rmse"] = rmse(y_te, p)
    return base


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=38.0, help="每次執行的秒數預算")
    args = ap.parse_args()

    df = pd.read_parquet(HERE / "model_data.parquet")
    data = {
        "tr": df[(df.year >= SPLIT["train"][0]) & (df.year <= SPLIT["train"][1])],
        "va": df[(df.year >= SPLIT["val"][0]) & (df.year <= SPLIT["val"][1])],
        "te": df[(df.year >= SPLIT["test"][0]) & (df.year <= SPLIT["test"][1])],
    }

    exps = build_experiments()
    done = load_done()
    pending = [e for e in exps if e["id"] not in done]
    print(f"實驗總數 {len(exps)}，已完成 {len(done)}，待跑 {len(pending)}")

    t0 = time.time()
    n = 0
    for exp in pending:
        if time.time() - t0 > args.budget:
            break
        row = run_one(exp, data)
        append_row(row)
        n += 1
        tag = f"{row['r2']:.4f}" if np.isfinite(row["r2"]) else "  -  "
        print(f"  ✓ {exp['id']:38s} R²={tag}")

    remain = len(pending) - n
    if remain == 0:
        print(f"\n=== 全部完成（{len(exps)} 個實驗）===")
    else:
        print(f"\n本次跑了 {n} 個，還剩 {remain} 個，請再執行一次。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
