#!/usr/bin/env python3
"""
正式訓練（可續跑）——以消融勝出的設定，在完整資料上訓練最終模型。

消融結論
--------
  * 最佳預測方向：湍流強度 TI = σ/U（陣風因子 gust/U、p99/U 幾乎同等好）
  * 特徵：風向最關鍵；veer_97_35、AT_95、RH_95 反而有害 → 列入待剪清單

流程
----
  階段 1  特徵集選擇：在 TI 上比較 full17 / pruned14 / core8，用驗證集選最佳
  階段 2  正式訓練：用勝出特徵集，對 TI / gustfac / p99n 各訓練一個最終模型，
          存模型、預測、指標、特徵重要性
  階段 3  產圖：預測vs實測、特徵重要性、殘差對風向、分季節誤差（以 TI 為主）

以 cache.json 記錄進度，可分多次在 45 秒批次限制下跑完。

用法
----
    python final_train.py            # 重複執行直到印出「全部完成」
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ablation"))
from prepare_data import FEATURES, GROUPS   # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "ablation" / "model_data.parquet"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
CACHE = HERE / "cache.json"

SPLIT = {"train": (2016, 2018), "val": (2019, 2019), "test": (2020, 2021)}

FEATURE_SETS = {
    "full17": list(FEATURES),
    "pruned14": [f for f in FEATURES if f not in
                 ("veer_97_35", "AT_95_mean", "RH_95_mean")],
    "core8": ["WS_100_mean", "WS_69W_mean", "WS_38W_mean", "shear_alpha",
              "WD_97_sin", "WD_97_cos", "WD_35_sin", "WD_35_cos"],
}

FINAL_TARGETS = {
    "tgt_ti":      "湍流強度 TI = σ/U",
    "tgt_gustfac": "3 秒陣風因子 gust/U",
    "tgt_p99n":    "p99 / U",
}

# 正式訓練用較保守、較充分的參數
PARAMS = dict(
    objective="regression", metric="rmse", learning_rate=0.03,
    num_leaves=63, min_data_in_leaf=60, feature_fraction=0.85,
    bagging_fraction=0.85, bagging_freq=1, lambda_l2=1.0, verbose=-1,
    num_threads=2, seed=42, bagging_seed=42, feature_fraction_seed=42,
    data_random_seed=42,
)
NUM_ROUNDS = 4000
EARLY_STOP = 150


def r2(y, p):
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    ss = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - p) ** 2).sum() / ss)


def rmse(y, p):
    m = np.isfinite(y) & np.isfinite(p)
    e = y[m] - p[m]
    return float(np.sqrt((e ** 2).mean()))


def mae(y, p):
    m = np.isfinite(y) & np.isfinite(p)
    return float(np.abs(y[m] - p[m]).mean())


def load_cache() -> dict:
    if CACHE.exists():
        return json.loads(CACHE.read_text(encoding="utf-8"))
    return {"fs_select": {}, "final": {}, "figs_done": False}


def save_cache(c: dict) -> None:
    CACHE.write_text(json.dumps(c, ensure_ascii=False, indent=2), encoding="utf-8")


def splits(df):
    def sel(k):
        a, b = SPLIT[k]
        return df[(df.year >= a) & (df.year <= b)]
    return sel("train"), sel("val"), sel("test")


def train_eval(df, target, feats, want_model=False):
    import lightgbm as lgb
    tr, va, te = splits(df)
    ytr, yva, yte = (tr[target].to_numpy(), va[target].to_numpy(), te[target].to_numpy())
    mtr, mva = np.isfinite(ytr), np.isfinite(yva)
    ds = lgb.Dataset(tr.loc[mtr, feats], ytr[mtr])
    dv = lgb.Dataset(va.loc[mva, feats], yva[mva], reference=ds)
    model = lgb.train(PARAMS, ds, NUM_ROUNDS, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
    pv = model.predict(va[feats], num_iteration=model.best_iteration)
    pt = model.predict(te[feats], num_iteration=model.best_iteration)
    out = {"val_r2": r2(yva, pv), "test_r2": r2(yte, pt),
           "test_rmse": rmse(yte, pt), "test_mae": mae(yte, pt),
           "best_iter": int(model.best_iteration), "n_features": len(feats)}
    if want_model:
        return out, model, (te, yte, pt)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=38.0)
    args = ap.parse_args()

    df = pd.read_parquet(DATA)
    c = load_cache()
    t0 = time.time()

    # ---------- 階段 1：特徵集選擇（在 TI 上）----------
    for name, feats in FEATURE_SETS.items():
        if name in c["fs_select"]:
            continue
        if time.time() - t0 > args.budget:
            print("時間用完，請再執行一次（階段 1 進行中）"); save_cache(c); return 0
        res = train_eval(df, "tgt_ti", feats)
        c["fs_select"][name] = res
        save_cache(c)
        print(f"  [特徵集] {name:9s} val R²={res['val_r2']:.4f}  test R²={res['test_r2']:.4f}")

    best_fs = max(c["fs_select"], key=lambda k: c["fs_select"][k]["val_r2"])
    c["best_fs"] = best_fs
    save_cache(c)

    # ---------- 階段 2：正式訓練三個目標 ----------
    feats = FEATURE_SETS[best_fs]
    for tgt in FINAL_TARGETS:
        if tgt in c["final"]:
            continue
        if time.time() - t0 > args.budget:
            print("時間用完，請再執行一次（階段 2 進行中）"); save_cache(c); return 0
        res, model, (te, yte, pt) = train_eval(df, tgt, feats, want_model=True)
        model.save_model(str(HERE / f"model_{tgt}.txt"))
        imp = pd.DataFrame({"feature": feats,
                            "gain": model.feature_importance("gain")}).sort_values(
            "gain", ascending=False)
        imp["gain_pct"] = (100 * imp.gain / imp.gain.sum()).round(2)
        imp.to_csv(HERE / f"importance_{tgt}.csv", index=False, encoding="utf-8-sig")
        if tgt == "tgt_ti":                       # 存 TI 的測試集預測供畫圖
            pred_df = te[["ts", "year", "WD_97_sin", "WD_97_cos"]].copy()
            # 由 sin/cos 還原風向角度（度）供殘差對風向圖使用
            pred_df["WD_97"] = (np.degrees(np.arctan2(
                te["WD_97_sin"], te["WD_97_cos"])) + 360) % 360
            pred_df["y_true"] = yte
            pred_df["y_pred"] = pt
            pred_df.to_parquet(HERE / "pred_ti_test.parquet", index=False)
        c["final"][tgt] = {**res, "feature_set": best_fs}
        save_cache(c)
        print(f"  [正式] {FINAL_TARGETS[tgt]:16s} test R²={res['test_r2']:.4f}  "
              f"RMSE={res['test_rmse']:.4f}  ({best_fs}, {res['best_iter']} 輪)")

    # ---------- 階段 3：產圖 ----------
    if not c.get("figs_done"):
        if time.time() - t0 > args.budget:
            print("時間用完，請再執行一次（進入產圖階段）"); save_cache(c); return 0
        make_figures(df, feats, c)
        c["figs_done"] = True
        save_cache(c)

    # 完成 → 寫最終指標彙整
    summary = pd.DataFrame([{"target": FINAL_TARGETS[k], **v} for k, v in c["final"].items()])
    summary.to_csv(HERE / "final_metrics.csv", index=False, encoding="utf-8-sig")
    print("\n=== 全部完成 ===")
    print(f"勝出特徵集：{best_fs}（{len(feats)} 個特徵）")
    print(summary[["target", "test_r2", "test_rmse", "test_mae", "best_iter"]].to_string(index=False))
    return 0


def make_figures(df, feats, c):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.sans-serif": ["Noto Sans CJK JP", "Droid Sans Fallback", "DejaVu Sans"],
        "axes.unicode_minus": False, "figure.dpi": 150, "savefig.dpi": 150,
        "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 11})
    B = {"blue": "#2E5E8C", "red": "#C1584B", "warm": "#D69A3C", "grey": "#8A8A8A"}

    pred = pd.read_parquet(HERE / "pred_ti_test.parquet")
    y, p = pred.y_true.to_numpy(), pred.y_pred.to_numpy()

    # 圖 1：預測 vs 實測
    fig, ax = plt.subplots(figsize=(6.4, 6))
    m = (y < 0.3) & (p < 0.3)
    ax.hexbin(y[m], p[m], gridsize=70, cmap="Blues", bins="log", mincnt=1)
    ax.plot([0, 0.3], [0, 0.3], "--", color=B["red"], lw=1.5, label="1:1")
    ax.set_xlim(0, 0.3); ax.set_ylim(0, 0.3)
    ax.set_xlabel("實測 TI"); ax.set_ylabel("預測 TI")
    ax.set_title(f"正式模型　預測 vs 實測（測試集 2020–2021）\n"
                 f"R² = {c['final']['tgt_ti']['test_r2']:.3f}　"
                 f"RMSE = {c['final']['tgt_ti']['test_rmse']:.4f}", fontsize=12)
    ax.legend(loc="upper left")
    fig.savefig(FIG / "final1_pred_vs_obs.png"); plt.close(fig)

    # 圖 2：特徵重要性
    imp = pd.read_csv(HERE / "importance_tgt_ti.csv").sort_values("gain_pct").tail(14)
    cols = [B["red"] if f.startswith("WD_") or f == "veer_97_35" else B["blue"]
            for f in imp.feature]
    fig, ax = plt.subplots(figsize=(8, 5.6))
    ax.barh(imp.feature, imp.gain_pct, color=cols)
    for i, v in enumerate(imp.gain_pct):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("特徵重要性（gain, %）")
    ax.set_title("正式模型　特徵重要性（目標＝TI）\n紅＝風向", fontsize=12)
    fig.savefig(FIG / "final2_importance.png"); plt.close(fig)

    # 圖 3：殘差對風向（需要風向欄）
    if "WD_97" in pred:
        wd = pred.WD_97.to_numpy()
        res = y - p
        b = pd.cut(wd, np.arange(0, 361, 15))
        g = pd.Series(res).groupby(b, observed=True).median()
        x = np.arange(0, 360, 15) + 7.5
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.axhline(0, color="k", lw=1)
        ax.plot(x[: len(g)], g.values, "o-", color=B["blue"], lw=2, ms=5)
        ax.set_xlabel("100 m 風向（度）"); ax.set_ylabel("殘差中位數（實測−預測）")
        ax.set_xlim(0, 360); ax.set_xticks(np.arange(0, 361, 45))
        ax.set_title("正式模型　殘差對風向：接近平坦代表風向資訊已被學到", fontsize=12)
        fig.savefig(FIG / "final3_residual_by_dir.png"); plt.close(fig)

    # 圖 4：分季節誤差
    pred2 = pred.copy()
    pred2["month"] = pd.to_datetime(pred2.ts).dt.month
    pred2["season"] = pred2.month.map(lambda mo: "冬" if mo in (12, 1, 2) else
                                      "春" if mo in (3, 4, 5) else
                                      "夏" if mo in (6, 7, 8) else "秋")
    pred2["ae"] = np.abs(pred2.y_true - pred2.y_pred)
    g = pred2.groupby("season").ae.mean().reindex(["冬", "春", "夏", "秋"])
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.bar(g.index, g.values, color=B["blue"])
    for i, v in enumerate(g.values):
        ax.text(i, v + 0.0004, f"{v:.4f}", ha="center", fontsize=10)
    ax.set_ylabel("平均絕對誤差 MAE")
    ax.set_title("正式模型　分季節誤差（目標＝TI）", fontsize=12)
    fig.savefig(FIG / "final4_error_by_season.png"); plt.close(fig)
    print("  ✓ 已產出 4 張正式模型圖")


if __name__ == "__main__":
    raise SystemExit(main())
