#!/usr/bin/env python3
"""超短期發電預測圖表。"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE / "results"; FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.sans-serif": ["Microsoft JhengHei", "SimHei", "Noto Sans CJK JP", "Droid Sans Fallback", "DejaVu Sans"],
    "axes.unicode_minus": False, "figure.dpi": 150, "savefig.dpi": 150,
    "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": .25,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11})
B = {"blue": "#2E5E8C", "red": "#C1584B", "green": "#4E9A6B",
     "warm": "#D69A3C", "grey": "#8A8A8A", "purple": "#7B5AA6"}


def fig_skill():
    m = pd.read_csv(RES / "forecast_metrics.csv")
    order = ["30min", "1h", "2h", "3h", "6h"]
    m["o"] = m.horizon.map({h: i for i, h in enumerate(order)})
    m = m.sort_values("o")
    x = np.arange(len(m))
    fig, ax = plt.subplots(figsize=(8.4, 5))
    ax.axhline(m.clim_nrmse.mean(), color=B["grey"], ls=":", lw=1.2,
               label=f"氣候平均基準 ≈ {m.clim_nrmse.mean():.0f}%")
    ax.plot(x, m.persist_nrmse, "o-", color=B["red"], lw=2, label="Persistence（只用現在值）")
    ax.plot(x, m.ml_nrmse, "s-", color=B["blue"], lw=2, label="LightGBM（塔的近時特徵）")
    for xi, (a, b_) in enumerate(zip(m.persist_nrmse, m.ml_nrmse)):
        ax.annotate(f"−{100*(1-b_/a):.0f}%", (xi, b_), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8.5, color=B["blue"])
    ax.set_xticks(x); ax.set_xticklabels(m.horizon)
    ax.set_xlabel("預測提前量"); ax.set_ylabel("發電量預測 nRMSE（% 額定）")
    ax.set_title("圖 1　超短期發電預測技術得分\n"
                 "ML 相對 persistence 的改善隨時程增大（30分 8% → 6h 24%）", fontsize=12)
    ax.legend(); fig.savefig(FIG / "fc1_skill_curve.png"); plt.close(fig)


def fig_pred_obs():
    d = pd.read_parquet(RES / "pred_3h_test.parquet")
    y, p = d.y.to_numpy(), d.pred.to_numpy()
    r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    fig, ax = plt.subplots(figsize=(6.2, 6))
    ax.hexbin(y, p, gridsize=60, cmap="Blues", bins="log", mincnt=1)
    ax.plot([0, 1], [0, 1], "--", color=B["red"], lw=1.5, label="1:1")
    ax.set_xlabel("實際出力（% 額定）"); ax.set_ylabel("預測出力")
    ax.set_title(f"圖 2　3 小時預測 vs 實際（測試 2020–2021）\n"
                 f"R² = {r2:.3f}　nRMSE = {np.sqrt(((y-p)**2).mean())*100:.1f}%", fontsize=12)
    ax.legend(loc="upper left"); fig.savefig(FIG / "fc2_pred_vs_obs.png"); plt.close(fig)


def fig_timeseries():
    pt = pd.read_parquet(RES / "pred_3h_test.parquet")[["ts", "y", "pred", "persist"]]
    q = pd.read_parquet(RES / "pred_3h_quantiles.parquet")
    d = pt.merge(q[["ts", "q10", "q90"]], on="ts", how="inner").sort_values("ts")
    d["ts"] = pd.to_datetime(d.ts)
    # 選一段有起伏的冬季週
    win = d[(d.ts >= "2021-01-11") & (d.ts < "2021-01-18")]
    if len(win) < 100:
        win = d.iloc[:1000]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.fill_between(win.ts, win.q10 * 100, win.q90 * 100, color=B["blue"], alpha=.18,
                    label="p10–p90 不確定區間")
    ax.plot(win.ts, win.y * 100, color="k", lw=1.6, label="實際出力")
    ax.plot(win.ts, win.pred * 100, color=B["blue"], lw=1.4, label="ML 預測 (3h 前)")
    ax.plot(win.ts, win.persist * 100, color=B["red"], lw=1, ls="--", alpha=.7, label="Persistence")
    ax.set_ylabel("出力（% 額定）")
    ax.set_title("圖 3　實際 vs 3 小時前預測（範例：2021 年 1 月一週）\n"
                 "ML 抓得住升降趨勢，不確定區間涵蓋大部分實際值", fontsize=12)
    ax.legend(ncol=2, fontsize=9); fig.autofmt_xdate()
    fig.savefig(FIG / "fc3_example_week.png"); plt.close(fig)


def fig_season():
    s = pd.read_csv(RES / "forecast_by_season.csv")
    order = ["冬", "春", "夏", "秋"]
    s["o"] = s.season.map({v: i for i, v in enumerate(order)}); s = s.sort_values("o")
    x = np.arange(len(s)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.bar(x - w / 2, s.persist_nrmse_pct, w, label="Persistence", color=B["red"])
    ax.bar(x + w / 2, s.ml_nrmse_pct, w, label="LightGBM", color=B["blue"])
    for i, (a, b_) in enumerate(zip(s.persist_nrmse_pct, s.ml_nrmse_pct)):
        ax.text(i + w / 2, b_ + 0.4, f"−{100*(1-b_/a):.0f}%", ha="center",
                fontsize=9, color=B["blue"], fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(s.season)
    ax.set_ylabel("3h 預測 nRMSE（% 額定）")
    ax.set_title("圖 4　分季節 3 小時預測誤差\n冬季風大波動大、誤差高，但改善幅度各季一致", fontsize=12)
    ax.legend(); fig.savefig(FIG / "fc4_by_season.png"); plt.close(fig)


def fig_importance():
    import lightgbm as lgb
    mdl = lgb.Booster(model_file=str(HERE / "models" / "point_3h.txt"))
    imp = pd.DataFrame({"f": mdl.feature_name(), "g": mdl.feature_importance("gain")})
    imp["pct"] = 100 * imp.g / imp.g.sum()
    imp = imp.sort_values("pct").tail(14)
    def col(f):
        if f.startswith("P_"): return B["blue"]
        if f.startswith("WS") or f.startswith("WD"): return B["green"]
        return B["grey"]
    fig, ax = plt.subplots(figsize=(8, 5.6))
    ax.barh(imp.f, imp.pct, color=[col(f) for f in imp.f])
    for i, v in enumerate(imp.pct):
        ax.text(v + 0.4, i, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("特徵重要性（gain, %）")
    ax.set_title("圖 5　3 小時預測模型特徵重要性\n藍＝出力歷史　綠＝風速/風向", fontsize=12)
    fig.savefig(FIG / "fc5_importance.png"); plt.close(fig)


if __name__ == "__main__":
    fig_skill(); fig_pred_obs(); fig_timeseries(); fig_season(); fig_importance()
    print("✓ 5 張預測圖完成")
