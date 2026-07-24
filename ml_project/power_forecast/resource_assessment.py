#!/usr/bin/env python3
"""
虛擬風場資源評估 —— 只用 BSMI 塔資料就能做的靜態發電潛力分析。

產出：
  results/resource_stats.csv / .json
  figures/res1_monthly_cf.png       各月容量因數
  figures/res2_power_curve.png      功率曲線 vs 風速分布
  figures/res3_power_duration.png   功率延時曲線
  figures/res4_diurnal.png          分季節日夜出力
  figures/res5_energy_rose.png      能量風花圖（各方向貢獻多少發電）
  figures/res6_interannual.png      逐年容量因數（資源穩定度）
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from virtual_power import (load_power_table, power_curve, density_correct,
                           _ALT_CURVES, RHO_REF)

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "BSMI_10min.parquet"
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
RES = HERE / "results"; RES.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.sans-serif": ["Noto Sans CJK JP", "Droid Sans Fallback", "DejaVu Sans"],
    "axes.unicode_minus": False, "figure.dpi": 150, "savefig.dpi": 150,
    "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": .25,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11})
B = {"blue": "#2E5E8C", "red": "#C1584B", "green": "#4E9A6B",
     "warm": "#D69A3C", "grey": "#8A8A8A", "purple": "#7B5AA6"}
SEASON_COL = {"冬 (12-2)": B["blue"], "春 (3-5)": B["green"],
              "夏 (6-8)": B["red"], "秋 (9-11)": B["warm"]}


def season_of(m):
    return ("冬 (12-2)" if m in (12, 1, 2) else "春 (3-5)" if m in (3, 4, 5)
            else "夏 (6-8)" if m in (6, 7, 8) else "秋 (9-11)")


def main() -> int:
    d = load_power_table(str(DATA))
    d["season"] = d.month.map(season_of)
    cf = float(d.P.mean())
    stats = {
        "n_samples": int(len(d)),
        "period": f"{d.ts.min():%Y-%m} ~ {d.ts.max():%Y-%m}",
        "capacity_factor_pct": round(100 * cf, 2),
        "full_load_hours": round(cf * 8760),
        "frac_below_cutin_pct": round(100 * (d.WS_100_mean < 3).mean(), 2),
        "frac_at_rated_pct": round(100 * (d.P >= 0.99).mean(), 2),
        "frac_cutout_pct": round(100 * (d.WS_100_mean > 25).mean(), 3),
        "best_month_cf": round(100 * d.groupby("month").P.mean().max(), 1),
        "worst_month_cf": round(100 * d.groupby("month").P.mean().min(), 1),
    }
    # 功率曲線敏感度
    ueff = density_correct(d.WS_100_mean.to_numpy(), d.air_density.to_numpy())
    stats["cf_sensitivity"] = {name: round(100 * power_curve(ueff, ru).mean(), 1)
                               for name, (ru,) in _ALT_CURVES.items()}
    # 逐年 CF
    stats["cf_by_year"] = {int(y): round(100 * v, 1)
                           for y, v in d.groupby("year").P.mean().items()}
    (RES / "resource_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{"metric": k, "value": v} for k, v in stats.items()
                 if not isinstance(v, dict)]).to_csv(
        RES / "resource_stats.csv", index=False, encoding="utf-8-sig")

    # ---- 圖 1 月容量因數 ----
    mcf = d.groupby("month").P.mean() * 100
    fig, ax = plt.subplots(figsize=(8, 4.6))
    cols = [B["blue"] if v >= 40 else B["warm"] if v >= 25 else B["red"] for v in mcf]
    ax.bar(mcf.index, mcf.values, color=cols)
    ax.axhline(100 * cf, color="k", ls="--", lw=1, label=f"全年平均 {100*cf:.0f}%")
    for x, v in zip(mcf.index, mcf.values):
        ax.text(x, v + 1, f"{v:.0f}", ha="center", fontsize=8.5)
    ax.set_xlabel("月"); ax.set_ylabel("容量因數 (%)"); ax.set_xticks(range(1, 13))
    ax.set_title("圖 1　虛擬風機各月容量因數\n"
                 f"冬季東北季風 {stats['best_month_cf']:.0f}%，夏季僅 {stats['worst_month_cf']:.0f}%"
                 f" —— {stats['best_month_cf']/stats['worst_month_cf']:.0f} 倍季節落差", fontsize=12)
    ax.legend(); fig.savefig(FIG / "res1_monthly_cf.png"); plt.close(fig)

    # ---- 圖 2 功率曲線 vs 風速分布 ----
    from virtual_power import _PC_U, _PC_P
    fig, ax = plt.subplots(figsize=(8, 4.6))
    U = np.linspace(0, 28, 281)
    ax.plot(U, power_curve(U) * 100, color=B["blue"], lw=2.4, label="代表性功率曲線")
    ax.axvspan(0, 3, color=B["grey"], alpha=.12); ax.axvspan(25, 28, color=B["red"], alpha=.12)
    ax.set_xlabel("100 m 風速 (m/s)"); ax.set_ylabel("出力（% 額定）", color=B["blue"])
    ax2 = ax.twinx(); ax2.hist(d.WS_100_mean, bins=56, color=B["warm"], alpha=.30); ax2.grid(False)
    ax2.set_ylabel("風速出現次數", color=B["warm"])
    ax.set_title("圖 2　風速 → 出力：功率曲線 vs 本站風速分布\n"
                 "大量風速落在陡升與額定區間，是高容量因數的來源", fontsize=12)
    ax.legend(loc="center right"); fig.savefig(FIG / "res2_power_curve.png"); plt.close(fig)

    # ---- 圖 3 功率延時曲線 ----
    ps = np.sort(d.P.to_numpy())[::-1] * 100
    x = np.linspace(0, 100, len(ps))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(x, ps, color=B["purple"], lw=2.2)
    ax.fill_between(x, ps, color=B["purple"], alpha=.15)
    ax.axhline(100 * cf, color="k", ls="--", lw=1, label=f"平均 {100*cf:.0f}%")
    p_at = {"滿載時間占比": 100 * (d.P >= 0.99).mean(), "零出力占比": 100 * (d.P <= 0.01).mean()}
    ax.set_xlabel("時間占比 (%)"); ax.set_ylabel("出力（% 額定）")
    ax.set_title("圖 3　功率延時曲線（出力由高到低排序）\n"
                 f"約 {p_at['滿載時間占比']:.0f}% 時間滿載、{p_at['零出力占比']:.0f}% 時間零出力",
                 fontsize=12)
    ax.legend(); fig.savefig(FIG / "res3_power_duration.png"); plt.close(fig)

    # ---- 圖 4 分季節日夜出力 ----
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for s, col in SEASON_COL.items():
        sub = d[d.season == s]
        g = sub.groupby("hour").P.mean() * 100
        ax.plot(g.index, g.values, "o-", color=col, lw=2, ms=4, label=s)
    ax.set_xlabel("當地時間（時）"); ax.set_ylabel("平均出力（% 額定）"); ax.set_xticks(range(0, 24, 3))
    ax.set_title("圖 4　分季節日夜出力形態\n冬季全日高檔，夏季白天略升但整體偏低", fontsize=12)
    ax.legend(ncol=2); fig.savefig(FIG / "res4_diurnal.png"); plt.close(fig)

    # ---- 圖 5 能量風花圖（各方向的發電貢獻）----
    nsec = 16; width = 360 / nsec
    edges = np.arange(-width / 2, 360, width)
    sec = pd.cut((d.WD_97_vecmean + width / 2) % 360, bins=edges, labels=False, include_lowest=True)
    energy = d.groupby(sec).P.sum()
    energy = energy / energy.sum() * 100
    theta = np.deg2rad(np.arange(nsec) * width)
    fig = plt.figure(figsize=(6.6, 6.4)); ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    vals = np.array([energy.get(i, 0) for i in range(nsec)])
    ax.bar(theta, vals, width=np.deg2rad(width) * 0.9, color=B["blue"], edgecolor="white")
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
    ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    top = int(np.argmax(vals))
    ax.set_title(f"圖 5　能量風花圖：發電量的方向來源\n"
                 f"最大貢獻方向約 {top*width:.0f}°（{vals[top]:.0f}% 的總發電）", pad=20, fontsize=12)
    fig.savefig(FIG / "res5_energy_rose.png"); plt.close(fig)

    # ---- 圖 6 逐年容量因數 ----
    ycf = d.groupby("year").P.mean() * 100
    n_by_year = d.groupby("year").size()
    fig, ax = plt.subplots(figsize=(8, 4.4))
    full_years = n_by_year[n_by_year > 40000].index  # 只標資料較完整的年
    ax.bar(ycf.index, ycf.values, color=[B["blue"] if y in full_years else B["grey"] for y in ycf.index])
    for x, v in zip(ycf.index, ycf.values):
        ax.text(x, v + 0.8, f"{v:.0f}", ha="center", fontsize=9)
    ax.set_xlabel("年"); ax.set_ylabel("容量因數 (%)")
    ax.set_title("圖 6　逐年容量因數（灰＝該年資料不完整）\n"
                 "年際變化不大，代表風資源穩定", fontsize=12)
    fig.savefig(FIG / "res6_interannual.png"); plt.close(fig)

    print("=== 資源評估 ===")
    for k, v in stats.items():
        if not isinstance(v, dict):
            print(f"  {k}: {v}")
    print("  CF 敏感度:", stats["cf_sensitivity"])
    print("  逐年 CF:", stats["cf_by_year"])
    print("✓ 6 張圖完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
