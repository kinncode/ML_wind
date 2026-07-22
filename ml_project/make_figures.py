#!/usr/bin/env python3
"""
產出專案 ① 的所有圖表 —— 每張圖各自獨立存成一個 PNG 檔。

用法
----
    python make_figures.py --data "D:/ML_wind/ml_project/data" --out "D:/ML_wind/ml_project/results/figures"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.sans-serif": ["Noto Sans CJK JP", "Droid Sans Fallback", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})

C = {"main": "#2E5E8C", "accent": "#C1584B", "third": "#4E9A6B",
     "grey": "#8A8A8A", "warm": "#D69A3C", "purple": "#7B5AA6"}

FIGSIZE = (8.2, 5.0)


def save(fig, out: Path, name: str) -> None:
    p = out / name
    fig.savefig(p)
    plt.close(fig)
    print(f"  ✓ {name}")


# --------------------------------------------------------------------------
def load(data_dir: Path) -> pd.DataFrame:
    base = pd.read_parquet(data_dir / "BSMI_10min.parquet")
    turb = pd.read_parquet(data_dir / "BSMI_turb.parquet")
    d = base.merge(turb, on="ts", how="inner")
    d = d[d["is_valid"]].copy()
    d["ti"] = d.WS_100E_std / d.WS_100E_mean
    d["gust_factor"] = d.WS_100E_gust3s / d.WS_100E_mean
    d["month"] = d.ts.dt.month
    d["hour"] = d.ts.dt.hour
    d["year"] = d.ts.dt.year
    d["season"] = np.where(d.month.isin([12, 1, 2]), "冬 (12-2月)",
                  np.where(d.month.isin([3, 4, 5]), "春 (3-5月)",
                  np.where(d.month.isin([6, 7, 8]), "夏 (6-8月)", "秋 (9-11月)")))
    return d


# ==========================================================================
# A. 特徵觀察
# ==========================================================================
def fig_wind_rose(d, out):
    dd = d[d.WS_100_mean >= 2]
    nsec = 16
    width = 360 / nsec
    edges = np.arange(-width / 2, 360, width)
    sec = pd.cut((dd.WD_97_vecmean + width / 2) % 360, bins=edges,
                 labels=False, include_lowest=True)
    bands = [(2, 5), (5, 8), (8, 12), (12, 16), (16, 40)]
    labels = ["2–5", "5–8", "8–12", "12–16", ">16 m/s"]
    colors = ["#CFE0EC", "#9BC0DA", "#5E97C4", "#2E5E8C", "#16354F"]

    fig = plt.figure(figsize=(6.8, 6.6))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    theta = np.deg2rad(np.arange(nsec) * width)
    bottom = np.zeros(nsec)
    for (lo, hi), lab, col in zip(bands, labels, colors):
        m = (dd.WS_100_mean >= lo) & (dd.WS_100_mean < hi)
        cnt = np.array([((sec == i) & m).sum() for i in range(nsec)]) / len(dd) * 100
        ax.bar(theta, cnt, width=np.deg2rad(width) * 0.92, bottom=bottom,
               color=col, edgecolor="white", linewidth=0.5, label=lab)
        bottom += cnt

    ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
    ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    ax.set_title("圖 1　風花圖：風向與風速聯合分布\n"
                 "東北（NE）與南至西南（S–SW）雙峰結構，強風幾乎全來自 NE", pad=22, fontsize=12)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.05), title="100 m 風速", fontsize=9)
    save(fig, out, "01_wind_rose.png")


def fig_ti_by_direction(d, out):
    dd = d[(d.WS_100_mean >= 4) & d.ti.notna()]
    width = 15
    b = pd.cut(dd.WD_97_vecmean, np.arange(0, 361, width))
    g = dd.groupby(b, observed=True).ti.agg(["median", "count",
                                             lambda s: s.quantile(.25),
                                             lambda s: s.quantile(.75)])
    g.columns = ["med", "n", "q25", "q75"]
    x = np.arange(0, 360, width) + width / 2
    g = g.reindex(range(len(x)), fill_value=np.nan) if len(g) != len(x) else g

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.fill_between(x, g.q25, g.q75, color=C["main"], alpha=0.20, label="四分位距")
    ax.plot(x, g.med, "o-", color=C["main"], lw=2, ms=5, label="TI 中位數")
    ax.set_xlabel("100 m 風向（度，0 = 北）")
    ax.set_ylabel("湍流強度 TI = σ$_u$ / U")
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 45))

    ax2 = ax.twinx()
    ax2.bar(x, g.n, width=width * 0.85, color=C["grey"], alpha=0.18, zorder=0)
    ax2.set_ylabel("樣本數", color=C["grey"])
    ax2.grid(False)
    ax2.tick_params(colors=C["grey"])

    lo_s, hi_s = g.med.min(), g.med.max()
    ax.set_title("圖 2　湍流強度隨風向的變化\n"
                 f"東北扇區最低 {lo_s:.3f}，南向扇區最高 {hi_s:.3f} —— 相差 {hi_s / lo_s:.1f} 倍",
                 fontsize=12)
    ax.legend(loc="upper left")
    save(fig, out, "02_ti_by_direction.png")


def fig_ti_vs_speed(d, out):
    dd = d[(d.WS_100_mean >= 3) & d.ti.notna()]
    bins = np.arange(3, 26, 1.0)
    b = pd.cut(dd.WS_100_mean, bins)
    g = dd.groupby(b, observed=True).ti.agg(
        med="median", q10=lambda s: s.quantile(.10), q90=lambda s: s.quantile(.90), n="count")
    x = bins[:-1] + 0.5
    g = g.iloc[: len(x)]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.fill_between(x[: len(g)], g.q10, g.q90, color=C["accent"], alpha=0.18,
                    label="10–90 百分位")
    ax.plot(x[: len(g)], g.med, "o-", color=C["accent"], lw=2, ms=5, label="TI 中位數")
    ax.set_xlabel("100 m 平均風速 (m/s)")
    ax.set_ylabel("湍流強度 TI")
    ax.set_title("圖 3　湍流強度 vs 平均風速\n"
                 "分散帶很寬 —— 同一個風速對應的 TI 可以差三倍以上", fontsize=12)
    ax.legend()
    save(fig, out, "03_ti_vs_windspeed.png")


def fig_seasonal(d, out):
    g = d[d.WS_100_mean > 0].groupby("month").agg(
        ws=("WS_100_mean", "mean"), ti=("ti", "median"), pd_=("power_density", "mean"))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(g.index, g.ws, "o-", color=C["main"], lw=2.2, ms=6, label="平均風速")
    ax.set_xlabel("月份")
    ax.set_ylabel("100 m 平均風速 (m/s)", color=C["main"])
    ax.tick_params(axis="y", colors=C["main"])
    ax.set_xticks(range(1, 13))

    ax2 = ax.twinx()
    ax2.plot(g.index, g.ti, "s--", color=C["accent"], lw=2, ms=5, label="湍流強度")
    ax2.set_ylabel("湍流強度 TI（中位數）", color=C["accent"])
    ax2.tick_params(axis="y", colors=C["accent"])
    ax2.grid(False)

    ax.set_title("圖 4　季節循環：風速與湍流反向變化\n"
                 "冬季東北季風強而穩定，夏季風弱但湍流強", fontsize=12)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center")
    save(fig, out, "04_seasonal_cycle.png")


def fig_diurnal(d, out):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    cols = {"冬 (12-2月)": C["main"], "春 (3-5月)": C["third"],
            "夏 (6-8月)": C["accent"], "秋 (9-11月)": C["warm"]}
    for s, col in cols.items():
        sub = d[d.season == s]
        if len(sub) < 200:
            continue
        g = sub.groupby("hour").WS_100_mean.mean()
        ax.plot(g.index, g - g.mean(), "o-", color=col, lw=2, ms=4, label=s)
    ax.axhline(0, color="k", lw=0.8, alpha=0.4)
    ax.set_xlabel("當地時間（時）")
    ax.set_ylabel("風速距平 (m/s)")
    ax.set_xticks(range(0, 24, 3))
    ax.set_title("圖 5　日夜變化（各季節去除月平均後）\n"
                 "振幅達 3 m/s —— 時間特徵必須用 sin/cos 編碼", fontsize=12)
    ax.legend(ncol=2)
    save(fig, out, "05_diurnal_cycle.png")


def fig_shear_by_direction(d, out):
    dd = d[(d.WS_100_mean >= 4) & d.shear_alpha.notna() & d.shear_alpha.between(-0.3, 0.8)]
    width = 15
    b = pd.cut(dd.WD_97_vecmean, np.arange(0, 361, width))
    g = dd.groupby(b, observed=True).shear_alpha.agg(
        med="median", q25=lambda s: s.quantile(.25), q75=lambda s: s.quantile(.75))
    x = np.arange(0, 360, width) + width / 2
    g = g.iloc[: len(x)]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.fill_between(x[: len(g)], g.q25, g.q75, color=C["third"], alpha=0.22, label="四分位距")
    ax.plot(x[: len(g)], g.med, "o-", color=C["third"], lw=2, ms=5, label="風切指數中位數")
    ax.axhline(1 / 7, color=C["accent"], ls="--", lw=1.6, label="業界預設 α = 1/7")
    ax.set_xlabel("100 m 風向（度）")
    ax.set_ylabel("風切指數 α")
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 45))
    ax.set_title("圖 6　風切指數隨風向的變化\n"
                 "固定 α = 1/7 在多數扇區都偏高，是垂直外推誤差的來源", fontsize=12)
    ax.legend()
    save(fig, out, "06_shear_by_direction.png")


def fig_integral_scale(d, out):
    dd = d[d.WS_100E_int_scale_s.notna() & (d.WS_100E_int_scale_s < 70)]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(dd.WS_100E_int_scale_s, bins=70, color=C["purple"], alpha=0.80, edgecolor="white")
    med = dd.WS_100E_int_scale_s.median()
    ax.axvline(med, color=C["accent"], ls="--", lw=1.8, label=f"中位數 {med:.1f} s")
    ax.set_xlabel("湍流積分時間尺度 T$_u$ (s)")
    ax.set_ylabel("次數")
    ax.set_title("圖 7　湍流積分時間尺度分布\n"
                 "量測穩定（雙感測器 r = 0.98），但無法由 10 分鐘平均狀態預測", fontsize=12)
    ax.legend()
    save(fig, out, "07_integral_scale_dist.png")


def fig_noise_ceiling(d, out):
    dd = d[(d.WS_100_mean >= 4)].copy()
    dd["ti_w"] = dd.WS_100W_std / dd.WS_100W_mean
    m = dd.ti.notna() & dd.ti_w.notna() & (dd.ti < 0.35) & (dd.ti_w < 0.35)
    x, y = dd.loc[m, "ti"], dd.loc[m, "ti_w"]
    r = np.corrcoef(x, y)[0, 1]

    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    ax.hexbin(x, y, gridsize=70, cmap="Blues", bins="log", mincnt=1)
    lim = [0, 0.35]
    ax.plot(lim, lim, color=C["accent"], ls="--", lw=1.5, label="1:1")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("TI（100 m 東側風速計）")
    ax.set_ylabel("TI（100 m 西側風速計）")
    ax.set_title(f"圖 8　噪聲天花板：兩支同高度風速計的一致性\n"
                 f"r = {r:.4f} → 任何模型的 R² 上限約為 {r:.3f}", fontsize=12)
    ax.legend(loc="upper left")
    save(fig, out, "08_noise_ceiling.png")


# ==========================================================================
# B. 預測結果
# ==========================================================================
FEATS = ["WS_100_mean", "WS_69W_mean", "WS_38W_mean", "shear_alpha",
         "WD_97_sin", "WD_97_cos", "WD_35_sin", "WD_35_cos", "veer_97_35",
         "AT_95_mean", "RH_95_mean", "BP_93_mean", "air_density",
         "hour_sin", "hour_cos", "doy_sin", "doy_cos"]


def train_ti_model(d):
    import lightgbm as lgb
    d = d[(d.WS_100_mean >= 4) & d.ti.notna()].copy()
    h = d.ts.dt.hour + d.ts.dt.minute / 60
    doy = d.ts.dt.dayofyear
    d["hour_sin"], d["hour_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    d["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    d["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    tr = d[d.year <= 2018]
    va = d[d.year == 2019]
    te = d[d.year >= 2020].copy()
    # 固定所有亂數種子 —— bagging 與 feature_fraction 都會引入隨機性，
    # 不固定的話每次跑出來的 R² 會在 ±0.02 間浮動，圖表數字對不上文件
    p = dict(objective="regression", metric="rmse", learning_rate=0.05, num_leaves=63,
             min_data_in_leaf=40, feature_fraction=0.85, bagging_fraction=0.85,
             bagging_freq=1, verbose=-1, num_threads=4,
             seed=42, bagging_seed=42, feature_fraction_seed=42, data_random_seed=42)
    out = {}
    for tag, f in [("風速", ["WS_100_mean"]), ("完整", FEATS)]:
        ds = lgb.Dataset(tr[f], tr.ti)
        mdl = lgb.train(p, ds, 3000, valid_sets=[lgb.Dataset(va[f], va.ti, reference=ds)],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
        out[tag] = (mdl, f, mdl.predict(te[f], num_iteration=mdl.best_iteration))
    # 現地線性
    A = np.column_stack([tr.WS_100_mean, np.ones(len(tr))])
    coef, *_ = np.linalg.lstsq(A, tr.ti, rcond=None)
    out["線性"] = (None, None, coef[0] * te.WS_100_mean.to_numpy() + coef[1])
    out["常數"] = (None, None, np.full(len(te), tr.ti.mean()))
    return te, out


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def fig_baseline_ladder(te, mo, out):
    names = ["常數", "線性", "風速", "完整"]
    labels = ["B0 常數", "B2 現地線性\n(僅風速)", "B3 LightGBM\n(僅風速)",
              "B4 LightGBM\n(完整平均狀態)"]
    vals = [r2(te.ti.to_numpy(), mo[n][2]) for n in names]
    cols = [C["grey"], C["grey"], C["warm"], C["main"]]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars = ax.bar(labels, vals, color=cols, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                ha="center", fontsize=11, fontweight="bold")
    ax.axhline(0.991, color=C["accent"], ls="--", lw=1.6)
    ax.text(-0.42, 0.945, "噪聲天花板 R² = 0.991（雙感測器信度）",
            color=C["accent"], ha="left", va="top", fontsize=9.5)
    ax.set_ylabel("測試集 R²")
    ax.set_ylim(-0.15, 1.10)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("圖 9　基準線階梯：預測湍流強度 TI\n"
                 f"只用風速的模型停在 {vals[2]:.2f}；加入大氣狀態後跳到 {vals[3]:.2f}",
                 fontsize=12)
    save(fig, out, "09_baseline_ladder.png")


def fig_pred_vs_obs(te, mo, out):
    y = te.ti.to_numpy()
    p = mo["完整"][2]
    m = (y < 0.30) & (p < 0.30)
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    ax.hexbin(y[m], p[m], gridsize=70, cmap="Blues", bins="log", mincnt=1)
    lim = [0, 0.30]
    ax.plot(lim, lim, color=C["accent"], ls="--", lw=1.5, label="1:1")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("實測 TI")
    ax.set_ylabel("預測 TI")
    ax.set_title(f"圖 10　B4 模型：預測 vs 實測（測試集 2020–2021）\n"
                 f"R² = {r2(y, p):.3f}　RMSE = {np.sqrt(((y - p) ** 2).mean()):.4f}", fontsize=12)
    ax.legend(loc="upper left")
    save(fig, out, "10_pred_vs_obs_ti.png")


def fig_importance(mo, out):
    mdl, feats, _ = mo["完整"]
    imp = pd.DataFrame({"f": feats, "g": mdl.feature_importance("gain")})
    imp["pct"] = 100 * imp.g / imp.g.sum()
    imp = imp.sort_values("pct").tail(12)
    cols = [C["accent"] if f.startswith("WD_") else C["main"] for f in imp.f]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.barh(imp.f, imp.pct, color=cols)
    for i, (f, v) in enumerate(zip(imp.f, imp.pct)):
        ax.text(v + 0.6, i, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("特徵重要性（gain，%）")
    ax.set_title("圖 11　特徵重要性：風向（紅）主導湍流強度\n"
                 "四個風向分量合計約 80%", fontsize=12)
    ax.grid(axis="y", alpha=0)
    save(fig, out, "11_feature_importance.png")


def fig_residual_by_direction(te, mo, out):
    res_full = te.ti.to_numpy() - mo["完整"][2]
    res_ws = te.ti.to_numpy() - mo["風速"][2]
    width = 15
    b = pd.cut(te.WD_97_vecmean, np.arange(0, 361, width))
    x = np.arange(0, 360, width) + width / 2

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for res, lab, col in [(res_ws, "B3 只用風速", C["warm"]),
                          (res_full, "B4 完整平均狀態", C["main"])]:
        g = pd.Series(res).groupby(b.values, observed=True).median()
        ax.plot(x[: len(g)], g.values, "o-", color=col, lw=2, ms=5, label=lab)
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("100 m 風向（度）")
    ax.set_ylabel("殘差中位數（實測 − 預測）")
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 45))
    ax.set_title("圖 12　殘差的風向結構\n"
                 "只用風速時殘差隨風向大幅擺盪；加入風向後被壓平", fontsize=12)
    ax.legend()
    save(fig, out, "12_residual_by_direction.png")


def fig_iec(d, out):
    dd = d[(d.WS_100_mean >= 4) & d.WS_100E_std.notna()]
    bins = np.arange(4, 26, 1.0)
    b = pd.cut(dd.WS_100_mean, bins)
    g = dd.groupby(b, observed=True).WS_100E_std.agg(
        med="median", q90=lambda s: s.quantile(.90))
    x = bins[:-1] + 0.5
    g = g.iloc[: len(x)]
    u = np.linspace(4, 25, 100)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for iref, cls, col in [(0.16, "A", "#B03A2E"), (0.14, "B", "#D68910"), (0.12, "C", "#7D6608")]:
        ax.plot(u, iref * (0.75 * u + 5.6), "--", color=col, lw=1.8,
                label=f"IEC NTM Class {cls}")
    ax.plot(x[: len(g)], g.med, "o-", color=C["main"], lw=2.4, ms=6, label="本站實測中位數")
    ax.plot(x[: len(g)], g.q90, "s:", color=C["main"], lw=1.6, ms=4, alpha=0.7,
            label="本站實測 90 百分位")
    ax.set_xlabel("100 m 平均風速 (m/s)")
    ax.set_ylabel("湍流標準差 σ$_u$ (m/s)")
    ax.set_title("圖 13　IEC 標準湍流模型 vs 本站實測\n"
                 "連最低的 Class C 都高估約 0.94 m/s —— 這是一個極低湍流站址", fontsize=12)
    ax.legend(fontsize=9)
    save(fig, out, "13_iec_ntm_vs_obs.png")


def fig_normalized_effect(out):
    labels = ["3 秒陣風\n(m/s)", "p99\n(m/s)", "陣風因子\ngust/U", "TI\nσ/U", "p99/U"]
    lin = [0.983, 0.986, 0.084, 0.053, 0.077]
    ml = [0.985, 0.987, 0.635, 0.608, 0.640]
    x = np.arange(len(labels))
    w = 0.36

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(x - w / 2, lin, w, label="B2 現地線性（僅風速）", color=C["grey"])
    ax.bar(x + w / 2, ml, w, label="B4 LightGBM（完整平均狀態）", color=C["main"])
    ax.axvline(1.5, color=C["accent"], ls="--", lw=1.5)
    ax.text(0.5, 1.06, "有量綱目標\n線性就有 0.98", ha="center", fontsize=9.5, color=C["accent"])
    ax.text(3.0, 1.06, "正規化目標\n真實難度才顯現", ha="center", fontsize=9.5, color=C["accent"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("測試集 R²")
    ax.set_ylim(0, 1.22)
    ax.set_title("圖 14　為什麼一定要用正規化目標\n"
                 "陣風 ≈ 風速 × 1.16，直接預測會得到虛高的 R²", fontsize=12)
    ax.legend(loc="center left", fontsize=9)
    save(fig, out, "14_normalized_vs_raw.png")


def fig_error_by_season(te, mo, out):
    te = te.copy()
    te["res"] = np.abs(te.ti.to_numpy() - mo["完整"][2])
    te["res_ws"] = np.abs(te.ti.to_numpy() - mo["風速"][2])
    order = ["冬 (12-2月)", "春 (3-5月)", "夏 (6-8月)", "秋 (9-11月)"]
    g = te.groupby("season")[["res_ws", "res"]].mean().reindex(order).dropna()
    x = np.arange(len(g))
    w = 0.36

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(x - w / 2, g.res_ws, w, label="B3 只用風速", color=C["warm"])
    ax.bar(x + w / 2, g.res, w, label="B4 完整平均狀態", color=C["main"])
    for i, (a, b_) in enumerate(zip(g.res_ws, g.res)):
        ax.text(i + w / 2, b_ + 0.0006, f"−{100 * (1 - b_ / a):.0f}%",
                ha="center", fontsize=10, color=C["main"], fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(g.index)
    ax.set_ylabel("平均絕對誤差 MAE")
    ax.set_title("圖 15　分季節誤差分解\n"
                 "夏季湍流大、誤差也大，但改善幅度同樣顯著", fontsize=12)
    ax.legend()
    save(fig, out, "15_error_by_season.png")


# ==========================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    d = load(Path(args.data))
    print(f"載入 {len(d):,} 筆\n特徵觀察圖：")
    fig_wind_rose(d, out)
    fig_ti_by_direction(d, out)
    fig_ti_vs_speed(d, out)
    fig_seasonal(d, out)
    fig_diurnal(d, out)
    fig_shear_by_direction(d, out)
    fig_integral_scale(d, out)
    fig_noise_ceiling(d, out)

    print("預測結果圖：")
    te, mo = train_ti_model(d)
    fig_baseline_ladder(te, mo, out)
    fig_pred_vs_obs(te, mo, out)
    fig_importance(mo, out)
    fig_residual_by_direction(te, mo, out)
    fig_iec(d, out)
    fig_normalized_effect(out)
    fig_error_by_season(te, mo, out)
    print(f"\n完成 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
