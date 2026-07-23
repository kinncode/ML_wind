"""
BSMI 測風塔 — 紊流特徵探索圖組
============================
產出兩張多面板圖：
  results/figures/turb_physics.png  — 紊流物理特徵
  results/figures/turb_spectral.png — 頻譜結構與品管

用法: python plot_turbulence_features.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def _setup_cjk_font():
    """讓中文標題不會變成方框；找不到 CJK 字型就沉默略過。"""
    import glob
    cands = (glob.glob("/usr/share/fonts/**/NotoSerifCJK*.ttc", recursive=True)
             + glob.glob("/usr/share/fonts/**/NotoSansCJK*.ttc", recursive=True)
             + glob.glob("C:/Windows/Fonts/msjh*.ttc")
             + glob.glob("C:/Windows/Fonts/msyh*.ttc"))
    for p in cands:
        try:
            fm.fontManager.addfont(p)
        except Exception:
            pass
    names = {f.name for f in fm.fontManager.ttflist}
    for n in ["Noto Sans CJK TC", "Noto Sans CJK SC", "Noto Sans CJK JP",
              "Noto Serif CJK TC", "Noto Serif CJK SC", "Noto Serif CJK JP",
              "Microsoft JhengHei", "Microsoft YaHei"]:
        if n in names:
            plt.rcParams["font.family"] = [n, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            print(f"[font] using {n}")
            return
    print("[font] 找不到 CJK 字型，中文標題可能顯示為方框")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIGS = os.path.join(HERE, "results", "figures")
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 140,
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.titlesize": 11, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
})
_setup_cjk_font()

S = "WS_100E"   # 主感測器
S2 = "WS_100W"  # 備援感測器


# --------------------------------------------------------------------------
# 1. 載入與合併
# --------------------------------------------------------------------------
def load():
    turb = pd.read_parquet(os.path.join(DATA, "BSMI_turb.parquet"))
    ten = pd.read_parquet(os.path.join(DATA, "BSMI_10min.parquet"))
    df = ten.merge(turb, on="ts", how="inner")
    df = df[df["is_valid"]].copy()
    df["ts"] = pd.to_datetime(df["ts"])
    df["hour"] = df["ts"].dt.hour
    df["month"] = df["ts"].dt.month
    # 譜擬合品質過濾旗標（不刪列，只標記）
    df["spec_ok"] = df[f"{S}_spec_r2"] > 0.85
    print(f"[load] merged rows = {len(df):,}  "
          f"({df['ts'].min():%Y-%m} ~ {df['ts'].max():%Y-%m})")
    print(f"[load] spec_r2>0.85 佔比 = {df['spec_ok'].mean():.1%}")
    return df


def binned_stat(x, y, bins, qs=(0.25, 0.5, 0.75)):
    """回傳每個 bin 的中心點與分位數，樣本 <30 的 bin 丟棄。"""
    idx = pd.cut(x, bins)
    g = pd.DataFrame({"y": y}).groupby(idx, observed=False)["y"]
    out = g.quantile(list(qs)).unstack()
    out["n"] = g.size()
    out = out[out["n"] >= 30]
    centers = np.array([iv.mid for iv in out.index])
    return centers, out


# --------------------------------------------------------------------------
# 2. 物理特徵圖組
# --------------------------------------------------------------------------
def fig_physics(df):
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 15))
    fig.suptitle("BSMI 100 m — Turbulence feature diagnostics (physics)",
                 fontsize=14, y=0.995)

    U = df["WS_100_mean"]
    TI = df[f"{S}_ti"]

    # (a) TI vs 風速 + IEC NTM 參考 -----------------------------------------
    ax = axes[0, 0]
    m = (U > 0.5) & TI.between(0, 0.6)
    ax.hexbin(U[m], TI[m], gridsize=70, cmap="Blues", bins="log", mincnt=1)
    ubins = np.arange(0, 26, 1.0)
    c, st = binned_stat(U[m], TI[m], ubins)
    ax.plot(c, st[0.5], "r-", lw=2, label="median")
    ax.fill_between(c, st[0.25], st[0.75], color="r", alpha=0.15,
                    label="IQR (p25–p75)")
    uu = np.linspace(2, 25, 100)
    for iref, ls, lb in [(0.16, "--", "IEC A"), (0.14, "-.", "IEC B"),
                         (0.12, ":", "IEC C")]:
        ax.plot(uu, iref * (0.75 * uu + 5.6) / uu, ls, c="k", lw=1.1, label=lb)
    ax.set(xlim=(0, 25), ylim=(0, 0.45),
           xlabel="Wind speed 100 m (m/s)", ylabel="TI  (σ/ū)",
           title="(a) TI vs wind speed — 最核心的一張")
    ax.legend(loc="upper right", ncol=2)

    # (b) 風向 × 風速 → TI 熱圖 ----------------------------------------------
    ax = axes[0, 1]
    wd = df["WD_97_vecmean"] % 360
    wdb = np.arange(0, 361, 15)
    ub = np.array([2, 4, 6, 8, 10, 13, 16, 20, 26])
    piv = (pd.DataFrame({"wd": pd.cut(wd, wdb), "u": pd.cut(U, ub), "ti": TI})
           .groupby(["u", "wd"], observed=False)["ti"]
           .agg(["median", "size"]))
    med = piv["median"].where(piv["size"] >= 20).unstack()
    im = ax.pcolormesh(wdb, np.arange(len(ub)), med.values,
                       cmap="magma_r", shading="flat")
    ax.set_yticks(np.arange(len(ub) - 1) + 0.5)
    ax.set_yticklabels([f"{ub[i]}–{ub[i+1]}" for i in range(len(ub) - 1)])
    ax.set(xticks=np.arange(0, 361, 45), xlabel="Wind direction 97 m (deg)",
           ylabel="Wind speed bin (m/s)",
           title="(b) median TI by direction × speed — 找地形/尾流扇區")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="median TI")

    # (c) 日夜 × 月份 TI 熱圖 -----------------------------------------------
    ax = axes[1, 0]
    piv = df.pivot_table(index="hour", columns="month",
                         values=f"{S}_ti", aggfunc="median")
    months = list(piv.columns)          # 資料只涵蓋部分月份
    im = ax.pcolormesh(np.arange(len(months) + 1), np.arange(-0.5, 24.5),
                       piv.values, cmap="viridis", shading="flat")
    ax.set_xticks(np.arange(len(months)) + 0.5)
    ax.set_xticklabels(months)
    ax.set(yticks=range(0, 24, 3), xlabel="Month (資料僅涵蓋這幾個月)",
           ylabel="Hour of day",
           title="(c) median TI — diurnal × seasonal 大氣穩定度指紋")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="median TI")

    # (d) 積分長度尺度 vs 風速 ----------------------------------------------
    ax = axes[1, 1]
    L = df[f"{S}_int_len_m"]
    m = (U > 1) & L.between(1, 800)
    ax.hexbin(U[m], L[m], gridsize=70, cmap="Greens", bins="log", mincnt=1)
    c, st = binned_stat(U[m], L[m], ubins)
    ax.plot(c, st[0.5], "r-", lw=2, label="median")
    ax.fill_between(c, st[0.25], st[0.75], color="r", alpha=0.15)
    ax.axhline(340.2, color="k", ls="--", lw=1.1,
               label="IEC Λ₁ ≈ 340 m (z>60 m)")
    ax.set(xlim=(0, 25), ylim=(0, 600),
           xlabel="Wind speed 100 m (m/s)",
           ylabel="Integral length scale L (m)",
           title="(d) 積分長度尺度 vs 風速 — 渦流大小")
    ax.legend()

    # (e) 陣風因子 vs TI ----------------------------------------------------
    ax = axes[2, 0]
    G = df[f"{S}_gust_factor"]
    m = TI.between(0.005, 0.5) & G.between(1.0, 2.5)
    ax.hexbin(TI[m], G[m], gridsize=70, cmap="Purples", bins="log", mincnt=1)
    tb = np.arange(0, 0.51, 0.02)
    c, st = binned_stat(TI[m], G[m], tb)
    ax.plot(c, st[0.5], "r-", lw=2, label="median")
    ti_th = np.linspace(0, 0.5, 50)
    ax.plot(ti_th, 1 + 3.0 * ti_th, "k--", lw=1.2, label="G = 1 + 3·TI (理論)")
    ax.set(xlim=(0, 0.4), ylim=(1, 2.2), xlabel="TI", ylabel="Gust factor",
           title="(e) 陣風因子 vs TI — 兩者是否冗餘特徵")
    ax.legend()

    # (f) 風切指數 vs TI（穩定度） -------------------------------------------
    ax = axes[2, 1]
    a = df["shear_alpha"]
    m = a.between(-0.2, 0.8) & TI.between(0, 0.4) & (U > 3)
    ax.hexbin(a[m], TI[m], gridsize=70, cmap="Oranges", bins="log", mincnt=1)
    ab = np.arange(-0.2, 0.81, 0.04)
    c, st = binned_stat(a[m], TI[m], ab)
    ax.plot(c, st[0.5], "r-", lw=2, label="median")
    ax.axvline(1 / 7, color="k", ls="--", lw=1.1, label="α = 1/7 (中性)")
    ax.set(xlabel="Shear exponent α", ylabel="TI",
           title="(f) 風切 vs TI — 穩定層結 ⇒ 高風切、低紊流")
    ax.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.985])
    p = os.path.join(FIGS, "turb_physics.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {p}")


# --------------------------------------------------------------------------
# 3. 頻譜與品管圖組
# --------------------------------------------------------------------------
def fig_spectral(df):
    fig, axes = plt.subplots(4, 2, figsize=(13.5, 20))
    fig.suptitle("BSMI 100 m — Spectral structure & feature QC",
                 fontsize=14, y=0.995)

    U = df["WS_100_mean"]
    sl = df[f"{S}_spec_slope"]
    r2 = df[f"{S}_spec_r2"]

    # (a) 譜斜率分布 vs -5/3 ------------------------------------------------
    ax = axes[0, 0]
    b = np.arange(-3.5, 0.51, 0.05)
    ax.hist(sl.dropna(), bins=b, color="lightsteelblue",
            label=f"all (n={sl.notna().sum():,})")
    ax.hist(sl[df["spec_ok"]].dropna(), bins=b, color="steelblue",
            label=f"r²>0.85 (n={df['spec_ok'].sum():,})")
    ax.axvline(-5 / 3, color="r", lw=2, label="Kolmogorov −5/3")
    ax.axvline(sl.median(), color="k", ls="--", lw=1.5,
               label=f"median = {sl.median():.3f}")
    ax.set(xlabel="Spectral slope", ylabel="count",
           title="(a) 慣性副區斜率 — 是否落在 −5/3")
    ax.legend()

    # (b) slope vs r² — 品管門檻 -------------------------------------------
    ax = axes[0, 1]
    m = sl.notna() & r2.notna()
    hb = ax.hexbin(sl[m], r2[m], gridsize=70, cmap="cividis_r",
                   bins="log", mincnt=1)
    ax.axvline(-5 / 3, color="r", lw=1.5)
    ax.axhline(0.85, color="k", ls="--", lw=1.5, label="建議門檻 r²=0.85")
    ax.set(xlim=(-3.5, 0.5), ylim=(0, 1),
           xlabel="Spectral slope", ylabel="Fit r²",
           title="(b) 斜率 vs 擬合品質 — 低 r² 的斜率不可信")
    ax.legend(loc="lower left")
    fig.colorbar(hb, ax=ax, label="count")

    # (c) 譜斜率 vs 風速 ----------------------------------------------------
    ax = axes[1, 0]
    d = df[df["spec_ok"]]
    m = d["WS_100_mean"] > 1
    ax.hexbin(d["WS_100_mean"][m], d[f"{S}_spec_slope"][m],
              gridsize=70, cmap="Blues", bins="log", mincnt=1)
    ub = np.arange(0, 26, 1.0)
    c, st = binned_stat(d["WS_100_mean"][m], d[f"{S}_spec_slope"][m], ub)
    ax.plot(c, st[0.5], "r-", lw=2, label="median")
    ax.fill_between(c, st[0.25], st[0.75], color="r", alpha=0.15)
    ax.axhline(-5 / 3, color="k", ls="--", lw=1.2, label="−5/3")
    ax.set(xlim=(0, 25), ylim=(-3, 0),
           xlabel="Wind speed 100 m (m/s)", ylabel="Spectral slope",
           title="(c) 斜率 vs 風速 — 低風速時擬合會偏離")
    ax.legend()

    # (d) 積分時間尺度分布（依風速分層） -------------------------------------
    ax = axes[1, 1]
    T = df[f"{S}_int_scale_s"]
    for lo, hi, col in [(2, 6, "#4c72b0"), (6, 12, "#55a868"),
                        (12, 20, "#c44e52"), (20, 30, "#8172b2")]:
        s = T[(U >= lo) & (U < hi) & T.between(0, 70)]
        if len(s) > 100:
            ax.hist(s, bins=np.arange(0, 70, 1.5), histtype="step", lw=1.8,
                    density=True, color=col, label=f"U {lo}–{hi} m/s (n={len(s):,})")
    ax.set(xlabel="Integral time scale T (s)", ylabel="density",
           title="(d) 積分時間尺度分布 — 決定特徵時間窗長度")
    ax.legend()

    # (e) 分位數展距 vs σ — 內部一致性檢查 ----------------------------------
    ax = axes[2, 0]
    spread = df[f"{S}_p99"] - df[f"{S}_p01"]
    sig = df[f"{S}_std"]
    m = sig.between(0.01, 6) & spread.between(0, 20)
    ax.hexbin(sig[m], spread[m], gridsize=70, cmap="Reds",
              bins="log", mincnt=1)
    xx = np.linspace(0, 6, 50)
    ax.plot(xx, 4.65 * xx, "k--", lw=1.5, label="高斯期望 (p99−p01 = 4.65σ)")
    r = np.corrcoef(sig[m], spread[m])[0, 1]
    ax.set(xlim=(0, 5), ylim=(0, 20),
           xlabel="σ from 10-min stats (m/s)",
           ylabel="p99 − p01 from turb (m/s)",
           title=f"(e) 兩個管線一致性檢查 — r = {r:.4f}")
    ax.legend()

    # (f) 特徵相關矩陣 ------------------------------------------------------
    ax = axes[2, 1]
    cols = ["WS_100_mean", f"{S}_ti", f"{S}_std", f"{S}_gust_factor",
            f"{S}_gust3s", f"{S}_int_scale_s", f"{S}_int_len_m",
            f"{S}_spec_slope", f"{S}_p99", "shear_alpha", "WD_97_sigma",
            "AT_95_mean", "air_density"]
    lab = ["U", "TI", "σ", "G", "gust3s", "T_int", "L_int",
           "slope", "p99", "α", "σ_WD", "T_air", "ρ"]
    C = df[cols].corr(method="spearman")
    im = ax.imshow(C.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set(xticks=range(len(lab)), yticks=range(len(lab)))
    ax.set_xticklabels(lab, rotation=90)
    ax.set_yticklabels(lab)
    ax.grid(False)
    for i in range(len(lab)):
        for j in range(len(lab)):
            v = C.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(v) > 0.55 else "black")
    ax.set_title("(f) Spearman 相關 — 抓冗餘特徵")
    fig.colorbar(im, ax=ax, label="ρ", shrink=0.8)

    # (g) 固定頻帶 vs 約化頻帶 — 風速假訊號 ---------------------------------
    ax = axes[3, 0]
    ub = np.arange(0, 26, 1.0)
    for col, c_, lb in [(f"{S}_spec_slope", "#2166ac", "reduced band  n=f·z/U"),
                        (f"{S}_spec_slope_fixed", "#b2182b", "fixed band  0.05–0.25 Hz")]:
        m = df[col].notna() & (U > 1)
        cc, st = binned_stat(U[m], df[col][m], ub)
        r = np.corrcoef(U[m], df[col][m])[0, 1]
        ax.plot(cc, st[0.5], "-", color=c_, lw=2,
                label=f"{lb}\n  median={df[col].median():.3f}, corr(U)={r:+.3f}")
        ax.fill_between(cc, st[0.25], st[0.75], color=c_, alpha=0.13)
    ax.axhline(-5 / 3, color="k", ls="--", lw=1.3, label="−5/3")
    ax.set(xlim=(0, 25), ylim=(-2.6, -1.0),
           xlabel="Wind speed 100 m (m/s)", ylabel="Spectral slope",
           title="(g) 頻帶選擇的影響 — 固定頻帶會把風速當成湍流訊號")
    ax.legend(loc="lower right", fontsize=7.5)

    # (h) 雙感測器一致性 = 目標變數噪聲天花板 --------------------------------
    ax = axes[3, 1]
    targets = [("gust3s", "3 s gust"), ("p99", "p99"), ("int_len_m", "L_int"),
               ("int_scale_s", "T_int"), ("spec_slope", "spec slope"),
               ("spec_slope_fixed", "slope (fixed)")]
    names, vals = [], []
    for f, lb in targets:
        a, b = f"{S}_{f}", f"{S2}_{f}"
        if a in df and b in df:
            m = df[a].notna() & df[b].notna()
            names.append(f"{lb}\n(n={m.sum():,})")
            vals.append(np.corrcoef(df[a][m], df[b][m])[0, 1] ** 2)
    y = np.arange(len(vals))
    bars = ax.barh(y, vals, color=["#2a9d8f" if v > 0.95 else
                                   "#e9c46a" if v > 0.85 else "#e76f51"
                                   for v in vals])
    for yi, v in zip(y, vals):
        ax.text(v - 0.02, yi, f"{v:.3f}", va="center", ha="right",
                fontsize=8.5, color="white", fontweight="bold")
    ax.set(yticks=y, xlim=(0, 1.0), xlabel="R² ceiling  = r²(100E, 100W)",
           title="(h) 目標變數的可達 R² 上限 — 雙感測器一致性")
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.99])
    p = os.path.join(FIGS, "turb_spectral.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {p}")


# --------------------------------------------------------------------------
# 4. 驗證
# --------------------------------------------------------------------------
def verify(df):
    print("\n" + "=" * 58)
    print("驗證")
    print("=" * 58)
    sl = df[f"{S}_spec_slope"]
    ok = df.loc[df["spec_ok"], f"{S}_spec_slope"]
    print(f"譜斜率中位數 (全部)      = {sl.median():.4f}   (理論 -1.6667)")
    print(f"譜斜率中位數 (r²>0.85)   = {ok.median():.4f}")
    print(f"積分長度尺度中位數        = {df[f'{S}_int_len_m'].median():.1f} m")
    print(f"積分時間尺度中位數        = {df[f'{S}_int_scale_s'].median():.1f} s")
    print("\n各月 TI 中位數（對照 PLAN.md：1月 0.053 / 4月 0.060 / "
          "7月 0.098 / 10月 0.061）")
    print(df.groupby("month")[f"{S}_ti"].median().round(4).to_string())
    print("\n兩支 100 m 感測器紊流特徵一致性 (Pearson r):")
    for f in ["ti", "int_scale_s", "spec_slope", "int_len_m", "gust3s"]:
        a, b = f"{S}_{f}", f"{S2}_{f}"
        if a in df and b in df:
            m = df[a].notna() & df[b].notna()
            print(f"  {f:<14} r = {np.corrcoef(df[a][m], df[b][m])[0,1]:.4f}")


if __name__ == "__main__":
    df = load()
    fig_physics(df)
    fig_spectral(df)
    verify(df)
