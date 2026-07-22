#!/usr/bin/env python3
"""
頻譜相關圖表 —— 需要回讀 1 Hz 原始檔，所以獨立成一支程式。

這兩張圖是整個專案方法學上最關鍵的證據：
  圖 16  系集平均頻譜 —— 說明本站 1 Hz 杯式風速計的可用頻帶到哪裡
  圖 17  固定頻帶 vs 約化頻帶 —— 說明為什麼頻譜斜率一定要用約化頻率

用法
----
    python make_figures_spectrum.py --root "D:/ML_wind" --out "D:/ML_wind/ml_project/results/figures"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from extract_turbulence import to_blocks, welch_psd, Z_REF, REDUCED_BAND, F_LIMITS
from preprocess import read_toa5

plt.rcParams.update({
    "font.sans-serif": ["Noto Sans CJK JP", "Droid Sans Fallback", "DejaVu Sans"],
    "axes.unicode_minus": False, "figure.dpi": 150, "savefig.dpi": 150,
    "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
})
C = {"main": "#2E5E8C", "accent": "#C1584B", "third": "#4E9A6B",
     "grey": "#8A8A8A", "warm": "#D69A3C"}

MONTHS = [("2020-01", "BSMI wind raw data 2020.01-2021.05/Raw_BSMI_Wind_Hz_2020-01.csv"),
          ("2020-07", "BSMI wind raw data 2020.01-2021.05/Raw_BSMI_Wind_Hz_2020-07.csv")]


def collect(root: Path):
    U, P, F = [], None, None
    for _, rel in MONTHS:
        arr, _ = to_blocks(read_toa5(root / rel), "WS_100E")
        psd, f = welch_psd(np.nan_to_num(arr, nan=0.0))
        U.append(np.nanmean(arr, axis=1))
        P = psd if P is None else np.vstack([P, psd])
        F = f
    return np.concatenate(U), P, F


def fig_mean_spectrum(U, psd, f, out):
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    bands = [((5, 8), C["third"], "U = 5–8 m/s"),
             ((10, 14), C["main"], "U = 10–14 m/s"),
             ((16, 24), C["accent"], "U = 16–24 m/s")]
    m0 = f > 0
    for (lo, hi), col, lab in bands:
        m = (U >= lo) & (U < hi)
        if m.sum() < 50:
            continue
        S = psd[m].mean(axis=0)
        ax.loglog(f[m0], S[m0], color=col, lw=1.8, label=f"{lab}  (n={m.sum():,})")

    # -5/3 參考線
    ref_f = np.array([0.03, 0.3])
    anchor = psd[(U >= 10) & (U < 14)].mean(axis=0)[np.argmin(np.abs(f - 0.1))]
    ax.loglog(ref_f, anchor * (ref_f / 0.1) ** (-5 / 3), "k--", lw=1.6,
              label="理論慣性次區間 $f^{-5/3}$")

    ax.axvspan(0.30, 0.5, color=C["grey"], alpha=0.18)
    ax.text(0.36, ax.get_ylim()[1] * 0.25, "杯式風速計\n衰減與雜訊底噪",
            fontsize=9, ha="center", color="#5A5A5A")
    ax.axvspan(F_LIMITS[0], F_LIMITS[1], color=C["warm"], alpha=0.10)
    ax.text(0.075, ax.get_ylim()[0] * 4, "可信頻帶 0.02–0.30 Hz",
            fontsize=9, ha="center", color="#8A6A20")

    ax.set_xlabel("頻率 f (Hz)")
    ax.set_ylabel("功率譜密度 S(f)  (m²/s²/Hz)")
    ax.set_title("圖 16　系集平均風速頻譜（Welch，100 秒分段）\n"
                 "慣性次區間隨風速往高頻移動 —— 這是固定頻帶會失準的原因", fontsize=12)
    ax.legend(fontsize=9, loc="lower left")
    fig.savefig(out / "16_mean_spectrum.png")
    plt.close(fig)
    print("  ✓ 16_mean_spectrum.png")


def slopes(U, psd, f, mode: str):
    """mode='fixed' 用固定 0.05–0.25 Hz；mode='reduced' 用 n = f·z/U。"""
    out = np.full(len(U), np.nan)
    if mode == "fixed":
        b = (f >= 0.05) & (f <= 0.25)
        lf = np.log(f[b])
        lfc = lf - lf.mean()
        lp = np.log(psd[:, b])
        return (lp * lfc).sum(axis=1) / (lfc**2).sum()
    n1, n2 = REDUCED_BAND
    for i, u in enumerate(U):
        if not np.isfinite(u) or u < 2:
            continue
        lo = max(n1 * u / Z_REF, F_LIMITS[0])
        hi = min(n2 * u / Z_REF, F_LIMITS[1])
        if hi / lo < 2.0:
            continue
        b = (f >= lo) & (f <= hi)
        if b.sum() < 5:
            continue
        lf, lp = np.log(f[b]), np.log(psd[i, b])
        if not np.isfinite(lp).all():
            continue
        out[i] = np.polyfit(lf, lp, 1)[0]
    return out


def fig_slope_correction(U, psd, f, out):
    d = pd.DataFrame({"U": U, "fixed": slopes(U, psd, f, "fixed"),
                      "reduced": slopes(U, psd, f, "reduced")})
    d = d[(d.U > 3) & (d.U < 26)]
    bins = np.arange(3, 27, 1.5)
    b = pd.cut(d.U, bins)
    g = d.groupby(b, observed=True)[["fixed", "reduced"]].median()
    x = bins[:-1] + 0.75
    g = g.iloc[: len(x)]

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.axhline(-5 / 3, color="k", ls="--", lw=1.6, label="理論值 −5/3 = −1.667")
    ax.plot(x[: len(g)], g.fixed, "o-", color=C["accent"], lw=2.2, ms=6,
            label="固定頻帶 0.05–0.25 Hz")
    ax.plot(x[: len(g)], g.reduced, "s-", color=C["main"], lw=2.2, ms=6,
            label="約化頻帶 n = f·z/U ∈ [0.3, 2.0]")

    r_fix = np.corrcoef(d.dropna(subset=["fixed"]).U, d.dropna(subset=["fixed"]).fixed)[0, 1]
    dr = d.dropna(subset=["reduced"])
    r_red = np.corrcoef(dr.U, dr.reduced)[0, 1]
    ax.text(0.03, 0.05,
            f"固定頻帶  corr(斜率, U) = {r_fix:+.3f}，跨風速擺盪 {g.fixed.max() - g.fixed.min():.3f}\n"
            f"約化頻帶  corr(斜率, U) = {r_red:+.3f}，跨風速擺盪 {g.reduced.max() - g.reduced.min():.3f}",
            transform=ax.transAxes, fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=C["grey"], alpha=0.9))

    ax.set_xlabel("100 m 平均風速 (m/s)")
    ax.set_ylabel("頻譜斜率")
    ax.set_title("圖 17　為什麼頻譜斜率必須用約化頻率\n"
                 "固定頻帶的斜率隨風速大幅漂移，量到的是風速而不是湍流結構", fontsize=12)
    ax.legend(fontsize=9.5, loc="upper left")
    fig.savefig(out / "17_spectral_slope_correction.png")
    plt.close(fig)
    print("  ✓ 17_spectral_slope_correction.png")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    U, psd, f = collect(Path(args.root))
    print(f"載入 {len(U):,} 個 10 分鐘區塊")
    fig_mean_spectrum(U, psd, f, out)
    fig_slope_correction(U, psd, f, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
