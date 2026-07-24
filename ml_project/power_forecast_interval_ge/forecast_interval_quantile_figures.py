#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""窗口平均出力 機率區間圖表。"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.sans-serif": ["Noto Sans CJK JP", "Droid Sans Fallback", "DejaVu Sans"],
    "axes.unicode_minus": False, "figure.dpi": 150, "savefig.dpi": 150,
    "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": .25,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11})
B = {"blue": "#2E5E8C", "red": "#C1584B", "green": "#4E9A6B", "warm": "#D69A3C", "grey": "#8A8A8A"}


def load_metrics():
    return pd.read_csv(HERE / "results" / "interval_prob_metrics.csv")


def fig_coverage():
    m = load_metrics()
    x = np.arange(len(m)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - w / 2, m.raw_cov80_pct, w, label="原始 p10–p90", color=B["warm"])
    ax.bar(x + w / 2, m.cqr_cov80_pct, w, label="CQR 校正後", color=B["blue"])
    ax.axhline(80, color="k", ls=":", lw=1.4, label="名目 80% 目標")
    ax.set_xticks(x); ax.set_xticklabels([f"H{h}" for h in m.H])
    ax.set_ylabel("實際涵蓋率 (%)"); ax.set_ylim(50, 100)
    for i, r in m.iterrows():
        ax.text(i - w / 2, r.raw_cov80_pct + 0.6, f"{r.raw_cov80_pct:.0f}", ha="center", fontsize=8)
        ax.text(i + w / 2, r.cqr_cov80_pct + 0.6, f"{r.cqr_cov80_pct:.0f}", ha="center", fontsize=8)
    ax.set_title("圖 A　窗口平均出力：p10–p90 區間涵蓋率\n"
                 "短時程略偏保守(≈90%)；H24 原本不足、CQR 校正回 80%", fontsize=12)
    ax.legend(fontsize=9); fig.savefig(FIG / "qiv_A_coverage.png"); plt.close(fig)


def fig_example():
    h = 3
    te = pd.read_parquet(HERE / "data" / f"pred_qtest_power_H{h}.parquet")
    M = np.sort(te[["q10", "q50", "q90"]].to_numpy(), axis=1)
    te[["q10", "q50", "q90"]] = M
    te["ts"] = pd.to_datetime(te.ts)
    win = te[(te.ts >= "2021-01-11") & (te.ts < "2021-01-18")]
    if len(win) < 50:
        win = te.iloc[:1000]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.fill_between(win.ts, win.q10 * 100, win.q90 * 100, color=B["blue"], alpha=.20,
                    label="80% 區間 (p10–p90)")
    ax.plot(win.ts, win.y * 100, color="k", lw=1.6, label="實際窗口平均出力")
    ax.plot(win.ts, win.q50 * 100, color=B["red"], lw=1.3, label="中位數預測 p50")
    ax.set_ylabel("未來 3h 平均出力（% 額定）")
    ax.set_title("圖 B　窗口平均出力 + 不確定區間（H3，範例：2021 年 1 月一週）\n"
                 "注意：目標是平滑的窗口平均，比瞬時值穩定，區間也較貼合", fontsize=12)
    ax.legend(ncol=3, fontsize=9); fig.autofmt_xdate()
    fig.savefig(FIG / "qiv_B_example.png"); plt.close(fig)


def fig_sharpness():
    m = load_metrics()
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(m.H, m.cqr_width_pct, "o-", color=B["blue"], lw=2, ms=7)
    for _, r in m.iterrows():
        ax.annotate(f"{r.cqr_width_pct:.0f}%", (r.H, r.cqr_width_pct),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xlabel("提前量 (小時)"); ax.set_ylabel("80% 區間平均寬度（% 額定）")
    ax.set_xticks(m.H)
    ax.set_title("圖 C　區間寬度 vs 時程\n"
                 "H1–H6 越遠越寬（越不確定）；H24 因『整日平均』被平滑反而較窄", fontsize=12)
    fig.savefig(FIG / "qiv_C_sharpness.png"); plt.close(fig)


if __name__ == "__main__":
    fig_coverage(); fig_example(); fig_sharpness()
    print("✓ 3 張窗口平均區間圖完成")
