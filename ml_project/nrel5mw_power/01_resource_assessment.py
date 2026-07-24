#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_resource_assessment.py — NREL 5MW 虛擬風場資源評估 + 功率曲線對照
====================================================================

用 NREL 5MW（絕對 MW）把 BSMI 測風塔算成虛擬風場，產出：
  1. 全期資源評估：容量因數、平均出力、等效滿載時數、年發電量
  2. 月別 / 小時別 出力形態
  3. 功率曲線對照：官方 NREL 5MW  vs  正規化 8MW 代表曲線

輸出
  results/resource_summary.csv       全期關鍵指標
  results/monthly_profile.csv        月別出力
  results/hourly_profile.csv         小時別出力
  results/curve_comparison.csv       兩條曲線的容量因數對照
  results/figures/fig1_power_curves.png
  results/figures/fig2_monthly_cf.png
  results/figures/fig3_hourly_cf.png
  results/figures/fig4_cf_distribution.png
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nrel_5mw as N

N.setup_cjk_font()

MONTH_NAMES = ["1月", "2月", "3月", "4月", "5月", "6月",
               "7月", "8月", "9月", "10月", "11月", "12月"]


def main():
    print("[1/4] 載入資料並以 NREL 5MW 換算虛擬出力（絕對 MW）...")
    d = N.load_power_table()
    n = len(d)
    cf = d["P_cf"].mean()
    p_mean_mw = d["P_mw"].mean()
    flh = cf * 8760.0                       # 等效滿載時數 (h/年)
    aep_mwh = p_mean_mw * 8760.0            # 單機年發電量 (MWh)

    # 運轉區間佔比
    below_cutin = (d["WS_100_mean"] < 3.0).mean()
    rated_frac = (d["P_mw"] >= N.RATED_MW - 1e-6).mean()
    zero_frac = (d["P_mw"] <= 1e-6).mean()

    summary = {
        "有效樣本數": n,
        "資料期間起": str(d["ts"].min()),
        "資料期間迄": str(d["ts"].max()),
        "平均出力_MW": round(p_mean_mw, 4),
        "容量因數_%": round(100 * cf, 2),
        "等效滿載時數_h/年": round(flh, 1),
        "單機年發電量_MWh": round(aep_mwh, 1),
        "出力為0佔比_%": round(100 * zero_frac, 2),
        "低於切入風速佔比_%": round(100 * below_cutin, 2),
        "滿載(5MW)佔比_%": round(100 * rated_frac, 2),
        "對照8MW正規化容量因數_%": round(100 * d["P_rep8_cf"].mean(), 2),
    }
    df_sum = pd.DataFrame([summary]).T.rename(columns={0: "值"})
    df_sum.to_csv(N.RESULTS_DIR / "resource_summary.csv", encoding="utf-8-sig")
    print(df_sum.to_string())

    # --- 月別 / 小時別 ---
    print("\n[2/4] 計算月別與小時別出力形態...")
    monthly = d.groupby("month").agg(
        平均出力_MW=("P_mw", "mean"),
        容量因數_pct=("P_cf", lambda x: 100 * x.mean()),
        平均風速_ms=("WS_100_mean", "mean"),
    ).reset_index()
    monthly["月"] = monthly["month"].map(lambda m: MONTH_NAMES[m - 1])
    monthly.to_csv(N.RESULTS_DIR / "monthly_profile.csv", index=False, encoding="utf-8-sig")

    hourly = d.groupby("hour").agg(
        平均出力_MW=("P_mw", "mean"),
        容量因數_pct=("P_cf", lambda x: 100 * x.mean()),
    ).reset_index()
    hourly.to_csv(N.RESULTS_DIR / "hourly_profile.csv", index=False, encoding="utf-8-sig")

    # --- 功率曲線對照 ---
    print("[3/4] 建立 NREL 5MW vs 8MW 正規化曲線對照...")
    u = np.linspace(0, 27, 271)
    rho_std = np.full_like(u, N.RHO_REF)
    nrel_cf = N.nrel_5mw_power_kw(u, rho_std) / N.RATED_KW      # 0–1
    rep8_cf = N.rep8mw_cf(u, rho_std)                            # 0–1
    curve = pd.DataFrame({"風速_ms": u, "NREL5MW_CF": nrel_cf, "REP8MW_CF": rep8_cf})
    curve.to_csv(N.RESULTS_DIR / "curve_comparison.csv", index=False, encoding="utf-8-sig")

    # --- 繪圖 ---
    print("[4/4] 繪製圖表...")

    # 圖1：功率曲線對照
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(u, nrel_cf, lw=2.4, color="#1f77b4", label="NREL 5MW（官方查表）")
    ax.plot(u, rep8_cf, lw=2.0, color="#ff7f0e", ls="--", label="8MW 正規化代表曲線")
    ax.axvline(3.0, color="grey", ls=":", alpha=0.6)
    ax.axvline(25.0, color="grey", ls=":", alpha=0.6)
    ax.text(3.0, 1.03, "切入 3", ha="center", fontsize=9, color="grey")
    ax.text(25.0, 1.03, "切出 25", ha="center", fontsize=9, color="grey")
    ax.set_xlabel("等效風速 (m/s)")
    ax.set_ylabel("正規化出力 / 額定 (0–1)")
    ax.set_title("功率曲線對照：NREL 5MW vs 8MW 正規化代表曲線")
    ax.set_ylim(-0.03, 1.1)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(N.FIG_DIR / "fig1_power_curves.png", dpi=140)
    plt.close(fig)

    # 圖2：月別容量因數
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    bars = ax1.bar(monthly["月"], monthly["容量因數_pct"], color="#2ca02c", alpha=0.8)
    ax1.set_ylabel("容量因數 (%)", color="#2ca02c")
    ax1.set_title("NREL 5MW 虛擬風場 — 月別容量因數與平均風速")
    ax1.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, monthly["容量因數_pct"]):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.0f}", ha="center", fontsize=8)
    ax2 = ax1.twinx()
    ax2.plot(monthly["月"], monthly["平均風速_ms"], "o-", color="#d62728", lw=2, label="平均風速")
    ax2.set_ylabel("100m 平均風速 (m/s)", color="#d62728")
    fig.tight_layout()
    fig.savefig(N.FIG_DIR / "fig2_monthly_cf.png", dpi=140)
    plt.close(fig)

    # 圖3：小時別容量因數
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(hourly["hour"], hourly["容量因數_pct"], "o-", color="#9467bd", lw=2)
    ax.set_xlabel("小時 (當地時間)")
    ax.set_ylabel("容量因數 (%)")
    ax.set_title("NREL 5MW 虛擬風場 — 日內出力形態（全期平均）")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(N.FIG_DIR / "fig3_hourly_cf.png", dpi=140)
    plt.close(fig)

    # 圖4：出力分布直方圖
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(d["P_mw"], bins=50, color="#17becf", alpha=0.85, edgecolor="white")
    ax.axvline(p_mean_mw, color="#d62728", ls="--", lw=2, label=f"平均 {p_mean_mw:.2f} MW")
    ax.set_xlabel("瞬時出力 (MW)")
    ax.set_ylabel("10 分鐘格數")
    ax.set_title(f"NREL 5MW 出力分布（容量因數 {100*cf:.1f}%，滿載佔 {100*rated_frac:.0f}%）")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(N.FIG_DIR / "fig4_cf_distribution.png", dpi=140)
    plt.close(fig)

    print("\n完成。輸出於 results/ 與 results/figures/")
    print(f"  → 容量因數 {100*cf:.1f}%｜等效滿載 {flh:.0f} h/年｜單機年發電 {aep_mwh:,.0f} MWh")


if __name__ == "__main__":
    main()
