#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nrel_5mw.py — 可重用的 NREL 5MW 虛擬風場出力模組（絕對 MW 版）
============================================================

把 BSMI 測風塔的 100 m 風速，用 **NREL 5MW Reference Offshore Wind Turbine**
的官方功率曲線換算成「虛擬風機出力」（絕對值，單位 MW，額定 5.0 MW）。

為什麼獨立成一個模組
--------------------
專案裡原本有兩套彼此不一致的功率曲線：
  * train_power_forecast_pipeline.py：NREL 5MW，用立方近似 3.704·v³（絕對 kW）
  * power_forecast / interval / advanced：一條「正規化 8MW 代表曲線」（0–1）

本模組把 **官方 NREL 5MW 查表曲線** 抽成單一可信來源，讓任何子專案
`from nrel_5mw import nrel_5mw_power_mw` 就能得到一致的絕對 MW 出力，
並附一條 8MW 正規化曲線供對照分析。

重要聲明
--------
  * 出力是「風速 × 功率曲線」的物理推算，非實測（測風塔無 SCADA）。
  * 已做 IEC 61400-12-1 空氣密度修正：v_eff = v · (ρ/1.225)^(1/3)。
  * 假設塔高 100 m ≈ 輪轂高度；要換更高輪轂可用 hub_extrapolate。
  * 25 m/s 為理想硬切出（真實風機有遲滯，此處不建模）。
"""
from __future__ import annotations
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# --------------------------------------------------------------------------
# 路徑
# --------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent                 # ml_project/
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = HERE / "results"
FIG_DIR = RESULTS_DIR / "figures"
for _d in (RESULTS_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RHO_REF = 1.225      # kg/m^3，海平面標準空氣密度
RATED_KW = 5000.0    # NREL 5MW 額定
RATED_MW = 5.0

# --------------------------------------------------------------------------
# NREL 5MW 官方功率曲線 (kW) — 查表 + 線性內插
# 來源同 advanced_forecasting/common.py，為官方參考機型曲線。
# 切入 3.0、額定 11.4、切出 25.0 m/s。
# --------------------------------------------------------------------------
NREL_5MW_WS = np.array([
    0.0, 2.9, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5,
    7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.4,
    12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0,
    22.0, 23.0, 24.0, 25.0, 25.1,
])
NREL_5MW_KW = np.array([
    0.0, 0.0, 27.3, 56.6, 93.6, 144.5, 208.3, 289.7, 399.6, 518.8,
    655.1, 811.7, 1007.0, 1211.0, 1458.0, 1726.0, 1984.0, 2267.0, 2587.0, 5000.0,
    5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0,
    5000.0, 5000.0, 5000.0, 5000.0, 0.0,
])

# --------------------------------------------------------------------------
# 對照用：正規化 8MW 代表曲線 (0–1)，即專案其他子模組原本用的那條。
# S 形，切入 3、額定約 12、切出 25 m/s。純供「機型敏感度」對照。
# --------------------------------------------------------------------------
REP8_U = np.array([0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                   14, 20, 24, 25, 25.001, 40])
REP8_P = np.array([0, 0, 0.010, 0.040, 0.100, 0.200, 0.330, 0.500, 0.680,
                   0.830, 0.940, 0.990, 1.000, 1.000, 1.000, 1.000, 1.000,
                   0.0, 0.0])


# --------------------------------------------------------------------------
# 物理換算
# --------------------------------------------------------------------------
def density_correct(u, rho) -> np.ndarray:
    """IEC 61400-12-1 空氣密度修正後的等效風速。"""
    return np.asarray(u, dtype="float64") * (np.asarray(rho, dtype="float64") / RHO_REF) ** (1.0 / 3.0)


def hub_extrapolate(u100, alpha, hub: float = 100.0) -> np.ndarray:
    """用冪次律風切把 100 m 風速外推到輪轂高度 hub（m）。hub=100 時不變。"""
    return np.asarray(u100, dtype="float64") * (hub / 100.0) ** np.asarray(alpha, dtype="float64")


def nrel_5mw_power_kw(ws100, air_density, hub: float = 100.0, alpha=None) -> np.ndarray:
    """NREL 5MW 出力（kW）。含空氣密度修正；hub!=100 時需給 alpha 做外推。"""
    u = np.asarray(ws100, dtype="float64")
    if hub != 100.0:
        if alpha is None:
            raise ValueError("hub != 100 時需提供 shear_alpha 才能外推到輪轂高度")
        u = hub_extrapolate(u, alpha, hub)
    v_eff = density_correct(u, air_density)
    return np.interp(v_eff, NREL_5MW_WS, NREL_5MW_KW)


def nrel_5mw_power_mw(ws100, air_density, hub: float = 100.0, alpha=None) -> np.ndarray:
    """NREL 5MW 出力（MW，0–5）。"""
    return nrel_5mw_power_kw(ws100, air_density, hub=hub, alpha=alpha) / 1000.0


def rep8mw_cf(ws100, air_density) -> np.ndarray:
    """對照用 8MW 正規化出力（0–1，容量因數瞬時值），含密度修正。"""
    v_eff = density_correct(ws100, air_density)
    return np.interp(v_eff, REP8_U, REP8_P, left=0.0, right=0.0)


# --------------------------------------------------------------------------
# 資料載入
# --------------------------------------------------------------------------
def load_power_table(parquet_path: str | Path | None = None, hub: float = 100.0) -> pd.DataFrame:
    """讀 10 分鐘表、篩有效格，加上 NREL 5MW 出力欄。

    產生欄位：
      P_mw      NREL 5MW 出力（MW，0–5）
      P_cf      正規化容量因數（0–1）＝ P_mw / 5.0
      P_rep8_cf 對照 8MW 正規化出力（0–1）
      month/hour/year
    """
    if parquet_path is None:
        parquet_path = DATA_DIR / "BSMI_10min.parquet"
    d = pd.read_parquet(parquet_path)
    d = d[d["is_valid"]].copy()
    alpha = d["shear_alpha"].to_numpy() if hub != 100.0 else None
    d["P_mw"] = nrel_5mw_power_mw(d["WS_100_mean"].to_numpy(), d["air_density"].to_numpy(),
                                  hub=hub, alpha=alpha)
    d["P_cf"] = d["P_mw"] / RATED_MW
    d["P_rep8_cf"] = rep8mw_cf(d["WS_100_mean"].to_numpy(), d["air_density"].to_numpy())
    d["month"] = d["ts"].dt.month
    d["hour"] = d["ts"].dt.hour
    d["year"] = d["ts"].dt.year
    return d


# --------------------------------------------------------------------------
# CJK 字型（供繪圖腳本共用）
# --------------------------------------------------------------------------
def setup_cjk_font():
    cands = (glob.glob("/usr/share/fonts/**/NotoSansCJK*.ttc", recursive=True)
             + glob.glob("C:/Windows/Fonts/msjh*.ttc")
             + glob.glob("C:/Windows/Fonts/msyh*.ttc"))
    for p in cands:
        try:
            fm.fontManager.addfont(p)
        except Exception:
            pass
    names = {f.name for f in fm.fontManager.ttflist}
    for n in ["Microsoft JhengHei", "Microsoft YaHei", "Noto Sans CJK TC", "DejaVu Sans"]:
        if n in names:
            plt.rcParams["font.family"] = [n, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


if __name__ == "__main__":
    d = load_power_table()
    cf = d["P_cf"].mean()
    print(f"NREL 5MW 虛擬風場｜有效樣本 {len(d):,}")
    print(f"  平均出力 {d['P_mw'].mean():.3f} MW｜容量因數 {100*cf:.1f}%｜等效滿載 {cf*8760:.0f} h/年")
    print(f"  對照 8MW 正規化曲線容量因數 {100*d['P_rep8_cf'].mean():.1f}%")
