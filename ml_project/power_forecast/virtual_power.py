#!/usr/bin/env python3
"""
虛擬風場功率模組 —— 把 BSMI 測風塔的風速轉成「虛擬風機出力」。

因為塔本身沒有發電量資料，我們用一條代表性的功率曲線，把 100 m 風速
換算成出力（額定的百分比）。這樣塔就變成一座「虛擬風場」，可用來做
資源評估與超短期發電預測。

重要聲明
--------
  * 出力是「風速 × 功率曲線」的物理推算，不是實測。絕對 MW 數值取決於
    機型；本模組回傳「正規化出力」（0–1，即容量因數的瞬時值），與額定
    功率無關，因此結論（容量因數、季節形態、預測技術得分）對機型穩健。
  * 已做 IEC 空氣密度修正（冬夏密度差約 5%，直接影響出力）。
  * 假設塔高 100 m ≈ 輪轂高度。若要換到更高輪轂，可用實測風切指數
    shear_alpha 外推（見 hub_extrapolate）。
"""
from __future__ import annotations
import numpy as np
import pandas as pd

RHO_REF = 1.225   # kg/m^3，海平面標準空氣密度

# 代表性現代離岸風機（約 8 MW 級）正規化功率曲線：出力 / 額定
# S 形，切入 3、額定約 12、切出 25 m/s。數值為典型值，非特定機型。
_PC_U = np.array([0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                  14, 20, 24, 25, 25.001, 40])
_PC_P = np.array([0, 0, 0.010, 0.040, 0.100, 0.200, 0.330, 0.500, 0.680,
                  0.830, 0.940, 0.990, 1.000, 1.000, 1.000, 1.000, 1.000,
                  0.0, 0.0])

# 敏感度用的兩條對照曲線（低風速高 CF 機型 vs 高風速機型）
_ALT_CURVES = {
    "低風速機型(額定10)": (10.0,),
    "標準機型(額定12)": (12.0,),
    "高風速機型(額定14)": (14.0,),
}


def power_curve(u: np.ndarray, rated_u: float | None = None) -> np.ndarray:
    """正規化出力（0–1）。rated_u 若給定，用簡化立方模型換算到該額定風速。"""
    u = np.asarray(u, dtype="float64")
    if rated_u is None:
        return np.interp(u, _PC_U, _PC_P, left=0.0, right=0.0)
    # 簡化立方模型（敏感度分析用）：切入 3、切出 25
    cin, cout = 3.0, 25.0
    p = np.where(u < cin, 0.0,
        np.where(u < rated_u, (u**3 - cin**3) / (rated_u**3 - cin**3),
        np.where(u < cout, 1.0, 0.0)))
    return np.clip(p, 0.0, 1.0)


def density_correct(u: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """IEC 空氣密度修正後的等效風速。"""
    return np.asarray(u) * (np.asarray(rho) / RHO_REF) ** (1.0 / 3.0)


def hub_extrapolate(u100: np.ndarray, alpha: np.ndarray, hub: float = 100.0,
                    fallback_alpha: float = 0.14) -> np.ndarray:
    """用冪次律風切把 100 m 風速外推到輪轂高度 hub（m）。
    
    若 alpha 為 NaN 或超出物理合理範圍 (-0.2 ~ 0.8)，自動啟動 fallback_alpha (預設 0.14)。
    """
    u100 = np.asarray(u100, dtype="float64")
    if hub == 100.0:
        return u100
    alpha_clean = np.asarray(alpha, dtype="float64").copy()
    invalid = np.isnan(alpha_clean) | (alpha_clean < -0.2) | (alpha_clean > 0.8)
    alpha_clean[invalid] = fallback_alpha
    return u100 * (hub / 100.0) ** alpha_clean


def virtual_power(df: pd.DataFrame, rated_u: float | None = None,
                  hub: float = 100.0, fallback_alpha: float = 0.14) -> np.ndarray:
    """從 10 分鐘平均表算正規化虛擬出力。

    需要欄位：WS_100_mean, air_density；hub!=100 時另需 shear_alpha。
    """
    u = df["WS_100_mean"].to_numpy(dtype="float64")
    if hub != 100.0 and "shear_alpha" in df:
        u = hub_extrapolate(u, df["shear_alpha"].to_numpy(), hub, fallback_alpha=fallback_alpha)
    u_eff = density_correct(u, df["air_density"].to_numpy())
    return power_curve(u_eff, rated_u=rated_u)


def load_power_table(parquet_path: str, hub: float = 100.0) -> pd.DataFrame:
    """讀 10 分鐘表、篩有效、加上正規化出力欄 P（0–1）。"""
    d = pd.read_parquet(parquet_path)
    d = d[d["is_valid"]].copy()
    d["P"] = virtual_power(d, hub=hub)
    d["month"] = d["ts"].dt.month
    d["hour"] = d["ts"].dt.hour
    d["year"] = d["ts"].dt.year
    return d


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else "../data/BSMI_10min.parquet"
    d = load_power_table(p)
    cf = d.P.mean()
    print(f"樣本 {len(d):,}｜正規化容量因數 = {100*cf:.1f}%｜等效滿載 {cf*8760:.0f} h/年")
    print("功率曲線敏感度（不同額定風速機型）：")
    for name, (ru,) in _ALT_CURVES.items():
        pw = power_curve(density_correct(d.WS_100_mean.to_numpy(), d.air_density.to_numpy()), ru)
        print(f"  {name:16s} CF = {100*pw.mean():.1f}%")
