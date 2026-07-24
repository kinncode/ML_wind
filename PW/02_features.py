#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2 —— 特徵提取 / 轉換 + 建立多時程目標

特徵（皆只用 t 時刻(含)以前資訊，避免洩漏）：
  * 當前氣象態：WS_100_mean、四高度風速、風切 shear_alpha、TI、陣風因子、
    風向 sin/cos、風向 sigma、氣溫/濕度/氣壓、空氣密度
  * 滯後 lag：WS_100_mean 在 t-10min, -20, -30, -60, -120, -180min
  * 滾動統計（過去 1h/3h/6h）：均值、標準差、趨勢(線性斜率)、最小/最大
  * 變化率：近 1h、3h 的風速差分
  * 時間週期編碼：小時、年內日 的 sin/cos（季節與日夜）
  * 當前正規化出力 P（由功率曲線）

目標（t+H，H ∈ {1,3,6}h）：
  * y_ws100_h   : 100 m 風速
  * y_power_h   : 正規化虛擬出力 P

洩漏防護：目標由未來值 shift 取得；只有當 t..t+H 全段連續且 is_ok 才保留樣本。
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import config as C

def roll_slope_vec(y: pd.Series, w: int) -> pd.Series:
    """向量化滾動線性斜率（每步）：對固定窗 x=0..w-1，
    slope = Σ(x-x̄)(y-ȳ) / Σ(x-x̄)² = Σ c_i · y_i，c 為固定權重（卷積）。"""
    x = np.arange(w)
    c = (x - x.mean())
    denom = (c**2).sum()
    c = c / denom                      # 固定權重
    vals = y.values.astype("float64")
    # 卷積：對每個結束於 t 的窗做加權和；有 NaN 的窗設 NaN
    kernel = c[::-1]
    conv = np.convolve(vals, kernel, mode="full")[:len(vals)]
    conv[:w-1] = np.nan
    # 標記含 NaN 的窗
    nan_run = pd.Series(np.isnan(vals).astype(float)).rolling(w, min_periods=w).sum().values
    conv[nan_run > 0] = np.nan
    return pd.Series(conv, index=y.index)

def main():
    df = pd.read_parquet(C.CLEAN_PARQUET).set_index("ts").sort_index()
    print(f"讀入 clean：{df.shape}")

    # 只在 is_ok 的觀測上取值；其餘設 NaN（保留連續網格）
    ok = df["is_ok"].fillna(False).values
    base = df.copy()
    num_cols = [c for c in df.columns if c not in ("is_ok",)]
    base.loc[~ok, num_cols] = np.nan

    step = C.STEP_MIN
    def steps(minutes): return minutes // step

    feat = pd.DataFrame(index=base.index)

    # --- 當前態特徵 ---
    cur_cols = ["WS_100_mean","WS_100E_mean","WS_100W_mean","WS_69W_mean","WS_38W_mean",
                "WS_100E_std","WS_100E_ti","WS_100E_gust_factor",
                "WD_97_sin","WD_97_cos","WD_97_sigma",
                "AT_95_mean","RH_95_mean","BP_93_mean","shear_alpha","air_density"]
    cur_cols = [c for c in cur_cols if c in base.columns]
    for c in cur_cols:
        feat[c] = base[c]

    # 當前正規化出力
    feat["P_now"] = C.virtual_power(base["WS_100_mean"], base["air_density"])

    ws = base["WS_100_mean"]

    # --- 滯後 ---
    for m in [10, 20, 30, 60, 120, 180]:
        feat[f"ws_lag{m}"] = ws.shift(steps(m))

    # --- 滾動統計（過去視窗，closed='left' 不含當前避免與當前態重複）---
    for win_m in [60, 180, 360]:
        w = steps(win_m)
        r = ws.rolling(w, min_periods=max(3, w//2))
        feat[f"ws_rmean_{win_m}"] = r.mean()
        feat[f"ws_rstd_{win_m}"]  = r.std()
        feat[f"ws_rmin_{win_m}"]  = r.min()
        feat[f"ws_rmax_{win_m}"]  = r.max()
        feat[f"ws_slope_{win_m}"] = roll_slope_vec(ws, w)

    # --- 變化率 ---
    feat["ws_diff_60"]  = ws - ws.shift(steps(60))
    feat["ws_diff_180"] = ws - ws.shift(steps(180))

    # --- 時間週期編碼 ---
    hour = base.index.hour + base.index.minute/60.0
    doy  = base.index.dayofyear
    feat["hour_sin"] = np.sin(2*np.pi*hour/24)
    feat["hour_cos"] = np.cos(2*np.pi*hour/24)
    feat["doy_sin"]  = np.sin(2*np.pi*doy/365.25)
    feat["doy_cos"]  = np.cos(2*np.pi*doy/365.25)

    # --- 目標（未來值）---
    P_now_full = C.virtual_power(base["WS_100_mean"], base["air_density"])
    P_series = pd.Series(P_now_full, index=base.index)
    # 向量化「未來 t..t+k 全段有效」：用 not-ok 的累積和判斷窗內是否有無效點
    notok = (~ok).astype(np.int64)
    cs = np.concatenate([[0], np.cumsum(notok)])   # cs[i] = sum(notok[:i])
    N = len(ok)
    for h in C.HORIZONS_H:
        k = C.HORIZON_STEPS[h]
        feat[f"y_ws100_{h}"]  = ws.shift(-k)
        feat[f"y_power_{h}"]  = P_series.shift(-k)
        # 窗 [t, t+k]（含端點，共 k+1 點）內無效點數量 = cs[t+k+1]-cs[t]
        m = np.zeros(N, dtype=bool)
        valid_end = N - k
        idx = np.arange(valid_end)
        bad = cs[idx + k + 1] - cs[idx]
        m[idx] = (bad == 0)
        feat[f"m_{h}"] = m

    feat["is_ok"] = ok
    feat["year"]  = base.index.year
    feat["month"] = base.index.month
    feat["hour_i"]= base.index.hour

    os.makedirs(C.DATA_DIR, exist_ok=True)
    feat.reset_index().to_parquet(C.FEAT_PARQUET, index=False)
    n_feat = len([c for c in feat.columns if not (c.startswith("y_") or c.startswith("m_")
                  or c in ("is_ok","year","month","hour_i"))])
    print(f"特徵欄位數：{n_feat}")
    print(f"輸出：{C.FEAT_PARQUET}  shape={feat.shape}")
    # 各目標可用樣本數
    for h in C.HORIZONS_H:
        valid = feat["is_ok"] & feat[f"m_{h}"] & feat[f"y_ws100_{h}"].notna() & feat[cur_cols].notna().all(axis=1)
        print(f"  H={h}h 可用樣本：{int(valid.sum()):,}")

if __name__ == "__main__":
    main()
