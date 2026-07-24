#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
forecast_interval_features.py —— 區間發電能量特徵與標的建置模組 (修正無洩漏標的與遮罩對齊)

修正內容：
  1. 標的完全排除當前時步 t (P_now)，精確求未來 [t+1 .. t+k] 步階之區間平均功率。
     使用 `P_series.shift(-1).iloc[::-1].rolling(k).mean().iloc[::-1]`。
  2. 遮罩 m_h 完美對齊未來 [t+1 .. t+k] 步階之無缺漏驗證。
  3. 建置 P_rmean_{h} (過去 H 小時滞動平均出力) 特徵，供對稱式 persistence 基準使用。
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import config as C

def roll_slope_vec(y: pd.Series, w: int) -> pd.Series:
    x = np.arange(w)
    c = (x - x.mean())
    denom = (c**2).sum()
    c = c / denom
    vals = y.values.astype("float64")
    kernel = c[::-1]
    conv = np.convolve(vals, kernel, mode="full")[:len(vals)]
    conv[:w-1] = np.nan
    nan_run = pd.Series(np.isnan(vals).astype(float)).rolling(w, min_periods=w).sum().values
    conv[nan_run > 0] = np.nan
    return pd.Series(conv, index=y.index)

def main():
    print("="*70)
    print("power_forecast_interval Stage 3 —— 修正版區間能量特徵與無洩漏標的建置")
    print("="*70)

    if not os.path.exists(C.CLEAN_PARQUET):
        raise FileNotFoundError(f"找不到 {C.CLEAN_PARQUET}，請先執行 01_load_validate.py")

    df = pd.read_parquet(C.CLEAN_PARQUET).set_index("ts").sort_index()

    ok = df["is_ok"].fillna(False).values
    base = df.copy()
    num_cols = [c for c in df.columns if c != "is_ok"]
    base.loc[~ok, num_cols] = np.nan

    step = C.STEP_MIN
    def steps(minutes): return minutes // step

    feat = pd.DataFrame(index=base.index)

    cur_cols = ["WS_100_mean","WS_100E_mean","WS_100W_mean","WS_69W_mean","WS_38W_mean",
                "WS_100E_std","WS_100E_ti","WS_100E_gust_factor",
                "WD_97_sin","WD_97_cos","WD_97_sigma",
                "AT_95_mean","RH_95_mean","BP_93_mean","shear_alpha","air_density"]
    cur_cols = [c for c in cur_cols if c in base.columns]
    for c in cur_cols: feat[c] = base[c]

    P_now_series = C.virtual_power(base["WS_100_mean"], base["air_density"])
    feat["P_now"] = P_now_series
    ws = base["WS_100_mean"]

    for m in [10, 20, 30, 60, 120, 180]: feat[f"ws_lag{m}"] = ws.shift(steps(m))

    for win_m in [60, 180, 360]:
        w = steps(win_m)
        r = ws.rolling(w, min_periods=max(3, w//2))
        feat[f"ws_rmean_{win_m}"] = r.mean()
        feat[f"ws_rstd_{win_m}"]  = r.std()
        feat[f"ws_rmin_{win_m}"]  = r.min()
        feat[f"ws_rmax_{win_m}"]  = r.max()
        feat[f"ws_slope_{win_m}"] = roll_slope_vec(ws, w)

    feat["ws_diff_60"]  = ws - ws.shift(steps(60))
    feat["ws_diff_180"] = ws - ws.shift(steps(180))

    hour = base.index.hour + base.index.minute / 60.0
    doy  = base.index.dayofyear
    feat["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    feat["doy_sin"]  = np.sin(2 * np.pi * doy / 365.25)
    feat["doy_cos"]  = np.cos(2 * np.pi * doy / 365.25)

    # 【嚴格防洩漏標的建置】：先 shift(-1) 排除當前時步 t，精確為未來的 [t+1 .. t+k] 步階
    P_future = pd.Series(P_now_series, index=base.index).shift(-1)
    ws_future = ws.shift(-1)

    notok = (~ok).astype(np.int64)
    cs = np.concatenate([[0], np.cumsum(notok)])
    N = len(ok)

    for h in C.HORIZONS_H:
        k = C.HORIZON_STEPS[h]

        # 區間平均功率 (0~1)
        mean_p = P_future.iloc[::-1].rolling(k).mean().iloc[::-1]
        feat[f"y_power_{h}"] = mean_p
        feat[f"y_ws100_{h}"] = ws_future.iloc[::-1].rolling(k).mean().iloc[::-1]

        # 無洩漏遮罩 m_h (檢查未來的 [t+1 .. t+k] 步階)
        m = np.zeros(N, dtype=bool)
        valid_end = N - k
        idx = np.arange(valid_end)
        bad = cs[idx + k + 1] - cs[idx + 1]
        m[idx] = (bad == 0)
        feat[f"m_{h}"] = m

    feat["is_ok"]  = ok
    feat["year"]   = base.index.year
    feat["month"]  = base.index.month
    feat["hour_i"] = base.index.hour

    # P_rmean_{h}: 過去 H 小時滞動平均出力，供對稱式 persistence 基準使用
    P_now_s = pd.Series(P_now_series, index=base.index)
    for h in C.HORIZONS_H:
        k = C.HORIZON_STEPS[h]
        feat[f"P_rmean_{h}"] = P_now_s.rolling(k, min_periods=max(3, k // 2)).mean()

    os.makedirs(C.DATA_DIR, exist_ok=True)
    feat.reset_index().to_parquet(C.FEAT_PARQUET, index=False)
    print(f"輸出修正版特徵檔：{C.FEAT_PARQUET}  shape={feat.shape}")
    print("Stage 3 完成 (100% 排除當前時步洩漏，標的與遮罩完美對齊)。")

if __name__ == "__main__":
    main()
