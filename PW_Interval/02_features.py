#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2 —— 特徵提取 / 轉換 + 區間累積能量標的建立 (Interval Target Engineering)

核心修改：
  將 PW 原本的單點瞬時標的 y(t+H) 改為 未來時間區間 [t+1 .. t+k] 內：
  - y_power_h : 未來 H 小時區間累積總發電能量 / 平均出力 E_[t, t+H]
  - y_ws100_h : 未來 H 小時區間平均 100m 風速 U_[t, t+H]
  - m_h       : 區間內全段連續且無缺漏有效性遮罩

輸出：data/features.parquet
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import config as C

def roll_slope_vec(y: pd.Series, w: int) -> pd.Series:
    """向量化滾動線性斜率。"""
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
    print("PW_Interval Stage 2 —— 特徵提取 + 區間發電能量標的建立")
    print("="*70)

    if not os.path.exists(C.CLEAN_PARQUET):
        raise FileNotFoundError(f"找不到 {C.CLEAN_PARQUET}，請先執行 01_load_validate.py")

    df = pd.read_parquet(C.CLEAN_PARQUET).set_index("ts").sort_index()
    print(f"讀入 clean 網格資料：{df.shape}")

    ok = df["is_ok"].fillna(False).values
    base = df.copy()
    num_cols = [c for c in df.columns if c != "is_ok"]
    base.loc[~ok, num_cols] = np.nan

    step = C.STEP_MIN
    def steps(minutes): return minutes // step

    feat = pd.DataFrame(index=base.index)

    # 1) 當前氣象態特徵
    cur_cols = ["WS_100_mean","WS_100E_mean","WS_100W_mean","WS_69W_mean","WS_38W_mean",
                "WS_100E_std","WS_100E_ti","WS_100E_gust_factor",
                "WD_97_sin","WD_97_cos","WD_97_sigma",
                "AT_95_mean","RH_95_mean","BP_93_mean","shear_alpha","air_density"]
    cur_cols = [c for c in cur_cols if c in base.columns]
    for c in cur_cols:
        feat[c] = base[c]

    feat["P_now"] = C.virtual_power(base["WS_100_mean"], base["air_density"])
    ws = base["WS_100_mean"]

    # 2) 滯後 lag
    for m in [10, 20, 30, 60, 120, 180]:
        feat[f"ws_lag{m}"] = ws.shift(steps(m))

    # 3) 滾動統計
    for win_m in [60, 180, 360]:
        w = steps(win_m)
        r = ws.rolling(w, min_periods=max(3, w//2))
        feat[f"ws_rmean_{win_m}"] = r.mean()
        feat[f"ws_rstd_{win_m}"]  = r.std()
        feat[f"ws_rmin_{win_m}"]  = r.min()
        feat[f"ws_rmax_{win_m}"]  = r.max()
        feat[f"ws_slope_{win_m}"] = roll_slope_vec(ws, w)

    # 4) 變化率
    feat["ws_diff_60"]  = ws - ws.shift(steps(60))
    feat["ws_diff_180"] = ws - ws.shift(steps(180))

    # 5) 時間週期編碼
    hour = base.index.hour + base.index.minute / 60.0
    doy  = base.index.dayofyear
    feat["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    feat["doy_sin"]  = np.sin(2 * np.pi * doy / 365.25)
    feat["doy_cos"]  = np.cos(2 * np.pi * doy / 365.25)

    # 6) 【核心修改】區間標的 (Interval Energy Targets - 100% 無洩漏)
    # 計算未來 k 個步階 [t+1 .. t+k] 之累積平均發電能量與風速 (先 shift(-1) 排除當前時步 t)
    P_now_full = C.virtual_power(base["WS_100_mean"], base["air_density"])
    P_future = pd.Series(P_now_full, index=base.index).shift(-1)
    ws_future = ws.shift(-1)

    notok = (~ok).astype(np.int64)
    cs = np.concatenate([[0], np.cumsum(notok)])
    N = len(ok)

    for h in C.HORIZONS_H:
        k = C.HORIZON_STEPS[h]

        # 反轉滾動以求未來 [t+1 .. t+k] 步階之平均值 (等於區間累積能量)
        feat[f"y_power_{h}"] = P_future.iloc[::-1].rolling(k).mean().iloc[::-1]
        feat[f"y_ws100_{h}"] = ws_future.iloc[::-1].rolling(k).mean().iloc[::-1]

        # 嚴格無洩漏連續性遮罩 m_h：要求 [t+1, t+k] 全段皆為 is_ok
        m = np.zeros(N, dtype=bool)
        valid_end = N - k
        idx = np.arange(valid_end)
        bad = cs[idx + k + 1] - cs[idx + 1]  # 注意端點：[t+1..t+k]
        m[idx] = (bad == 0)
        feat[f"m_{h}"] = m

    feat["is_ok"]  = ok
    feat["year"]   = base.index.year
    feat["month"]  = base.index.month
    feat["hour_i"] = base.index.hour

    os.makedirs(C.DATA_DIR, exist_ok=True)
    feat.reset_index().to_parquet(C.FEAT_PARQUET, index=False)

    n_feat = len([c for c in feat.columns if not (c.startswith("y_") or c.startswith("m_")
                  or c in ("is_ok","year","month","hour_i"))])
    print(f"特徵欄位總數：{n_feat}")
    print(f"輸出：{C.FEAT_PARQUET}  shape={feat.shape}")

    for h in C.HORIZONS_H:
        valid = feat["is_ok"] & feat[f"m_{h}"] & feat[f"y_ws100_{h}"].notna() & feat[cur_cols].notna().all(axis=1)
        print(f"  區間 H={h}h 可用樣本數：{int(valid.sum()):,}")

    print("\nStage 2 完成。")

if __name__ == "__main__":
    main()
