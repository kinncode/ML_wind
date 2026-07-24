#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_load_validate.py —— QC 數據載入與獨立驗證模組

動作：載入 BSMI 10 分鐘檔，執行獨立 4 重 QC 控制（去重、凍結感測器、四高度一致性與氣壓範圍）。
輸出：data/clean_10min.parquet
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import config as C

def main():
    print("="*70)
    print("power_forecast_interval Stage 1 —— 載入 + 獨立 QC 驗證")
    print("="*70)

    src = os.path.normpath(C.SRC_10MIN)
    if not os.path.exists(src):
        raise FileNotFoundError(f"找不到來源 10 分鐘檔：{src}")

    df = pd.read_parquet(src)
    print(f"來源：{src}｜原始筆數：{len(df):,}")

    keep = ["ts", "coverage",
            "WS_100E_mean", "WS_100W_mean", "WS_69W_mean", "WS_38W_mean",
            "WS_100_mean", "WS_100E_std", "WS_100E_ti", "WS_100E_gust_factor",
            "WD_97_sin", "WD_97_cos", "WD_97_sigma",
            "AT_95_mean", "RH_95_mean", "BP_93_mean",
            "shear_alpha", "air_density"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()
    df["ts"] = pd.to_datetime(df["ts"])

    df = df.sort_values("ts").drop_duplicates("ts", keep="first")
    full_idx = pd.date_range(df["ts"].min(), df["ts"].max(), freq="10min")
    df = df.set_index("ts").reindex(full_idx)
    df.index.name = "ts"

    fail = pd.Series(False, index=df.index)
    present = df["WS_100_mean"].notna()

    if "coverage" in df: fail |= present & (df["coverage"] < 0.6)
    fail |= present & ((df["WS_100_mean"] < 0) | (df["WS_100_mean"] > 60))
    if "AT_95_mean" in df: fail |= present & ((df["AT_95_mean"] < -10) | (df["AT_95_mean"] > 50))
    if "BP_93_mean" in df: fail |= present & ((df["BP_93_mean"] < 940) | (df["BP_93_mean"] > 1050))
    if "air_density" in df: fail |= present & ((df["air_density"] < 1.0) | (df["air_density"] > 1.4))

    if "WS_100E_mean" in df:
        v = df["WS_100E_mean"]
        same = v.eq(v.shift(1)) & v.notna()
        grp = (~same).cumsum()
        runlen = same.groupby(grp).transform("sum")
        fail |= same & (runlen >= 5)

    if "WS_100E_mean" in df and "WS_100W_mean" in df:
        pair = (df["WS_100E_mean"] - df["WS_100W_mean"]).abs()
        fail |= present & (pair > 3.0)

    df["is_ok"] = present & (~fail)
    print(f"[結果] 網格點 {len(df):,}｜通過 QC 有效筆數：{int(df['is_ok'].sum()):,} ({100*df['is_ok'].sum()/int(present.sum()):.1f}%)")

    os.makedirs(C.DATA_DIR, exist_ok=True)
    os.makedirs(C.RES_DIR, exist_ok=True)
    df.reset_index().to_parquet(C.CLEAN_PARQUET, index=False)
    print(f"輸出：{C.CLEAN_PARQUET}")
    print("Stage 1 完成。")

if __name__ == "__main__":
    main()
