#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1 —— 載入 + 獨立 QC 驗證 (PW_Interval 專案)

輸入：已驗證之 10 分鐘 BSMI 測風塔資料 (BSMI_10min.parquet)
動作：獨立跑四重 QC 檢查，標記 is_ok，輸出至 data/clean_10min.parquet。
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import config as C

REPORT = []
def log(msg=""):
    print(msg)
    REPORT.append(str(msg))

def main():
    log("="*70)
    log("PW_Interval Stage 1 —— 載入 + 獨立 QC 驗證")
    log("="*70)

    src = os.path.normpath(C.SRC_10MIN)
    if not os.path.exists(src):
        log(f"[錯誤] 找不到來源 10 分鐘檔：{src}")
        sys.exit(1)
    df = pd.read_parquet(src)
    log(f"來源：{src}")
    log(f"原始筆數：{len(df):,}  欄位數：{df.shape[1]}")

    keep = ["ts", "coverage",
            "WS_100E_mean", "WS_100W_mean", "WS_69W_mean", "WS_38W_mean",
            "WS_100_mean", "WS_100E_std", "WS_100E_ti", "WS_100E_gust_factor",
            "WD_97_sin", "WD_97_cos", "WD_97_sigma",
            "AT_95_mean", "RH_95_mean", "BP_93_mean",
            "shear_alpha", "air_density"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()
    df["ts"] = pd.to_datetime(df["ts"])

    df = df.sort_values("ts")
    dup = df["ts"].duplicated().sum()
    df = df.drop_duplicates("ts", keep="first")
    log(f"\n[時間軸] 去除重複時間戳：{dup:,}")

    full_idx = pd.date_range(df["ts"].min(), df["ts"].max(), freq="10min")
    df = df.set_index("ts").reindex(full_idx)
    df.index.name = "ts"
    n_grid = len(df)
    n_missing_grid = df["WS_100_mean"].isna().sum()
    log(f"[時間軸] 連續網格點數：{n_grid:,}｜其中缺漏：{n_missing_grid:,} ({100*n_missing_grid/n_grid:.1f}%)")

    fail = pd.Series(False, index=df.index)
    def mark(cond, name):
        cond = cond.fillna(False)
        cnt = int(cond.sum())
        log(f"[QC] {name:32s} 觸發 {cnt:,}")
        return cond

    present = df["WS_100_mean"].notna()

    if "coverage" in df:
        fail |= mark(present & (df["coverage"] < 0.6), "覆蓋率 < 0.6")
    fail |= mark(present & ((df["WS_100_mean"] < 0) | (df["WS_100_mean"] > 60)), "100m風速超出 0–60 m/s")
    for c in ["WS_100E_mean","WS_100W_mean","WS_69W_mean","WS_38W_mean"]:
        if c in df:
            fail |= mark(present & ((df[c] < 0) | (df[c] > 60)), f"{c} 超出 0–60 m/s")
    if "AT_95_mean" in df:
        fail |= mark(present & ((df["AT_95_mean"] < -10) | (df["AT_95_mean"] > 50)), "氣溫超出 -10–50°C")
    if "RH_95_mean" in df:
        fail |= mark(present & ((df["RH_95_mean"] < 0) | (df["RH_95_mean"] > 105)), "相對濕度超出 0–105%")
    if "BP_93_mean" in df:
        fail |= mark(present & ((df["BP_93_mean"] < 940) | (df["BP_93_mean"] > 1050)), "氣壓超出 940–1050 hPa")
    if "air_density" in df:
        fail |= mark(present & ((df["air_density"] < 1.0) | (df["air_density"] > 1.4)), "空氣密度超出 1.0–1.4")
    if "WD_97_sin" in df:
        r = np.sqrt(df["WD_97_sin"]**2 + df["WD_97_cos"]**2)
        fail |= mark(present & (r < 0.5), "風向向量長度異常(<0.5)")

    if "WS_100E_mean" in df:
        v = df["WS_100E_mean"]
        same = v.eq(v.shift(1)) & v.notna()
        grp = (~same).cumsum()
        runlen = same.groupby(grp).transform("sum")
        frozen = same & (runlen >= 5)
        fail |= mark(frozen, "感測器凍結(>=1h 定值)")

    if "WS_100E_mean" in df and "WS_100W_mean" in df:
        pair = (df["WS_100E_mean"] - df["WS_100W_mean"]).abs()
        fail |= mark(present & (pair > 3.0), "100E/100W 配對差 >3 m/s")

    df["is_ok"] = present & (~fail)
    n_ok = int(df["is_ok"].sum())
    log(f"\n[結果] 網格點 {n_grid:,}｜有觀測 {int(present.sum()):,}｜"
        f"通過 QC 有效 {n_ok:,} ({100*n_ok/int(present.sum()):.1f}% of 有觀測)")

    os.makedirs(C.DATA_DIR, exist_ok=True)
    os.makedirs(C.RES_DIR, exist_ok=True)
    df.reset_index().to_parquet(C.CLEAN_PARQUET, index=False)
    log(f"\n輸出：{C.CLEAN_PARQUET}")

    rp = os.path.join(C.RES_DIR, "validation_report.txt")
    with open(rp, "w", encoding="utf-8") as f:
        f.write("\n".join(REPORT))
    log(f"輸出：{rp}")
    log("\nStage 1 完成。")

if __name__ == "__main__":
    main()
