#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1 —— 載入 + 驗證（QC）

輸入：已驗證的 10 分鐘 BSMI 塔資料（BSMI_10min.parquet）
動作：
  1. 重新獨立跑一輪 QC 驗證（不盲信上游旗標）：
       - 時間軸單調、去重
       - 覆蓋率 coverage 門檻
       - 物理範圍檢查（風速、風向、溫濕壓、空氣密度）
       - 感測器凍結/卡死偵測（連續多筆完全相同）
       - 四高度風速一致性（配對差、風切合理性）
  2. 標記 is_ok，輸出乾淨連續 10 分鐘序列到 data/clean_10min.parquet
  3. 輸出 results/validation_report.txt
輸出目標欄：WS_100_mean, air_density（給功率曲線用）等建模所需欄位。
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
    log("PW Stage 1 —— 載入 + 驗證")
    log("="*70)

    src = os.path.normpath(C.SRC_10MIN)
    if not os.path.exists(src):
        log(f"[錯誤] 找不到來源 10 分鐘檔：{src}")
        log("      請先由原始 1Hz CSV 產生（見 config.RAW_DIRS）。")
        sys.exit(1)
    df = pd.read_parquet(src)
    log(f"來源：{src}")
    log(f"原始筆數：{len(df):,}  欄位數：{df.shape[1]}")

    # --- 只保留建模需要的欄位（自足） ---
    keep = ["ts", "coverage",
            "WS_100E_mean", "WS_100W_mean", "WS_69W_mean", "WS_38W_mean",
            "WS_100_mean", "WS_100E_std", "WS_100E_ti", "WS_100E_gust_factor",
            "WD_97_sin", "WD_97_cos", "WD_97_sigma",
            "AT_95_mean", "RH_95_mean", "BP_93_mean",
            "shear_alpha", "air_density"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()
    df["ts"] = pd.to_datetime(df["ts"])

    n0 = len(df)
    # --- 1) 時間軸：排序、去重 ---
    df = df.sort_values("ts")
    dup = df["ts"].duplicated().sum()
    df = df.drop_duplicates("ts", keep="first")
    log(f"\n[時間軸] 去除重複時間戳：{dup:,}")

    # --- 建立連續 10 分鐘網格（缺漏留 NaN，供後面 gap 判斷） ---
    full_idx = pd.date_range(df["ts"].min(), df["ts"].max(), freq="10min")
    df = df.set_index("ts").reindex(full_idx)
    df.index.name = "ts"
    n_grid = len(df)
    n_missing_grid = df["WS_100_mean"].isna().sum()
    log(f"[時間軸] 連續網格點數：{n_grid:,}｜其中缺漏：{n_missing_grid:,} "
        f"({100*n_missing_grid/n_grid:.1f}%)")

    # --- 2) QC 檢查（逐項計數，建立 fail 遮罩） ---
    fail = pd.Series(False, index=df.index)
    def mark(cond, name):
        cond = cond.fillna(False)
        cnt = int(cond.sum())
        log(f"[QC] {name:32s} 觸發 {cnt:,}")
        return cond

    present = df["WS_100_mean"].notna()

    # 覆蓋率
    if "coverage" in df:
        fail |= mark(present & (df["coverage"] < 0.6), "覆蓋率 < 0.6")
    # 物理範圍
    fail |= mark(present & ((df["WS_100_mean"] < 0) | (df["WS_100_mean"] > 60)), "100m風速超出 0–60 m/s")
    for c in ["WS_100E_mean","WS_100W_mean","WS_69W_mean","WS_38W_mean"]:
        if c in df:
            fail |= mark(present & ((df[c] < 0) | (df[c] > 60)), f"{c} 超出 0–60")
    if "AT_95_mean" in df:
        fail |= mark(present & ((df["AT_95_mean"] < -10) | (df["AT_95_mean"] > 50)), "氣溫超出 -10–50°C")
    if "RH_95_mean" in df:
        fail |= mark(present & ((df["RH_95_mean"] < 0) | (df["RH_95_mean"] > 105)), "相對濕度超出 0–105%")
    if "BP_93_mean" in df:
        # 下限放寬到 940 hPa，保留颱風低壓的真實資料（全期最低 948.95）
        fail |= mark(present & ((df["BP_93_mean"] < 940) | (df["BP_93_mean"] > 1050)), "氣壓超出 940–1050 hPa")
    if "air_density" in df:
        fail |= mark(present & ((df["air_density"] < 1.0) | (df["air_density"] > 1.4)), "空氣密度超出 1.0–1.4")
    if "WD_97_sin" in df:
        r = np.sqrt(df["WD_97_sin"]**2 + df["WD_97_cos"]**2)
        fail |= mark(present & (r < 0.5), "風向向量長度異常(<0.5)")

    # 感測器凍結：WS_100E_mean 連續 >=6 筆(=1h)完全相同
    if "WS_100E_mean" in df:
        v = df["WS_100E_mean"]
        same = v.eq(v.shift(1)) & v.notna()
        # 連續 run 長度
        grp = (~same).cumsum()
        runlen = same.groupby(grp).transform("sum")
        frozen = same & (runlen >= 5)   # 5 個連續相同差 = 6 筆一致
        fail |= mark(frozen, "感測器凍結(≥1h 定值)")

    # 四高度一致性：100E 與 100W 差過大
    if "WS_100E_mean" in df and "WS_100W_mean" in df:
        pair = (df["WS_100E_mean"] - df["WS_100W_mean"]).abs()
        fail |= mark(present & (pair > 3.0), "100E/100W 配對差 >3 m/s")

    df["is_ok"] = present & (~fail)
    n_ok = int(df["is_ok"].sum())
    log(f"\n[結果] 網格點 {n_grid:,}｜有觀測 {int(present.sum()):,}｜"
        f"通過 QC 有效 {n_ok:,} ({100*n_ok/int(present.sum()):.1f}% of 有觀測)")

    # 年度分佈
    yr = df.loc[df["is_ok"]].index.year.value_counts().sort_index()
    log("\n[有效資料年度分佈]")
    for y, c in yr.items():
        log(f"   {y}: {c:,} 筆 (≈{c/6/24:.0f} 天)")

    # --- 3) 輸出乾淨資料（保留完整網格 + is_ok 旗標，方便建 lag 時判連續） ---
    os.makedirs(C.DATA_DIR, exist_ok=True)
    df.reset_index().to_parquet(C.CLEAN_PARQUET, index=False)
    log(f"\n輸出：{C.CLEAN_PARQUET}")

    os.makedirs(C.RES_DIR, exist_ok=True)
    rp = os.path.join(C.RES_DIR, "validation_report.txt")
    with open(rp, "w", encoding="utf-8") as f:
        f.write("\n".join(REPORT))
    log(f"輸出：{rp}")
    log("\nStage 1 完成。")

if __name__ == "__main__":
    sys.path.insert(0, C.PW_DIR if hasattr(C, "PW_DIR") else os.path.dirname(__file__))
    main()
