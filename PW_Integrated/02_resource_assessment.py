#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2 —— 風場資源評估 (Wind Resource Assessment)

輸入：data/clean_10min.parquet
動作：
  1. 篩選通過 QC (is_ok == True) 之資料，以 8 MW 級離岸機型功率曲線 + 空氣密度修正計算正規化虛擬出力 P (0–1)。
  2. 計算風場核心資源指標：
     - 容量因數 (Capacity Factor, CF %)
     - 等效滿載時數 (Equivalent Full Load Hours)
     - 滿載占比 (ws >= 12 m/s)、零出力占比 (ws < 3 m/s)、切出停機占比 (ws > 25 m/s)
     - 逐月/分季節 CF、年際穩定度
     - 機率分佈與功率曲線敏感度 (額定風速 10 m/s, 12 m/s, 14 m/s)
  3. 輸出 results/resource_stats.json
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import config as C

def main():
    print("="*70)
    print("PW_Integrated Stage 2 —— 風場資源評估")
    print("="*70)

    if not os.path.exists(C.CLEAN_PARQUET):
        raise FileNotFoundError(f"找不到 {C.CLEAN_PARQUET}，請先執行 01_load_validate.py")

    df = pd.read_parquet(C.CLEAN_PARQUET)
    df["ts"] = pd.to_datetime(df["ts"])
    valid = df[df["is_ok"]].copy()

    ws = valid["WS_100_mean"].to_numpy(dtype="float64")
    rho = valid["air_density"].to_numpy(dtype="float64")

    # 正規化虛擬出力 P (0-1)
    p = C.virtual_power(ws, rho)
    valid["P"] = p
    valid["year"] = valid["ts"].dt.year
    valid["month"] = valid["ts"].dt.month
    valid["hour"] = valid["ts"].dt.hour

    cf_overall = float(p.mean())
    full_load_hours = cf_overall * 8760.0

    ratio_zero = float((ws < 3.0).mean())
    ratio_full = float((ws >= 12.0).mean())
    ratio_cutout = float((ws > 25.0).mean())

    # 逐月 CF
    monthly_cf = valid.groupby("month")["P"].mean().to_dict()
    best_month = int(max(monthly_cf, key=monthly_cf.get))
    worst_month = int(min(monthly_cf, key=monthly_cf.get))

    # 逐年 CF
    yearly_cf = valid.groupby("year")["P"].mean().to_dict()

    # 機型敏感度 (立方簡化模型對照)
    def alt_pc(u_raw, r_raw, rated_u):
        u_eff = C.density_correct(u_raw, r_raw)
        cin, cout = 3.0, 25.0
        res = np.where(u_eff < cin, 0.0,
              np.where(u_eff < rated_u, (u_eff**3 - cin**3)/(rated_u**3 - cin**3),
              np.where(u_eff < cout, 1.0, 0.0)))
        return float(np.clip(res, 0.0, 1.0).mean())

    sensitivity = {
        "rated_10ms_CF": alt_pc(ws, rho, 10.0),
        "rated_12ms_CF": alt_pc(ws, rho, 12.0),
        "rated_14ms_CF": alt_pc(ws, rho, 14.0),
    }

    stats = {
        "n_samples": int(len(valid)),
        "cf_overall": round(cf_overall, 4),
        "full_load_hours_per_year": round(full_load_hours, 1),
        "ratio_zero_power": round(ratio_zero, 4),
        "ratio_full_power": round(ratio_full, 4),
        "ratio_cutout_shutdown": round(ratio_cutout, 4),
        "best_month": {"month": best_month, "cf": round(monthly_cf[best_month], 4)},
        "worst_month": {"month": worst_month, "cf": round(monthly_cf[worst_month], 4)},
        "monthly_cf": {int(k): round(v, 4) for k, v in monthly_cf.items()},
        "yearly_cf": {int(k): round(v, 4) for k, v in yearly_cf.items()},
        "sensitivity": {k: round(v, 4) for k, v in sensitivity.items()}
    }

    print(f"有效樣本數：{stats['n_samples']:,}")
    print(f"整體容量因數 (CF)：{stats['cf_overall']*100:.2f}%")
    print(f"等效滿載時數：{stats['full_load_hours_per_year']:.0f} h/年")
    print(f"滿載時間占比 (≥12 m/s)：{stats['ratio_full_power']*100:.1f}%")
    print(f"零出力占比 (<3 m/s)：{stats['ratio_zero_power']*100:.1f}%")
    print(f"切出停機占比 (>25 m/s)：{stats['ratio_cutout_shutdown']*100:.2f}%")
    print(f"最佳月 (12月/1月) vs 最差月 (8月)：{monthly_cf[best_month]*100:.1f}% vs {monthly_cf[worst_month]*100:.1f}%")
    print("機型敏感度（不同額定風速 CF）：")
    for k, v in sensitivity.items():
        print(f"  {k:20s}: {v*100:.1f}%")

    os.makedirs(C.RES_DIR, exist_ok=True)
    with open(C.RESOURCE_JSON, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n資源評估數據已寫入：{C.RESOURCE_JSON}")
    print("Stage 2 完成。")

if __name__ == "__main__":
    main()
