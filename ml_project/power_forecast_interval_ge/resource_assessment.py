#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
resource_assessment.py —— 風資源評估模組 (Wind Resource Assessment)

計算站點之容量因數 (CF %)、年等效滿載小時數 (h/year)、月度與晝夜風資源分佈。
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import config as C

def main():
    print("="*70)
    print("power_forecast_interval Stage 2 —— 風資源評估 (Resource Assessment)")
    print("="*70)

    if not os.path.exists(C.CLEAN_PARQUET):
        raise FileNotFoundError(f"找不到 {C.CLEAN_PARQUET}，請先執行 01_load_validate.py")

    df = pd.read_parquet(C.CLEAN_PARQUET).set_index("ts").sort_index()
    valid = df[df["is_ok"]].copy()

    valid["P_norm"] = C.virtual_power(valid["WS_100_mean"], valid["air_density"])
    cf_mean = valid["P_norm"].mean()
    cf_pct = cf_mean * 100.0
    full_load_hours = cf_mean * 8760.0

    print(f"有效樣本數：{len(valid):,} 筆")
    print(f"全期平均容量因數 (Capacity Factor)：{cf_pct:.2f}%")
    print(f"年等效滿載發電時數 (Full Load Hours)：{full_load_hours:,.0f} 小時/年")

    monthly_cf = valid.groupby(valid.index.month)["P_norm"].mean() * 100.0
    diurnal_cf = valid.groupby(valid.index.hour)["P_norm"].mean() * 100.0

    stats = {
        "n_samples": len(valid),
        "capacity_factor_pct": round(cf_pct, 2),
        "full_load_hours": round(full_load_hours, 0),
        "monthly_cf": {int(k): round(v, 2) for k, v in monthly_cf.items()},
        "diurnal_cf": {int(k): round(v, 2) for k, v in diurnal_cf.items()}
    }

    res_json = os.path.join(C.RES_DIR, "resource_stats.json")
    with open(res_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"資源評估結果已寫入：{res_json}")
    print("Stage 2 完成。")

if __name__ == "__main__":
    main()
