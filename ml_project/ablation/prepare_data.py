#!/usr/bin/env python3
"""
消融實驗第 0 步：把 10 分鐘平均表與湍流目標表合併成一張「建模用資料表」。

輸出 ablation/model_data.parquet，含：
  - 17 個「平均狀態」特徵（唯一允許當輸入的欄位）
  - 各種預測目標（正規化與未正規化）
  - year 欄位供時間切分

之後所有消融與正式訓練都直接讀這張表，不必再碰原始檔。
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"

# 唯一允許當輸入的 17 個特徵：全部來自 10 分鐘平均狀態
# （標準氣象檔或中尺度模式給得出來的東西），不含任何從 1 Hz 導出的量
FEATURES = [
    "WS_100_mean", "WS_69W_mean", "WS_38W_mean",              # 風速剖面
    "shear_alpha",                                            # 風切
    "WD_97_sin", "WD_97_cos", "WD_35_sin", "WD_35_cos", "veer_97_35",  # 風向
    "AT_95_mean", "RH_95_mean", "BP_93_mean", "air_density",  # 熱力
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",             # 時間
]

# 特徵分群（供群組消融）
GROUPS = {
    "風速剖面": ["WS_100_mean", "WS_69W_mean", "WS_38W_mean"],
    "風向":     ["WD_97_sin", "WD_97_cos", "WD_35_sin", "WD_35_cos", "veer_97_35"],
    "風切":     ["shear_alpha"],
    "熱力":     ["AT_95_mean", "RH_95_mean", "BP_93_mean", "air_density"],
    "時間":     ["hour_sin", "hour_cos", "doy_sin", "doy_cos"],
}


def build() -> pd.DataFrame:
    b = pd.read_parquet(DATA / "BSMI_10min.parquet")
    t = pd.read_parquet(DATA / "BSMI_turb.parquet")
    d = b.merge(t, on="ts", how="inner")
    d = d[d["is_valid"]].copy()
    d = d[d["WS_100_mean"] >= 4.0]                # IEC 湍流只在運轉風速範圍有定義

    # 時間週期特徵
    h = d.ts.dt.hour + d.ts.dt.minute / 60
    doy = d.ts.dt.dayofyear
    d["hour_sin"] = np.sin(2 * np.pi * h / 24)
    d["hour_cos"] = np.cos(2 * np.pi * h / 24)
    d["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    d["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    d["year"] = d.ts.dt.year

    # ---- 預測目標 ----
    e = d["WS_100E_mean"]
    # 正規化目標（把風速影響除掉，才是真正在預測「湍流形狀」）
    d["tgt_ti"]        = d["WS_100E_std"]    / e
    d["tgt_gustfac"]   = d["WS_100E_gust3s"] / e
    d["tgt_p99n"]      = d["WS_100E_p99"]    / e
    d["tgt_p01n"]      = d["WS_100E_p01"]    / e
    # 天生無量綱的目標
    d["tgt_specslope"] = d["WS_100E_spec_slope"]
    d["tgt_intscale"]  = d["WS_100E_int_scale_s"]
    # 未正規化目標（用來示範「虛高 R²」陷阱）
    d["tgt_gust_raw"]  = d["WS_100E_gust3s"]
    d["tgt_p99_raw"]   = d["WS_100E_p99"]

    # 供評估雙感測器天花板用（西側對應欄位）
    w = d["WS_100W_mean"]
    d["twin_ti"]        = d["WS_100W_std"]    / w
    d["twin_gustfac"]   = d["WS_100W_gust3s"] / w
    d["twin_p99n"]      = d["WS_100W_p99"]    / w
    d["twin_p01n"]      = d["WS_100W_p01"]    / w
    d["twin_specslope"] = d["WS_100W_spec_slope"]
    d["twin_intscale"]  = d["WS_100W_int_scale_s"]
    d["twin_gust_raw"]  = d["WS_100W_gust3s"]
    d["twin_p99_raw"]   = d["WS_100W_p99"]

    keep = ["ts", "year"] + FEATURES + \
           [c for c in d.columns if c.startswith(("tgt_", "twin_"))]
    return d[keep].reset_index(drop=True)


if __name__ == "__main__":
    df = build()
    out = HERE / "model_data.parquet"
    df.to_parquet(out, index=False, compression="snappy")
    print(f"✓ {out}  {df.shape[0]:,} 列 × {df.shape[1]} 欄")
    print(f"  期間 year {df.year.min()}–{df.year.max()}")
    print("  各目標非空比例：")
    for c in [c for c in df.columns if c.startswith("tgt_")]:
        print(f"    {c:16s} {100 * df[c].notna().mean():5.1f}%  "
              f"平均 {df[c].mean():.4f}")
