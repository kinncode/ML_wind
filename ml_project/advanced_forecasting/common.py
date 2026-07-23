#!/usr/bin/env python3
"""
共用模組 — 資料載入、特徵工程、NREL 5MW 風機模擬、年分拆分
所有進階預測腳本 (interval / quantile / trajectory) 共用此模組。
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import glob

# --------------------------------------------------------------------------
# 路徑設定
# --------------------------------------------------------------------------
PROJECT_DIR = Path("d:/wind_d/ML_wind/ml_project")
DATA_DIR = PROJECT_DIR / "data"
ADV_DIR = PROJECT_DIR / "advanced_forecasting"
ADV_RESULTS = ADV_DIR / "results"
ADV_FIGURES = ADV_RESULTS / "figures"

for d in [ADV_DIR, ADV_RESULTS, ADV_FIGURES]:
    d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# CJK 字型
# --------------------------------------------------------------------------
def setup_cjk_font():
    cands = (glob.glob("/usr/share/fonts/**/NotoSerifCJK*.ttc", recursive=True)
             + glob.glob("/usr/share/fonts/**/NotoSansCJK*.ttc", recursive=True)
             + glob.glob("C:/Windows/Fonts/msjh*.ttc")
             + glob.glob("C:/Windows/Fonts/msyh*.ttc"))
    for p in cands:
        try:
            fm.fontManager.addfont(p)
        except Exception:
            pass
    names = {f.name for f in fm.fontManager.ttflist}
    for n in ["Microsoft JhengHei", "Microsoft YaHei", "Noto Sans CJK TC", "DejaVu Sans"]:
        if n in names:
            plt.rcParams["font.family"] = [n, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return

setup_cjk_font()

# --------------------------------------------------------------------------
# NREL 5MW 官方功率曲線 (kW) — 查表 + 線性內插
# --------------------------------------------------------------------------
NREL_5MW_WS = np.array([
    0.0, 2.9, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5,
    7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.4,
    12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0,
    22.0, 23.0, 24.0, 25.0, 25.1
])
NREL_5MW_KW = np.array([
    0.0, 0.0, 27.3, 56.6, 93.6, 144.5, 208.3, 289.7, 399.6, 518.8,
    655.1, 811.7, 1007.0, 1211.0, 1458.0, 1726.0, 1984.0, 2267.0, 2587.0, 5000.0,
    5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0,
    5000.0, 5000.0, 5000.0, 5000.0, 0.0
])

RATED_MW = 5.0  # 額定容量


def simulate_nrel_5mw_power(ws_100, air_density):
    """NREL 5MW 功率模擬 + IEC 61400-12-1 空氣密度修正"""
    rho_0 = 1.225
    v_eff = ws_100 * (air_density / rho_0) ** (1.0 / 3.0)
    power_kw = np.interp(v_eff, NREL_5MW_WS, NREL_5MW_KW)
    return power_kw, v_eff


# --------------------------------------------------------------------------
# 預測時程定義 (名稱, shift 步數, 期望時間差)
# --------------------------------------------------------------------------
HORIZONS = [
    ("10min", 1,   pd.Timedelta(minutes=10)),
    ("1h",    6,   pd.Timedelta(minutes=60)),
    ("6h",    36,  pd.Timedelta(minutes=360)),
    ("24h",   144, pd.Timedelta(minutes=1440)),
]

# --------------------------------------------------------------------------
# 完整特徵清單 (42 維)
# --------------------------------------------------------------------------
FEATURES = [
    "sim_power_mw", "power_density_kw", "pd_roll_mean_1h", "pd_roll_max_1h",
    "Power_lag_1", "Power_lag_2", "Power_lag_3", "Power_lag_6", "Power_lag_12", "Power_lag_18",
    "Power_diff1", "Power_diff6", "Power_roll_mean_1h", "Power_roll_std_1h",
    "Power_roll_max_1h", "Power_roll_mean_3h",
    "WS_100_mean", "WS100_lag_1", "WS100_lag_2", "WS100_lag_3",
    "WS100_lag_6", "WS100_lag_12", "WS100_lag_18",
    "WS100_diff1", "WS100_diff6",
    "WS100_roll_mean_1h", "WS100_roll_std_1h", "WS100_roll_max_1h",
    "WS100_roll_mean_3h", "WS100_roll_std_3h",
    "shear_alpha", "WS_100E_ti", "WS_100E_gust_factor",
    "WD_97_sin", "WD_97_cos",
    "WD_sin_diff1", "WD_cos_diff1", "WD_sin_diff6", "WD_cos_diff6",
    "BP_93_mean", "delta_BP_1h", "delta_BP_3h", "delta_BP_6h",
    "AT_95_mean", "RH_95_mean", "air_density",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
]


# --------------------------------------------------------------------------
# 資料載入與特徵工程 (年分拆分)
# --------------------------------------------------------------------------
def load_data():
    """
    載入 BSMI 測風塔資料，建構特徵工程與預測標的。
    年分拆分: Train 2016-2018, Test 2020-2021
    回傳: df_full, df_train, df_test
    """
    print("[DATA] 載入 BSMI 10min + turbulence 資料...")
    df_10m = pd.read_parquet(DATA_DIR / "BSMI_10min.parquet")
    df_turb = pd.read_parquet(DATA_DIR / "BSMI_turb.parquet")
    df = df_10m.merge(df_turb, on="ts", how="inner")
    df = df[df["is_valid"]].sort_values("ts").reset_index(drop=True)

    # 發電量模擬
    df["sim_power_kw"], df["v_eff"] = simulate_nrel_5mw_power(
        df["WS_100_mean"].values, df["air_density"].values
    )
    df["sim_power_mw"] = df["sim_power_kw"] / 1000.0
    df["power_density_kw"] = df["power_density"] / 1000.0

    # 週期時間特徵
    h = df.ts.dt.hour + df.ts.dt.minute / 60.0
    doy = df.ts.dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * h / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * h / 24.0)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # 多階滯後與滾動統計
    for step in [1, 2, 3, 6, 12, 18]:
        df[f"WS100_lag_{step}"] = df["WS_100_mean"].shift(step)
        df[f"Power_lag_{step}"] = df["sim_power_mw"].shift(step)

    df["WS100_diff1"] = df["WS_100_mean"] - df["WS100_lag_1"]
    df["WS100_diff6"] = df["WS_100_mean"] - df["WS100_lag_6"]
    df["WS100_roll_mean_1h"] = df["WS_100_mean"].shift(1).rolling(6).mean()
    df["WS100_roll_std_1h"]  = df["WS_100_mean"].shift(1).rolling(6).std()
    df["WS100_roll_max_1h"]  = df["WS_100_mean"].shift(1).rolling(6).max()
    df["WS100_roll_mean_3h"] = df["WS_100_mean"].shift(1).rolling(18).mean()
    df["WS100_roll_std_3h"]  = df["WS_100_mean"].shift(1).rolling(18).std()

    df["Power_diff1"] = df["sim_power_mw"] - df["Power_lag_1"]
    df["Power_diff6"] = df["sim_power_mw"] - df["Power_lag_6"]
    df["Power_roll_mean_1h"] = df["sim_power_mw"].shift(1).rolling(6).mean()
    df["Power_roll_std_1h"]  = df["sim_power_mw"].shift(1).rolling(6).std()
    df["Power_roll_max_1h"]  = df["sim_power_mw"].shift(1).rolling(6).max()
    df["Power_roll_mean_3h"] = df["sim_power_mw"].shift(1).rolling(18).mean()

    df["pd_roll_mean_1h"] = df["power_density_kw"].shift(1).rolling(6).mean()
    df["pd_roll_max_1h"]  = df["power_density_kw"].shift(1).rolling(6).max()

    df["delta_BP_1h"] = df["BP_93_mean"] - df["BP_93_mean"].shift(6)
    df["delta_BP_3h"] = df["BP_93_mean"] - df["BP_93_mean"].shift(18)
    df["delta_BP_6h"] = df["BP_93_mean"] - df["BP_93_mean"].shift(36)

    df["WD_sin_diff1"] = df["WD_97_sin"] - df["WD_97_sin"].shift(1)
    df["WD_cos_diff1"] = df["WD_97_cos"] - df["WD_97_cos"].shift(1)
    df["WD_sin_diff6"] = df["WD_97_sin"] - df["WD_97_sin"].shift(6)
    df["WD_cos_diff6"] = df["WD_97_cos"] - df["WD_97_cos"].shift(6)

    # 多時程預測標的 + 時序斷點偵測
    for hor_name, shift_n, expected_gap in HORIZONS:
        col = f"target_power_{shift_n}"
        df[col] = df["sim_power_mw"].shift(-shift_n)
        # 清除跨時序斷點的錯位 target
        actual_gap = df["ts"].shift(-shift_n) - df["ts"]
        bad_mask = actual_gap != expected_gap
        n_bad = bad_mask.sum()
        df.loc[bad_mask, col] = np.nan
        print(f"  [GAP] {col}: 清除 {n_bad} 筆跨斷點 target")

        df[f"target_delta_{shift_n}"] = df[col] - df["sim_power_mw"]

    df["year"] = df.ts.dt.year

    # 清除特徵 NaN (lag/rolling 的前幾筆)
    feature_na = df[FEATURES].isna().any(axis=1)
    df = df[~feature_na].reset_index(drop=True)

    # 依年分拆分
    df_train = df[df["year"] <= 2018].copy()
    df_test  = df[df["year"] >= 2020].copy()

    print(f"[DATA] 特徵有效筆數: {len(df)}")
    print(f"[DATA] 訓練集 (2016-2018): {len(df_train)}")
    print(f"[DATA] 測試集 (2020-2021): {len(df_test)}")

    return df, df_train, df_test
