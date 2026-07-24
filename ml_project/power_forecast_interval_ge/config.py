#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
power_forecast_interval 設定檔 —— 獨立區間發電能量預測與資源評估專案

基於 ml_project/power_forecast 改編：
  結合風資源評估 + 區間平均功率 P_[t+1, t+k] 預測模型（嚴格排除當前時步 t）。
"""
from __future__ import annotations
import os
import numpy as np

# 路徑
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(PROJECT_DIR, "data")
MODEL_DIR   = os.path.join(PROJECT_DIR, "models")
RES_DIR     = os.path.join(PROJECT_DIR, "results")
FIG_DIR     = os.path.join(PROJECT_DIR, "figures")

# 來源 BSMI 10 分鐘檔
SRC_10MIN = os.path.normpath(os.path.join(PROJECT_DIR, "..", "data", "BSMI_10min.parquet"))

CLEAN_PARQUET = os.path.join(DATA_DIR, "clean_10min.parquet")
FEAT_PARQUET  = os.path.join(DATA_DIR, "features_interval.parquet")

STEP_MIN   = 10                             # 10 分鐘解析度
HORIZONS_H = [1, 3, 6, 24]                  # 區間預測提前量 (1h, 3h, 6h, 24h)
HORIZON_STEPS = {h: int(h * 60 // STEP_MIN) for h in HORIZONS_H}

TARGETS = ["ws100", "power"]
TEST_START = "2020-06-01"                   # 測試集起始點
N_CV_SPLITS = 4
RANDOM_SEED = 42

# 功率曲線
RHO_REF = 1.225
_PC_U = np.array([0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                  14, 20, 24, 25, 25.001, 40], dtype=float)
_PC_P = np.array([0, 0, 0.010, 0.040, 0.100, 0.200, 0.330, 0.500, 0.680,
                  0.830, 0.940, 0.990, 1.000, 1.000, 1.000, 1.000, 1.000,
                  0.0, 0.0], dtype=float)

def power_curve(u: np.ndarray | float) -> np.ndarray | float:
    return np.interp(np.asarray(u, dtype="float64"), _PC_U, _PC_P, left=0.0, right=0.0)

def density_correct(u: np.ndarray | float, rho: np.ndarray | float) -> np.ndarray | float:
    return np.asarray(u, dtype="float64") * (np.asarray(rho, dtype="float64") / RHO_REF) ** (1.0 / 3.0)

def virtual_power(ws100: np.ndarray | float, air_density: np.ndarray | float) -> np.ndarray | float:
    ws_arr = np.asarray(ws100, dtype="float64")
    rho_arr = np.asarray(air_density, dtype="float64")
    u_eff = density_correct(ws_arr, rho_arr)
    p = power_curve(u_eff)
    # NaN 保護：輸入含 NaN 時保留 NaN，避免 np.interp 靜默回傳 0
    nan_mask = np.isnan(ws_arr) | np.isnan(rho_arr)
    if np.any(nan_mask):
        p = np.where(nan_mask, np.nan, p)
    return p
