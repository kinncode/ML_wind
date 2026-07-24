#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PW_Interval 設定檔 —— 獨立區間總發電能量預測管線 (Interval Energy Forecast)

基於 PW 專案架構改編：
  將傳統單點 (t+H) 瞬時預測改為未來區間 (t..t+H) 累積總發電能量 E (kWh/MW) 與平均風速預測。
"""
from __future__ import annotations
import os
import numpy as np

# ---------------------------------------------------------------- 路徑
PW_INTV_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(PW_INTV_DIR, "data")
MODEL_DIR   = os.path.join(PW_INTV_DIR, "models")
RES_DIR     = os.path.join(PW_INTV_DIR, "results")
FIG_DIR     = os.path.join(PW_INTV_DIR, "figures")

# 來源 10 分鐘檔 (與 PW 相同)
SRC_10MIN = os.path.normpath(os.path.join(PW_INTV_DIR, "..", "ml_project", "data", "BSMI_10min.parquet"))

CLEAN_PARQUET = os.path.join(DATA_DIR, "clean_10min.parquet")
FEAT_PARQUET  = os.path.join(DATA_DIR, "features.parquet")

# ---------------------------------------------------------------- 時間與預測區間
STEP_MIN   = 10                             # 基礎解析度（分鐘）
HORIZONS_H = [1, 3, 6]                      # 預測時間區間（小時）
HORIZON_STEPS = {h: int(h * 60 // STEP_MIN) for h in HORIZONS_H}   # {1:6, 3:18, 6:36}

TARGETS = ["ws100", "power"]                # 區間預測標的：區間平均 100m 風速與區間總發電能量
TEST_START = "2020-06-01"                   # 保留測試期起始點
N_CV_SPLITS = 4
RANDOM_SEED = 42

# ---------------------------------------------------------------- 功率曲線
RHO_REF = 1.225
_PC_U = np.array([0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                  14, 20, 24, 25, 25.001, 40], dtype=float)
_PC_P = np.array([0, 0, 0.010, 0.040, 0.100, 0.200, 0.330, 0.500, 0.680,
                  0.830, 0.940, 0.990, 1.000, 1.000, 1.000, 1.000, 1.000,
                  0.0, 0.0], dtype=float)


def power_curve(u: np.ndarray | float) -> np.ndarray | float:
    """正規化出力（0–1）。"""
    return np.interp(np.asarray(u, dtype="float64"), _PC_U, _PC_P, left=0.0, right=0.0)


def density_correct(u: np.ndarray | float, rho: np.ndarray | float) -> np.ndarray | float:
    """IEC 空氣密度修正。"""
    return np.asarray(u, dtype="float64") * (np.asarray(rho, dtype="float64") / RHO_REF) ** (1.0 / 3.0)


def virtual_power(ws100: np.ndarray | float, air_density: np.ndarray | float) -> np.ndarray | float:
    """100 m 風速 + 空氣密度 ➔ 正規化虛擬出力（0–1）。"""
    u_eff = density_correct(ws100, air_density)
    return power_curve(u_eff)
