#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PW_Integrated 設定檔 —— 整合風場資源評估、0–6h ML 點預測與機率預測之獨立管線
"""
from __future__ import annotations
import os
import numpy as np

# ---------------------------------------------------------------- 路徑設定
PW_INT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(PW_INT_DIR, "data")
MODEL_DIR  = os.path.join(PW_INT_DIR, "models")
RES_DIR    = os.path.join(PW_INT_DIR, "results")
FIG_DIR    = os.path.join(PW_INT_DIR, "figures")

# 來源資料（已聚合之 10 分鐘 BSMI 測風塔 parquet 檔）
SRC_10MIN = os.path.normpath(os.path.join(PW_INT_DIR, "..", "ml_project", "data", "BSMI_10min.parquet"))

CLEAN_PARQUET = os.path.join(DATA_DIR, "clean_10min.parquet")
RESOURCE_JSON = os.path.join(RES_DIR, "resource_stats.json")
FEAT_PARQUET  = os.path.join(DATA_DIR, "features.parquet")

# ---------------------------------------------------------------- 時間與預測參數
STEP_MIN   = 10                             # 基礎解析度（分鐘）
HORIZONS_H = [1, 3, 6]                      # 預測提前量（小時）
HORIZON_STEPS = {h: int(h * 60 // STEP_MIN) for h in HORIZONS_H}   # {1:6, 3:18, 6:36}

TARGETS = ["ws100", "power"]                # 雙目標評估：100m 風速與正規化出力 P
TEST_START = "2020-06-01"                   # 保留測試集起始點
N_CV_SPLITS = 4                             # Expanding-window 折數
RANDOM_SEED = 42

# 機率預測分位點
QUANTILES = [0.1, 0.5, 0.9]

# ---------------------------------------------------------------- 功率曲線與物理修正
RHO_REF = 1.225   # kg/m^3 海平面標準空氣密度

# 代表性 8 MW 級離岸風機正規化功率曲線（切入 3 m/s、額定 12 m/s、切出 25 m/s）
_PC_U = np.array([0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                  14, 20, 24, 25, 25.001, 40], dtype=float)
_PC_P = np.array([0, 0, 0.010, 0.040, 0.100, 0.200, 0.330, 0.500, 0.680,
                  0.830, 0.940, 0.990, 1.000, 1.000, 1.000, 1.000, 1.000,
                  0.0, 0.0], dtype=float)


def power_curve(u: np.ndarray | float) -> np.ndarray | float:
    """正規化出力（0–1，即容量因數瞬時值）。"""
    return np.interp(np.asarray(u, dtype="float64"), _PC_U, _PC_P, left=0.0, right=0.0)


def density_correct(u: np.ndarray | float, rho: np.ndarray | float) -> np.ndarray | float:
    """IEC 空氣密度修正後的等效風速。"""
    return np.asarray(u, dtype="float64") * (np.asarray(rho, dtype="float64") / RHO_REF) ** (1.0 / 3.0)


def virtual_power(ws100: np.ndarray | float, air_density: np.ndarray | float) -> np.ndarray | float:
    """100 m 風速 + 空氣密度 → 正規化虛擬出力（0–1）。"""
    u_eff = density_correct(ws100, air_density)
    return power_curve(u_eff)
