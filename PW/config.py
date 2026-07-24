#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PW 專案設定檔 —— 單一風場超短期（0–6h）風力發電預測（獨立、可重現）

前提
----
BSMI 100 m 測風塔沒有實測發電量，因此本專案用一條代表性離岸風機功率曲線
（含 IEC 空氣密度修正）把 100 m 風速換算成「正規化虛擬出力」P（0–1，即容量
因數的瞬時值）。P 對機型穩健，適合做預測技術評估。

管線
----
  01_load_validate.py  →  02_features.py  →  03_train_select.py  →  04_evaluate_report.py

驗證策略
--------
時序前推（expanding-window time-series CV）＋ 保留最後 12 個月為測試年，
嚴格避免時間洩漏。
"""
from __future__ import annotations
import os
import numpy as np

# ---------------------------------------------------------------- 路徑
PW_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PW_DIR, "data")
MODEL_DIR= os.path.join(PW_DIR, "models")
RES_DIR  = os.path.join(PW_DIR, "results")
FIG_DIR  = os.path.join(PW_DIR, "figures")

# 已驗證的 10 分鐘來源資料（由原始 1Hz 塔資料聚合而來）。
# 若不存在，01 會嘗試從原始 CSV 重建（見 RAW_DIRS）。
SRC_10MIN = os.path.join(PW_DIR, "..", "ml_project", "data", "BSMI_10min.parquet")
RAW_DIRS  = [
    os.path.join(PW_DIR, "..", "BSMI wind raw data 2016.03~2017.12"),
    os.path.join(PW_DIR, "..", "BSMI wind raw data 2018.01-2019.12"),
    os.path.join(PW_DIR, "..", "BSMI wind raw data 2020.01-2021.05"),
]

CLEAN_PARQUET = os.path.join(DATA_DIR, "clean_10min.parquet")
FEAT_PARQUET  = os.path.join(DATA_DIR, "features.parquet")

# ---------------------------------------------------------------- 時間解析度與預測時程
STEP_MIN   = 10                      # 基礎解析度（分鐘）
HORIZONS_H = [1, 3, 6]               # 預測提前量（小時）
HORIZON_STEPS = {h: h * 60 // STEP_MIN for h in HORIZONS_H}   # {1:6, 3:18, 6:36}

# ---------------------------------------------------------------- 預測目標
# 兩個目標：100 m 風速（有塔真值）與正規化出力 P（風速經功率曲線）
TARGETS = ["ws100", "power"]

# ---------------------------------------------------------------- 驗證切分
# 資料涵蓋 2016-03 ~ 2021-10；保留 2020-06 起（約 17 個月，2021-06 缺測）為測試期。
TEST_START = "2020-06-01"
N_CV_SPLITS = 4                      # 訓練期內的 expanding-window 折數
RANDOM_SEED = 42

# ---------------------------------------------------------------- 功率曲線（自足，與 ml_project 一致）
RHO_REF = 1.225   # kg/m^3 海平面標準空氣密度
# 代表性 ~8 MW 級離岸機型正規化功率曲線：切入 3、額定 ~12、切出 25 m/s
_PC_U = np.array([0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                  14, 20, 24, 25, 25.001, 40])
_PC_P = np.array([0, 0, 0.010, 0.040, 0.100, 0.200, 0.330, 0.500, 0.680,
                  0.830, 0.940, 0.990, 1.000, 1.000, 1.000, 1.000, 1.000,
                  0.0, 0.0])


def power_curve(u):
    """正規化出力（0–1）。"""
    return np.interp(np.asarray(u, dtype="float64"), _PC_U, _PC_P, left=0.0, right=0.0)


def density_correct(u, rho):
    """IEC 空氣密度修正後的等效風速。"""
    return np.asarray(u) * (np.asarray(rho) / RHO_REF) ** (1.0 / 3.0)


def virtual_power(ws100, air_density):
    """100 m 風速 + 空氣密度 → 正規化虛擬出力（0–1）。"""
    u_eff = density_correct(ws100, air_density)
    return power_curve(u_eff)
