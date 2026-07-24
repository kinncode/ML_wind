#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
virtual_power.py —— 風速空氣密度修正與虛擬風場功率計算模組

包含 IEC 61400-12 空氣密度修正與標準離岸風機功率曲線。
"""
from __future__ import annotations
import numpy as np
import config as C

def density_correct(u: np.ndarray | float, rho: np.ndarray | float) -> np.ndarray | float:
    """IEC 61400-12 空氣密度修正。"""
    return C.density_correct(u, rho)

def power_curve(u: np.ndarray | float) -> np.ndarray | float:
    """標準 8 MW 離岸風機功率曲線 (0-1 額定值)。"""
    return C.power_curve(u)

def virtual_power(ws100: np.ndarray | float, air_density: np.ndarray | float) -> np.ndarray | float:
    """計算 100m 風速對應之虛擬風場正規化出力 (0-1)。"""
    return C.virtual_power(ws100, air_density)
