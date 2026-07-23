#!/usr/bin/env python3
"""
BSMI 測風塔 — 單一 XGBoost 發電量預測超參數調校腳本 (Single XGBoost Hyperparameter Tuning)

本腳本展示單一 XGBoost 模型在風力發電預測上的關鍵調校參數設定與最佳化範例。
"""

from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

DATA_DIR = Path("d:/wind_d/ML_wind/ml_project/data")

# 1. NREL 5MW 離岸風機發電量模擬
def simulate_nrel_5mw_power(ws_100, air_density):
    rho_0 = 1.225
    v_eff = ws_100 * (air_density / rho_0) ** (1.0 / 3.0)
    power = np.zeros_like(v_eff)
    mask_reg2 = (v_eff >= 3.0) & (v_eff < 11.4)
    power[mask_reg2] = np.minimum(5000.0, 3.704 * (v_eff[mask_reg2] ** 3))
    mask_reg3 = (v_eff >= 11.4) & (v_eff <= 25.0)
    power[mask_reg3] = 5000.0
    return power

# 2. 載入資料與特徵
df_10m = pd.read_parquet(DATA_DIR / "BSMI_10min.parquet")
df_turb = pd.read_parquet(DATA_DIR / "BSMI_turb.parquet")
df = df_10m.merge(df_turb, on="ts", how="inner")
df = df[df["is_valid"]].sort_values("ts").reset_index(drop=True)

df["sim_power_mw"] = simulate_nrel_5mw_power(df["WS_100_mean"].values, df["air_density"].values) / 1000.0

h = df.ts.dt.hour + df.ts.dt.minute / 60.0
doy = df.ts.dt.dayofyear
df["hour_sin"] = np.sin(2 * np.pi * h / 24.0)
df["hour_cos"] = np.cos(2 * np.pi * h / 24.0)
df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

# Lags
for s in [1, 2, 3, 6]:
    df[f"WS100_lag_{s}"] = df["WS_100_mean"].shift(s)
    df[f"Power_lag_{s}"] = df["sim_power_mw"].shift(s)

df["WS100_diff1"] = df["WS_100_mean"] - df["WS100_lag_1"]
df["WS100_roll_mean_1h"] = df["WS_100_mean"].shift(1).rolling(6).mean()
df["WS100_roll_std_1h"] = df["WS_100_mean"].shift(1).rolling(6).std()

df["delta_BP_1h"] = df["BP_93_mean"] - df["BP_93_mean"].shift(6)
df["delta_BP_6h"] = df["BP_93_mean"] - df["BP_93_mean"].shift(36)

df["target_power_1h"] = df["sim_power_mw"].shift(-6)
df["year"] = df.ts.dt.year

df_clean = df.dropna().copy()
df_train = df_clean[df_clean["year"] <= 2018].copy()
df_test = df_clean[df_clean["year"] >= 2020].copy()

features = [
    "sim_power_mw", "Power_lag_1", "Power_lag_2", "Power_lag_3", "Power_lag_6",
    "WS_100_mean", "WS100_lag_1", "WS100_lag_2", "WS100_lag_3", "WS100_lag_6",
    "WS100_diff1", "WS100_roll_mean_1h", "WS100_roll_std_1h",
    "shear_alpha", "WS_100E_ti", "WS_100E_gust_factor",
    "WD_97_sin", "WD_97_cos", "BP_93_mean", "delta_BP_1h", "delta_BP_6h",
    "AT_95_mean", "RH_95_mean", "air_density",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos"
]

X_train, y_train = df_train[features], df_train["target_power_1h"]
X_test, y_test = df_test[features], df_test["target_power_1h"]

# 3. 單一 XGBoost 最佳調校參數範例 (Single XGBoost Config)
xgb_best_params = {
    # 樹結構控制 (Tree Architecture)
    "max_depth": 6,                # 樹深度：建議 4~8 (避免過擬合極端陣風噪訊)
    "min_child_weight": 10,        # 節點最小權重和：建議 5~30 (過濾極端單一離群點)
    "gamma": 0.1,                  # 分裂所需最小損失減少量：建議 0.0~0.5
    
    # 學習率與正則化 (Learning & Regularization)
    "learning_rate": 0.03,         # 步長學習率：建議 0.01~0.05
    "n_estimators": 300,           # 樹的數量：建議 200~500
    "reg_alpha": 0.5,              # L1 正則化：建議 0.1~5.0 (自動稀疏化不重要 Lag 特徵)
    "reg_lambda": 2.0,             # L2 正則化：建議 1.0~10.0 (平滑樹葉節點權重)
    
    # 隨機採樣 (Sampling)
    "subsample": 0.8,              # 橫向資料抽樣比例：建議 0.7~0.85
    "colsample_bytree": 0.8,       # 縱向特徵抽樣比例：建議 0.7~0.85
    
    # 算力加速與目標函數 (Acceleration & Objective)
    "tree_method": "hist",         # 使用直方圖加速 (速度提升 10 倍以上)
    "objective": "reg:squarederror", # 平方誤差 (若離群點多可改為 reg:absoluteerror)
    "eval_metric": "rmse",
    "random_state": 42,
    "n_jobs": 4
}

print("開始訓練單一最佳化 XGBoost 風力發電預測模型 (+1h Target)...")
model = xgb.XGBRegressor(**xgb_best_params)
model.fit(X_train, y_train)

preds = np.clip(model.predict(X_test), 0.0, 5.0)

rmse = np.sqrt(mean_squared_error(y_test, preds))
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)
nmae = (mae / 5.0) * 100.0

# 物理 Persistence Baseline 對比
base_power = df_test["sim_power_mw"].values
rmse_p = np.sqrt(mean_squared_error(y_test, base_power))
skill = (1.0 - (rmse / rmse_p)) * 100.0

print(f"\n--- 單一 XGBoost 預測結果 ---")
print(f"RMSE (MW)     : {rmse:.4f}")
print(f"MAE (MW)      : {mae:.4f}")
print(f"NMAE (%)      : {nmae:.2f}%")
print(f"R2           : {r2:.4f}")
print(f"Skill Score   : +{skill:.2f}% (相較於 Persistence)")
