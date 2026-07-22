#!/usr/bin/env python3
"""
BSMI 測風塔 — 特徵與預測目標適配性驗證 (Feature Suitability Analysis)

本腳本系統化評估各類氣象特徵在不同預測任務上的貢獻度與適用性：
1. 多時程風速預測 (t+10min, t+1h, t+3h, t+6h) — 使用 Delta ML (殘差預測) 與 Direct Regression
2. 垂直風速外推 (38m -> 100m) — 比較物理 Power Law (alpha=1/7, alpha=0.060) 與 LightGBM 機器學習
3. 湍流強度 (TI) 降尺度特徵貢獻
"""

from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# --------------------------------------------------------------------------
# 0. 設定與字型 Setup
# --------------------------------------------------------------------------
def setup_cjk_font():
    import glob
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

DATA_DIR = Path("d:/wind_d/ML_wind/ml_project/data")
OUT_DIR = Path("d:/wind_d/ML_wind/ml_project/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# 1. 載入資料與構建 Lag & 差分特徵
# --------------------------------------------------------------------------
print("[1/4] 載入資料並構建特徵工程...")
df_10m = pd.read_parquet(DATA_DIR / "BSMI_10min.parquet")
df_turb = pd.read_parquet(DATA_DIR / "BSMI_turb.parquet")
df = df_10m.merge(df_turb, on="ts", how="inner")
df = df[df["is_valid"]].sort_values("ts").reset_index(drop=True)

# 時間週期編碼
h = df.ts.dt.hour + df.ts.dt.minute / 60.0
doy = df.ts.dt.dayofyear
df["hour_sin"] = np.sin(2 * np.pi * h / 24.0)
df["hour_cos"] = np.cos(2 * np.pi * h / 24.0)
df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

# 過去風速差分與滾動統計
df["WS100_diff1"] = df["WS_100_mean"] - df["WS_100_mean"].shift(1)
df["WS100_diff6"] = df["WS_100_mean"] - df["WS_100_mean"].shift(6)
df["WS100_roll_mean_1h"] = df["WS_100_mean"].shift(1).rolling(6).mean()
df["WS100_roll_std_1h"] = df["WS_100_mean"].shift(1).rolling(6).std()

# 氣壓變率 (Barometric Pressure Tendency)
df["delta_BP_1h"] = df["BP_93_mean"] - df["BP_93_mean"].shift(6)
df["delta_BP_3h"] = df["BP_93_mean"] - df["BP_93_mean"].shift(18)
df["delta_BP_6h"] = df["BP_93_mean"] - df["BP_93_mean"].shift(36)

# 預測目標：真實風速與 Delta 風速變化 (t+10m, t+1h, t+3h, t+6h)
df["target_ws_10m"] = df["WS_100_mean"].shift(-1)
df["target_ws_1h"] = df["WS_100_mean"].shift(-6)
df["target_ws_3h"] = df["WS_100_mean"].shift(-18)
df["target_ws_6h"] = df["WS_100_mean"].shift(-36)

df["target_delta_10m"] = df["target_ws_10m"] - df["WS_100_mean"]
df["target_delta_1h"] = df["target_ws_1h"] - df["WS_100_mean"]
df["target_delta_3h"] = df["target_ws_3h"] - df["WS_100_mean"]
df["target_delta_6h"] = df["target_ws_6h"] - df["WS_100_mean"]

df["year"] = df.ts.dt.year

df_clean = df.dropna().copy()
df_train = df_clean[df_clean["year"] <= 2018]
df_test = df_clean[df_clean["year"] >= 2020]

print(f"訓練集筆數 (2016-2018): {len(df_train)}, 測試集筆數 (2020-2021): {len(df_test)}")

# --------------------------------------------------------------------------
# 2. 實驗 A: 多時程風速預測 (Delta ML vs Persistence)
# --------------------------------------------------------------------------
print("\n[2/4] 執行實驗 A: 多時程風速預測與特徵重要性 (Gain%) 分析...")

features_forecast = [
    "WS_100_mean", "WS100_diff1", "WS100_diff6",
    "WS100_roll_mean_1h", "WS100_roll_std_1h",
    "WD_97_sin", "WD_97_cos", "shear_alpha",
    "BP_93_mean", "delta_BP_1h", "delta_BP_3h", "delta_BP_6h",
    "AT_95_mean", "RH_95_mean", "air_density",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos"
]

horizons_info = [
    ("10 min (+10m)", "target_ws_10m", "target_delta_10m"),
    ("1 Hour (+1h)",  "target_ws_1h",  "target_delta_1h"),
    ("3 Hours (+3h)", "target_ws_3h",  "target_delta_3h"),
    ("6 Hours (+6h)", "target_ws_6h",  "target_delta_6h"),
]

perf_list = []
imp_list = []

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": 42,
    "n_estimators": 150,
    "verbose": -1
}

for name, target_ws_col, target_delta_col in horizons_info:
    X_tr = df_train[features_forecast]
    y_tr_delta = df_train[target_delta_col]
    
    X_te = df_test[features_forecast]
    y_te_true = df_test[target_ws_col]
    base_ws = df_test["WS_100_mean"]
    
    # 物理基準: Persistence
    rmse_persist = np.sqrt(mean_squared_error(y_te_true, base_ws))
    mae_persist = mean_absolute_error(y_te_true, base_ws)
    
    # ML 模型: Delta Predictor
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(X_tr, y_tr_delta)
    
    pred_delta = model.predict(X_te)
    pred_final = base_ws + pred_delta
    
    rmse_ml = np.sqrt(mean_squared_error(y_te_true, pred_final))
    mae_ml = mean_absolute_error(y_te_true, pred_final)
    r2_ml = r2_score(y_te_true, pred_final)
    skill_score = (1.0 - (rmse_ml / rmse_persist)) * 100.0
    
    perf_list.append({
        "Horizon": name,
        "Persistence_RMSE": rmse_persist,
        "LGBM_Delta_RMSE": rmse_ml,
        "Persistence_MAE": mae_persist,
        "LGBM_Delta_MAE": mae_ml,
        "LGBM_R2": r2_ml,
        "Skill_Score (%)": skill_score
    })
    
    gain = model.booster_.feature_importance(importance_type="gain")
    gain_pct = (gain / gain.sum()) * 100.0
    for f, g in zip(features_forecast, gain_pct):
        imp_list.append({"Horizon": name, "Feature": f, "Gain_Pct": g})

df_perf = pd.DataFrame(perf_list)
df_imp = pd.DataFrame(imp_list)

print("\n--- 多時程預測 (Delta ML vs Persistence) 表現比較 ---")
print(df_perf.to_string(index=False))

df_perf.to_csv(OUT_DIR / "forecast_suitability_perf.csv", index=False)
df_imp.to_csv(OUT_DIR / "forecast_suitability_importance.csv", index=False)

# --------------------------------------------------------------------------
# 3. 實驗 B: 垂直風速外推 (38m -> 100m) 特徵適配性
# --------------------------------------------------------------------------
print("\n[3/4] 執行實驗 B: 垂直風速外推 (38m -> 100m) 特徵與模型比較...")

features_extrap = [
    "WS_38W_mean", "WS_38W_ti",
    "WD_35_sin", "WD_35_cos",
    "AT_95_mean", "RH_95_mean", "BP_93_mean", "air_density",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos"
]

y_tr_ex = df_train["WS_100_mean"]
y_te_ex = df_test["WS_100_mean"]

# 物理基準 1: Standard Power Law (alpha=1/7)
pred_pl17 = df_test["WS_38W_mean"] * ((100.0 / 38.0) ** (1.0 / 7.0))
rmse_pl17 = np.sqrt(mean_squared_error(y_te_ex, pred_pl17))
mae_pl17 = mean_absolute_error(y_te_ex, pred_pl17)
bias_pl17 = np.mean(pred_pl17 - y_te_ex)

# 物理基準 2: Site-fitted Power Law (alpha=0.060)
pred_plmed = df_test["WS_38W_mean"] * ((100.0 / 38.0) ** 0.060)
rmse_plmed = np.sqrt(mean_squared_error(y_te_ex, pred_plmed))
mae_plmed = mean_absolute_error(y_te_ex, pred_plmed)
bias_plmed = np.mean(pred_plmed - y_te_ex)

# ML 模型: LightGBM (Full Atmospheric State)
model_ex = lgb.LGBMRegressor(**lgb_params)
model_ex.fit(df_train[features_extrap], y_tr_ex)
pred_ex = model_ex.predict(df_test[features_extrap])

rmse_ex = np.sqrt(mean_squared_error(y_te_ex, pred_ex))
mae_ex = mean_absolute_error(y_te_ex, pred_ex)
bias_ex = np.mean(pred_ex - y_te_ex)
r2_ex = r2_score(y_te_ex, pred_ex)

df_extrap_perf = pd.DataFrame([
    {"Model": "Standard Power Law (alpha=1/7)", "RMSE": rmse_pl17, "MAE": mae_pl17, "Bias": bias_pl17, "R2": r2_score(y_te_ex, pred_pl17)},
    {"Model": "Site-fitted Power Law (alpha=0.060)", "RMSE": rmse_plmed, "MAE": mae_plmed, "Bias": bias_plmed, "R2": r2_score(y_te_ex, pred_plmed)},
    {"Model": "LightGBM (Full Atmospheric State)", "RMSE": rmse_ex, "MAE": mae_ex, "Bias": bias_ex, "R2": r2_ex},
])

print("\n--- 垂直風速外推 (38m -> 100m) 表現比較 ---")
print(df_extrap_perf.to_string(index=False))

gain_ex = model_ex.booster_.feature_importance(importance_type="gain")
df_imp_ex = pd.DataFrame({
    "Feature": features_extrap,
    "Gain_Pct": (gain_ex / gain_ex.sum()) * 100.0
}).sort_values("Gain_Pct", ascending=False)

df_extrap_perf.to_csv(OUT_DIR / "extrapolation_suitability_perf.csv", index=False)
df_imp_ex.to_csv(OUT_DIR / "extrapolation_suitability_importance.csv", index=False)

# --------------------------------------------------------------------------
# 4. 繪製特徵適配性綜合分析圖表
# --------------------------------------------------------------------------
print("\n[4/4] 繪製特徵適配性綜合分析圖表...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (1) 多時程預測 RMSE 與 Skill Score (%)
ax1 = axes[0, 0]
horizons = df_perf["Horizon"]
ax1.plot(horizons, df_perf["Persistence_RMSE"], 'o--', color='crimson', label="Persistence Baseline RMSE")
ax1.plot(horizons, df_perf["LGBM_Delta_RMSE"], 's-', color='navy', linewidth=2, label="Delta LightGBM RMSE")
ax1.set_ylabel("RMSE (m/s)", color="navy")
ax1.set_title("(a) 多時程預測 (Delta ML) RMSE 與 Persistence 比對")
ax1.legend(loc="upper left")

ax1_twin = ax1.twinx()
bars = ax1_twin.bar(horizons, df_perf["Skill_Score (%)"], alpha=0.3, color="teal", width=0.35, label="Skill Score (%)")
ax1_twin.set_ylabel("Skill Score (%) = (1 - RMSE_ml/RMSE_persist)*100", color="teal")
ax1_twin.set_ylim(0, 12)
for bar in bars:
    yval = bar.get_height()
    ax1_twin.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"+{yval:.2f}%", ha='center', va='bottom', fontsize=9, color="darkcyan", fontweight='bold')
ax1_twin.legend(loc="upper right")

# (2) 多時程預測: 關鍵特徵重要性 (Gain%) 隨 Horizon 變化
ax2 = axes[0, 1]
piv_imp = df_imp.pivot(index="Feature", columns="Horizon", values="Gain_Pct")
top_features = piv_imp.mean(axis=1).sort_values(ascending=False).head(7).index
piv_imp.loc[top_features].T.plot(kind="bar", stacked=False, ax=ax2, colormap="viridis")
ax2.set_title("(b) 預測特徵重要性 (Gain%) 隨 Forecast Horizon 移轉")
ax2.set_ylabel("Feature Importance Gain (%)")
ax2.set_xlabel("Forecast Horizon")
ax2.legend(title="Features", bbox_to_anchor=(1.02, 1), loc="upper left")

# (3) 垂直風速外推: 預測與真實值散佈比對
ax3 = axes[1, 0]
idx_sample = np.random.choice(len(y_te_ex), size=5000, replace=False)
ax3.scatter(y_te_ex.iloc[idx_sample], pred_pl17.iloc[idx_sample], alpha=0.15, s=8, color="crimson", label="Standard Power Law (alpha=1/7)")
ax3.scatter(y_te_ex.iloc[idx_sample], pred_ex[idx_sample], alpha=0.15, s=8, color="teal", label="LightGBM (Full State)")
ax3.plot([0, 30], [0, 30], 'k--', alpha=0.7)
ax3.set_xlim(0, 25)
ax3.set_ylim(0, 25)
ax3.set_xlabel("Measured 100m Hub Height Wind Speed (m/s)")
ax3.set_ylabel("Extrapolated 100m Wind Speed (m/s)")
ax3.set_title("(c) 垂直風速外推 (38m -> 100m) 散佈與偏差比較")
ax3.legend()

# (4) 垂直風速外推: 特徵重要性 (Gain%)
ax4 = axes[1, 1]
ax4.barh(df_imp_ex["Feature"], df_imp_ex["Gain_Pct"], color="darkcyan", alpha=0.85)
ax4.set_xlabel("Feature Importance Gain (%)")
ax4.set_title("(d) 垂直外推之特徵貢獻 (38m 風速 + 風向 + 大氣狀態)")
ax4.invert_yaxis()

plt.tight_layout()
fig_path = FIG_DIR / "feature_suitability_analysis.png"
plt.savefig(fig_path, dpi=150)
plt.close()

print(f"\n[完成] 特徵適配性驗證圖表已儲存至: {fig_path}")
