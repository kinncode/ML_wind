#!/usr/bin/env python3
"""
BSMI 測風塔 — 風力發電預測可行性驗證與特徵重要性分析 (Wind Power Forecasting & Feature Importance Analysis)

本腳本執行以下核心分析：
1. 模擬 NREL 5MW 離岸風機功率曲線 (含空氣密度動態修正)
2. 建立多時程預測標的 (t+10m, t+1h, t+3h, t+6h)
3. 特徵相關性分析 (Pearson 線性相關 & Spearman 秩相關)
4. 機器學習特徵重要性 (LightGBM Gain % & Split count)
5. 可行性評估 (Skill Score 相較於 Persistence 物理基準線)
6. 繪製並保存高解析度視覺化圖表
"""

from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import os

# --------------------------------------------------------------------------
# 0. 字型與路徑設定 Setup
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
# 1. NREL 5MW 離岸風機功率曲線模擬 (IEC 空氣密度修正)
# --------------------------------------------------------------------------
def simulate_nrel_5mw_power(ws_100, air_density):
    """
    依據 IEC 61400-12-1 計算空氣密度修正後之有效風速 (Effective Wind Speed)，
    並透過 NREL 5MW 離岸參考風機功率曲線計算發電功率 (kW)。
    Rotor diameter = 126m, Rated power = 5000 kW.
    """
    rho_0 = 1.225  # 標準空氣密度 (kg/m3)
    # 空氣密度修正有效風速
    v_eff = ws_100 * (air_density / rho_0) ** (1.0 / 3.0)
    
    # 功率曲線邏輯
    power = np.zeros_like(v_eff)
    
    # Region 2 (3.0 m/s <= v_eff < 11.4 m/s): 三次方漸進功率曲線
    mask_reg2 = (v_eff >= 3.0) & (v_eff < 11.4)
    # Cp_max 約 0.485, Swept area A = pi * (63^2) = 12469 m2
    # P_mech = 0.5 * 1.225 * 12469 * 0.485 * v_eff^3 / 1000 (kW) ~ 3.704 * v_eff^3
    p_reg2 = 3.704 * (v_eff[mask_reg2] ** 3)
    power[mask_reg2] = np.minimum(5000.0, p_reg2)
    
    # Region 3 (11.4 m/s <= v_eff <= 25.0 m/s): 額定功率 5000 kW
    mask_reg3 = (v_eff >= 11.4) & (v_eff <= 25.0)
    power[mask_reg3] = 5000.0
    
    # Cut-in (<3 m/s) 與 Cut-out (>25 m/s) 皆為 0 kW
    return power

# --------------------------------------------------------------------------
# 2. 資料載入與特徵工程
# --------------------------------------------------------------------------
print("[1/5] 載入 BSMI 測風塔資料並生成風機發電量標的與特徵...")
df_10m = pd.read_parquet(DATA_DIR / "BSMI_10min.parquet")
df_turb = pd.read_parquet(DATA_DIR / "BSMI_turb.parquet")
df = df_10m.merge(df_turb, on="ts", how="inner")
df = df[df["is_valid"]].sort_values("ts").reset_index(drop=True)

# 模擬 5MW 發電量 (kW)
df["sim_power_kw"] = simulate_nrel_5mw_power(df["WS_100_mean"].values, df["air_density"].values)
df["sim_power_mw"] = df["sim_power_kw"] / 1000.0

# 時間週期特徵
h = df.ts.dt.hour + df.ts.dt.minute / 60.0
doy = df.ts.dt.dayofyear
df["hour_sin"] = np.sin(2 * np.pi * h / 24.0)
df["hour_cos"] = np.cos(2 * np.pi * h / 24.0)
df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

# 風速與功率 Lag / 滾動統計特徵
df["WS100_diff1"] = df["WS_100_mean"] - df["WS_100_mean"].shift(1)
df["WS100_diff6"] = df["WS_100_mean"] - df["WS_100_mean"].shift(6)
df["WS100_roll_mean_1h"] = df["WS_100_mean"].shift(1).rolling(6).mean()
df["WS100_roll_std_1h"] = df["WS_100_mean"].shift(1).rolling(6).std()

df["Power_diff1"] = df["sim_power_mw"] - df["sim_power_mw"].shift(1)
df["Power_diff6"] = df["sim_power_mw"] - df["sim_power_mw"].shift(6)
df["Power_roll_mean_1h"] = df["sim_power_mw"].shift(1).rolling(6).mean()
df["Power_roll_std_1h"] = df["sim_power_mw"].shift(1).rolling(6).std()

# 氣壓趨勢 (Barometric Pressure Tendency)
df["delta_BP_1h"] = df["BP_93_mean"] - df["BP_93_mean"].shift(6)
df["delta_BP_3h"] = df["BP_93_mean"] - df["BP_93_mean"].shift(18)
df["delta_BP_6h"] = df["BP_93_mean"] - df["BP_93_mean"].shift(36)

# 多時程發電量目標 (t+10m, t+1h, t+3h, t+6h)
df["target_power_10m"] = df["sim_power_mw"].shift(-1)
df["target_power_1h"] = df["sim_power_mw"].shift(-6)
df["target_power_3h"] = df["sim_power_mw"].shift(-18)
df["target_power_6h"] = df["sim_power_mw"].shift(-36)

# Delta 殘差預測標的
df["target_delta_power_10m"] = df["target_power_10m"] - df["sim_power_mw"]
df["target_delta_power_1h"] = df["target_power_1h"] - df["sim_power_mw"]
df["target_delta_power_3h"] = df["target_power_3h"] - df["sim_power_mw"]
df["target_delta_power_6h"] = df["target_power_6h"] - df["sim_power_mw"]

df["year"] = df.ts.dt.year

df_clean = df.dropna().copy()
df_train = df_clean[df_clean["year"] <= 2018].copy()
df_test = df_clean[df_clean["year"] >= 2020].copy()

print(f"資料總比數: {len(df_clean)}, 訓練集 (2016-2018): {len(df_train)}, 測試集 (2020-2021): {len(df_test)}")

# --------------------------------------------------------------------------
# 3. 特徵相關性分析 (Pearson & Spearman Correlation)
# --------------------------------------------------------------------------
print("\n[2/5] 計算氣象/物理特徵與未來發電量之 Pearson 與 Spearman 相關係數...")

feature_cols = [
    "sim_power_mw", "WS_100_mean", "WS100_diff1", "WS100_diff6",
    "WS100_roll_mean_1h", "WS100_roll_std_1h",
    "Power_diff1", "Power_diff6", "Power_roll_mean_1h", "Power_roll_std_1h",
    "shear_alpha", "WS_100E_ti", "WS_100E_gust_factor",
    "WD_97_sin", "WD_97_cos",
    "BP_93_mean", "delta_BP_1h", "delta_BP_3h", "delta_BP_6h",
    "AT_95_mean", "RH_95_mean", "air_density",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos"
]

targets = ["target_power_10m", "target_power_1h", "target_power_3h", "target_power_6h"]

corr_records = []
for f in feature_cols:
    rec = {"Feature": f}
    for t in targets:
        p_corr = df_clean[f].corr(df_clean[t], method="pearson")
        s_corr = df_clean[f].corr(df_clean[t], method="spearman")
        rec[f"{t}_Pearson"] = p_corr
        rec[f"{t}_Spearman"] = s_corr
    corr_records.append(rec)

df_corr = pd.DataFrame(corr_records)
df_corr.to_csv(OUT_DIR / "power_feature_correlation.csv", index=False)
print("相關係數分析完成，已保存至 power_feature_correlation.csv")

# --------------------------------------------------------------------------
# 4. 機器學習特徵重要性 (LightGBM Multi-horizon Forecasting)
# --------------------------------------------------------------------------
print("\n[3/5] 執行多時程發電量預測與 LightGBM 特徵重要性 (Gain %) 分析...")

horizons_info = [
    ("10 min (+10m)", "target_power_10m", "target_delta_power_10m"),
    ("1 Hour (+1h)",  "target_power_1h",  "target_delta_power_1h"),
    ("3 Hours (+3h)", "target_power_3h",  "target_delta_power_3h"),
    ("6 Hours (+6h)", "target_power_6h",  "target_delta_power_6h"),
]

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": 42,
    "n_estimators": 150,
    "verbose": -1
}

perf_list = []
imp_list = []

model_predictions = {}

for name, target_power_col, target_delta_col in horizons_info:
    X_tr = df_train[feature_cols]
    y_tr_delta = df_train[target_delta_col]
    
    X_te = df_test[feature_cols]
    y_te_true = df_test[target_power_col]
    base_power = df_test["sim_power_mw"]
    
    # 物理基準 1: Persistence (發電量(t+h) = 發電量(t))
    rmse_persist = np.sqrt(mean_squared_error(y_te_true, base_power))
    mae_persist = mean_absolute_error(y_te_true, base_power)
    
    # ML 模型: Delta Regressor
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(X_tr, y_tr_delta)
    
    pred_delta = model.predict(X_te)
    pred_final = np.clip(base_power + pred_delta, 0.0, 5.0)  # 物理上限 5.0 MW
    
    model_predictions[name] = pred_final
    
    rmse_ml = np.sqrt(mean_squared_error(y_te_true, pred_final))
    mae_ml = mean_absolute_error(y_te_true, pred_final)
    r2_ml = r2_score(y_te_true, pred_final)
    
    # NMAE (%) 相對於風機容量 5MW
    nmae_persist = (mae_persist / 5.0) * 100.0
    nmae_ml = (mae_ml / 5.0) * 100.0
    
    skill_score = (1.0 - (rmse_ml / rmse_persist)) * 100.0
    
    perf_list.append({
        "Horizon": name,
        "Persistence_RMSE_MW": rmse_persist,
        "LGBM_Delta_RMSE_MW": rmse_ml,
        "Persistence_MAE_MW": mae_persist,
        "LGBM_Delta_MAE_MW": mae_ml,
        "Persistence_NMAE (%)": nmae_persist,
        "LGBM_Delta_NMAE (%)": nmae_ml,
        "LGBM_R2": r2_ml,
        "Skill_Score (%)": skill_score
    })
    
    gain = model.booster_.feature_importance(importance_type="gain")
    splits = model.booster_.feature_importance(importance_type="split")
    gain_pct = (gain / gain.sum()) * 100.0
    
    for f, g, s in zip(feature_cols, gain_pct, splits):
        imp_list.append({"Horizon": name, "Feature": f, "Gain_Pct": g, "Split_Count": s})

df_perf = pd.DataFrame(perf_list)
df_imp = pd.DataFrame(imp_list)

print("\n--- 風力發電預測模型表現與基準對比表 (MW) ---")
print(df_perf.to_string(index=False))

df_perf.to_csv(OUT_DIR / "power_forecast_perf.csv", index=False)
df_imp.to_csv(OUT_DIR / "power_feature_importance.csv", index=False)

# --------------------------------------------------------------------------
# 5. 繪製專業高解析度視覺化圖表
# --------------------------------------------------------------------------
print("\n[4/5] 繪製特徵相關性、重要性與可行性預測圖表...")

# (A) 圖 1: 特徵相關性熱圖 (Spearman Rank Correlation)
plt.figure(figsize=(12, 10))
corr_matrix = df_corr.set_index("Feature")[["target_power_10m_Spearman", "target_power_1h_Spearman", "target_power_3h_Spearman", "target_power_6h_Spearman"]]
corr_matrix.columns = ["+10min", "+1h", "+3h", "+6h"]
corr_matrix = corr_matrix.sort_values(by="+10min", ascending=False)

sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="coolwarm", cbar_kws={'label': 'Spearman Correlation ($r_s$)'}, vmin=-0.8, vmax=0.8)
plt.title("BSMI 風場特徵與未來自發電量 Spearman 相關係數矩陣", fontsize=14, pad=15)
plt.xlabel("預測時間提前量 (Horizon)", fontsize=12)
plt.ylabel("氣象 / 物理特徵", fontsize=12)
plt.tight_layout()
plt.savefig(FIG_DIR / "power_correlation_heatmap.png", dpi=300)
plt.close()
print("已生成圖表: power_correlation_heatmap.png")

# (B) 圖 2: 多時程特徵重要性 (Gain %) 比較
plt.figure(figsize=(14, 8))
# 取 1h 與 6h 前 14 大特徵
top_features = df_imp.groupby("Feature")["Gain_Pct"].mean().nlargest(14).index
df_imp_top = df_imp[df_imp["Feature"].isin(top_features)]

sns.barplot(data=df_imp_top, x="Gain_Pct", y="Feature", hue="Horizon", palette="viridis")
plt.title("LightGBM 風力發電預測特徵重要性 (Gain %) 隨預測時程變化", fontsize=14, pad=15)
plt.xlabel("特徵重要性貢獻度 Gain (%)", fontsize=12)
plt.ylabel("特徵名稱", fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(FIG_DIR / "power_feature_importance.png", dpi=300)
plt.close()
print("已生成圖表: power_feature_importance.png")

# (C) 圖 3: 預測可行性與 Skill Score 隨預測時程上升圖
fig, ax1 = plt.subplots(figsize=(10, 6))

horizons = df_perf["Horizon"].values
x = np.arange(len(horizons))

width = 0.35
rects1 = ax1.bar(x - width/2, df_perf["Persistence_RMSE_MW"], width, label='Persistence Baseline (RMSE MW)', color='#bdc3c7')
rects2 = ax1.bar(x + width/2, df_perf["LGBM_Delta_RMSE_MW"], width, label='LightGBM Delta ML (RMSE MW)', color='#2ecc71')

ax1.set_ylabel('預測均方根誤差 RMSE (MW)', fontsize=12)
ax1.set_title('風力發電預測可行性驗證：Persistence 物理基準 vs LightGBM (Skill Score)', fontsize=14, pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(horizons, fontsize=11)
ax1.legend(loc='upper left')
ax1.grid(axis='y', linestyle='--', alpha=0.5)

# 雙軸標註 Skill Score
ax2 = ax1.twinx()
ax2.plot(x, df_perf["Skill_Score (%)"], color='#e74c3c', marker='o', linewidth=2.5, markersize=8, label='Skill Score (%)')
ax2.set_ylabel('技術得分 Skill Score (% 改善率)', color='#e74c3c', fontsize=12)
ax2.tick_params(axis='y', labelcolor='#e74c3c')
ax2.set_ylim(0, 45)

for i, txt in enumerate(df_perf["Skill_Score (%)"]):
    ax2.annotate(f"+{txt:.1f}%", (x[i], txt + 1.2), ha='center', color='#c0392b', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig(FIG_DIR / "power_forecast_feasibility.png", dpi=300)
plt.close()
print("已生成圖表: power_forecast_feasibility.png")

# (D) 圖 4: 測試集實際發電量 vs ML 預測值時序曲線 (1 小時預測視角範例)
plt.figure(figsize=(14, 6))
sample_slice = df_test.iloc[1000:1500]  # 取約 3.5 天的連續數據
ts_sample = sample_slice["ts"]
true_power = sample_slice["target_power_1h"]
pred_power = model_predictions["1 Hour (+1h)"][1000:1500]
persist_power = sample_slice["sim_power_mw"]

plt.plot(ts_sample, true_power, label='真實模擬發電量 (t+1h Target)', color='#2c3e50', linewidth=2)
plt.plot(ts_sample, pred_power, label='LightGBM 預測發電量', color='#e67e22', linestyle='--', linewidth=1.8)
plt.plot(ts_sample, persist_power, label='Persistence 基準線', color='#95a5a6', linestyle=':', linewidth=1.2)

plt.title("離岸風機 (+1 小時前瞻) 發電量預測時序曲線對比 (2020 年測試集截取)", fontsize=14, pad=15)
plt.xlabel("時間 (Timestamp)", fontsize=12)
plt.ylabel("風機輸出功率 (MW)", fontsize=12)
plt.legend(loc='upper right', frameon=True)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(FIG_DIR / "power_prediction_timeseries.png", dpi=300)
plt.close()
print("已生成圖表: power_prediction_timeseries.png")

print("\n[5/5] 所有可行性驗證、特徵重要性分析與視覺化輸出完成！")
