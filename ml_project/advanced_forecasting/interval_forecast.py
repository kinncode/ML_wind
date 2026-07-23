#!/usr/bin/env python3
"""
區間預測 (Interval Forecast) — Conformal Prediction
瞬時功率信賴區間 + 累積發電量信賴區間
訓練: 2016-2017, 校準: 2018, 測試: 2020-2021
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

from common import (
    load_data, FEATURES, HORIZONS, RATED_MW,
    ADV_RESULTS, ADV_FIGURES,
)

# ==========================================================================
# 1. 載入資料
# ==========================================================================
print("=" * 70)
print("Interval Forecast - Conformal Prediction")
print("=" * 70)

df_full, df_train_all, df_test = load_data()

# 區間預測特殊拆分: Train 2016-2017, Cal 2018
df_train = df_train_all[df_train_all["year"] <= 2017].copy()
df_calib = df_train_all[df_train_all["year"] == 2018].copy()
print(f"[INTERVAL] Train(2016-2017): {len(df_train)}, Cal(2018): {len(df_calib)}, Test(2020-2021): {len(df_test)}")

# ==========================================================================
# 2. 訓練 & Conformal Prediction
# ==========================================================================
ALPHA = 0.10  # 90% CI

results_records = []
interval_data = {}

for hor_name, shift_n, expected_gap in HORIZONS:
    target_col = f"target_power_{shift_n}"
    print(f"\n--- Horizon: {hor_name} (shift={shift_n}) ---")

    mask_tr = df_train[target_col].notna()
    mask_cal = df_calib[target_col].notna()
    mask_te = df_test[target_col].notna()

    X_tr = df_train.loc[mask_tr, FEATURES]
    y_tr = df_train.loc[mask_tr, target_col]
    X_cal = df_calib.loc[mask_cal, FEATURES]
    y_cal = df_calib.loc[mask_cal, target_col]
    X_te = df_test.loc[mask_te, FEATURES]
    y_te = df_test.loc[mask_te, target_col].values
    ts_te = df_test.loc[mask_te, "ts"].values
    base_power = df_test.loc[mask_te, "sim_power_mw"].values

    print(f"  Train: {len(X_tr)}, Calib: {len(X_cal)}, Test: {len(X_te)}")

    # Train LightGBM
    model = lgb.LGBMRegressor(
        objective="regression", metric="rmse", learning_rate=0.03,
        num_leaves=45, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=20, random_state=42, n_estimators=250, verbose=-1,
    )
    model.fit(X_tr, y_tr)

    # Calibration residuals
    pred_cal = np.clip(model.predict(X_cal), 0.0, RATED_MW)
    residuals = np.abs(y_cal.values - pred_cal)

    # Conformal quantile
    q = np.ceil((1 - ALPHA) * (len(residuals) + 1)) / len(residuals)
    q = min(q, 1.0)
    conformal_q = np.quantile(residuals, q)
    print(f"  Conformal 90% half-width: {conformal_q:.4f} MW")

    # Test predictions + intervals
    pred_te = np.clip(model.predict(X_te), 0.0, RATED_MW)
    lower = np.clip(pred_te - conformal_q, 0.0, RATED_MW)
    upper = np.clip(pred_te + conformal_q, 0.0, RATED_MW)

    # Metrics
    coverage = np.mean((y_te >= lower) & (y_te <= upper))
    avg_width = np.mean(upper - lower)
    rmse = np.sqrt(mean_squared_error(y_te, pred_te))
    mae = mean_absolute_error(y_te, pred_te)
    r2 = r2_score(y_te, pred_te)

    # Winkler Score
    winkler_scores = np.where(
        y_te < lower, (upper - lower) + (2 / ALPHA) * (lower - y_te),
        np.where(
            y_te > upper, (upper - lower) + (2 / ALPHA) * (y_te - upper),
            upper - lower
        )
    )
    winkler = np.mean(winkler_scores)

    # Skill Score
    rmse_persist = np.sqrt(mean_squared_error(y_te, base_power))
    skill = (1.0 - rmse / rmse_persist) * 100.0

    results_records.append({
        "Horizon": hor_name, "Shift": shift_n,
        "RMSE_MW": rmse, "MAE_MW": mae, "R2": r2,
        "Skill_Score_%": skill,
        "Coverage_90%": coverage,
        "Avg_Interval_Width_MW": avg_width,
        "Conformal_HalfWidth_MW": conformal_q,
        "Winkler_Score": winkler,
    })

    interval_data[hor_name] = {
        "ts": ts_te, "true": y_te, "pred": pred_te,
        "lower": lower, "upper": upper, "base": base_power,
    }

    print(f"  RMSE={rmse:.3f} MW, MAE={mae:.3f} MW, R2={r2:.3f}")
    print(f"  Coverage={coverage:.3%}, Width={avg_width:.3f} MW, Winkler={winkler:.3f}")

# ==========================================================================
# 3. 累積發電量區間 (MWh)
# ==========================================================================
print("\n--- Cumulative Energy (MWh) Intervals ---")
DT_HOURS = 10.0 / 60.0

cum_records = []
cum_data = {}

for hor_name, shift_n, _ in HORIZONS:
    conformal_q_single = [r for r in results_records if r["Horizon"] == hor_name][0]["Conformal_HalfWidth_MW"]
    y_te_all = interval_data[hor_name]["true"]
    pred_te_all = interval_data[hor_name]["pred"]

    energy_actual = y_te_all * shift_n * DT_HOURS
    energy_pred = pred_te_all * shift_n * DT_HOURS
    energy_halfwidth = conformal_q_single * shift_n * DT_HOURS
    energy_lower = np.clip(energy_pred - energy_halfwidth, 0, None)
    energy_upper = energy_pred + energy_halfwidth

    coverage_e = np.mean((energy_actual >= energy_lower) & (energy_actual <= energy_upper))
    avg_width_e = np.mean(energy_upper - energy_lower)

    cum_records.append({
        "Horizon": hor_name,
        "Window_Hours": shift_n * DT_HOURS,
        "Energy_Coverage_90%": coverage_e,
        "Energy_Avg_Width_MWh": avg_width_e,
    })

    cum_data[hor_name] = {
        "actual_mwh": energy_actual, "pred_mwh": energy_pred,
        "lower_mwh": energy_lower, "upper_mwh": energy_upper,
    }

    print(f"  {hor_name}: Coverage={coverage_e:.3%}, Width={avg_width_e:.3f} MWh")

# ==========================================================================
# 4. 保存數據
# ==========================================================================
df_results = pd.DataFrame(results_records)
df_results.to_csv(ADV_RESULTS / "interval_forecast_metrics.csv", index=False)
df_cum = pd.DataFrame(cum_records)
df_cum.to_csv(ADV_RESULTS / "interval_cumulative_energy.csv", index=False)
print(f"\n[SAVE] Metrics saved to {ADV_RESULTS}")

# ==========================================================================
# 5. 圖表繪製
# ==========================================================================
print("\n[PLOT] Generating interval forecast figures...")

# Fig 1: Time series with 90% CI band
fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=False)
for ax_idx, (plot_hor, plot_title) in enumerate([("1h", "+1h"), ("6h", "+6h")]):
    ax = axes[ax_idx]
    d = interval_data[plot_hor]
    n_show = 500
    ts = d["ts"][:n_show]
    ax.fill_between(ts, d["lower"][:n_show], d["upper"][:n_show],
                    alpha=0.25, color="#3498db", label="90% CI")
    ax.plot(ts, d["true"][:n_show], color="#2c3e50", linewidth=1.5, label="Actual")
    ax.plot(ts, d["pred"][:n_show], color="#e74c3c", linewidth=1.2, linestyle="--", label="Prediction")
    ax.set_ylabel("Power (MW)", fontsize=11)
    ax.set_title(f"Conformal Prediction {plot_title} Power Interval", fontsize=13, pad=10)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(-0.3, 5.5)
axes[1].set_xlabel("Time", fontsize=11)
plt.tight_layout()
plt.savefig(ADV_FIGURES / "interval_timeseries_ci.png", dpi=300)
plt.close()
print("  [OK] interval_timeseries_ci.png")

# Fig 2: Coverage & Width bars
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
hor_labels = [r["Horizon"] for r in results_records]
coverages = [r["Coverage_90%"] * 100 for r in results_records]
widths = [r["Avg_Interval_Width_MW"] for r in results_records]

bars1 = ax1.bar(hor_labels, coverages, color=["#2ecc71" if c >= 88 else "#e74c3c" for c in coverages],
                edgecolor="white", linewidth=1.5)
ax1.axhline(y=90, color="#2c3e50", linestyle="--", linewidth=1.5, label="Nominal 90%")
ax1.set_ylim(0, 105)
ax1.set_ylabel("Coverage (%)", fontsize=11)
ax1.set_xlabel("Horizon", fontsize=11)
ax1.set_title("Conformal 90% CI Coverage", fontsize=13, pad=10)
ax1.legend(fontsize=10)
ax1.grid(axis="y", linestyle="--", alpha=0.4)
for bar, val in zip(bars1, coverages):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
             f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")

bars2 = ax2.bar(hor_labels, widths, color="#3498db", edgecolor="white", linewidth=1.5)
ax2.set_ylabel("Avg Interval Width (MW)", fontsize=11)
ax2.set_xlabel("Horizon", fontsize=11)
ax2.set_title("90% CI Average Width", fontsize=13, pad=10)
ax2.grid(axis="y", linestyle="--", alpha=0.4)
for bar, val in zip(bars2, widths):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
             f"{val:.2f}", ha="center", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig(ADV_FIGURES / "interval_coverage_width.png", dpi=300)
plt.close()
print("  [OK] interval_coverage_width.png")

# Fig 3: Width boxplot
fig, ax = plt.subplots(figsize=(10, 6))
box_data = [interval_data[h[0]]["upper"] - interval_data[h[0]]["lower"] for h in HORIZONS]
box_labels = [h[0] for h in HORIZONS]
bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True,
                boxprops=dict(facecolor="#3498db", alpha=0.6),
                medianprops=dict(color="#e74c3c", linewidth=2),
                whiskerprops=dict(color="#2c3e50"),
                flierprops=dict(marker=".", markersize=2, alpha=0.3))
ax.set_ylabel("Interval Width (MW)", fontsize=11)
ax.set_xlabel("Horizon", fontsize=11)
ax.set_title("90% CI Width Distribution", fontsize=13, pad=10)
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(ADV_FIGURES / "interval_width_boxplot.png", dpi=300)
plt.close()
print("  [OK] interval_width_boxplot.png")

# Fig 4: Power vs Energy intervals (6h example)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))
d_6h = interval_data["6h"]
n_show = 300
ts = d_6h["ts"][:n_show]

ax1.fill_between(ts, d_6h["lower"][:n_show], d_6h["upper"][:n_show],
                 alpha=0.25, color="#9b59b6", label="Power 90% CI")
ax1.plot(ts, d_6h["true"][:n_show], color="#2c3e50", linewidth=1.5, label="Actual (MW)")
ax1.plot(ts, d_6h["pred"][:n_show], color="#e67e22", linewidth=1.2, linestyle="--", label="Predicted (MW)")
ax1.set_ylabel("Instantaneous Power (MW)", fontsize=11)
ax1.set_title("+6h: Instantaneous Power Interval", fontsize=13, pad=10)
ax1.legend(loc="upper right", fontsize=9)
ax1.grid(True, linestyle="--", alpha=0.4)
ax1.set_ylim(-0.3, 5.5)

cd_6h = cum_data["6h"]
ax2.fill_between(ts, cd_6h["lower_mwh"][:n_show], cd_6h["upper_mwh"][:n_show],
                 alpha=0.25, color="#27ae60", label="Energy 90% CI")
ax2.plot(ts, cd_6h["actual_mwh"][:n_show], color="#2c3e50", linewidth=1.5, label="Actual (MWh)")
ax2.plot(ts, cd_6h["pred_mwh"][:n_show], color="#e67e22", linewidth=1.2, linestyle="--", label="Predicted (MWh)")
ax2.set_ylabel("Cumulative Energy (MWh)", fontsize=11)
ax2.set_xlabel("Time", fontsize=11)
ax2.set_title("+6h: Cumulative Energy (MWh) Interval", fontsize=13, pad=10)
ax2.legend(loc="upper right", fontsize=9)
ax2.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig(ADV_FIGURES / "interval_power_vs_energy.png", dpi=300)
plt.close()
print("  [OK] interval_power_vs_energy.png")

print("\n" + "=" * 70)
print("Interval Forecast DONE!")
print("=" * 70)
