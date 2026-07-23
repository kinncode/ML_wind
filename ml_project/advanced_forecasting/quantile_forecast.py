#!/usr/bin/env python3
"""
Probabilistic Forecast - LightGBM Quantile Regression
P10 / P50 / P90
Train: 2016-2018, Test: 2020-2021
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
print("=" * 70)
print("Probabilistic Forecast - P10 / P50 / P90 Quantile Regression")
print("=" * 70)

df_full, df_train, df_test = load_data()

QUANTILES = [0.10, 0.50, 0.90]
Q_NAMES = {0.10: "P10", 0.50: "P50", 0.90: "P90"}

# ==========================================================================
# Train P10/P50/P90 (4 horizons x 3 quantiles = 12 models)
# ==========================================================================
print(f"\n[TRAIN] Training {len(HORIZONS)} x {len(QUANTILES)} = {len(HORIZONS)*len(QUANTILES)} quantile models...")

quantile_preds = {}
pinball_records = []

for hor_name, shift_n, _ in HORIZONS:
    target_col = f"target_power_{shift_n}"
    mask_tr = df_train[target_col].notna()
    mask_te = df_test[target_col].notna()

    X_tr = df_train.loc[mask_tr, FEATURES]
    y_tr = df_train.loc[mask_tr, target_col]
    X_te = df_test.loc[mask_te, FEATURES]
    y_te = df_test.loc[mask_te, target_col].values
    ts_te = df_test.loc[mask_te, "ts"].values
    base_power = df_test.loc[mask_te, "sim_power_mw"].values

    print(f"\n--- {hor_name}: Train={len(X_tr)}, Test={len(X_te)} ---")

    preds = {"true": y_te, "ts": ts_te, "base": base_power}

    for q in QUANTILES:
        q_name = Q_NAMES[q]
        model = lgb.LGBMRegressor(
            objective="quantile", alpha=q,
            learning_rate=0.03, num_leaves=45,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, random_state=42,
            n_estimators=250, verbose=-1,
        )
        model.fit(X_tr, y_tr)
        pred_q = np.clip(model.predict(X_te), 0.0, RATED_MW)
        preds[q_name] = pred_q

        errors = y_te - pred_q
        pinball = np.mean(np.where(errors >= 0, q * errors, (q - 1) * errors))
        pinball_records.append({
            "Horizon": hor_name, "Quantile": q_name,
            "Alpha": q, "Pinball_Loss": pinball,
        })
        print(f"  {q_name}: Pinball Loss = {pinball:.4f}")

    # Enforce monotonicity: P10 <= P50 <= P90
    preds["P10"] = np.minimum(preds["P10"], preds["P50"])
    preds["P90"] = np.maximum(preds["P90"], preds["P50"])
    quantile_preds[hor_name] = preds

# ==========================================================================
# Evaluate: CRPS, Reliability, Sharpness
# ==========================================================================
print("\n[EVAL] Computing CRPS, Reliability, Sharpness...")

eval_records = []

for hor_name, shift_n, _ in HORIZONS:
    d = quantile_preds[hor_name]
    y = d["true"]
    p50 = d["P50"]
    base = d["base"]

    rmse = np.sqrt(mean_squared_error(y, p50))
    mae = mean_absolute_error(y, p50)
    r2 = r2_score(y, p50)
    rmse_persist = np.sqrt(mean_squared_error(y, base))
    skill = (1.0 - rmse / rmse_persist) * 100.0

    pl_vals = [r["Pinball_Loss"] for r in pinball_records if r["Horizon"] == hor_name]
    crps_approx = np.mean(pl_vals)

    actual_below_p10 = np.mean(y < d["P10"])
    actual_below_p50 = np.mean(y < d["P50"])
    actual_below_p90 = np.mean(y < d["P90"])
    sharpness = np.mean(d["P90"] - d["P10"])

    eval_records.append({
        "Horizon": hor_name,
        "P50_RMSE_MW": rmse, "P50_MAE_MW": mae, "P50_R2": r2,
        "P50_Skill_%": skill,
        "CRPS_approx": crps_approx,
        "Sharpness_MW": sharpness,
        "Actual_below_P10": actual_below_p10,
        "Actual_below_P50": actual_below_p50,
        "Actual_below_P90": actual_below_p90,
    })

    print(f"  {hor_name}: RMSE={rmse:.3f}, R2={r2:.3f}, CRPS={crps_approx:.4f}, "
          f"Sharpness={sharpness:.3f} MW")
    print(f"    Reliability: P(<P10)={actual_below_p10:.3f}(target 0.10), "
          f"P(<P50)={actual_below_p50:.3f}(target 0.50), "
          f"P(<P90)={actual_below_p90:.3f}(target 0.90)")

# ==========================================================================
# Save data
# ==========================================================================
pd.DataFrame(pinball_records).to_csv(ADV_RESULTS / "quantile_pinball_loss.csv", index=False)
pd.DataFrame(eval_records).to_csv(ADV_RESULTS / "quantile_forecast_metrics.csv", index=False)
print(f"\n[SAVE] Metrics saved to {ADV_RESULTS}")

# ==========================================================================
# Figures
# ==========================================================================
print("\n[PLOT] Generating quantile forecast figures...")

# Fig 1: Fan Chart
fig, axes = plt.subplots(2, 1, figsize=(16, 10))
for ax_idx, (plot_hor, plot_title) in enumerate([("1h", "+1h"), ("6h", "+6h")]):
    ax = axes[ax_idx]
    d = quantile_preds[plot_hor]
    n_show = 500
    ts = d["ts"][:n_show]
    ax.fill_between(ts, d["P10"][:n_show], d["P90"][:n_show],
                    alpha=0.20, color="#e74c3c", label="P10-P90 (80%)")
    ax.plot(ts, d["P50"][:n_show], color="#e74c3c", linewidth=1.3, linestyle="--", label="P50 (median)")
    ax.plot(ts, d["true"][:n_show], color="#2c3e50", linewidth=1.5, label="Actual")
    ax.set_ylabel("Power (MW)", fontsize=11)
    ax.set_title(f"Quantile Fan Chart - {plot_title}", fontsize=13, pad=10)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(-0.3, 5.5)
axes[1].set_xlabel("Time", fontsize=11)
plt.tight_layout()
plt.savefig(ADV_FIGURES / "quantile_fan_chart.png", dpi=300)
plt.close()
print("  [OK] quantile_fan_chart.png")

# Fig 2: Reliability Diagram
fig, ax = plt.subplots(figsize=(8, 8))
nominal = [0.10, 0.50, 0.90]
colors = ["#3498db", "#e67e22", "#2ecc71", "#e74c3c"]
for i, (hor_name, _, _) in enumerate(HORIZONS):
    rec = [r for r in eval_records if r["Horizon"] == hor_name][0]
    observed = [rec["Actual_below_P10"], rec["Actual_below_P50"], rec["Actual_below_P90"]]
    ax.plot(nominal, observed, "o-", color=colors[i], linewidth=2, markersize=8, label=hor_name)
ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.5, label="Perfect Reliability")
ax.set_xlabel("Nominal Quantile", fontsize=12)
ax.set_ylabel("Observed Frequency", fontsize=12)
ax.set_title("Quantile Reliability Diagram", fontsize=14, pad=12)
ax.legend(fontsize=10)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_aspect("equal")
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(ADV_FIGURES / "quantile_reliability.png", dpi=300)
plt.close()
print("  [OK] quantile_reliability.png")

# Fig 3: Pinball Loss bars
fig, ax = plt.subplots(figsize=(10, 6))
hor_labels = [h[0] for h in HORIZONS]
x = np.arange(len(hor_labels))
width = 0.25
for i, q in enumerate(QUANTILES):
    q_name = Q_NAMES[q]
    vals = [r["Pinball_Loss"] for r in pinball_records if r["Quantile"] == q_name]
    bars = ax.bar(x + i * width, vals, width, label=q_name,
                  color=["#3498db", "#e67e22", "#2ecc71"][i], edgecolor="white", linewidth=1)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", fontsize=8)
ax.set_xticks(x + width)
ax.set_xticklabels(hor_labels)
ax.set_ylabel("Pinball Loss (MW)", fontsize=11)
ax.set_xlabel("Horizon", fontsize=11)
ax.set_title("Pinball Loss by Quantile and Horizon", fontsize=13, pad=10)
ax.legend(fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(ADV_FIGURES / "quantile_pinball_loss.png", dpi=300)
plt.close()
print("  [OK] quantile_pinball_loss.png")

# Fig 4: CRPS
fig, ax = plt.subplots(figsize=(9, 5.5))
crps_vals = [r["CRPS_approx"] for r in eval_records]
bars = ax.bar(hor_labels, crps_vals, color="#9b59b6", edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, crps_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
            f"{val:.4f}", ha="center", fontsize=10, fontweight="bold")
ax.set_ylabel("CRPS (MW)", fontsize=11)
ax.set_xlabel("Horizon", fontsize=11)
ax.set_title("CRPS (Continuous Ranked Probability Score) by Horizon", fontsize=13, pad=10)
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(ADV_FIGURES / "quantile_crps.png", dpi=300)
plt.close()
print("  [OK] quantile_crps.png")

# Fig 5: Sharpness boxplot
fig, ax = plt.subplots(figsize=(10, 6))
sharp_data = [quantile_preds[h[0]]["P90"] - quantile_preds[h[0]]["P10"] for h in HORIZONS]
bp = ax.boxplot(sharp_data, tick_labels=hor_labels, patch_artist=True,
                boxprops=dict(facecolor="#e74c3c", alpha=0.5),
                medianprops=dict(color="#2c3e50", linewidth=2),
                whiskerprops=dict(color="#2c3e50"),
                flierprops=dict(marker=".", markersize=2, alpha=0.3))
ax.set_ylabel("P90 - P10 Width (MW)", fontsize=11)
ax.set_xlabel("Horizon", fontsize=11)
ax.set_title("Sharpness (P90-P10) Distribution by Horizon", fontsize=13, pad=10)
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(ADV_FIGURES / "quantile_sharpness.png", dpi=300)
plt.close()
print("  [OK] quantile_sharpness.png")

print("\n" + "=" * 70)
print("Probabilistic Forecast DONE!")
print("=" * 70)
