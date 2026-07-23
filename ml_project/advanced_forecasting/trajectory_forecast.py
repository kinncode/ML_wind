#!/usr/bin/env python3
"""
Trajectory Forecast - Recursive Multi-Step LightGBM
10-min steps, up to 24h (144 steps)
Train: 2016-2018, Test: 2020-2021
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from collections import deque

from common import (
    load_data, FEATURES, HORIZONS, RATED_MW,
    NREL_5MW_WS, NREL_5MW_KW,
    ADV_RESULTS, ADV_FIGURES,
)

# ==========================================================================
print("=" * 70)
print("Trajectory Forecast - Recursive Multi-Step LightGBM")
print("=" * 70)

df_full, df_train, df_test = load_data()

MAX_STEPS = 144
ANCHOR_STEPS = [1, 6, 36, 144]
ANCHOR_NAMES = {1: "10min", 6: "1h", 36: "6h", 144: "24h"}

# ==========================================================================
# Train 1-step-ahead model (t -> t+10m)
# ==========================================================================
print("\n[TRAIN] Training 1-step-ahead LightGBM (target = t+10min)...")

target_1step = "target_power_1"
mask_tr = df_train[target_1step].notna()

model_1step = lgb.LGBMRegressor(
    objective="regression", metric="rmse", learning_rate=0.03,
    num_leaves=45, subsample=0.8, colsample_bytree=0.8,
    min_child_samples=20, random_state=42, n_estimators=250, verbose=-1,
)
model_1step.fit(df_train.loc[mask_tr, FEATURES], df_train.loc[mask_tr, target_1step])
print(f"  1-step model trained on {mask_tr.sum()} samples")

# Train Direct models as baselines
print("[TRAIN] Training Direct baseline models...")
direct_models = {}
for hor_name, shift_n, _ in HORIZONS:
    t_col = f"target_power_{shift_n}"
    mask = df_train[t_col].notna()
    m = lgb.LGBMRegressor(
        objective="regression", metric="rmse", learning_rate=0.03,
        num_leaves=45, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=20, random_state=42, n_estimators=250, verbose=-1,
    )
    m.fit(df_train.loc[mask, FEATURES], df_train.loc[mask, t_col])
    direct_models[shift_n] = m
    print(f"  Direct {hor_name} trained")

# ==========================================================================
# Recursive prediction
# ==========================================================================
print(f"\n[PREDICT] Starting recursive prediction (max {MAX_STEPS} steps)...")

ts_test = df_test["ts"].values
dt_expected = pd.Timedelta(minutes=10)

# Find continuous segments
gaps = np.where(np.diff(ts_test.astype("datetime64[ns]").astype(np.int64))
                != dt_expected.value)[0]
seg_starts = np.concatenate([[0], gaps + 1])
seg_ends = np.concatenate([gaps + 1, [len(ts_test)]])
segments = [(s, e) for s, e in zip(seg_starts, seg_ends) if (e - s) >= MAX_STEPS + 18]
print(f"  Found {len(segments)} continuous segments (len >= {MAX_STEPS + 18})")

# Sample start indices (every 36 steps = 6h)
start_indices = []
for seg_s, seg_e in segments:
    for i in range(seg_s, seg_e - MAX_STEPS, 36):
        start_indices.append(i)
print(f"  Total {len(start_indices)} trajectory start points")

# Feature index map
feat_idx = {f: FEATURES.index(f) for f in FEATURES}

# Collect step-wise predictions
step_preds = {k: [] for k in range(1, MAX_STEPS + 1)}
step_trues = {k: [] for k in range(1, MAX_STEPS + 1)}

test_features = df_test[FEATURES].values
test_power = df_test["sim_power_mw"].values
test_ts = df_test["ts"].values


def compute_time_features(base_ts, step):
    future_ts = base_ts + np.timedelta64(step * 10, "m")
    future_dt = pd.Timestamp(future_ts)
    h = future_dt.hour + future_dt.minute / 60.0
    doy = future_dt.dayofyear
    return {
        "hour_sin": np.sin(2 * np.pi * h / 24.0),
        "hour_cos": np.cos(2 * np.pi * h / 24.0),
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
    }


def update_power_features(feat, power_history, pred_power):
    """Update power-related features in the feature vector."""
    ph = list(power_history)
    n_ph = len(ph)

    feat[feat_idx["sim_power_mw"]] = pred_power

    for lag, key in [(1, "Power_lag_1"), (2, "Power_lag_2"), (3, "Power_lag_3"),
                     (6, "Power_lag_6"), (12, "Power_lag_12"), (18, "Power_lag_18")]:
        if n_ph > lag:
            feat[feat_idx[key]] = ph[-(lag + 1)]

    if n_ph > 1:
        feat[feat_idx["Power_diff1"]] = pred_power - ph[-2]
    if n_ph > 6:
        feat[feat_idx["Power_diff6"]] = pred_power - ph[-7]

    if n_ph >= 7:
        recent6 = ph[-7:-1]
        feat[feat_idx["Power_roll_mean_1h"]] = np.mean(recent6)
        feat[feat_idx["Power_roll_std_1h"]] = np.std(recent6, ddof=1) if len(recent6) > 1 else 0.0
        feat[feat_idx["Power_roll_max_1h"]] = np.max(recent6)
    if n_ph >= 19:
        recent18 = ph[-19:-1]
        feat[feat_idx["Power_roll_mean_3h"]] = np.mean(recent18)

    return feat


n_total = len(start_indices)
report_interval = max(1, n_total // 10)

for prog, start_i in enumerate(start_indices):
    if prog % report_interval == 0:
        print(f"  Progress: {prog}/{n_total} ({prog/n_total*100:.0f}%)")

    feat = test_features[start_i].copy()

    power_history = deque(maxlen=20)
    for j in range(max(0, start_i - 19), start_i + 1):
        power_history.append(test_power[j])

    base_ts = test_ts[start_i]

    for step_k in range(1, MAX_STEPS + 1):
        target_idx = start_i + step_k
        if target_idx >= len(test_power):
            break

        pred_power = np.clip(model_1step.predict(feat.reshape(1, -1))[0], 0.0, RATED_MW)

        step_preds[step_k].append(pred_power)
        step_trues[step_k].append(test_power[target_idx])

        power_history.append(pred_power)
        feat = update_power_features(feat, power_history, pred_power)

        tf = compute_time_features(base_ts, step_k)
        for k, v in tf.items():
            feat[feat_idx[k]] = v

print("  Recursive prediction complete")

# ==========================================================================
# Direct model predictions (baselines)
# ==========================================================================
print("\n[DIRECT] Computing Direct baseline predictions...")
direct_preds = {}
direct_trues = {}
for hor_name, shift_n, _ in HORIZONS:
    t_col = f"target_power_{shift_n}"
    mask = df_test[t_col].notna()
    X_te = df_test.loc[mask, FEATURES]
    y_te = df_test.loc[mask, t_col].values
    pred = np.clip(direct_models[shift_n].predict(X_te), 0.0, RATED_MW)
    direct_preds[shift_n] = pred
    direct_trues[shift_n] = y_te

# ==========================================================================
# Evaluation
# ==========================================================================
print("\n[EVAL] Computing step-wise metrics...")

stepwise_records = []
for step_k in range(1, MAX_STEPS + 1):
    if len(step_preds[step_k]) < 10:
        continue
    p = np.array(step_preds[step_k])
    t = np.array(step_trues[step_k])
    rmse = np.sqrt(mean_squared_error(t, p))
    mae = mean_absolute_error(t, p)
    r2 = r2_score(t, p) if np.var(t) > 0 else 0.0
    stepwise_records.append({
        "Step": step_k, "Minutes": step_k * 10,
        "RMSE_MW": rmse, "MAE_MW": mae, "R2": r2,
        "N_samples": len(p),
    })

df_stepwise = pd.DataFrame(stepwise_records)

# Direct baselines at anchor points
direct_anchor = {}
for shift_n in ANCHOR_STEPS:
    if shift_n in direct_preds:
        p = direct_preds[shift_n]
        t = direct_trues[shift_n]
        direct_anchor[shift_n] = {
            "RMSE_MW": np.sqrt(mean_squared_error(t, p)),
            "MAE_MW": mean_absolute_error(t, p),
            "R2": r2_score(t, p),
        }

# Comparison table
print("\n--- Recursive vs Direct at anchor points ---")
compare_records = []
for step_n in ANCHOR_STEPS:
    name = ANCHOR_NAMES[step_n]
    rec_row = df_stepwise[df_stepwise["Step"] == step_n]
    if len(rec_row) == 0:
        continue
    rec = rec_row.iloc[0]
    dir_vals = direct_anchor.get(step_n, {})
    compare_records.append({
        "Horizon": name, "Step": step_n,
        "Recursive_RMSE": rec["RMSE_MW"],
        "Recursive_MAE": rec["MAE_MW"],
        "Recursive_R2": rec["R2"],
        "Direct_RMSE": dir_vals.get("RMSE_MW", np.nan),
        "Direct_MAE": dir_vals.get("MAE_MW", np.nan),
        "Direct_R2": dir_vals.get("R2", np.nan),
    })
    print(f"  {name}: Recursive RMSE={rec['RMSE_MW']:.3f} vs Direct RMSE={dir_vals.get('RMSE_MW', np.nan):.3f}")

df_compare = pd.DataFrame(compare_records)

# Trajectory smoothness
smoothness_recursive = []
for prog, start_i in enumerate(start_indices[:500]):
    traj = []
    for step_k in range(1, min(MAX_STEPS + 1, len(test_power) - start_i)):
        idx = len(step_preds[step_k]) - len(start_indices) + prog
        if 0 <= idx < len(step_preds[step_k]):
            traj.append(step_preds[step_k][idx])
    if len(traj) > 1:
        smoothness_recursive.extend(np.abs(np.diff(traj)).tolist())

# ==========================================================================
# Save
# ==========================================================================
df_stepwise.to_csv(ADV_RESULTS / "trajectory_stepwise_metrics.csv", index=False)
df_compare.to_csv(ADV_RESULTS / "trajectory_vs_direct.csv", index=False)
print(f"\n[SAVE] Metrics saved to {ADV_RESULTS}")

# ==========================================================================
# Figures
# ==========================================================================
print("\n[PLOT] Generating trajectory figures...")

# Fig 1: Trajectory time series (3 examples)
fig, axes = plt.subplots(3, 1, figsize=(16, 12))
show_indices = [start_indices[i] for i in [0, len(start_indices) // 3, len(start_indices) * 2 // 3]
                if i < len(start_indices)]

for ax_idx, si in enumerate(show_indices[:3]):
    ax = axes[ax_idx]
    n_steps = min(MAX_STEPS, len(test_power) - si - 1)

    true_traj = test_power[si:si + n_steps + 1]
    pred_traj = [test_power[si]]

    feat = test_features[si].copy()
    power_history = deque(maxlen=20)
    for j in range(max(0, si - 19), si + 1):
        power_history.append(test_power[j])
    base_ts = test_ts[si]

    for step_k in range(1, n_steps + 1):
        p = np.clip(model_1step.predict(feat.reshape(1, -1))[0], 0.0, RATED_MW)
        pred_traj.append(p)
        power_history.append(p)
        feat = update_power_features(feat, power_history, p)
        tf = compute_time_features(base_ts, step_k)
        for k, v in tf.items():
            feat[feat_idx[k]] = v

    time_axis = np.arange(len(true_traj)) * 10 / 60
    ax.plot(time_axis, true_traj, color="#2c3e50", linewidth=2, label="Actual")
    ax.plot(time_axis[:len(pred_traj)], pred_traj, color="#e74c3c", linewidth=1.5,
            linestyle="--", label="Recursive Prediction")
    ax.axvline(x=1, color="#95a5a6", linestyle=":", alpha=0.7)
    ax.axvline(x=6, color="#95a5a6", linestyle=":", alpha=0.7)

    start_time = pd.Timestamp(test_ts[si]).strftime("%Y-%m-%d %H:%M")
    ax.set_title(f"Trajectory #{ax_idx+1} (Start: {start_time})", fontsize=12, pad=8)
    ax.set_ylabel("Power (MW)", fontsize=10)
    ax.set_ylim(-0.3, 5.5)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

axes[2].set_xlabel("Forecast Lead Time (hours)", fontsize=11)
plt.suptitle("Recursive Trajectory vs Actual (24h)", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(ADV_FIGURES / "trajectory_timeseries.png", dpi=300, bbox_inches="tight")
plt.close()
print("  [OK] trajectory_timeseries.png")

# Fig 2: RMSE decay curve
fig, ax = plt.subplots(figsize=(12, 6))
steps = df_stepwise["Step"].values
rmse_vals = df_stepwise["RMSE_MW"].values
minutes = df_stepwise["Minutes"].values

ax.plot(minutes / 60, rmse_vals, color="#e74c3c", linewidth=2, label="Recursive (cumulative)")
for step_n in ANCHOR_STEPS:
    if step_n in direct_anchor:
        ax.scatter(step_n * 10 / 60, direct_anchor[step_n]["RMSE_MW"],
                   s=120, zorder=5, color="#3498db", edgecolors="#2c3e50", linewidth=1.5,
                   label=f"Direct {ANCHOR_NAMES[step_n]}")
ax.set_xlabel("Forecast Lead Time (hours)", fontsize=12)
ax.set_ylabel("RMSE (MW)", fontsize=12)
ax.set_title("Recursive RMSE Decay vs Direct Anchors", fontsize=14, pad=12)
ax.legend(fontsize=10)
ax.grid(True, linestyle="--", alpha=0.4)
ax.set_xlim(0, 25)
plt.tight_layout()
plt.savefig(ADV_FIGURES / "trajectory_rmse_decay.png", dpi=300)
plt.close()
print("  [OK] trajectory_rmse_decay.png")

# Fig 3: R2 decay curve
fig, ax = plt.subplots(figsize=(12, 6))
r2_vals = df_stepwise["R2"].values
ax.plot(minutes / 60, r2_vals, color="#27ae60", linewidth=2, label="Recursive R2")
for step_n in ANCHOR_STEPS:
    if step_n in direct_anchor:
        ax.scatter(step_n * 10 / 60, direct_anchor[step_n]["R2"],
                   s=120, zorder=5, color="#e67e22", edgecolors="#2c3e50", linewidth=1.5,
                   label=f"Direct {ANCHOR_NAMES[step_n]}")
ax.set_xlabel("Forecast Lead Time (hours)", fontsize=12)
ax.set_ylabel("R2", fontsize=12)
ax.set_title("Recursive R2 Decay Curve", fontsize=14, pad=12)
ax.axhline(y=0, color="#95a5a6", linestyle=":", alpha=0.7)
ax.legend(fontsize=10)
ax.grid(True, linestyle="--", alpha=0.4)
ax.set_xlim(0, 25)
plt.tight_layout()
plt.savefig(ADV_FIGURES / "trajectory_r2_decay.png", dpi=300)
plt.close()
print("  [OK] trajectory_r2_decay.png")

# Fig 4: Smoothness histogram
fig, ax = plt.subplots(figsize=(10, 6))
if smoothness_recursive:
    ax.hist(smoothness_recursive, bins=80, density=True, alpha=0.6,
            color="#e74c3c", edgecolor="white", label="Recursive step jumps")
real_jumps = np.abs(np.diff(test_power))
ax.hist(real_jumps, bins=80, density=True, alpha=0.5,
        color="#3498db", edgecolor="white", label="Actual step jumps")
ax.set_xlabel("|Delta Power| per step (MW)", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("Trajectory Smoothness - Step Jump Distribution", fontsize=13, pad=10)
ax.set_xlim(0, 3)
ax.legend(fontsize=10)
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(ADV_FIGURES / "trajectory_smoothness.png", dpi=300)
plt.close()
print("  [OK] trajectory_smoothness.png")

print("\n" + "=" * 70)
print("Trajectory Forecast DONE!")
print("=" * 70)
