#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 4 —— 區間發電能量預測評估報告與圖表生成 (PW_Interval 專案)

修正：採用繁體中文與相容字型設定 (Microsoft JhengHei, SimHei)，徹底解決字型無法顯示與 Glyph Warning 問題。
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import config as C

# 全局載入微軟正黑體字型檔
msjh_font = "C:/Windows/Fonts/msjh.ttc"
if os.path.exists(msjh_font):
    plt.rcParams['font.family'] = FontProperties(fname=msjh_font).get_name()
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'MingLiU', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def main():
    print("="*70)
    print("PW_Interval Stage 4 —— 評估報告與視覺化圖表生成 (修正字型顯示)")
    print("="*70)

    os.makedirs(C.FIG_DIR, exist_ok=True)
    os.makedirs(C.RES_DIR, exist_ok=True)

    test_metrics_path = os.path.join(C.RES_DIR, "test_metrics.csv")
    if not os.path.exists(test_metrics_path):
        raise FileNotFoundError(f"找不到 {test_metrics_path}，請先執行 03_train_select.py")

    df_m = pd.read_csv(test_metrics_path)

    fp = FontProperties(fname="C:/Windows/Fonts/msjh.ttc", size=10) if os.path.exists("C:/Windows/Fonts/msjh.ttc") else None
    fp_title = FontProperties(fname="C:/Windows/Fonts/msjh.ttc", size=12, weight="bold") if os.path.exists("C:/Windows/Fonts/msjh.ttc") else None

    # --- 圖 1：區間發電量 nRMSE 與 R² 比較 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df_p = df_m[df_m["target"] == "power"]

    # nRMSE
    pivot_nrmse = df_p.pivot(index="H", columns="model", values="nRMSE")
    pivot_nrmse.plot(kind="bar", ax=axes[0], colormap="viridis", width=0.75, edgecolor="black")
    axes[0].set_title("區間發電能量 (Power) 測試集 nRMSE 比較", fontproperties=fp_title)
    axes[0].set_xlabel("預測時間區間 H (小時)", fontproperties=fp)
    axes[0].set_ylabel("nRMSE (越低越好)", fontproperties=fp)
    axes[0].legend(prop=fp)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # R² 擬合度
    pivot_r2 = df_p.pivot(index="H", columns="model", values="R2")
    pivot_r2.plot(kind="bar", ax=axes[1], colormap="Set2", width=0.75, edgecolor="black")
    axes[1].set_title("區間發電能量 擬合度 R² (越接近 1.0 越精準)", fontproperties=fp_title)
    axes[1].set_xlabel("預測時間區間 H (小時)", fontproperties=fp)
    axes[1].set_ylabel("R² 得分", fontproperties=fp)
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(prop=fp)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig1_path = os.path.join(C.FIG_DIR, "fig1_interval_skill_comparison.png")
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    print(f"產出圖表：{fig1_path}")

    # --- 圖 2：特徵重要度 Top 15 ---
    imp_path = os.path.join(C.RES_DIR, "importance_power_H3.csv")
    if os.path.exists(imp_path):
        df_imp = pd.read_csv(imp_path).head(15)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(df_imp["feature"][::-1], df_imp["gain"][::-1], color='#2ca02c', alpha=0.85, edgecolor='black')
        ax.set_title("區間能量預測 (Power 3h) LightGBM 特徵重要度 Gain Top 15", fontproperties=fp_title)
        ax.set_xlabel("Gain 得分", fontproperties=fp)
        plt.tight_layout()
        fig2_path = os.path.join(C.FIG_DIR, "fig2_interval_feature_importance.png")
        plt.savefig(fig2_path, dpi=200)
        plt.close()
        print(f"產出圖表：{fig2_path}")

    # --- 圖 3：區間能量擬合散佈圖 (R²) ---
    pred_p3_path = os.path.join(C.DATA_DIR, "pred_power_H3.parquet")
    if os.path.exists(pred_p3_path):
        df_p3 = pd.read_parquet(pred_p3_path)
        fig, ax = plt.subplots(figsize=(7, 6))
        idx_sub = np.random.choice(len(df_p3), size=min(5000, len(df_p3)), replace=False)
        r2_val = df_m[(df_m["target"]=="power") & (df_m["H"]==3) & (df_m["model"]=="lightgbm")]["R2"].values[0]

        ax.scatter(df_p3["y_true"].values[idx_sub], df_p3["pred_lgbm"].values[idx_sub], alpha=0.25, color='#2ca02c', s=12, label='測試集樣本')
        ax.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=2, label='1:1 理想對角線')
        ax.set_title(f"區間能量 (3h) 真實值 vs 預測值 擬合散佈圖 (R² = {r2_val:.4f})", fontproperties=fp_title)
        ax.set_xlabel("真實 3h 區間累積發電能量", fontproperties=fp)
        ax.set_ylabel("LightGBM 預測 3h 區間累積能量", fontproperties=fp)
        ax.annotate(f'R² = {r2_val:.4f}\n(消弭雜訊後高度精準)', xy=(0.05, 0.82), xycoords='axes fraction',
                    fontproperties=fp_title, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.legend(prop=fp)
        plt.tight_layout()
        fig3_path = os.path.join(C.FIG_DIR, "fig3_interval_energy_scatter.png")
        plt.savefig(fig3_path, dpi=200)
        plt.close()
        print(f"產出圖表：{fig3_path}")

    # --- 圖 4：1 週測試集時序比較 ---
    if os.path.exists(pred_p3_path):
        df_p3["ts"] = pd.to_datetime(df_p3["ts"])
        snippet = df_p3.iloc[2000:2000 + 6 * 24 * 7].copy()

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(snippet["ts"], snippet["y_true"], color='black', label='真實 3h 區間發電能量', linewidth=2.0)
        ax.plot(snippet["ts"], snippet["pred_lgbm"], color='#2ca02c', label='LightGBM 區間能量預測', linewidth=1.8, linestyle='--')
        ax.plot(snippet["ts"], snippet["persist"], color='#d62728', linestyle=':', label='Persistence 基準', linewidth=1.2, alpha=0.7)
        ax.set_title("測試集 1 週區間能量預測時序比對 (Power H=3h)", fontproperties=fp_title)
        ax.set_xlabel("時間", fontproperties=fp)
        ax.set_ylabel("區間累積能量 (等效滿載小時數 h)", fontproperties=fp)
        ax.legend(prop=fp)
        plt.tight_layout()
        fig4_path = os.path.join(C.FIG_DIR, "fig4_interval_timeseries.png")
        plt.savefig(fig4_path, dpi=200)
        plt.close()
        print(f"產出圖表：{fig4_path}")

    print("\nStage 4 評估圖表產出完成。")

if __name__ == "__main__":
    main()
