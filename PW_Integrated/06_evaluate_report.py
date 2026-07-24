#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 6 —— 整合評估報告與視覺化圖表自動化生成

動作：
  1. 載入並彙整 Stage 1–5 之結果。
  2. 自動繪製 5 張高品質圖表至 figures/ 目錄：
     - fig1_resource_assessment.png : 風資源評估（月容量因數、日夜曲線、機型敏感度）
     - fig2_model_skill_comparison.png : 點預測模型誤差 (nRMSE) 與相對 Persistence 技術得分比較
     - fig3_quantile_intervals.png : p10/p50/p90 機率預測不確定性信賴區間時序圖
     - fig4_feature_importance.png : LightGBM 模型特徵重要度 Top 15
     - fig5_timeseries_forecast.png : 測試集真實出力 vs 預測出力時序比較
  3. 輸出 results/summary_metrics.csv
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C

# 設定中文字型與樣式（必須先套用 style，再設定字型，否則 style.use 會覆蓋字型設定）
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def main():
    print("="*70)
    print("PW_Integrated Stage 6 —— 整合評估報告與圖表生成")
    print("="*70)

    os.makedirs(C.FIG_DIR, exist_ok=True)
    os.makedirs(C.RES_DIR, exist_ok=True)

    # 1. 載入資源評估 JSON
    res_stats = {}
    if os.path.exists(C.RESOURCE_JSON):
        with open(C.RESOURCE_JSON, "r", encoding="utf-8") as f:
            res_stats = json.load(f)

    # --- 圖 1：風資源評估整合圖 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if "monthly_cf" in res_stats:
        m_data = res_stats["monthly_cf"]
        months = [int(k) for k in m_data.keys()]
        cfs = [m_data[str(k)] * 100 for k in months]
        bars = axes[0].bar(months, cfs, color='#1f77b4', alpha=0.85, edgecolor='black')
        axes[0].axhline(res_stats.get("cf_overall", 0.451)*100, color='red', linestyle='--', label=f'全期平均 CF ({res_stats.get("cf_overall", 0.451)*100:.1f}%)')
        axes[0].set_title("BSMI 測風塔 逐月容量因數 (CF %)", fontsize=13, fontweight='bold')
        axes[0].set_xlabel("月份", fontsize=11)
        axes[0].set_ylabel("容量因數 (%)", fontsize=11)
        axes[0].set_xticks(months)
        axes[0].legend()
        for bar in bars:
            h = bar.get_height()
            axes[0].annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, h),
                             xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    # 功率曲線示範
    u_seq = np.linspace(0, 30, 200)
    p_seq = C.power_curve(u_seq)
    axes[1].plot(u_seq, p_seq, color='#2ca02c', linewidth=2.5, label='代表性 8 MW 離岸風機功率曲線')
    axes[1].axvline(3, color='gray', linestyle=':', label='切入風速 (3 m/s)')
    axes[1].axvline(12, color='orange', linestyle=':', label='額定風速 (12 m/s)')
    axes[1].axvline(25, color='red', linestyle=':', label='切出風速 (25 m/s)')
    axes[1].set_title("離岸風機功率曲線 (風速 → 正規化出力 P)", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("等效風速 (m/s)", fontsize=11)
    axes[1].set_ylabel("正規化出力 (0–1)", fontsize=11)
    axes[1].legend()
    plt.tight_layout()
    fig1_path = os.path.join(C.FIG_DIR, "fig1_resource_assessment.png")
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    print(f"產出圖表：{fig1_path}")

    # --- 圖 2：模型誤差 nRMSE 與技術得分比較 ---
    test_metrics_path = os.path.join(C.RES_DIR, "test_metrics.csv")
    if os.path.exists(test_metrics_path):
        df_m = pd.read_csv(test_metrics_path)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 發電量預測發展
        df_p = df_m[df_m["target"] == "power"]
        pivot_nrmse = df_p.pivot(index="H", columns="model", values="nRMSE")
        pivot_nrmse.plot(kind="bar", ax=axes[0], colormap="viridis", width=0.75, edgecolor="black")
        axes[0].set_title("發電量預測 (Power) 測試集 nRMSE 比較", fontsize=13, fontweight='bold')
        axes[0].set_xlabel("預測時程 H (小時)", fontsize=11)
        axes[0].set_ylabel("nRMSE", fontsize=11)
        axes[0].grid(True, linestyle="--", alpha=0.5)

        # 相對 Persistence 技術得分
        df_lgb = df_p[df_p["model"] == "lightgbm"]
        axes[1].plot(df_lgb["H"], df_lgb["skill_vs_persist"] * 100, marker='o', linewidth=2.5, color='#d62728', label='LightGBM 勝 Persistence (%)')
        axes[1].set_title("LightGBM 相對 Persistence 技術得分改善", fontsize=13, fontweight='bold')
        axes[1].set_xlabel("預測提前量 H (小時)", fontsize=11)
        axes[1].set_ylabel("相對改善幅度 (%)", fontsize=11)
        axes[1].set_xticks(C.HORIZONS_H)
        for _, row in df_lgb.iterrows():
            axes[1].annotate(f"+{row['skill_vs_persist']*100:.1f}%",
                             xy=(row["H"], row["skill_vs_persist"] * 100),
                             xytext=(0, 8), textcoords="offset points", ha='center', fontweight='bold')
        axes[1].legend()
        plt.tight_layout()
        fig2_path = os.path.join(C.FIG_DIR, "fig2_model_skill_comparison.png")
        plt.savefig(fig2_path, dpi=200)
        plt.close()
        print(f"產出圖表：{fig2_path}")

    # --- 圖 3：p10/p50/p90 機率預測不確定性區間 ---
    pred_q_path = os.path.join(C.DATA_DIR, "pred_quantile_H3.parquet")
    if os.path.exists(pred_q_path):
        df_q = pd.read_parquet(pred_q_path)
        df_q["ts"] = pd.to_datetime(df_q["ts"])
        snippet = df_q.iloc[1000:1000 + 6 * 24 * 5].copy()  # 5 天片段

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(snippet["ts"], snippet["y_true"], color='black', label='真實出力 (True P)', linewidth=1.5)
        ax.plot(snippet["ts"], snippet["q50"], color='#1f77b4', linestyle='--', label='點預測 p50', linewidth=1.5)
        ax.fill_between(snippet["ts"], snippet["q10"], snippet["q90"], color='#1f77b4', alpha=0.25, label='p10–p90 不確定性信賴區間')
        ax.set_title("超短期 (+3h) 風電發電量分位數機率預測時序", fontsize=13, fontweight='bold')
        ax.set_xlabel("時間", fontsize=11)
        ax.set_ylabel("正規化出力 P (0–1)", fontsize=11)
        ax.legend(loc='upper right')
        plt.tight_layout()
        fig3_path = os.path.join(C.FIG_DIR, "fig3_quantile_intervals.png")
        plt.savefig(fig3_path, dpi=200)
        plt.close()
        print(f"產出圖表：{fig3_path}")

    # --- 圖 4：LightGBM 特徵重要度 ---
    imp_path = os.path.join(C.RES_DIR, "importance_power_H3.csv")
    if os.path.exists(imp_path):
        df_imp = pd.read_csv(imp_path).head(15)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(df_imp["feature"][::-1], df_imp["gain"][::-1], color='#8c564b', alpha=0.85, edgecolor='black')
        ax.set_title("LightGBM 發電預測 (+3h) 特徵重要度 (Gain) Top 15", fontsize=13, fontweight='bold')
        ax.set_xlabel("Gain 得分", fontsize=11)
        plt.tight_layout()
        fig4_path = os.path.join(C.FIG_DIR, "fig4_feature_importance.png")
        plt.savefig(fig4_path, dpi=200)
        plt.close()
        print(f"產出圖表：{fig4_path}")

    # --- 圖 5：測試集真實出力 vs 預測時序比較 ---
    pred_p3_path = os.path.join(C.DATA_DIR, "pred_power_H3.parquet")
    if os.path.exists(pred_p3_path):
        df_p3 = pd.read_parquet(pred_p3_path)
        df_p3["ts"] = pd.to_datetime(df_p3["ts"])
        snippet = df_p3.iloc[2000:2000 + 6 * 24 * 7].copy()  # 1 週片段

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(snippet["ts"], snippet["y_true"], color='black', label='真實出力', linewidth=1.8)
        ax.plot(snippet["ts"], snippet["pred_lgbm"], color='#1f77b4', label='LightGBM 預測', linewidth=1.5)
        ax.plot(snippet["ts"], snippet["persist"], color='#d62728', linestyle=':', label='Persistence 基準', linewidth=1.2, alpha=0.7)
        ax.set_title("測試集 1 週時序預測比較 (Power +3h)", fontsize=13, fontweight='bold')
        ax.set_xlabel("時間", fontsize=11)
        ax.set_ylabel("正規化出力 P (0–1)", fontsize=11)
        ax.legend()
        plt.tight_layout()
        fig5_path = os.path.join(C.FIG_DIR, "fig5_timeseries_forecast.png")
        plt.savefig(fig5_path, dpi=200)
        plt.close()
        print(f"產出圖表：{fig5_path}")

    print("\nStage 6 整合報告與圖表產出完成。")

if __name__ == "__main__":
    main()
