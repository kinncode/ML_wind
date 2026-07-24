#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 6 —— 區間發電能量預測特徵重要度 (Feature Importance) 深度分析與視覺化

功能：
  1. 跨預測時間區間 (1h, 3h, 6h, 24h) 提煉 LightGBM 特徵 Gain 得分與 Split 次數。
  2. 將 44 個特徵歸類為 5 大功能群組：
     - 風速現值與滯後 (Wind Speed Current & Lags)
     - 滾動趨勢與斜率 (Rolling Trends & Slopes)
     - 湍流與多高度風切 (Turbulence & Shear)
     - 氣象與空氣密度 (Atmosphere & Density)
     - 時間週期與風向 (Cyclical & Direction)
  3. 量化分析不同時間區間下「哪類特徵最關鍵」（如 1h 依賴滾動斜率，24h 依賴晝夜週期）。

輸出：
  results/interval_feature_importance_summary.csv
  figures/fig6_interval_feature_importance_top20.png
  figures/fig7_interval_feature_category_breakdown.png
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import lightgbm as lgb
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

def classify_feature(f_name: str) -> str:
    """將特徵歸類至 5 大功能群組。"""
    if f_name.startswith("hour_") or f_name.startswith("doy_") or f_name.startswith("WD_"):
        return "時間週期與風向"
    elif f_name.startswith("air_density") or f_name.startswith("AT_") or f_name.startswith("RH_") or f_name.startswith("BP_"):
        return "氣壓/溫度/密度"
    elif "shear" in f_name or "ti" in f_name or "gust" in f_name or "std" in f_name:
        return "湍流與風切"
    elif "rmean" in f_name or "rstd" in f_name or "rmin" in f_name or "rmax" in f_name or "slope" in f_name or "diff" in f_name:
        return "滾動趨勢與斜率"
    else:
        return "風速現值與滯後"

def main():
    print("="*70)
    print("PW_Interval Stage 6 —— 區間預測特徵重要度深度分析")
    print("="*70)

    horizons = [1, 3, 6, 24]
    all_imp_data = []

    # 1. 載入或重新計算各時間區間之 LightGBM 特徵 Gain 得分
    for h in horizons:
        model_path = os.path.join(C.MODEL_DIR, f"lgbm_power_H{h}.txt")
        imp_csv = os.path.join(C.RES_DIR, f"importance_power_H{h}.csv")

        if os.path.exists(imp_csv):
            df_imp = pd.read_csv(imp_csv)
        elif os.path.exists(model_path):
            bst = lgb.Booster(model_file=model_path)
            fnames = bst.feature_name()
            gains = bst.feature_importance("gain")
            df_imp = pd.DataFrame({"feature": fnames, "gain": gains}).sort_values("gain", ascending=False)
        else:
            print(f"提示：找不到 H={h}h 模型檔，跳過該時間視窗。")
            continue

        df_imp["horizon"] = f"{h}h"
        df_imp["category"] = df_imp["feature"].apply(classify_feature)
        # 正規化 Gain 比例 (%)
        df_imp["gain_pct"] = (df_imp["gain"] / df_imp["gain"].sum()) * 100.0
        all_imp_data.append(df_imp)

    if not all_imp_data:
        print("錯誤：找不到特徵重要度數據，請先執行 03_train_select.py 或 13_day_ahead_24h_settlement.py")
        return

    df_all_imp = pd.concat(all_imp_data, ignore_index=True)
    csv_out = os.path.join(C.RES_DIR, "interval_feature_importance_summary.csv")
    df_all_imp.to_csv(csv_out, index=False)
    print(f"特徵重要度總表已寫入：{csv_out}")

    fp = FontProperties(fname="C:/Windows/Fonts/msjh.ttc", size=10) if os.path.exists("C:/Windows/Fonts/msjh.ttc") else None
    fp_title = FontProperties(fname="C:/Windows/Fonts/msjh.ttc", size=12, weight="bold") if os.path.exists("C:/Windows/Fonts/msjh.ttc") else None

    # ------------------------------------------------------------------
    # 圖表 1: Top 20 核心特徵在 1h, 3h, 6h 區間的 Gain 得分排名
    # ------------------------------------------------------------------
    df_h3 = df_all_imp[df_all_imp["horizon"] == "3h"].sort_values("gain", ascending=True).tail(20)

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(df_h3["feature"], df_h3["gain_pct"], color="#2ca02c", alpha=0.85, edgecolor="black")

    ax.set_title("區間能量預測 (Power 3h) Top 20 核心特徵貢獻度 (Gain %)", fontproperties=fp_title)
    ax.set_xlabel("相對貢獻度比例 Gain (%)", fontproperties=fp)
    ax.grid(True, linestyle="--", alpha=0.5)

    for bar in bars:
        w_val = bar.get_width()
        ax.annotate(f"{w_val:.1f}%", xy=(w_val, bar.get_y() + bar.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points", ha="left", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig6_path = os.path.join(C.FIG_DIR, "fig6_interval_feature_importance_top20.png")
    plt.savefig(fig6_path, dpi=200)
    plt.close()
    print(f"Top 20 特徵重要度圖表已產出：{fig6_path}")

    # ------------------------------------------------------------------
    # 圖表 2: 5 大特徵功能群組在不同時間視窗 (1h, 3h, 6h, 24h) 的變化趨勢
    # ------------------------------------------------------------------
    cat_grp = df_all_imp.groupby(["horizon", "category"])["gain_pct"].sum().reset_index()
    piv_cat = cat_grp.pivot(index="horizon", columns="category", values="gain_pct").fillna(0)

    # 確保順序 1h -> 3h -> 6h -> 24h
    avail_h = [f"{h}h" for h in horizons if f"{h}h" in piv_cat.index]
    piv_cat = piv_cat.reindex(avail_h)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 子圖 A: 群組變化堆疊長條圖 (Stacked Bar)
    piv_cat.plot(kind="bar", stacked=True, ax=axes[0], colormap="tab10", width=0.55, edgecolor="black")
    axes[0].set_title("不同預測視窗下 5 大特徵群組權重佔比變化 (%)", fontproperties=fp_title)
    axes[0].set_xlabel("預測時間區間 H", fontproperties=fp)
    axes[0].set_ylabel("累積特徵貢獻度 (%)", fontproperties=fp)
    axes[0].set_xticks(range(len(avail_h)))
    axes[0].set_xticklabels(avail_h, fontproperties=fp)
    axes[0].legend(prop=fp, loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # 子圖 B: 3h 區間特徵群組圓餅圖 (Pie Chart)
    if "3h" in piv_cat.index:
        cat_3h = piv_cat.loc["3h"]
        axes[1].pie(cat_3h, labels=cat_3h.index, autopct="%1.1f%%", startangle=140,
                    colors=plt.cm.tab10(np.linspace(0, 1, len(cat_3h))),
                    textprops={'fontproperties': fp})
        axes[1].set_title("3 小時區間能量預測 特徵群組貢獻圓餅圖", fontproperties=fp_title)

    plt.tight_layout()
    fig7_path = os.path.join(C.FIG_DIR, "fig7_interval_feature_category_breakdown.png")
    plt.savefig(fig7_path, dpi=200)
    plt.close()
    print(f"特徵群組結構圖表已產出：{fig7_path}")

    # 打印文字摘要
    print("\n--- 各時間區間 5 大特徵群組貢獻比例摘要 (%) ---")
    print(piv_cat.to_string())

    print("\nStage 6 特徵重要度深度分析完成。")

if __name__ == "__main__":
    main()
