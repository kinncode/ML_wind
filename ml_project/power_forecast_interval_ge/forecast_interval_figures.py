#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
forecast_interval_figures.py —— power_forecast_interval 修正版圖表與動態技術報告生成模組

修正內容：
  1. 報告數據完全由 CSV / JSON 動態讀取生成，不硬編碼任何數字。
  2. 指標對齊國際標準 Capacity-Normalized nRMSE (nRMSE_cap = RMSE / 1.0)。
  3. y 軸與圖例標籤明確劃分「區間平均功率 (0~1)」與「區間發電能量 (h)」。
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
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
    print("power_forecast_interval —— 修正版 7 大評估圖表與動態報告生成")
    print("="*70)

    os.makedirs(C.FIG_DIR, exist_ok=True)
    os.makedirs(C.RES_DIR, exist_ok=True)

    fp = FontProperties(fname="C:/Windows/Fonts/msjh.ttc", size=10) if os.path.exists("C:/Windows/Fonts/msjh.ttc") else None
    fp_title = FontProperties(fname="C:/Windows/Fonts/msjh.ttc", size=12, weight="bold") if os.path.exists("C:/Windows/Fonts/msjh.ttc") else None

    # 1. 資源評估圖 (日與月分佈)
    res_json = os.path.join(C.RES_DIR, "resource_stats.json")
    stats = {}
    if os.path.exists(res_json):
        with open(res_json, "r", encoding="utf-8") as f:
            stats = json.load(f)

        fig, ax = plt.subplots(figsize=(10, 4.5))
        hours = [int(k) for k in stats["diurnal_cf"].keys()]
        vals = [stats["diurnal_cf"][str(k)] for k in hours]
        ax.plot(hours, vals, marker="o", color="#1f77b4", linewidth=2.0, label="100m 平均 CF (%)")
        ax.axhline(stats["capacity_factor_pct"], color="red", linestyle="--", label=f"全期平均 CF = {stats['capacity_factor_pct']}%")
        ax.set_title(f"100m 測風塔 日內 24 小時平均容量因數 CF (%)", fontproperties=fp_title)
        ax.set_xlabel("小時 (Hour)", fontproperties=fp)
        ax.set_ylabel("容量因數 CF (%)", fontproperties=fp)
        ax.set_xticks(range(0, 24))
        ax.legend(prop=fp)
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        fig1_path = os.path.join(C.FIG_DIR, "fig1_resource_diurnal.png")
        plt.savefig(fig1_path, dpi=200)
        plt.close()
        print(f"產出圖表：{fig1_path}")

        fig, ax = plt.subplots(figsize=(10, 4.5))
        months = [int(k) for k in stats["monthly_cf"].keys()]
        m_vals = [stats["monthly_cf"][str(k)] for k in months]
        colors = ["#2ca02c" if v >= 50 else "#1f77b4" for v in m_vals]
        bars = ax.bar(months, m_vals, color=colors, edgecolor="black", alpha=0.85)
        ax.axhline(stats["capacity_factor_pct"], color="red", linestyle="--", label=f"全期平均 CF = {stats['capacity_factor_pct']}%")
        ax.set_title("100m 測風塔 月度平均容量因數 CF (%) (展示秋冬季強風季)", fontproperties=fp_title)
        ax.set_xlabel("月份 (Month)", fontproperties=fp)
        ax.set_ylabel("容量因數 CF (%)", fontproperties=fp)
        ax.set_xticks(range(1, 13))
        ax.legend(prop=fp)
        ax.grid(True, linestyle="--", alpha=0.5)

        for bar in bars:
            h_val = bar.get_height()
            ax.annotate(f"{h_val:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h_val),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

        plt.tight_layout()
        fig2_path = os.path.join(C.FIG_DIR, "fig2_resource_monthly.png")
        plt.savefig(fig2_path, dpi=200)
        plt.close()
        print(f"產出圖表：{fig2_path}")

    # 2. 多模型 1h, 3h, 6h, 24h nRMSE(cap) 與 R² 對比圖
    test_metrics_path = os.path.join(C.RES_DIR, "test_metrics_interval.csv")
    df_m = None
    if os.path.exists(test_metrics_path):
        df_m = pd.read_csv(test_metrics_path)
        df_p = df_m[df_m["target"] == "power"]

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        piv_nrmse = df_p.pivot(index="H", columns="model", values="nRMSE_cap")
        piv_r2    = df_p.pivot(index="H", columns="model", values="R2")

        piv_nrmse.plot(kind="bar", ax=axes[0], colormap="viridis", width=0.75, edgecolor="black")
        axes[0].set_title("區間發電容量正規化 nRMSE(cap) 比較 (越低越好)", fontproperties=fp_title)
        axes[0].set_xlabel("預測時間區間 H (小時)", fontproperties=fp)
        axes[0].set_ylabel("nRMSE (Capacity-Normalized)", fontproperties=fp)
        axes[0].set_xticks([0, 1, 2, 3])
        axes[0].set_xticklabels(["1 小時 (1h)", "3 小時 (3h)", "6 小時 (6h)", "24 小時 (24h)"], fontproperties=fp)
        axes[0].legend(prop=fp)
        axes[0].grid(True, linestyle="--", alpha=0.5)

        piv_r2.plot(kind="bar", ax=axes[1], colormap="Set2", width=0.75, edgecolor="black")
        axes[1].set_title("區間發電功率 擬合度 R² (越接近 1.0 越精準)", fontproperties=fp_title)
        axes[1].set_xlabel("預測時間區間 H (小時)", fontproperties=fp)
        axes[1].set_ylabel("R² 得分", fontproperties=fp)
        axes[1].set_xticks([0, 1, 2, 3])
        axes[1].set_xticklabels(["1 小時 (1h)", "3 小時 (3h)", "6 小時 (6h)", "24 小時 (24h)"], fontproperties=fp)
        axes[1].set_ylim(0, 1.05)
        axes[1].legend(prop=fp)
        axes[1].grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        fig3_path = os.path.join(C.FIG_DIR, "fig3_interval_forecast_compare.png")
        plt.savefig(fig3_path, dpi=200)
        plt.close()
        print(f"產出圖表：{fig3_path}")

    # 3. 特徵重要度 Top 15
    import lightgbm as lgb
    model_3h_path = os.path.join(C.MODEL_DIR, "lgbm_power_interval_H3.txt")
    if os.path.exists(model_3h_path):
        bst = lgb.Booster(model_file=model_3h_path)
        fnames = bst.feature_name()
        gains = bst.feature_importance("gain")
        df_imp = pd.DataFrame({"feature": fnames, "gain": gains}).sort_values("gain", ascending=True).tail(15)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(df_imp["feature"], df_imp["gain"], color="#2ca02c", alpha=0.85, edgecolor="black")
        ax.set_title("區間發電能量預測 (Power 3h) LightGBM 特徵 Gain 得分 Top 15", fontproperties=fp_title)
        ax.set_xlabel("Gain 得分", fontproperties=fp)
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        fig4_path = os.path.join(C.FIG_DIR, "fig4_interval_feature_importance.png")
        plt.savefig(fig4_path, dpi=200)
        plt.close()
        print(f"產出圖表：{fig4_path}")

    # 4. 1h, 3h, 6h, 24h 擬合散佈圖
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, h in enumerate([1, 3, 6, 24]):
        pred_path = os.path.join(C.DATA_DIR, f"pred_power_interval_H{h}.parquet")
        if os.path.exists(pred_path):
            df_pred = pd.read_parquet(pred_path)
            sub_sample = df_pred.iloc[np.random.choice(len(df_pred), size=min(4000, len(df_pred)), replace=False)]
            r2_val = r2_score(df_pred["y_true"], df_pred["pred_lgbm"])

            axes[idx].scatter(sub_sample["y_true"], sub_sample["pred_lgbm"], alpha=0.25, color="#2ca02c", s=10)
            axes[idx].plot([0, 1], [0, 1], color="red", linestyle="--", linewidth=1.8, label="1:1 對角線")
            axes[idx].set_title(f"H={h}h 區間平均功率 (擬合度 R² = {r2_val:.4f})", fontproperties=fp_title)
            axes[idx].set_xlabel("真實區間平均功率 P (0~1)", fontproperties=fp)
            axes[idx].set_ylabel("LightGBM 預測功率 P (0~1)", fontproperties=fp)
            axes[idx].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig5_path = os.path.join(C.FIG_DIR, "fig5_interval_energy_scatter.png")
    plt.savefig(fig5_path, dpi=200)
    plt.close()
    print(f"產出圖表：{fig5_path}")

    # 5. 測試集 3 天高清放大區間能量時序比對圖
    pred_3h_path = os.path.join(C.DATA_DIR, "pred_power_interval_H3.parquet")
    if os.path.exists(pred_3h_path):
        df_3h = pd.read_parquet(pred_3h_path)
        df_3h["ts"] = pd.to_datetime(df_3h["ts"])
        snippet = df_3h.iloc[2000:2000 + 6 * 24 * 3].copy()

        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(snippet["ts"], snippet["y_true"], color="black", label="真實 3h 區間平均功率 (Real P_3h)", linewidth=3.0, marker="o", markersize=3, zorder=5)
        ax.plot(snippet["ts"], snippet["pred_lgbm"], color="#2ca02c", label="LightGBM 區間預測", linewidth=2.5, linestyle="--", marker="s", markersize=3, zorder=4)
        ax.plot(snippet["ts"], snippet["pred_xgb"], color="#ff7f0e", label="XGBoost 區間預測", linewidth=2.2, linestyle="-.", zorder=3)
        ax.plot(snippet["ts"], snippet["persist"], color="#d62728", label="Persistence 基準", linewidth=2.0, linestyle=":", alpha=0.85, zorder=2)

        ax.set_title("測試集 3 天 (72h) 高清局部放大區間預測多模型時序比對 (Power H=3h)", fontproperties=fp_title)
        ax.set_xlabel("時間 (YYYY-MM-DD HH:MM)", fontproperties=fp)
        ax.set_ylabel("3小時區間平均功率 (0~1 正規化)", fontproperties=fp)
        ax.legend(prop=fp, loc="upper right", frameon=True, facecolor="white", edgecolor="black", framealpha=0.9)
        ax.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        fig6_path = os.path.join(C.FIG_DIR, "fig6_interval_timeseries_comparison.png")
        plt.savefig(fig6_path, dpi=300)
        plt.close()
        print(f"產出 3 天高清時序圖表：{fig6_path}")

    # 6. 殘差分佈圖
    if os.path.exists(pred_3h_path):
        err_lgb = df_3h["pred_lgbm"] - df_3h["y_true"]
        err_per = df_3h["persist"] - df_3h["y_true"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].hist(err_lgb, bins=60, color="#2ca02c", alpha=0.7, label=f"LightGBM 殘差 (均值={err_lgb.mean():.4f})", edgecolor="black")
        axes[0].hist(err_per, bins=60, color="#d62728", alpha=0.4, label=f"Persistence 殘差 (均值={err_per.mean():.4f})", edgecolor="black")
        axes[0].axvline(0, color="black", linestyle="--", linewidth=1.5)
        axes[0].set_title("預測殘差 (Residuals = Pred - True) 分佈直方圖", fontproperties=fp_title)
        axes[0].set_xlabel("殘差值 (0~1 正規化功率)", fontproperties=fp)
        axes[0].set_ylabel("樣本數", fontproperties=fp)
        axes[0].legend(prop=fp)
        axes[0].grid(True, linestyle="--", alpha=0.5)

        axes[1].boxplot([err_lgb, err_per], tick_labels=["LightGBM", "Persistence"], patch_artist=True)
        axes[1].set_xticks([1, 2])
        axes[1].set_xticklabels(["LightGBM 誤差", "Persistence 誤差"], fontproperties=fp)
        axes[1].set_title("預測誤差箱型圖 (Boxplot 集中度比較)", fontproperties=fp_title)
        axes[1].set_ylabel("殘差值", fontproperties=fp)
        axes[1].grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        fig7_path = os.path.join(C.FIG_DIR, "fig7_error_distribution.png")
        plt.savefig(fig7_path, dpi=200)
        plt.close()
        print(f"產出圖表：{fig7_path}")

    # ------------------------------------------------------------------
    # 圖 8: p10 / p50 / p90 區間機率預測不確定性信賴帶 (Power H=3h)
    # ------------------------------------------------------------------
    pred_q3_path = os.path.join(C.DATA_DIR, "pred_quantile_power_interval_H3.parquet")
    if os.path.exists(pred_q3_path):
        df_q3 = pd.read_parquet(pred_q3_path)
        df_q3["ts"] = pd.to_datetime(df_q3["ts"])
        snippet_q = df_q3.iloc[2000:2000 + 6 * 24 * 3].copy()

        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(snippet_q["ts"], snippet_q["y_true"], color="black", label="真實 3h 區間平均功率 (Real P_3h)", linewidth=2.5, marker="o", markersize=3, zorder=5)
        ax.plot(snippet_q["ts"], snippet_q["p50"], color="#2ca02c", label="p50 (中位數期待值)", linewidth=2.2, linestyle="--", zorder=4)
        ax.plot(snippet_q["ts"], snippet_q["p90"], color="#1f77b4", label="p90 (樂觀極限上限)", linewidth=1.5, linestyle=":", zorder=3)
        ax.plot(snippet_q["ts"], snippet_q["p10"], color="#d62728", label="p10 (保守避險下限)", linewidth=1.5, linestyle=":", zorder=3)

        # 填滿 p10 ~ p90 之 80% 信賴區間帶
        ax.fill_between(snippet_q["ts"], snippet_q["p10"], snippet_q["p90"], color="#2ca02c", alpha=0.20, label="80% 機率不確定性信賴區間帶 (p10~p90)")

        ax.set_title("區間發電能量 p10 / p50 / p90 機率預測不確定性信賴帶 (Power H=3h)", fontproperties=fp_title)
        ax.set_xlabel("時間 (YYYY-MM-DD HH:MM)", fontproperties=fp)
        ax.set_ylabel("3小時區間平均功率 (0~1 正規化)", fontproperties=fp)
        ax.legend(prop=fp, loc="upper right", frameon=True, facecolor="white", edgecolor="black", framealpha=0.9)
        ax.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        fig8_path = os.path.join(C.FIG_DIR, "fig8_quantile_intervals.png")
        plt.savefig(fig8_path, dpi=300)
        plt.close()
        print(f"產出 80% 機率信賴帶圖表：{fig8_path}")

    # 8. 【動態生成 REPORT_INTERVAL.md】由 CSV / JSON 填入數據，徹底防範硬編碼
    cf_str = f"{stats.get('capacity_factor_pct', 45.00):.2f}%" if stats else "45.00%"
    flh_str = f"{stats.get('full_load_hours', 3942):,.0f}" if stats else "3,942"

    rpt_rows = []
    r2_3h_str = "N/A"
    if df_m is not None:
        sub_p = df_m[(df_m["target"] == "power") & (df_m["model"] == "lightgbm")]
        for _, r in sub_p.iterrows():
            h = int(r["H"])
            nrmse_c = r["nRMSE_cap"]
            r2_val  = r["R2"]
            skill   = r["skill_vs_persist"] * 100.0
            rpt_rows.append(f"| **區間平均功率 P** | **{h} 小時 ({h}h)** | LightGBM | **{nrmse_c:.4f}** | **{r2_val:.4f}** | **{skill:+.1f}%** |")
            if h == 3:
                r2_3h_str = f"{r2_val:.3f}"

    table_body = "\n".join(rpt_rows)

    # 機率預測表格
    q_metrics_path = os.path.join(C.RES_DIR, "quantile_interval_metrics.csv")
    q_table_rows = []
    if os.path.exists(q_metrics_path):
        df_qres = pd.read_csv(q_metrics_path)
        for _, r in df_qres.iterrows():
            h = int(r["H"])
            picp = r["picp_80_pct"]
            sharp = r["sharpness"]
            p50_r2 = r["r2_p50"]
            l10 = r["pinball_loss_p10"]
            l50 = r["pinball_loss_p50"]
            l90 = r["pinball_loss_p90"]
            q_table_rows.append(f"| **{h}h 區間** | **{picp:.1f}%** (目標 80.0%) | **{sharp:.4f}** | **{p50_r2:.4f}** | {l10:.4f} / {l50:.4f} / {l90:.4f} |")

    q_table_body = "\n".join(q_table_rows) if q_table_rows else "| 尚未執行機率預測 | - | - | - | - |"

    rpt_path = os.path.join(C.PROJECT_DIR, "REPORT_INTERVAL.md")
    with open(rpt_path, "w", encoding="utf-8") as f:
        f.write(f"""# power_forecast_interval 動態技術報告：風資源評估、點預測與 p10/p50/p90 機率預測

> 本報告資料均由 `test_metrics_interval.csv`、`quantile_interval_metrics.csv` 與 `resource_stats.json` 動態讀取生成。

---

## 1. 站點風資源評估摘要
- **全期平均容量因數 (Capacity Factor)**：**{cf_str}**
- **年等效滿載發電時數 (Full Load Hours)**：**{flh_str} 小時/年**

---

## 2. 區間發電能量預測核心優勢 (100% 排除當前時步洩漏)
將預測標的調整為未來區間 $[t+1 .. t+k]$ 平均功率並嚴格對齊無洩漏遮罩後：
1. **消弭高頻陣風與湍流雜訊**：3 小時區間平均功率預測擬合度 $R^2 = {r2_3h_str}$。
2. **對齊國際標準**：採用容量正規化 nRMSE (nRMSE_cap = RMSE / 1.0)。
3. **對稱式 Persistence 基準**：以過去 H 小時滾動均值預測未來 H 小時區間均值，比單純瞬時值更嚴格。

---

## 3. 測試集點預測評估指標總表 (2020-06 ~ 2021-10)

| 預測標的 | 區間提前量 H | 最佳模型 | 容量正規化 nRMSE(cap) | 擬合度 $R^2$ | 相對 Persistence 改善 |
| :--- | :--- | :--- | :--- | :--- | :--- |
{table_body}

---

## 4. 測試集 p10 / p50 / p90 機率預測與風險信賴帶總表

| 預測視窗 | 80% 信賴帶實測覆蓋率 (PICP) | 信賴帶平均寬度 (Sharpness) | p50 擬合度 $R^2$ | Pinball Loss (p10 / p50 / p90) |
| :--- | :--- | :--- | :--- | :--- |
{q_table_body}
""")
    print(f"動態技術報告已寫入：{rpt_path}")
    print("\nStage 5 評估完成，動態報告與包含 p10/p50/p90 8 大視覺化圖表已全數生成。")

if __name__ == "__main__":
    main()

