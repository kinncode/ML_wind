#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 4 —— 評估、圖表、報告

讀 Stage 3 的合併結果（results/test_metrics.csv, cv_scores.csv, importance_*, pred_*）
產出：
  figures/fig1_skill_by_horizon.png   各時程 nRMSE：4 模型比較
  figures/fig2_importance.png         最佳模型(LGBM)特徵重要度（ws100_H3 為例）
  figures/fig3_pred_scatter.png       測試年 預測 vs 實際（power_H3）
  figures/fig4_timeseries.png         測試年一段時間序列：實際/LGBM/persistence
  results/summary.csv                 最佳模型摘要
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C

# 中文字型（Noto Sans CJK TC）
from matplotlib import font_manager as fm
for _p in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    if os.path.exists(_p):
        fm.fontManager.addfont(_p)
        plt.rcParams["font.family"] = fm.FontProperties(fname=_p).get_name()
        break
plt.rcParams.update({"figure.dpi":120,"font.size":10,"axes.grid":True,
                     "grid.alpha":0.3,"axes.axisbelow":True,"axes.unicode_minus":False})
MC = {"persistence":"#999999","climatology":"#c44","ridge":"#4a7","lightgbm":"#26c"}

def main():
    tm = pd.read_csv(os.path.join(C.RES_DIR,"test_metrics.csv"))

    # ---- Fig 1: nRMSE by horizon, per target ----
    fig, axes = plt.subplots(1,2,figsize=(11,4.2))
    for ax,target,title in [(axes[0],"ws100","100 m 風速預測"),
                            (axes[1],"power","正規化發電量預測")]:
        d = tm[tm.target==target]
        for model in ["persistence","climatology","ridge","lightgbm"]:
            dd = d[d.model==model].sort_values("H")
            ax.plot(dd.H, dd.nRMSE, "-o", color=MC[model], label=model, lw=2)
        ax.set_title(title); ax.set_xlabel("預測提前量 (小時)")
        ax.set_ylabel("nRMSE (越低越好)"); ax.set_xticks(C.HORIZONS_H)
        ax.legend(fontsize=8)
    fig.suptitle("保留測試期（2020-06 ~ 2021-10）各模型預測誤差", fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(C.FIG_DIR,"fig1_skill_by_horizon.png"))
    plt.close(fig)

    # ---- Fig 2: feature importance (ws100_H3) ----
    imp = pd.read_csv(os.path.join(C.RES_DIR,"importance_ws100_H3.csv")).head(15)
    fig,ax = plt.subplots(figsize=(7,5))
    ax.barh(imp.feature[::-1], imp.gain[::-1], color="#26c")
    ax.set_title("LightGBM 特徵重要度 (ws100, +3h)  —— gain")
    ax.set_xlabel("total gain"); fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR,"fig2_importance.png")); plt.close(fig)

    # ---- Fig 3: pred vs actual scatter (power_H3) ----
    pr = pd.read_parquet(os.path.join(C.DATA_DIR,"pred_power_H3.parquet"))
    s = pr.sample(min(6000,len(pr)), random_state=1)
    fig,ax = plt.subplots(figsize=(5.2,5))
    ax.scatter(s.y_true, s.pred_lgbm, s=4, alpha=0.25, color="#26c")
    ax.plot([0,1],[0,1],"k--",lw=1)
    ax.set_xlabel("實際正規化出力"); ax.set_ylabel("LightGBM 預測")
    ax.set_title("測試年 出力預測 vs 實際 (+3h)"); ax.set_xlim(0,1); ax.set_ylim(0,1)
    fig.tight_layout(); fig.savefig(os.path.join(C.FIG_DIR,"fig3_pred_scatter.png")); plt.close(fig)

    # ---- Fig 4: time series slice (power_H3) ----
    pr["ts"]=pd.to_datetime(pr["ts"]); pr=pr.sort_values("ts")
    seg = pr[(pr.ts>="2020-12-01")&(pr.ts<"2020-12-11")]
    fig,ax = plt.subplots(figsize=(11,3.6))
    ax.plot(seg.ts, seg.y_true, color="k", lw=1.3, label="實際")
    ax.plot(seg.ts, seg.pred_lgbm, color="#26c", lw=1.2, label="LightGBM +3h")
    ax.plot(seg.ts, seg.persist, color="#999", lw=1, ls="--", label="persistence +3h")
    ax.set_title("測試年樣本段（2020-12）正規化出力預測"); ax.set_ylabel("P (0–1)")
    ax.legend(fontsize=8); fig.autofmt_xdate()
    fig.tight_layout(); fig.savefig(os.path.join(C.FIG_DIR,"fig4_timeseries.png")); plt.close(fig)

    # ---- summary.csv ----
    best = tm[tm.is_best].sort_values(["target","H"])[
        ["tag","target","H","model","RMSE","MAE","nRMSE","skill_vs_persist"]]
    best.to_csv(os.path.join(C.RES_DIR,"summary.csv"),index=False)
    print("圖表輸出至 figures/；摘要 results/summary.csv")
    print(best.to_string(index=False))

if __name__ == "__main__":
    main()
