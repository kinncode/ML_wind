#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 5 —— 同參數 LightGBM 雙任務適合度對比

目的：在「完全相同的 LightGBM 超參數」與「完全相同的資料切分、相同資料列」下，
比較 LightGBM 對兩種本質不同的風能任務的適合度：

  Task A  風場發電量預測（forecast, +1h）
          目標 = 正規化虛擬出力 P 在 t+1h。屬「時間預測」問題。
  Task B  風機結構安全 / 湍流特性（nowcast/downscaling）
          目標 = 100m 湍流強度 TI（10分鐘 std/mean），當前時刻。
          屬「以平均氣象態推估次網格湍流」問題（疲勞載重相關）。

共同設定
--------
  * 超參數（兩任務完全相同）：n_estimators=500, learning_rate=0.05,
    num_leaves=31, random_state=42（固定 500 棵，不早停，確保「同參數」）
  * 切分（依年份）：train 2016–2018 / val 2019 / test 2020–2021
  * 相同資料列：只用「兩任務目標皆有效」的交集列，確保公平對比

防洩漏
------
  Task B 的特徵排除當前視窗的 std / TI / gust_factor（那等同目標本身），
  只用平均風、風切、風向、溫濕壓、密度、時間、以及風速的滯後/滾動均值。
  Task A 為預測未來，可用當前所有特徵。

輸出
----
  results/suitability_metrics.csv     兩任務 train/val/test 指標
  results/suitability_importance_A.csv / _B.csv
  figures/fig5_suitability.png        R²/nRMSE 對比 + 散布圖
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import config as C

LGB_PARAMS = dict(objective="regression", n_estimators=500, learning_rate=0.05,
                  num_leaves=31, random_state=42, n_jobs=-1, verbose=-1)

META = {"ts","is_ok","year","month","hour_i"}
# Task B 需排除的「湍流洩漏」欄。
# 關鍵：任何「10分鐘視窗內」的擾動統計都與 TI 同源，會構成洩漏，必須排除，
# 包含風速 std/TI/陣風因子，以及風向視窗內標準差 WD_97_sigma（方向擾動＝湍流代理，
# 與陣風因子相關 0.95）。只留「平均氣象態」特徵做真正的降尺度估計。
TI_LEAK = {"WS_100E_ti","WS_100W_ti","WS_69W_ti","WS_38W_ti",
           "WS_100E_std","WS_100E_gust_factor","WD_97_sigma","P_now"}
TARGET_TI = "WS_100E_ti"

def all_feats(df):
    return [c for c in df.columns
            if c not in META and not c.startswith("y_") and not c.startswith("m_")]

def metric_block(y, p):
    rmse = float(np.sqrt(mean_squared_error(y, p)))
    denom = float(np.mean(y)) if np.mean(y) > 1e-9 else 1.0
    return {"R2": r2_score(y, p), "RMSE": rmse, "MAE": float(mean_absolute_error(y, p)),
            "nRMSE": rmse/denom, "mean_y": denom, "n": int(len(y))}

def split_years(df):
    yr = df["year"].values
    return (yr <= 2018), (yr == 2019), (yr >= 2020)

def main():
    df = pd.read_parquet(C.FEAT_PARQUET)
    df["ts"] = pd.to_datetime(df["ts"])

    feats_all = all_feats(df)
    featsA = feats_all                                   # 發電量預測：全特徵
    featsB = [c for c in feats_all if c not in TI_LEAK]  # 湍流估計：排除洩漏

    # 兩任務目標
    yA_col, yB_col = "y_power_1", TARGET_TI

    # 相同資料列：兩目標皆有效 + 兩任務特徵皆完整 + is_ok
    need = list(set(featsA) | set(featsB) | {yA_col, yB_col})
    common = (df["is_ok"].fillna(False) & df["m_1"].fillna(False)
              & df[yA_col].notna() & df[yB_col].notna()
              & df[need].notna().all(axis=1))
    d = df.loc[common].copy()
    tr, va, te = split_years(d)
    print(f"共同資料列：{len(d):,}")
    print(f"  train(2016-2018)={tr.sum():,}  val(2019)={va.sum():,}  test(2020-2021)={te.sum():,}")
    print(f"  Task A 特徵數={len(featsA)}  Task B 特徵數={len(featsB)}（排除 {len(featsA)-len(featsB)} 洩漏欄）")

    rows, preds, imps = [], {}, {}
    for tag, feats, ycol, clip in [("A_power", featsA, yA_col, True),
                                   ("B_turbTI", featsB, yB_col, False)]:
        X = d[feats].values
        y = d[ycol].values
        mdl = lgb.LGBMRegressor(**LGB_PARAMS)
        mdl.fit(X[tr], y[tr])                             # 固定 500 棵，不早停
        for split_name, m in [("train", tr), ("val", va), ("test", te)]:
            p = mdl.predict(X[m])
            if clip: p = np.clip(p, 0, 1)
            mb = metric_block(y[m], p)
            rows.append({"task": tag, "split": split_name, **mb})
        # 存測試預測與重要度
        pte = np.clip(mdl.predict(X[te]),0,1) if clip else mdl.predict(X[te])
        preds[tag] = pd.DataFrame({"y_true": y[te], "y_pred": pte})
        imp = pd.DataFrame({"feature": feats,
                            "gain": mdl.booster_.feature_importance("gain")}
                           ).sort_values("gain", ascending=False)
        imp.to_csv(os.path.join(C.RES_DIR, f"suitability_importance_{tag[0]}.csv"), index=False)
        imps[tag] = imp

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(C.RES_DIR, "suitability_metrics.csv"), index=False)
    print("\n===== 測試集(2020-2021)對比 =====")
    for tag in ["A_power","B_turbTI"]:
        r = res[(res.task==tag)&(res.split=="test")].iloc[0]
        print(f"  {tag:9s}  R²={r.R2:.3f}  RMSE={r.RMSE:.4f}  MAE={r.MAE:.4f}  nRMSE={r.nRMSE:.3f}  (n={r.n:,})")

    _make_fig(res, preds, imps)
    print("\n輸出：results/suitability_metrics.csv, suitability_importance_A/B.csv, figures/fig5_suitability.png")

def _make_fig(res, preds, imps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    fp="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if os.path.exists(fp):
        fm.fontManager.addfont(fp); plt.rcParams["font.family"]=fm.FontProperties(fname=fp).get_name()
    plt.rcParams.update({"figure.dpi":120,"font.size":10,"axes.grid":True,"grid.alpha":0.3,
                         "axes.axisbelow":True,"axes.unicode_minus":False})
    fig=plt.figure(figsize=(12,4.4))
    # (1) R2 by split
    ax1=fig.add_subplot(1,3,1)
    sp=["train","val","test"]; x=np.arange(3); w=0.36
    a=[res[(res.task=="A_power")&(res.split==s)].R2.iloc[0] for s in sp]
    b=[res[(res.task=="B_turbTI")&(res.split==s)].R2.iloc[0] for s in sp]
    ax1.bar(x-w/2,a,w,label="A 發電量預測(+1h)",color="#26c")
    ax1.bar(x+w/2,b,w,label="B 湍流TI估計",color="#e67")
    ax1.set_xticks(x); ax1.set_xticklabels(sp); ax1.set_ylabel("R²（越高越適合）")
    ax1.set_title("同參數 LightGBM：兩任務 R²"); ax1.legend(fontsize=8); ax1.set_ylim(0,1)
    for xi,(va_,vb_) in enumerate(zip(a,b)):
        ax1.text(xi-w/2,va_+0.02,f"{va_:.2f}",ha="center",fontsize=8)
        ax1.text(xi+w/2,vb_+0.02,f"{vb_:.2f}",ha="center",fontsize=8)
    # (2) scatter A
    ax2=fig.add_subplot(1,3,2)
    s=preds["A_power"].sample(min(5000,len(preds["A_power"])),random_state=1)
    ax2.scatter(s.y_true,s.y_pred,s=4,alpha=0.2,color="#26c"); ax2.plot([0,1],[0,1],"k--",lw=1)
    ax2.set_xlim(0,1); ax2.set_ylim(0,1); ax2.set_xlabel("實際 P"); ax2.set_ylabel("預測 P")
    r=res[(res.task=="A_power")&(res.split=="test")].iloc[0]
    ax2.set_title(f"A 發電量預測 test R²={r.R2:.2f}")
    # (3) scatter B
    ax3=fig.add_subplot(1,3,3)
    s=preds["B_turbTI"].sample(min(5000,len(preds["B_turbTI"])),random_state=1)
    hi=np.nanpercentile(preds["B_turbTI"].y_true,99)
    ax3.scatter(s.y_true,s.y_pred,s=4,alpha=0.2,color="#e67"); ax3.plot([0,hi],[0,hi],"k--",lw=1)
    ax3.set_xlim(0,hi); ax3.set_ylim(0,hi); ax3.set_xlabel("實際 TI"); ax3.set_ylabel("預測 TI")
    r=res[(res.task=="B_turbTI")&(res.split=="test")].iloc[0]
    ax3.set_title(f"B 湍流TI估計 test R²={r.R2:.2f}")
    fig.tight_layout(); fig.savefig(os.path.join(C.FIG_DIR,"fig5_suitability.png")); plt.close(fig)

if __name__ == "__main__":
    main()
