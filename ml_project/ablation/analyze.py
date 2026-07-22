#!/usr/bin/env python3
"""
消融結果分析：產出彙整表、圖，並判定「最佳預測方向」與「最精簡有效特徵集」。

輸出：
  results/summary_target_compare.csv   跨目標比較
  results/summary_feature_loo.csv      逐特徵消融（依重要性排序）
  results/summary_group.csv            特徵群消融
  results/noise_ceiling.csv            各目標的雙感測器天花板
  figures/ab1_target_compare.png
  figures/ab2_group_ablation.png
  figures/ab3_feature_loo.png
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prepare_data import GROUPS

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.sans-serif": ["Noto Sans CJK JP", "Droid Sans Fallback", "DejaVu Sans"],
    "axes.unicode_minus": False, "figure.dpi": 150, "savefig.dpi": 150,
    "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
})
C = {"blue": "#2E5E8C", "red": "#C1584B", "green": "#4E9A6B",
     "warm": "#D69A3C", "grey": "#8A8A8A", "purple": "#7B5AA6"}

SPLIT = {"test": (2020, 2021)}


def noise_ceiling() -> pd.DataFrame:
    df = pd.read_parquet(HERE / "model_data.parquet")
    te = df[(df.year >= 2020)]
    rows = []
    pairs = {"tgt_ti": "twin_ti", "tgt_gustfac": "twin_gustfac", "tgt_p99n": "twin_p99n",
             "tgt_p01n": "twin_p01n", "tgt_specslope": "twin_specslope",
             "tgt_intscale": "twin_intscale", "tgt_gust_raw": "twin_gust_raw",
             "tgt_p99_raw": "twin_p99_raw"}
    for t, w in pairs.items():
        m = te[t].notna() & te[w].notna()
        r = np.corrcoef(te.loc[m, t], te.loc[m, w])[0, 1]
        rows.append({"target": t, "twin_r": round(float(r), 4),
                     "ceiling_r2": round(float(r), 4)})
    return pd.DataFrame(rows)


def main() -> int:
    raw = pd.read_csv(RES / "ablation_raw.csv")
    nc = noise_ceiling()
    nc.to_csv(RES / "noise_ceiling.csv", index=False, encoding="utf-8-sig")

    # ---------- A. 跨目標比較 ----------
    A = raw[raw.block == "target_compare"].copy()
    A["kind"] = A.id.str.split("::").str[-1]
    piv = A.pivot_table(index=["target", "target_label"], columns="kind",
                        values="r2").reset_index()
    piv = piv.rename(columns={"B0_const": "B0常數", "B2_linear": "B2線性",
                              "B3_speed": "B3只用風速", "B4_full": "B4完整"})
    piv = piv.merge(nc[["target", "ceiling_r2"]], on="target", how="left")
    piv["gain_B2_to_B4"] = (piv["B4完整"] - piv["B2線性"]).round(3)
    order = ["tgt_ti", "tgt_gustfac", "tgt_p99n", "tgt_p01n",
             "tgt_gust_raw", "tgt_p99_raw", "tgt_specslope", "tgt_intscale"]
    piv["o"] = piv.target.map({t: i for i, t in enumerate(order)})
    piv = piv.sort_values("o").drop(columns="o")
    for c in ["B0常數", "B2線性", "B3只用風速", "B4完整"]:
        piv[c] = piv[c].round(4)
    piv.to_csv(RES / "summary_target_compare.csv", index=False, encoding="utf-8-sig")

    # 判定最佳「正規化」預測方向：B4 相對 B2 提升最大，且在噪聲天花板下有意義
    norm = piv[piv.target.isin(["tgt_ti", "tgt_gustfac", "tgt_p99n", "tgt_p01n",
                                "tgt_specslope", "tgt_intscale"])].copy()
    best = norm.sort_values("gain_B2_to_B4", ascending=False).iloc[0]

    # ---------- B. 特徵群消融 ----------
    B = raw[raw.block == "group"].copy()
    B_tbl = B[["label", "n_features", "r2"]].copy().round(4)
    full_r2 = float(B[B.id == "B::full"].r2.iloc[0])
    speed_r2 = float(B[B.id == "B::speed_only"].r2.iloc[0])
    # leave-one-group-out：移除該群造成的 R² 下降 = 該群的邊際貢獻
    loo = B[B.id.str.startswith("B::loo_group::")].copy()
    loo["group"] = loo.id.str.split("::").str[-1]
    loo["drop_vs_full"] = (full_r2 - loo["r2"]).round(4)
    loo = loo.sort_values("drop_vs_full", ascending=False)
    B_tbl.to_csv(RES / "summary_group.csv", index=False, encoding="utf-8-sig")

    # ---------- C. 逐特徵消融 ----------
    Cc = raw[raw.block == "feature_loo"].copy()
    Cc["feature"] = Cc.dropped
    Cc["r2_without"] = Cc.r2
    Cc["drop_vs_full"] = (full_r2 - Cc.r2).round(4)   # 移除後 R² 掉多少 = 該特徵不可取代性
    Cc = Cc[["feature", "r2_without", "drop_vs_full"]].sort_values(
        "drop_vs_full", ascending=False).round(4)
    Cc.to_csv(RES / "summary_feature_loo.csv", index=False, encoding="utf-8-sig")

    # ================= 圖 A：跨目標 =================
    fig, ax = plt.subplots(figsize=(9, 5.2))
    lab = norm["target_label"].tolist()
    x = np.arange(len(lab)); w = 0.26
    ax.bar(x - w, norm["B2線性"], w, label="B2 現地線性（只用風速）", color=C["grey"])
    ax.bar(x, norm["B3只用風速"], w, label="B3 LightGBM（只用單一風速）", color=C["warm"])
    ax.bar(x + w, norm["B4完整"], w, label="B4 LightGBM（完整特徵）", color=C["blue"])
    for i, (c, v) in enumerate(zip(norm["ceiling_r2"], norm["B4完整"])):
        ax.plot([i - 1.5 * w, i + 1.5 * w], [c, c], color=C["red"], lw=1.4, ls="--")
    ax.plot([], [], color=C["red"], ls="--", label="雙感測器天花板")
    ax.set_xticks(x); ax.set_xticklabels(lab, rotation=18, ha="right", fontsize=9.5)
    ax.set_ylabel("測試集 R²"); ax.set_ylim(-0.05, 1.05)
    ax.set_title("消融圖 1　哪個預測方向最值得做（正規化目標）\n"
                 "TI／陣風因子／p99 三者，加入完整特徵後大幅超越只用風速", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    fig.savefig(FIG / "ab1_target_compare.png"); plt.close(fig)

    # ================= 圖 B：特徵群 =================
    fig, ax = plt.subplots(figsize=(8.4, 5))
    names = ["只用風速剖面"] + [f"＋{g}" for g in GROUPS if g != "風速剖面"] + ["完整"]
    vals = [speed_r2]
    for g in GROUPS:
        if g == "風速剖面":
            continue
        vals.append(float(B[B.id == f"B::add::{g}"].r2.iloc[0]))
    vals.append(full_r2)
    cols = [C["grey"]] + [C["green"]] * (len(names) - 2) + [C["blue"]]
    xb = np.arange(len(names))
    bars = ax.bar(xb, vals, color=cols)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("測試集 R²（目標＝TI）"); ax.set_ylim(0, max(vals) * 1.12)
    ax.set_xticks(xb); ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9.5)
    ax.set_title("消融圖 2　從「只用風速剖面」開始，每加一群特徵的增益\n"
                 "風向貢獻最大；風切幾乎為零（因為它由三高度風速算出，與風速剖面重複）",
                 fontsize=11.5)
    fig.savefig(FIG / "ab2_group_ablation.png"); plt.close(fig)

    # ================= 圖 C：逐特徵 =================
    fig, ax = plt.subplots(figsize=(8.4, 6))
    cc = Cc.sort_values("drop_vs_full")
    colors = [C["red"] if f.startswith("WD_") or f == "veer_97_35" else
              (C["green"] if f.startswith("WS_") or f == "shear_alpha" else C["blue"])
              for f in cc.feature]
    ax.barh(cc.feature, cc.drop_vs_full, color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("移除該特徵後 R² 的下降量（越大＝越不可取代）")
    ax.set_title("消融圖 3　逐一移除單一特徵（目標＝TI）\n"
                 "紅＝風向　綠＝風速/風切　藍＝其他；負值代表移除後反而略好（該特徵在過擬合）",
                 fontsize=11.5)
    fig.savefig(FIG / "ab3_feature_loo.png"); plt.close(fig)

    # ---------- 主控台輸出 ----------
    print("=" * 70)
    print("跨目標比較（R²）：")
    print(piv[["target_label", "B0常數", "B2線性", "B3只用風速", "B4完整",
               "ceiling_r2", "gain_B2_to_B4"]].to_string(index=False))
    print(f"\n★ 最佳正規化預測方向：{best['target_label']}"
          f"（B2→B4 提升 {best['gain_B2_to_B4']:.3f}，B4 R²={best['B4完整']:.3f}，"
          f"天花板 {best['ceiling_r2']:.3f}）")
    print("\n" + "=" * 70)
    print("特徵群 leave-one-out（移除後 R² 下降 = 邊際貢獻，目標 TI）：")
    print(loo[["group", "r2", "drop_vs_full"]].to_string(index=False))
    print("\n逐特徵 leave-one-out 前 8 名（最不可取代）：")
    print(Cc.head(8).to_string(index=False))
    print("\n逐特徵 leave-one-out 後 5 名（移除後反而變好＝多餘）：")
    print(Cc.tail(5).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
