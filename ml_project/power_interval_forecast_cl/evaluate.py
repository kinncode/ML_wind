#!/usr/bin/env python3
"""
發電區間預測評估：保形校正 CQR + 機率評分 + 圖表。

保形校正 (Conformalized Quantile Regression, CQR)
------------------------------------------------
分位數模型本身不保證涵蓋率（例如宣稱 90% 區間，實際可能只涵蓋 82%）。
CQR 用一組獨立「校正集」修正區間寬度，讓測試集有接近保證的涵蓋率：

  對區間 (q_lo, q_hi)（例如 p05, p95，名目涵蓋 90%）：
    1. 在校正集算每點的「不符合分數」 E = max(q_lo − y, y − q_hi)
    2. 取 E 的 (1−α) 分位數（含有限樣本修正）作為調整量 Q
    3. 測試區間放寬成 [q_lo − Q, q_hi + Q]
  這樣測試集涵蓋率會 ≈ 名目值，且有理論保證（假設資料可交換）。

評分指標
--------
  涵蓋率 (coverage)、區間寬度 (sharpness)、pinball loss、CRPS(近似)、
  Winkler 區間分數、可靠度圖。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE / "results"; FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.sans-serif": ["Noto Sans CJK JP", "Droid Sans Fallback", "DejaVu Sans"],
    "axes.unicode_minus": False, "figure.dpi": 150, "savefig.dpi": 150,
    "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": .25,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11})
B = {"blue": "#2E5E8C", "red": "#C1584B", "green": "#4E9A6B",
     "warm": "#D69A3C", "grey": "#8A8A8A", "purple": "#7B5AA6"}

HORIZONS = ["1h", "3h", "6h"]
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
# 三個名目區間：(下分位, 上分位, 名目涵蓋率)
INTERVALS = [(0.25, 0.75, 0.50), (0.10, 0.90, 0.80), (0.05, 0.95, 0.90)]


def qc(q): return f"q{int(round(q*100)):02d}"


def clean_quantiles(df: pd.DataFrame) -> pd.DataFrame:
    """逐列排序，消除分位數交叉（q25>q75 之類）。

    獨立訓練的分位數模型會偶發交叉（此資料約 0.2% 的列），排序即可修正。
    刻意「不」做邊界吸附：把接近 0 的下界硬吸到 0 會讓下界變得不設限
    （出力恆 ≥ 0），反而把區間涵蓋率灌水。保留原始邊界行為才誠實。
    """
    cols = [qc(q) for q in QUANTILES]
    M = np.sort(df[cols].to_numpy(), axis=1)      # 逐列由小到大 → 保證單調
    out = df.copy()
    out[cols] = M
    return out


def pinball(y, pred, q):
    d = y - pred
    return np.mean(np.maximum(q * d, (q - 1) * d))


def crps_approx(y, Q):
    """用分位數集近似 CRPS = 2∫ pinball_q dq（梯形積分）。"""
    qs = QUANTILES
    pb = [pinball(y, Q[qc(q)].to_numpy(), q) for q in qs]
    return 2 * np.trapezoid(pb, qs)


def winkler(y, lo, hi, alpha):
    """Winkler 區間分數（越小越好）：獎勵窄區間、重罰未涵蓋。"""
    w = hi - lo
    below = y < lo; above = y > hi
    s = w.copy()
    s = s + np.where(below, 2 / alpha * (lo - y), 0)
    s = s + np.where(above, 2 / alpha * (y - hi), 0)
    return np.mean(s)


def cqr_Q(cal, q_lo, q_hi, nominal):
    """從校正集算 CQR 調整量 Q。"""
    y = cal.y.to_numpy()
    lo = cal[qc(q_lo)].to_numpy(); hi = cal[qc(q_hi)].to_numpy()
    E = np.maximum(lo - y, y - hi)
    n = len(E)
    level = np.ceil((n + 1) * nominal) / n
    level = min(level, 1.0)
    return float(np.quantile(E, level, method="higher"))


def main():
    rows_int = []      # 區間層級指標
    rows_prob = []     # 機率整體指標
    cqr_store = {}

    for hz in HORIZONS:
        cal = clean_quantiles(pd.read_parquet(RES / f"pred_cal_{hz}.parquet"))
        te = clean_quantiles(pd.read_parquet(RES / f"pred_test_{hz}.parquet"))
        y = te.y.to_numpy()

        # 機率整體：pinball 平均、CRPS
        pb_mean = np.mean([pinball(y, te[qc(q)].to_numpy(), q) for q in QUANTILES])
        rows_prob.append({"horizon": hz, "pinball_mean": round(pb_mean, 5),
                          "crps_approx": round(crps_approx(y, te), 5)})

        for q_lo, q_hi, nom in INTERVALS:
            lo = te[qc(q_lo)].to_numpy(); hi = te[qc(q_hi)].to_numpy()
            # 原始（未校正）
            cov0 = np.mean((y >= lo) & (y <= hi)); w0 = np.mean(hi - lo)
            wink0 = winkler(y, lo, hi, 1 - nom)
            # CQR 校正
            Q = cqr_Q(cal, q_lo, q_hi, nom)
            lo2 = np.clip(lo - Q, 0, 1); hi2 = np.clip(hi + Q, 0, 1)
            cov1 = np.mean((y >= lo2) & (y <= hi2)); w1 = np.mean(hi2 - lo2)
            wink1 = winkler(y, lo2, hi2, 1 - nom)
            cqr_store[f"{hz}:{int(nom*100)}"] = round(Q, 4)
            rows_int.append({
                "horizon": hz, "nominal_pct": int(nom * 100),
                "raw_coverage_pct": round(100 * cov0, 1), "raw_width_pct": round(100 * w0, 1),
                "raw_winkler": round(wink0, 4), "cqr_Q_pct": round(100 * Q, 1),
                "cqr_coverage_pct": round(100 * cov1, 1), "cqr_width_pct": round(100 * w1, 1),
                "cqr_winkler": round(wink1, 4)})

    df_int = pd.DataFrame(rows_int); df_prob = pd.DataFrame(rows_prob)
    df_int.to_csv(RES / "interval_metrics.csv", index=False, encoding="utf-8-sig")
    df_prob.to_csv(RES / "prob_metrics.csv", index=False, encoding="utf-8-sig")
    (RES / "cqr_adjustments.json").write_text(
        json.dumps(cqr_store, ensure_ascii=False, indent=2), encoding="utf-8")

    # ===== 圖 1：可靠度圖（3h，校正前後）=====
    hz = "3h"; te = clean_quantiles(pd.read_parquet(RES / f"pred_test_{hz}.parquet"))
    cal = clean_quantiles(pd.read_parquet(RES / f"pred_cal_{hz}.parquet")); y = te.y.to_numpy()
    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    ax.plot([0, 1], [0, 1], "--", color=B["grey"], lw=1.4, label="理想（完美校正）")
    raw_cov = [np.mean(y <= te[qc(q)].to_numpy()) for q in QUANTILES]
    ax.plot(QUANTILES, raw_cov, "o-", color=B["red"], lw=2, label="原始分位數")
    # CQR 校正後的分位（僅對稱區間端點示意，用區間涵蓋近似）
    cqr_pts_x, cqr_pts_y = [], []
    for q_lo, q_hi, nom in INTERVALS:
        Q = cqr_Q(cal, q_lo, q_hi, nom)
        lo2 = np.clip(te[qc(q_lo)].to_numpy() - Q, 0, 1)
        hi2 = np.clip(te[qc(q_hi)].to_numpy() + Q, 0, 1)
        cqr_pts_x += [(1 - nom) / 2, 1 - (1 - nom) / 2]
        cqr_pts_y += [np.mean(y <= lo2), np.mean(y <= hi2)]
    idx = np.argsort(cqr_pts_x)
    ax.plot(np.array(cqr_pts_x)[idx], np.array(cqr_pts_y)[idx], "s-", color=B["blue"],
            lw=2, label="CQR 保形校正後")
    ax.set_xlabel("名目分位 / 累積機率"); ax.set_ylabel("實際涵蓋比例")
    ax.set_title("圖 1　可靠度圖（3 小時）\n"
                 "曲線略在對角線上方＝預測分布略偏保守；對稱 CQR 不改變位置偏移", fontsize=12)
    ax.legend(loc="upper left"); fig.savefig(FIG / "iv1_reliability.png"); plt.close(fig)

    # ===== 圖 2：校正前後涵蓋率長條 =====
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    labels = [f"{r['horizon']}\n{r['nominal_pct']}%" for _, r in df_int.iterrows()]
    x = np.arange(len(df_int)); w = 0.38
    ax.bar(x - w / 2, df_int.raw_coverage_pct, w, label="原始分位數", color=B["red"])
    ax.bar(x + w / 2, df_int.cqr_coverage_pct, w, label="CQR 校正後", color=B["blue"])
    for _, r in df_int.iterrows():
        i = r.name
        ax.plot([i - 0.7, i + 0.7], [r.nominal_pct, r.nominal_pct], color="k", lw=1.2, ls=":")
    ax.plot([], [], "k:", label="名目涵蓋率（目標）")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("實際涵蓋率 (%)"); ax.set_ylim(30, 100)
    ax.set_title("圖 2　區間涵蓋率 vs 名目目標（校正前後）\n"
                 "80%／90% 區間接近目標；50% 區間受零出力點質量影響偏低", fontsize=12)
    ax.legend(fontsize=9); fig.savefig(FIG / "iv2_coverage.png"); plt.close(fig)

    # ===== 圖 3：範例週的區間帶（3h, CQR 80%）=====
    te = clean_quantiles(pd.read_parquet(RES / "pred_test_3h.parquet"))
    cal = clean_quantiles(pd.read_parquet(RES / "pred_cal_3h.parquet"))
    te["ts"] = pd.to_datetime(te.ts)
    Q80 = cqr_Q(cal, 0.10, 0.90, 0.80); Q50 = cqr_Q(cal, 0.25, 0.75, 0.50)
    win = te[(te.ts >= "2021-01-11") & (te.ts < "2021-01-18")].copy()
    if len(win) < 100: win = te.iloc[:1000].copy()
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.fill_between(win.ts, np.clip(win.q10 - Q80, 0, 1) * 100, np.clip(win.q90 + Q80, 0, 1) * 100,
                    color=B["blue"], alpha=.15, label="80% 區間 (CQR)")
    ax.fill_between(win.ts, np.clip(win.q25 - Q50, 0, 1) * 100, np.clip(win.q75 + Q50, 0, 1) * 100,
                    color=B["blue"], alpha=.28, label="50% 區間 (CQR)")
    ax.plot(win.ts, win.y * 100, color="k", lw=1.5, label="實際出力")
    ax.plot(win.ts, win.q50 * 100, color=B["red"], lw=1.2, label="中位數預測 (p50)")
    ax.set_ylabel("出力（% 額定）")
    ax.set_title("圖 3　3 小時前的機率預測區間（範例：2021 年 1 月一週）\n"
                 "深帶=50% 區間、淺帶=80% 區間，實際值大多落在帶內", fontsize=12)
    ax.legend(ncol=2, fontsize=9); fig.autofmt_xdate()
    fig.savefig(FIG / "iv3_example_bands.png"); plt.close(fig)

    # ===== 圖 4：涵蓋率–銳利度權衡（校正後）=====
    fig, ax = plt.subplots(figsize=(7.6, 5))
    marker = {"1h": "o", "3h": "s", "6h": "^"}
    for hz in HORIZONS:
        sub = df_int[df_int.horizon == hz]
        ax.plot(sub.cqr_width_pct, sub.cqr_coverage_pct, marker[hz] + "-", color=B["blue"],
                alpha=.5 + 0.15 * HORIZONS.index(hz), label=f"{hz}")
        for _, r in sub.iterrows():
            ax.annotate(f"{r.nominal_pct}%", (r.cqr_width_pct, r.cqr_coverage_pct),
                        fontsize=8, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("平均區間寬度（% 額定，越窄越好）")
    ax.set_ylabel("實際涵蓋率 (%)")
    ax.set_title("圖 4　涵蓋率 vs 區間寬度（CQR 校正後）\n"
                 "時程越長，要達到同樣涵蓋率就需要越寬的區間", fontsize=12)
    ax.legend(title="提前量"); fig.savefig(FIG / "iv4_sharpness.png"); plt.close(fig)

    print("=== 區間指標（校正前後）===")
    print(df_int.to_string(index=False))
    print("\n=== 機率整體指標 ===")
    print(df_prob.to_string(index=False))
    print("\nCQR 調整量:", cqr_store)
    print("✓ 4 張圖完成")


if __name__ == "__main__":
    raise SystemExit(main())
