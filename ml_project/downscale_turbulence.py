#!/usr/bin/env python3
"""
專案 ①：湍流降尺度 —— 從 10 分鐘平均狀態預測次分鐘湍流特性

核心問題
--------
再分析資料、中尺度模式、以及絕大多數的測風塔記錄，都只給你 10 分鐘平均值。
但風機載重計算（IEC 61400-1）需要的是湍流：σ_u、陣風、積分尺度、頻譜形狀。
這中間的落差，能不能用機器學習補起來？

這份資料能回答這個問題，是因為它有 1 Hz 的逐秒真值可以當標準答案。

基準線階梯（每一階都必須跑，這是整個專案的方法學骨幹）
------------------------------------------------------
    B0  常數           —— 永遠猜訓練集平均
    B1  IEC NTM        —— 國際標準的湍流公式，未校準
    B2  現地校準 NTM   —— 同樣的函數形式，但用本站資料擬合
    B3  LightGBM (U)   —— 只餵平均風速
    B4  LightGBM (全)  —— 餵完整平均狀態

真正要回答的問題不是「機器學習好不好」，而是：
  (a) B1→B2 現地校準能拿回多少？（不用 ML 就能拿到的）
  (b) B2→B3 樹模型對「風速」這個單一變數的非線性，能再拿多少？
  (c) B3→B4 平均風速以外的大氣狀態，還藏著多少獨立資訊？（這才是本專案的論點）

用法
----
    python downscale_turbulence.py --data "D:/ML_wind/ml_project/data" --out "D:/ML_wind/ml_project/results"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# 特徵定義
# --------------------------------------------------------------------------
# 只能用「10 分鐘平均狀態」—— 也就是標準氣象檔或中尺度模式給得出來的東西。
# 任何從 1 Hz 算出來的量（_std / _ti / _max / _min / _gust_factor / _sigma）
# 都是目標的近親，用了就是洩漏。
MEAN_STATE_FEATURES = [
    "WS_100_mean", "WS_69W_mean", "WS_38W_mean",   # 風速剖面
    "shear_alpha",                                  # 風切指數
    "WD_97_sin", "WD_97_cos", "WD_35_sin", "WD_35_cos", "veer_97_35",
    "AT_95_mean", "RH_95_mean", "BP_93_mean", "air_density",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
]

# 這些欄位絕對不能當特徵 —— 集中列出來，程式會實際檢查一次
LEAKY_PATTERNS = ("_std", "_ti", "_max", "_min", "_gust", "_sigma",
                  "_p01", "_p05", "_p10", "_p25", "_p50", "_p75", "_p90",
                  "_p95", "_p99", "_int_", "_spec_", "power_density")

TARGETS = {
    "WS_100E_std":         "湍流標準差 sigma_u (m/s)",
    "WS_100E_gust3s":      "3 秒陣風極值 (m/s)",
    "WS_100E_p99":         "1 秒風速 p99 (m/s)",
    "WS_100E_p01":         "1 秒風速 p01 (m/s)",
    "WS_100E_int_scale_s": "湍流積分時間尺度 T_u (s)",
    "WS_100E_spec_slope":  "頻譜斜率 (理論 -5/3)",
}

# 兩支同高度風速計互相比對，得到的相關係數即為目標的「信度」。
# 它是任何模型可達 R² 的物理上限 —— 見 README。
NOISE_CEILING = {}

SPLIT = {"train": (2016, 2018), "val": (2019, 2019), "test": (2020, 2021)}


# --------------------------------------------------------------------------
def load(data_dir: Path) -> pd.DataFrame:
    base = pd.read_parquet(data_dir / "BSMI_10min.parquet")
    turb = pd.read_parquet(data_dir / "BSMI_turb.parquet")
    df = base.merge(turb, on="ts", how="inner")
    df = df[df["is_valid"]].copy()

    # 時間週期特徵
    h = df.ts.dt.hour + df.ts.dt.minute / 60
    doy = df.ts.dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * h / 24)
    df["hour_cos"] = np.cos(2 * np.pi * h / 24)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # IEC 的湍流模型只在運轉風速範圍內有定義，低風速的 TI 也沒有物理意義
    df = df[df.WS_100_mean >= 4.0]
    df["year"] = df.ts.dt.year
    return df


def check_no_leakage(features: list[str]) -> None:
    bad = [f for f in features if any(p in f for p in LEAKY_PATTERNS)]
    if bad:
        raise ValueError(f"特徵中含有從 1 Hz 導出的洩漏欄位：{bad}")


def split_by_time(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    for name, (y0, y1) in SPLIT.items():
        out[name] = df[(df.year >= y0) & (df.year <= y1)]
    return out


# --------------------------------------------------------------------------
# 基準線
# --------------------------------------------------------------------------
def iec_ntm_sigma(u: np.ndarray, i_ref: float = 0.12) -> np.ndarray:
    """IEC 61400-1 正常湍流模型 (NTM)。

    sigma_1 = I_ref * (0.75 * V_hub + b),  b = 5.6 m/s

    I_ref 是 15 m/s 時的期望湍流強度，標準等級 A=0.16 / B=0.14 / C=0.12。
    這個公式給的是 90% 分位值（特徵值），不是平均值 —— 所以直接拿來當
    平均值的預測，本來就會系統性高估。這個偏差本身就是要展示的重點：
    國際標準是為了保守設計而訂的，不是為了無偏估計。
    """
    return i_ref * (0.75 * u + 5.6)


def fit_site_ntm(u: np.ndarray, sigma: np.ndarray) -> tuple[float, float]:
    """用本站資料擬合 NTM 的函數形式 sigma = a*u + b（最小平方）。"""
    A = np.column_stack([u, np.ones_like(u)])
    coef, *_ = np.linalg.lstsq(A, sigma, rcond=None)
    return float(coef[0]), float(coef[1])


# --------------------------------------------------------------------------
# 評估
# --------------------------------------------------------------------------
def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    e = y - p
    ss_tot = ((y - y.mean()) ** 2).sum()
    return {
        "n": int(len(y)),
        "rmse": float(np.sqrt((e**2).mean())),
        "mae": float(np.abs(e).mean()),
        "bias": float(e.mean()),
        "r2": float(1 - (e**2).sum() / ss_tot) if ss_tot > 0 else np.nan,
    }


def noise_ceiling(df: pd.DataFrame, target: str) -> float | None:
    """用兩支同高度風速計估目標變數的信度（可達 R² 上限）。

    兩支感測器量的是同一團空氣，若它們彼此只相關到 r，就代表目標有
    (1-r) 比例的變異是量測噪聲加上小尺度空間去相關，任何模型都學不到。
    這是 R² 的物理天花板 —— 注意它是「下界估計」，因為兩支感測器相距
    數公尺，本來就會有真實的空間去相關，不全然是儀器誤差。
    """
    twin = target.replace("WS_100E", "WS_100W")
    if twin not in df.columns:
        return None
    m = df[target].notna() & df[twin].notna()
    if m.sum() < 100:
        return None
    return float(np.corrcoef(df.loc[m, target], df.loc[m, twin])[0, 1])


# --------------------------------------------------------------------------
def run_target(df: pd.DataFrame, target: str, label: str, out_dir: Path) -> pd.DataFrame:
    import lightgbm as lgb

    d = df[df[target].notna()].copy()
    parts = split_by_time(d)
    tr, va, te = parts["train"], parts["val"], parts["test"]
    if min(len(tr), len(va), len(te)) < 500:
        print(f"  [跳過] {target}：切分後樣本不足 "
              f"(train={len(tr)}, val={len(va)}, test={len(te)})")
        return pd.DataFrame()

    ceil = noise_ceiling(d, target)
    NOISE_CEILING[target] = ceil

    print(f"\n{'=' * 78}\n目標：{label}  [{target}]")
    print(f"  訓練 {len(tr):,} / 驗證 {len(va):,} / 測試 {len(te):,}"
          f"   測試集平均 {te[target].mean():.4f}  標準差 {te[target].std():.4f}")
    if ceil is not None:
        print(f"  噪聲天花板（雙感測器信度）R² ≤ {ceil:.4f}")

    y_tr, y_te = tr[target].to_numpy(), te[target].to_numpy()
    rows = []

    # ---- B0 常數 ----
    rows.append({"model": "B0 常數（訓練集平均）",
                 **metrics(y_te, np.full(len(y_te), y_tr.mean()))})

    # ---- B1 / B2 只對 sigma_u 有定義 ----
    if target == "WS_100E_std":
        for cls, iref in [("A", 0.16), ("B", 0.14), ("C", 0.12)]:
            rows.append({"model": f"B1 IEC NTM Class {cls} (I_ref={iref})",
                         **metrics(y_te, iec_ntm_sigma(te.WS_100_mean.to_numpy(), iref))})
        a, b = fit_site_ntm(tr.WS_100_mean.to_numpy(), y_tr)
        rows.append({"model": f"B2 現地校準 NTM (σ={a:.4f}·U{b:+.4f})",
                     **metrics(y_te, a * te.WS_100_mean.to_numpy() + b)})
    else:
        a, b = fit_site_ntm(tr.WS_100_mean.to_numpy(), y_tr)
        rows.append({"model": f"B2 現地線性 (y={a:.4f}·U{b:+.4f})",
                     **metrics(y_te, a * te.WS_100_mean.to_numpy() + b)})

    # ---- B3 / B4 LightGBM ----
    # 固定亂數種子。bagging 與 feature_fraction 都是隨機的，不固定的話
    # 每次重跑的 R² 會有 ±0.02 的浮動，報告裡的數字就對不起來。
    params = dict(objective="regression", metric="rmse", learning_rate=0.05,
                  num_leaves=63, min_data_in_leaf=40, feature_fraction=0.85,
                  bagging_fraction=0.85, bagging_freq=1, verbose=-1, num_threads=4,
                  seed=42, bagging_seed=42, feature_fraction_seed=42, data_random_seed=42)

    fitted = {}
    for tag, feats in [("B3 LightGBM（只用風速）", ["WS_100_mean"]),
                       ("B4 LightGBM（完整平均狀態）", MEAN_STATE_FEATURES)]:
        check_no_leakage(feats)
        ds_tr = lgb.Dataset(tr[feats], y_tr)
        ds_va = lgb.Dataset(va[feats], va[target].to_numpy(), reference=ds_tr)
        model = lgb.train(params, ds_tr, num_boost_round=3000, valid_sets=[ds_va],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
        pred = model.predict(te[feats], num_iteration=model.best_iteration)
        rows.append({"model": f"{tag} [{model.best_iteration} 輪]", **metrics(y_te, pred)})
        fitted[tag] = (model, feats)

    res = pd.DataFrame(rows)
    best_base = res.loc[res.model.str.startswith(("B0", "B1", "B2")), "rmse"].min()
    res["vs_最佳物理基準"] = (100 * (1 - res.rmse / best_base)).round(2)
    if ceil is not None:
        res["佔可達上限比例"] = (100 * res.r2 / ceil).round(1)

    print(res.to_string(index=False))

    # 特徵重要性
    model, feats = fitted["B4 LightGBM（完整平均狀態）"]
    imp = pd.DataFrame({"feature": feats,
                        "gain": model.feature_importance("gain")}).sort_values("gain", ascending=False)
    imp["gain_pct"] = (100 * imp.gain / imp.gain.sum()).round(2)
    print("\n  前 8 名特徵（gain）：",
          ", ".join(f"{r.feature} {r.gain_pct}%" for r in imp.head(8).itertuples()))

    res.insert(0, "target", target)
    imp.insert(0, "target", target)
    imp.to_csv(out_dir / f"importance_{target}.csv", index=False, encoding="utf-8-sig")
    return res


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data_dir, out_dir = Path(args.data), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load(data_dir)
    print(f"樣本 {len(df):,}（已篩 is_valid 且 U ≥ 4 m/s）")
    print(f"期間 {df.ts.min()} ~ {df.ts.max()}")
    print(f"切分 訓練 {SPLIT['train']} / 驗證 {SPLIT['val']} / 測試 {SPLIT['test']}")

    all_res = [run_target(df, t, lab, out_dir) for t, lab in TARGETS.items()]
    res = pd.concat([r for r in all_res if len(r)], ignore_index=True)
    res.to_csv(out_dir / "benchmark.csv", index=False, encoding="utf-8-sig")
    (out_dir / "noise_ceiling.json").write_text(
        json.dumps(NOISE_CEILING, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n結果 → {out_dir / 'benchmark.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
