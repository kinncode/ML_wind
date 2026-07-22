#!/usr/bin/env python3
"""
BSMI 100 m 測風塔 —— 1 Hz 原始檔 → 10 分鐘統計 Parquet

把 Campbell TOA5 格式的原始逐秒資料（約 14 GB、1.6 億筆）壓成
每 10 分鐘一列的特徵表（約 29 萬列、幾十 MB），之後所有機器學習
題目都以這張表為起點。

用法
----
    # 全部處理（預設）
    python preprocess.py --root "D:/ML_wind" --out "D:/ML_wind/ml_project/data"

    # 只跑一個月試水溫
    python preprocess.py --root "D:/ML_wind" --out "./data" --only 2020-06

    # 改成 1 分鐘解析度（檔案會大 10 倍）
    python preprocess.py --root "D:/ML_wind" --out "./data" --freq 1min

輸出
----
    data/monthly/BSMI_10min_YYYY-MM.parquet   每月一檔
    data/BSMI_10min.parquet                   全部合併
    data/qc_report.csv                        每月資料完整度與異常統計

需求：pandas >= 1.5, numpy, pyarrow
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# 感測器配置（來自 TOA5 檔頭）
# --------------------------------------------------------------------------
WS_COLS = {"WS_100E": 100.0, "WS_100W": 100.0, "WS_69W": 69.0, "WS_38W": 38.0}
WD_COLS = ["WD_97", "WD_35"]
SCALAR_COLS = ["AT_95", "RH_95", "BP_93"]  # 氣溫(degC) / 相對濕度(%) / 氣壓(hPa)
DATA_COLS = list(WS_COLS) + WD_COLS + SCALAR_COLS

# 用來擬合風切指數的三個高度（100 m 取東西兩支平均）
SHEAR_HEIGHTS = np.array([100.0, 69.0, 38.0])

# 物理常數
R_DRY = 287.058   # J/(kg K)
R_VAP = 461.495   # J/(kg K)

FNAME_RE = re.compile(r"Raw_BSMI_Wind_Hz_(\d{4})-(\d{2})\.(csv|txt)$", re.IGNORECASE)


# --------------------------------------------------------------------------
# 讀檔
# --------------------------------------------------------------------------
def read_toa5(path: Path) -> pd.DataFrame:
    """讀一個 TOA5 月檔。

    檔案結構：
        line 0  TOA5 中繼資料
        line 1  欄位名稱      <- 當表頭
        line 2  單位
        line 3  取樣方式
        line 4+ 資料

    需要處理的格式差異：
      * 2016–2017 的時間戳有雙引號，2018 之後沒有
      * 欄名 'Record' (2016–2017) vs 'RECORD' (2018+)
      * 2021-07 之後副檔名變成 .txt，內容格式相同
    """
    df = pd.read_csv(
        path,
        skiprows=[0, 2, 3],      # 保留 line 1 當表頭
        header=0,
        na_values=["NAN", "NaN", "nan", "INF", "-INF", ""],
        engine="c",
        low_memory=False,
    )
    df.columns = [c.strip().upper() for c in df.columns]

    missing = [c for c in DATA_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} 缺少欄位：{missing}")

    ts = pd.to_datetime(df["TIMESTAMP"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    out = pd.DataFrame({"ts": ts})
    for c in DATA_COLS:
        out[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    out = out.dropna(subset=["ts"])
    # 逐秒資料偶爾會有重複時戳；保留第一筆並確保時序遞增
    out = out.drop_duplicates(subset="ts", keep="first").sort_values("ts")
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------
# 物理量計算
# --------------------------------------------------------------------------
def yamartino_sigma(sin_mean: np.ndarray, cos_mean: np.ndarray) -> np.ndarray:
    """風向標準差（Yamartino 法），單位為度。

    一般算術標準差在 0/360 度接縫處會爆炸，所以風向必須用向量法。
    """
    eps = np.sqrt(np.clip(1.0 - (sin_mean**2 + cos_mean**2), 0.0, 1.0))
    sigma = np.arcsin(eps) * (1.0 + 0.1547 * eps**3)
    return np.degrees(sigma)


def air_density(temp_c: np.ndarray, rh_pct: np.ndarray, press_hpa: np.ndarray) -> np.ndarray:
    """濕空氣密度 kg/m^3（Tetens 飽和水氣壓公式）。

    風功率正比於密度，冬夏差異可達 5%，是很有用的特徵。
    """
    t_k = temp_c + 273.15
    p_sat = 6.1078 * 10 ** (7.5 * temp_c / (temp_c + 237.3))   # hPa
    p_vap = np.clip(rh_pct, 0, 100) / 100.0 * p_sat
    p_dry = press_hpa - p_vap
    return (p_dry * 100.0) / (R_DRY * t_k) + (p_vap * 100.0) / (R_VAP * t_k)


def shear_exponent(u_stack: np.ndarray) -> np.ndarray:
    """冪次律風切指數 alpha，對 ln(z) vs ln(U) 做最小平方擬合。

    u_stack : shape (n_bins, 3)，欄位順序對應 SHEAR_HEIGHTS
    """
    ln_z = np.log(SHEAR_HEIGHTS)
    ln_z_c = ln_z - ln_z.mean()
    denom = (ln_z_c**2).sum()

    with np.errstate(divide="ignore", invalid="ignore"):
        ln_u = np.log(np.where(u_stack > 0.1, u_stack, np.nan))
    alpha = np.nansum(ln_z_c * ln_u, axis=1) / denom
    # 任一高度無效就整列作廢
    alpha[np.isnan(ln_u).any(axis=1)] = np.nan
    return alpha


def wrap180(deg: np.ndarray) -> np.ndarray:
    """把角度差摺到 (-180, 180]，用於風向剪切 (veer)。"""
    return (deg + 180.0) % 360.0 - 180.0


# --------------------------------------------------------------------------
# 聚合
# --------------------------------------------------------------------------
def aggregate(df: pd.DataFrame, freq: str = "10min") -> pd.DataFrame:
    """把逐秒資料聚成每 freq 一列的特徵表。"""
    expected = pd.Timedelta(freq).total_seconds()
    g = df.groupby(pd.Grouper(key="ts", freq=freq), sort=True)

    out = pd.DataFrame(index=g.size().index)
    out.index.name = "ts"

    n_raw = g.size()
    out["n_samples"] = n_raw.astype("int32")
    out["coverage"] = (n_raw / expected).astype("float32")

    # ---- 風速：平均 / 標準差 / 最小 / 最大(=1 秒陣風) / 亂流強度 ----
    for c in WS_COLS:
        s = g[c]
        mean = s.mean()
        std = s.std(ddof=0)
        out[f"{c}_mean"] = mean.astype("float32")
        out[f"{c}_std"] = std.astype("float32")
        out[f"{c}_min"] = s.min().astype("float32")
        out[f"{c}_max"] = s.max().astype("float32")     # 1 秒陣風
        # TI 在低風速下沒有物理意義，IEC 61400 慣例只在 >3 m/s 採用
        ti = (std / mean.where(mean > 3.0)).astype("float32")
        out[f"{c}_ti"] = ti
        # 陣風因子
        out[f"{c}_gust_factor"] = (s.max() / mean.where(mean > 3.0)).astype("float32")

    # ---- 風向：向量平均 + Yamartino 標準差 ----
    for c in WD_COLS:
        rad = np.radians(df[c].to_numpy(dtype="float64"))
        tmp = pd.DataFrame({"ts": df["ts"], "sin": np.sin(rad), "cos": np.cos(rad)})
        gg = tmp.groupby(pd.Grouper(key="ts", freq=freq), sort=True)
        sin_m = gg["sin"].mean().reindex(out.index)
        cos_m = gg["cos"].mean().reindex(out.index)
        wd = (np.degrees(np.arctan2(sin_m, cos_m)) + 360.0) % 360.0
        out[f"{c}_vecmean"] = wd.astype("float32")
        out[f"{c}_sigma"] = yamartino_sigma(sin_m.to_numpy(), cos_m.to_numpy()).astype("float32")
        # sin/cos 分量直接留著當模型輸入，免得下游又要處理環狀變數
        out[f"{c}_sin"] = sin_m.astype("float32")
        out[f"{c}_cos"] = cos_m.astype("float32")

    # ---- 純量氣象 ----
    for c in SCALAR_COLS:
        out[f"{c}_mean"] = g[c].mean().astype("float32")

    # ---- 衍生特徵 ----
    ws100 = out[["WS_100E_mean", "WS_100W_mean"]].mean(axis=1)
    out["WS_100_mean"] = ws100.astype("float32")
    # 兩支 100 m 風速計的差異：塔影、結冰、感測器故障都會在這裡露出馬腳
    out["ws100_pair_diff"] = (out["WS_100E_mean"] - out["WS_100W_mean"]).astype("float32")

    u_stack = np.column_stack([
        ws100.to_numpy(dtype="float64"),
        out["WS_69W_mean"].to_numpy(dtype="float64"),
        out["WS_38W_mean"].to_numpy(dtype="float64"),
    ])
    out["shear_alpha"] = shear_exponent(u_stack).astype("float32")
    out["veer_97_35"] = wrap180(
        (out["WD_97_vecmean"] - out["WD_35_vecmean"]).to_numpy(dtype="float64")
    ).astype("float32")

    out["air_density"] = air_density(
        out["AT_95_mean"].to_numpy(dtype="float64"),
        out["RH_95_mean"].to_numpy(dtype="float64"),
        out["BP_93_mean"].to_numpy(dtype="float64"),
    ).astype("float32")
    # 風功率密度 0.5 * rho * U^3 (W/m^2)
    out["power_density"] = (0.5 * out["air_density"] * out["WS_100_mean"] ** 3).astype("float32")

    # ---- 品質旗標 ----
    # 感測器卡死：整段標準差為 0
    out["flag_frozen"] = (
        (out[[f"{c}_std" for c in WS_COLS]] == 0).any(axis=1)
    )
    # 兩支 100 m 差太多（>1 m/s 且 >20%）
    rel = (out["ws100_pair_diff"].abs() / out["WS_100_mean"].where(out["WS_100_mean"] > 2))
    out["flag_pair_mismatch"] = (out["ws100_pair_diff"].abs() > 1.0) & (rel > 0.20)
    # 資料不足（少於 90% 的秒數）
    out["flag_low_coverage"] = out["coverage"] < 0.90
    out["is_valid"] = ~(out["flag_frozen"] | out["flag_pair_mismatch"] | out["flag_low_coverage"])

    return out.reset_index()


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def find_files(root: Path) -> list[tuple[str, Path]]:
    found = {}
    for p in sorted(root.rglob("Raw_BSMI_Wind_Hz_*")):
        m = FNAME_RE.search(p.name)
        if m:
            found[f"{m.group(1)}-{m.group(2)}"] = p
    return sorted(found.items())


def main() -> int:
    ap = argparse.ArgumentParser(description="BSMI 測風塔 1 Hz 原始資料前處理")
    ap.add_argument("--root", required=True, help="放三個原始資料夾的上層目錄")
    ap.add_argument("--out", required=True, help="輸出目錄")
    ap.add_argument("--freq", default="10min", help="聚合解析度，預設 10min")
    ap.add_argument("--only", default=None, help="只處理指定月份，如 2020-06")
    ap.add_argument("--force", action="store_true", help="重跑已存在的月份")
    args = ap.parse_args()

    root, out_dir = Path(args.root), Path(args.out)
    monthly_dir = out_dir / "monthly"
    monthly_dir.mkdir(parents=True, exist_ok=True)

    files = find_files(root)
    if args.only:
        files = [(k, v) for k, v in files if k == args.only]
    if not files:
        print(f"在 {root} 找不到符合的原始檔", file=sys.stderr)
        return 1

    print(f"找到 {len(files)} 個月份，解析度 {args.freq}\n")
    qc_rows = []
    t_start = time.time()

    for i, (month, path) in enumerate(files, 1):
        dst = monthly_dir / f"BSMI_{args.freq}_{month}.parquet"
        if dst.exists() and not args.force:
            print(f"[{i:2d}/{len(files)}] {month}  已存在，略過")
            qc_rows.append({"month": month, "status": "skipped"})
            continue

        t0 = time.time()
        try:
            raw = read_toa5(path)
            agg = aggregate(raw, freq=args.freq)
            agg.to_parquet(dst, index=False, compression="snappy")
        except Exception as exc:                       # noqa: BLE001
            print(f"[{i:2d}/{len(files)}] {month}  失敗：{exc}")
            qc_rows.append({"month": month, "status": f"error: {exc}"})
            continue

        dt = time.time() - t0
        n_bins = len(agg)
        n_valid = int(agg["is_valid"].sum())
        expected_bins = pd.Period(month, freq="M").days_in_month * (
            86400 / pd.Timedelta(args.freq).total_seconds()
        )
        qc_rows.append({
            "month": month,
            "status": "ok",
            "raw_rows": len(raw),
            "bins": n_bins,
            "bins_expected": int(expected_bins),
            "bins_valid": n_valid,
            "valid_pct": round(100 * n_valid / expected_bins, 2),
            "ws100_mean": round(float(agg.loc[agg.is_valid, "WS_100_mean"].mean()), 3),
            "ws100_max_gust": round(float(agg.loc[agg.is_valid, "WS_100E_max"].max()), 2),
            "shear_alpha_median": round(float(agg.loc[agg.is_valid, "shear_alpha"].median()), 4),
            "flag_frozen": int(agg["flag_frozen"].sum()),
            "flag_pair_mismatch": int(agg["flag_pair_mismatch"].sum()),
            "flag_low_coverage": int(agg["flag_low_coverage"].sum()),
            "seconds": round(dt, 1),
        })
        print(
            f"[{i:2d}/{len(files)}] {month}  {len(raw):>9,} 筆 → {n_bins:>5,} 格"
            f"  有效 {100 * n_valid / expected_bins:5.1f}%"
            f"  平均風速 {qc_rows[-1]['ws100_mean']:5.2f} m/s"
            f"  ({dt:.1f}s)"
        )

    # 合併
    parts = sorted(monthly_dir.glob(f"BSMI_{args.freq}_*.parquet"))
    if parts:
        combined = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        combined = combined.sort_values("ts").reset_index(drop=True)
        combo_path = out_dir / f"BSMI_{args.freq}.parquet"
        combined.to_parquet(combo_path, index=False, compression="snappy")
        size_mb = combo_path.stat().st_size / 1e6
        print(
            f"\n合併完成：{combo_path}"
            f"\n  {len(combined):,} 列 × {combined.shape[1]} 欄，{size_mb:.1f} MB"
            f"\n  時間範圍 {combined.ts.min()} ~ {combined.ts.max()}"
            f"\n  有效格數 {int(combined.is_valid.sum()):,}"
            f" ({100 * combined.is_valid.mean():.1f}%)"
        )

    qc_path = out_dir / "qc_report.csv"
    pd.DataFrame(qc_rows).to_csv(qc_path, index=False, encoding="utf-8-sig")
    print(f"  QC 報告 → {qc_path}")
    print(f"\n總耗時 {(time.time() - t_start) / 60:.1f} 分鐘")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
