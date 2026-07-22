#!/usr/bin/env python3
"""
1 Hz 湍流特徵萃取 —— 專案 ① 湍流降尺度的目標變數

回頭讀原始逐秒檔，對每個 10 分鐘區塊（600 個樣本）計算「10 分鐘平均值裡
看不到」的次分鐘尺度湍流特性，作為降尺度模型的預測目標：

  1. 1 秒風速的分位數分佈  p01 ~ p99
  2. 3 秒陣風極值          （工程上的標準陣風定義）
  3. 湍流積分時間尺度      T_u（自相關函數積分到首次過零）
  4. 頻譜斜率             （慣性次區間理論值 -5/3）

輸出可用 ts 直接跟 preprocess.py 的 10 分鐘表 join。

用法
----
    python extract_turbulence.py --root "D:/ML_wind" --out "D:/ML_wind/ml_project/data" --only 2020-01
    python extract_turbulence.py --root "D:/ML_wind" --out "D:/ML_wind/ml_project/data"

⚠️ 物理限制請務必讀 README 的「杯式風速計頻率響應」一節 ——
   1 Hz 取樣 + 杯式風速計的慣性，讓頻譜斜率只在有限頻帶內可信。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess import FNAME_RE, read_toa5

# --------------------------------------------------------------------------
# 參數
# --------------------------------------------------------------------------
FS = 1.0            # 取樣頻率 Hz
BLOCK = 600         # 每區塊秒數（10 分鐘）
GUST_WIN = 3        # 陣風定義：3 秒移動平均

# 頻譜斜率的擬合頻帶與 Welch 分段長度。
#
# 這兩個參數是實測校準出來的，不是隨便設的。2020-01 的系集平均頻譜，
# 分頻帶擬合局部斜率結果如下（理論慣性次區間 = -1.667）：
#
#     0.005-0.02 Hz   -1.18   含能區，理論上就比 -5/3 淺
#     0.02 -0.05 Hz   -1.39   仍在過渡帶
#     0.05 -0.10 Hz   -1.51
#     0.10 -0.20 Hz   -1.76   最接近 -5/3
#     0.20 -0.35 Hz   -1.95   杯式風速計慣性衰減，頻譜被壓陡
#     0.35 -0.50 Hz   -1.34   儀器雜訊底噪，頻譜反轉變平
#
# 也就是說：本站 1 Hz + 杯式風速計的資料，含能區與儀器截止頻率之間的
# 窗口很窄，沒有乾淨的 -5/3 平台。0.05–0.25 Hz 是兼顧「夠像冪次律」與
# 「還沒進入儀器衰減」的最佳折衷（實測斜率中位數 -1.704）。
#
# 但固定頻帶還有第二個更嚴重的問題：慣性次區間的位置會隨風速移動
# （f ~ U/z），所以固定頻帶在低風速時會取到過渡帶、高風速時取到儀器
# 衰減帶。實測固定頻帶斜率隨風速從 -2.03（3-5 m/s）單調變到 -1.59
# （20-30 m/s），corr(斜率, U) = +0.40 —— 它主要在反映風速，不是湍流。
#
# 正解是用約化頻率 n = f·z/U 定義頻帶，讓頻帶跟著風速伸縮。實測比較：
#
#     方法                corr(U)   中位斜率   跨風速擺盪   擬合 R²
#     固定 0.05-0.25 Hz    +0.398    -1.832      0.418      0.895
#     約化 n = 0.3-2.0     -0.035    -1.649      0.036      0.904   ← 採用
#     約化 n = 0.5-2.5     -0.051    -1.736      0.079      0.880
#
# 約化頻帶讓風速相關性從 +0.40 降到 -0.035，跨風速擺盪從 0.418 收斂到
# 0.036，而且中位斜率 -1.649 落在理論值 -1.667 的 1% 以內。
FIT_BAND_FIXED = (0.05, 0.25)     # 保留作為對照診斷，說明為何不能用固定頻帶
REDUCED_BAND = (0.3, 2.0)         # 主要定義：n = f·z/U 的下上限
Z_REF = 100.0                     # WS_100E / WS_100W 的量測高度 (m)

# 約化頻帶換算回實際頻率後，仍必須落在儀器可信的範圍內：
#   下限 0.02 Hz —— 再低則 10 分鐘區塊內週期數不足
#   上限 0.30 Hz —— 再高則進入杯式風速計衰減帶與雜訊底噪
F_LIMITS = (0.02, 0.30)

# Welch 分段平均。單一 10 分鐘週期圖每個頻率點只有 2 個自由度，逐區塊
# 斜率的雜訊極大（實測逐區塊擬合 R² 中位數僅 0.27），拿來當機器學習的
# 目標變數會有巨大的不可約噪聲。改用 100 秒分段、50% 重疊（11 段）後，
# R² 中位數提升到 0.88，斜率 IQR 從 0.565 收斂到 0.387。
WELCH_SEG = 100
WELCH_OVERLAP = 0.5

# 分位數
QUANTILES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

TURB_CHANNELS = ["WS_100E", "WS_100W"]

# 每個區塊至少要有多少有效樣本才計算（600 的 95%）
MIN_VALID = 570


# --------------------------------------------------------------------------
# 區塊化
# --------------------------------------------------------------------------
def to_blocks(df: pd.DataFrame, col: str) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """把不規則的逐秒序列切成 (n_blocks, 600) 的規則陣列。

    先重建完整的 1 Hz 時間軸（缺的補 NaN），再 reshape。這樣區塊邊界
    永遠對齊整點的 10 分鐘格，跟 preprocess.py 的輸出可以直接 join。
    """
    s = df.set_index("ts")[col].astype("float32")
    start = s.index.min().floor("10min")
    end = s.index.max().ceil("10min")
    full = pd.date_range(start, end, freq="1s", inclusive="left")
    s = s.reindex(full)

    n_blocks = len(s) // BLOCK
    arr = s.to_numpy()[: n_blocks * BLOCK].reshape(n_blocks, BLOCK)
    ts = full[: n_blocks * BLOCK : BLOCK]
    return arr, ts


def detrend(x: np.ndarray) -> np.ndarray:
    """逐列移除線性趨勢。

    做頻譜分析前一定要去趨勢：區塊內若有整體加速或減速，會在低頻端
    造成大量洩漏，把 -5/3 的斜率估歪。
    """
    n = x.shape[1]
    t = np.arange(n, dtype=np.float64)
    t_c = t - t.mean()
    denom = (t_c**2).sum()
    slope = (x * t_c).sum(axis=1, keepdims=True) / denom
    return x - (x.mean(axis=1, keepdims=True) + slope * t_c)


# --------------------------------------------------------------------------
# 湍流量
# --------------------------------------------------------------------------
def integral_time_scale(x_det: np.ndarray) -> np.ndarray:
    """湍流積分時間尺度 T_u（秒），自相關函數積分到首次過零。

    做法是 Wiener–Khinchin：用 FFT 算自相關比直接迴圈快兩個數量級。
    「積分到首次過零」是最常用的穩健截斷法 —— 自相關的長尾在有限
    樣本下純粹是雜訊，全部積下去會發散。
    """
    n_blocks, n = x_det.shape
    nfft = 1 << int(np.ceil(np.log2(2 * n)))

    f = np.fft.rfft(x_det, n=nfft, axis=1)
    acf = np.fft.irfft(f * np.conj(f), n=nfft, axis=1)[:, :n]

    var = acf[:, [0]]
    with np.errstate(divide="ignore", invalid="ignore"):
        acf = acf / var                       # 正規化成 rho(0)=1

    out = np.full(n_blocks, np.nan)
    for i in range(n_blocks):
        r = acf[i]
        if not np.isfinite(r[0]):
            continue
        neg = np.flatnonzero(r <= 0)
        k0 = neg[0] if neg.size else n        # 首次過零；沒過零就用整段
        # 梯形積分，dt = 1 s
        out[i] = np.trapezoid(r[:k0], dx=1.0 / FS) if k0 > 1 else 0.0
    return out


def welch_psd(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Welch 分段平均功率譜密度。

    每段各自去趨勢再加 Hann 窗 —— 分段後才去趨勢很重要，否則長時間
    尺度的擺盪會在每一段裡留下殘餘趨勢。
    """
    n = x.shape[1]
    seg = WELCH_SEG
    step = max(1, int(seg * (1 - WELCH_OVERLAP)))
    starts = list(range(0, n - seg + 1, step))

    win = np.hanning(seg)
    norm = (win**2).sum() * FS

    acc = None
    for s in starts:
        d = detrend(x[:, s : s + seg])
        p = (np.abs(np.fft.rfft(d * win, axis=1)) ** 2) / norm
        acc = p if acc is None else acc + p

    psd = acc / len(starts)
    freqs = np.fft.rfftfreq(seg, d=1.0 / FS)
    return psd, freqs


def _fit_loglog(lf: np.ndarray, lp: np.ndarray) -> tuple[float, float]:
    """對單一區塊做 log-log 最小平方，回傳 (斜率, R²)。"""
    lf_c = lf - lf.mean()
    b = float((lp * lf_c).sum() / (lf_c**2).sum())
    a = lp.mean() - b * lf.mean()
    resid = lp - (a + b * lf)
    ss_tot = ((lp - lp.mean()) ** 2).sum()
    return b, float(1 - (resid**2).sum() / ss_tot) if ss_tot > 0 else np.nan


def spectral_slope_fixed(psd: np.ndarray, freqs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """固定頻帶版本 —— 只作為對照診斷，不建議當主要指標。"""
    band = (freqs >= FIT_BAND_FIXED[0]) & (freqs <= FIT_BAND_FIXED[1])
    lf = np.log(freqs[band])
    lf_c = lf - lf.mean()

    with np.errstate(divide="ignore", invalid="ignore"):
        lp = np.log(psd[:, band])

    ok = np.isfinite(lp).all(axis=1)
    slope = np.full(len(psd), np.nan)
    r2 = np.full(len(psd), np.nan)
    if ok.any():
        lo = lp[ok]
        b = (lo * lf_c).sum(axis=1) / (lf_c**2).sum()
        pred = (lo.mean(axis=1) - b * lf.mean())[:, None] + b[:, None] * lf
        ss_tot = ((lo - lo.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
        slope[ok] = b
        with np.errstate(divide="ignore", invalid="ignore"):
            r2[ok] = 1 - ((lo - pred) ** 2).sum(axis=1) / ss_tot
    return slope, r2


def spectral_slope_reduced(
    psd: np.ndarray, freqs: np.ndarray, u_mean: np.ndarray, z: float = Z_REF
) -> tuple[np.ndarray, np.ndarray]:
    """約化頻率版本 —— 主要指標。

    頻帶隨風速伸縮：f ∈ [n1·U/z, n2·U/z]，再夾到儀器可信範圍 F_LIMITS 內。
    這樣量到的才是湍流結構本身，而不是風速的代理。依據見上方註解。
    """
    n1, n2 = REDUCED_BAND
    slope = np.full(len(psd), np.nan)
    r2 = np.full(len(psd), np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_psd = np.log(psd)

    for i, u in enumerate(u_mean):
        if not np.isfinite(u) or u < 2.0:
            continue
        lo = max(n1 * u / z, F_LIMITS[0])
        hi = min(n2 * u / z, F_LIMITS[1])
        if hi / lo < 2.0:                       # 不足一個八度就放棄，擬合沒意義
            continue
        band = (freqs >= lo) & (freqs <= hi)
        if band.sum() < 5:
            continue
        lp = log_psd[i, band]
        if not np.isfinite(lp).all():
            continue
        slope[i], r2[i] = _fit_loglog(np.log(freqs[band]), lp)
    return slope, r2


def peak_gust(x: np.ndarray, win: int = GUST_WIN) -> np.ndarray:
    """區塊內的 3 秒陣風極值（移動平均後取最大）。"""
    k = np.ones(win, dtype=np.float64) / win
    # 對每列做 'valid' 卷積
    sm = np.apply_along_axis(lambda r: np.convolve(r, k, mode="valid"), 1, x)
    return np.nanmax(sm, axis=1)


# --------------------------------------------------------------------------
# 單月處理
# --------------------------------------------------------------------------
def process_month(df: pd.DataFrame) -> pd.DataFrame:
    out = None
    for ch in TURB_CHANNELS:
        arr, ts = to_blocks(df, ch)
        n_valid = np.isfinite(arr).sum(axis=1)
        good = n_valid >= MIN_VALID

        res = {"ts": ts, f"{ch}_n_valid": n_valid.astype("int32")}

        # 分位數（允許少量缺值）
        with np.errstate(invalid="ignore"):
            qs = np.nanpercentile(np.where(good[:, None], arr, np.nan),
                                  [q * 100 for q in QUANTILES], axis=1)
        for q, row in zip(QUANTILES, qs):
            res[f"{ch}_p{int(q * 100):02d}"] = row.astype("float32")

        # 需要完整序列的量：先用線性內插補掉零星缺值
        filled = arr.copy()
        for i in np.flatnonzero(good & (n_valid < BLOCK)):
            r = filled[i]
            idx = np.arange(BLOCK)
            m = np.isfinite(r)
            r[~m] = np.interp(idx[~m], idx[m], r[m])
        filled[~good] = np.nan

        res[f"{ch}_gust3s"] = np.where(good, peak_gust(filled), np.nan).astype("float32")

        clean = np.nan_to_num(filled, nan=0.0)

        # 積分尺度用整段去趨勢後的序列
        det = detrend(clean)
        tu = integral_time_scale(det)
        tu[~good] = np.nan
        res[f"{ch}_int_scale_s"] = tu.astype("float32")

        # 頻譜：Welch 內部會逐段去趨勢，所以餵未去趨勢的序列
        u_mean = np.where(good, np.nanmean(arr, axis=1), np.nan)
        psd, freqs = welch_psd(clean)

        sl, r2 = spectral_slope_reduced(psd, freqs, u_mean)
        sl[~good] = np.nan
        r2[~good] = np.nan
        res[f"{ch}_spec_slope"] = sl.astype("float32")
        res[f"{ch}_spec_r2"] = r2.astype("float32")

        slf, r2f = spectral_slope_fixed(psd, freqs)
        slf[~good] = np.nan
        r2f[~good] = np.nan
        res[f"{ch}_spec_slope_fixed"] = slf.astype("float32")
        res[f"{ch}_spec_r2_fixed"] = r2f.astype("float32")

        # 積分長度尺度 L_u = T_u * U（Taylor 凍結湍流假說）
        res[f"{ch}_int_len_m"] = (tu * u_mean).astype("float32")

        part = pd.DataFrame(res)
        out = part if out is None else out.merge(part, on="ts", how="outer")
    return out


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="1 Hz 湍流特徵萃取")
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-merge", action="store_true",
                    help="跳過合併步驟。平行跑多個月份時必加，否則多個行程會同時寫同一個合併檔")
    ap.add_argument("--merge-only", action="store_true",
                    help="只做合併，不重新萃取")
    args = ap.parse_args()

    root, out_dir = Path(args.root), Path(args.out)
    turb_dir = out_dir / "turb"
    turb_dir.mkdir(parents=True, exist_ok=True)

    files = {}
    for p in sorted(root.rglob("Raw_BSMI_Wind_Hz_*")):
        m = FNAME_RE.search(p.name)
        if m:
            files[f"{m.group(1)}-{m.group(2)}"] = p
    items = sorted(files.items())
    if args.only:
        items = [(k, v) for k, v in items if k == args.only]
    if not items:
        print("找不到原始檔")
        return 1

    t_all = time.time()
    if args.merge_only:
        items = []
    for i, (month, path) in enumerate(items, 1):
        dst = turb_dir / f"BSMI_turb_{month}.parquet"
        if dst.exists() and not args.force:
            print(f"[{i:2d}/{len(items)}] {month}  已存在，略過")
            continue
        t0 = time.time()
        raw = read_toa5(path)
        res = process_month(raw)
        tmp = dst.with_suffix(".parquet.tmp")   # 原子寫入，避免平行批次被砍時留下半截檔
        res.to_parquet(tmp, index=False, compression="snappy")
        tmp.replace(dst)
        ok = res["WS_100E_int_scale_s"].notna()
        print(
            f"[{i:2d}/{len(items)}] {month}  {len(res):>5,} 區塊  可用 {ok.sum():>5,}"
            f"  T_u 中位數 {res.loc[ok, 'WS_100E_int_scale_s'].median():5.1f}s"
            f"  頻譜斜率中位數 {res.loc[ok, 'WS_100E_spec_slope'].median():+.3f}"
            f"  ({time.time() - t0:.1f}s)"
        )

    parts = sorted(turb_dir.glob("BSMI_turb_*.parquet"))
    if parts and not args.no_merge:
        comb = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        comb = comb.sort_values("ts").reset_index(drop=True)
        dst = out_dir / "BSMI_turb.parquet"
        comb.to_parquet(dst, index=False, compression="snappy")
        print(f"\n合併 → {dst}  ({len(comb):,} 列 × {comb.shape[1]} 欄)")
    print(f"總耗時 {(time.time() - t_all) / 60:.1f} 分鐘")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
