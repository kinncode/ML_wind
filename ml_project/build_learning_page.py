#!/usr/bin/env python3
"""
把湍流降尺度專案做成一頁「零基礎也能看懂」的教學網頁。

- 所有圖片以 base64 內嵌，產出單一 HTML，可離線開、可寄給別人
- 每個專業觀念都配一段白話解釋與生活比喻
- 附名詞小辭典與已查證的延伸閱讀連結

用法
----
    python build_learning_page.py --figures "D:/ML_wind/ml_project/results/figures" \
                                  --out "D:/ML_wind/ml_project/湍流降尺度_教學網頁.html"
"""

from __future__ import annotations

import argparse
import base64
from pathlib import Path


def data_uri(path: Path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{b64}"


# 每張圖：檔名、章節標題、白話解釋（HTML）
FIGURES = [
    ("01_wind_rose.png", "風從哪裡來？",
     "這叫「風花圖」。想像你站在塔中央，花瓣往哪個方向長，就代表風常從那個方向吹來；花瓣越長代表越常見，"
     "顏色越深代表風越強。這座塔有兩個主要來風方向：<b>東北</b>和<b>南到西南</b>。最重要的是——"
     "<b>最強的風（深藍）幾乎全來自東北</b>，那就是台灣冬天的東北季風。"),
    ("02_ti_by_direction.png", "關鍵發現：風的「顛簸程度」看方向",
     "藍線是「湍流強度」（風有多顛簸）的中位數，隨風向怎麼變。看出來了嗎？東北方向來的風最平順（TI≈0.05），"
     "南方來的風最顛簸（TI≈0.18），<b>相差三倍以上</b>。這是整個專案的靈魂：<b>風平不平順，主要看它從哪裡來</b>，"
     "而不是看它多快。上風處地表越平滑（很可能是開闊海面），風就越平順。"),
    ("03_ti_vs_windspeed.png", "為什麼「風速」不能決定顛簸",
     "如果顛簸只由風速決定，這條帶子應該很窄。但實際上淺紅色的分布帶<b>非常寬</b>——"
     "同一個風速，湍流可以差三倍。這張圖反過來證明：<b>光看風速，你猜不出風顛不顛簸</b>，一定還有別的線索。"),
    ("04_seasonal_cycle.png", "一年四季的節奏",
     "藍線是風速、紅線是顛簸程度，兩者<b>一整年剛好相反</b>：冬天東北季風又強又穩（風大、顛簸小），"
     "夏天風弱但亂（風小、顛簸大）。這也是為什麼訓練 AI 一定要用「完整一年」的資料，"
     "否則它會把「現在幾月」誤當成規律。"),
    ("05_diurnal_cycle.png", "白天和晚上也不一樣",
     "把每個月的平均扣掉後，看一天之內風速怎麼起伏。振幅可以到 3 m/s，而且每一季的高低點時間都不同。"
     "這解釋了一個小技巧：餵給模型「幾點」時，不能直接給 0 到 23 的數字（因為 23 點和 0 點其實相鄰），"
     "要用三角函數編碼成一個圓。"),
    ("06_shear_by_direction.png", "風速隨高度長高的速度，也看方向",
     "風越高越快，「長高的速度」叫風切指數 α。業界常直接用一個固定值 α=1/7（紅虛線）把矮塔的風速換算到"
     "高塔。但這張圖顯示 α 隨風向大幅變化，固定值在多數方向都偏高——這正是另一個可以用 AI 改進的題目。"),
    ("16_mean_spectrum.png", "風裡藏著一個宇宙級的規律",
     "這張圖把風的抖動拆解成「大慢波」到「小快波」的能量分布（叫功率頻譜）。神奇的是，黑虛線那個 −5/3 斜率"
     "是物理學家 Kolmogorov 在 1941 年推導出來的<b>普世規律</b>：大漩渦碎成小漩渦、能量一層層往下傳，"
     "從大氣到咖啡杯裡的漩渦都遵守。灰色區塊是我們的杯狀風速計轉不夠快、量不準的頻段——"
     "誠實標出儀器的極限，是做研究的基本功。"),
    ("07_integral_scale_dist.png", "一陣風有多「長」",
     "湍流積分尺度大致是「一陣風持續多久」，這裡中位數約 13 秒。這個量<b>量得很穩</b>"
     "（等一下圖 8 會解釋怎麼知道的），但後面會看到它幾乎<b>無法預測</b>——這是個誠實的失敗，我們照實說。"),
    ("08_noise_ceiling.png", "天才的驗證法：兩支溫度計互相對答案",
     "塔上同一個高度裝了兩支風速計，量的是同一團空氣。它們彼此有多一致（r=0.99），就是這個量的"
     "「可信度上限」——<b>任何模型再厲害，也不可能比這兩支儀器彼此的一致性更準</b>。"
     "有了這條天花板，後面說「R²=0.66」才有意義：它是滿分 0.99 裡的 0.66，而不是憑空的數字。"),
    ("09_baseline_ladder.png", "全片高潮：一步一步證明 AI 真的有用",
     "這是最重要的一張圖。目標是預測風的顛簸程度，四種方法由笨到聰明疊上去："
     "<b>B0</b> 永遠猜平均（比亂猜還差）→ <b>B2</b> 只用風速畫直線（0.05）→ "
     "<b>B3</b> 用風速但允許 AI 玩非線性（0.12）→ <b>B4</b> 加入風向等完整資訊（<b>0.66</b>）。"
     "只用風速怎麼調都卡在 0.12，<b>一加入風向就跳到 0.66</b>。紅虛線是天花板 0.99。這一跳，就是結論。"),
    ("14_normalized_vs_raw.png", "最常見的自我感覺良好陷阱",
     "這張圖在拆穿一個陷阱。左邊兩組（直接預測陣風、極大值）用最笨的直線就有 R²=0.98，看起來超強——"
     "但那只是因為<b>陣風 ≈ 風速 × 1.16</b>，一條直線當然準，AI 根本沒出力。右邊三組把風速的影響除掉後，"
     "真實難度才現形（直線只剩 0.05）。<b>網路上很多「風速預測準確率 98%」的專案，其實都掉進這個陷阱。</b>"),
    ("10_pred_vs_obs_ti.png", "AI 猜的 vs 真實答案",
     "橫軸是真實顛簸程度、縱軸是 AI 猜的，點越貼近紅色對角線越準。關鍵是：這是用"
     "<b>2020–2021 的資料測的，而 AI 只看過 2016–2018</b>——等於用「過去」預測「未來」，沒有作弊。"),
    ("11_feature_importance.png", "AI 到底在看什麼？",
     "紅色是風向、藍色是其他。四個風向欄位加起來佔了<b>約 80%</b>的重要性，其中一個（風向的南北分量）"
     "自己就佔 48%。而「風速」完全掉出前四名——因為我們要預測的顛簸程度，早就把風速除掉了。"
     "這張圖用數據坐實了「顛簸由風向決定」不是隨口說的。"),
    ("12_residual_by_direction.png", "檢查模型有沒有偷懶",
     "這是「錯誤診斷圖」。橘線（只用風速）的錯誤隨風向大幅上下擺盪，代表它<b>系統性地漏看了風向</b>。"
     "藍線（加入風向後）把這些擺盪大致砍半、壓平。還有殘留代表仍有進步空間——這正是下一步該做的。"),
    ("15_error_by_season.png", "分季節體檢",
     "不能只報一個整體平均。夏天湍流大、錯誤也大，但每一季裡「完整模型」相對「只用風速」的"
     "改善幅度（柱子上的百分比）都很明顯。這證明模型的價值不是只靠某個好做的季節撐場面。"),
    ("13_iec_ntm_vs_obs.png", "國際標準在這座塔上「太保守」",
     "三條虛線是國際風機設計標準 IEC 61400-1 的湍流等級，藍線是本站實測。連最低等級的 Class C 都比"
     "實測高出約 0.94 m/s——代表<b>這是一個湍流極低的優質站址</b>（風況接近離岸）。實務意義：直接套標準會"
     "高估風機的疲勞負荷。用本站資料校準加上 AI，把誤差從 1.01 一路降到 0.23。"),
    ("17_spectral_slope_correction.png", "特徵工程要尊重物理",
     "這是一個修 bug 的故事。一開始用固定頻段算 −5/3 斜率（紅線），發現它隨風速一直漂移——"
     "代表我們<b>量到的其實是風速，不是湍流</b>。改用「跟著風速伸縮」的頻段（藍線）後，斜率就穩穩貼在"
     "理論值 −5/3 上。教訓：做特徵時要懂背後的物理，不然會量到假東西。"),
]

GLOSSARY = [
    ("測風塔", "一根幾十到上百公尺高的桿子，不同高度裝了風速計、風向計、溫濕度計，用來長期記錄一個地點的風況。蓋風場前的必要功課。"),
    ("1 Hz / 逐秒資料", "每秒記錄一筆。這批資料是 1 Hz，所以看得到「秒等級」的抖動——這正是它珍貴的地方，大多數公開資料只有 10 分鐘一筆。"),
    ("平均風速 (U)", "一段時間（這裡是 10 分鐘）內風速的平均。標準氣象報告給的就是這個。"),
    ("湍流 / 亂流", "風在平均之外的雜亂抖動。同樣平均 8 m/s，可以很平順也可以忽大忽小。"),
    ("湍流強度 TI", "衡量「顛簸程度」的標準數字 = 這段時間風速的標準差 ÷ 平均風速。越大越亂。風機工程最關心的量之一。"),
    ("陣風 (Gust)", "短時間內的風速尖峰。這裡用「3 秒陣風」，是工程上常用的定義。"),
    ("風切指數 α", "風速隨高度增加的快慢。用來把矮處的風速換算到風機輪轂高度。"),
    ("降尺度 (Downscaling)", "從「粗」的資訊（10 分鐘平均）推算出「細」的資訊（秒級湍流）。本專案的核心動作。"),
    ("基準線 (Baseline)", "一個刻意很笨的對照組（例如「永遠猜平均」）。任何模型都要先贏過它，才算真的有用。"),
    ("R²（判定係數）", "衡量預測好壞：1 = 完美，0 = 跟亂猜平均一樣，負的 = 比亂猜還差。"),
    ("資料洩漏 (Leakage)", "不小心把「答案的近親」偷偷放進了題目，導致分數虛高。時間序列最常見的致命錯。"),
    ("噪聲天花板", "用兩支同高度儀器互相對答案，得出這個量「最多能被預測到多準」的物理上限。"),
    ("功率頻譜", "把風的抖動依「快慢」拆解，看每個頻率藏了多少能量。"),
    ("Kolmogorov −5/3 律", "湍流的普世規律：能量從大漩渦一層層傳到小漩渦，頻譜呈 −5/3 斜率。"),
    ("LightGBM", "一種又快又準的機器學習模型（梯度提升決策樹），特別適合這種表格資料，一般筆電就能跑。"),
]

# 已查證的延伸閱讀（2026-07 查核）
REFERENCES = [
    ("湍流強度是什麼、為什麼重要（風能入門）", [
        ("Turbulence Intensity — 清潔能源商會術語表（最白話）",
         "https://cleanenergybusinesscouncil.com/wind-energy-glossary/turbulence-intensity/"),
        ("Turbulence Intensity — ScienceDirect Topics（進階總覽）",
         "https://www.sciencedirect.com/topics/engineering/turbulence-intensity"),
    ]),
    ("Kolmogorov −5/3 律與能量串級", [
        ("The Kolmogorov ‘5/3’ spectrum and why it is important — 愛丁堡大學 McComb 教授部落格",
         "https://blogs.ed.ac.uk/physics-of-turbulence/2020/04/09/the-kolmogorov-5-3-spectrum-and-why-it-is-important/"),
        ("Turbulence Scales and Energy Cascade — Altair 教學",
         "https://help.altair.com/hwcfdsolvers/acusolve/topics/acusolve/training_manual/turb_scales_energy_cascasde_r.htm"),
    ]),
    ("IEC 61400-1 風機設計標準與湍流模型", [
        ("The IEC 61400-1 turbine safety standard — 丹麥科技大學 DTU WAsP（權威且好讀）",
         "https://wasp.dtu.dk/software/windfarm-assessment-tool/iec-61400-1"),
        ("IEC 61400-1 正常湍流模型在「台灣西海岸」的評估（與本站高度相關）",
         "https://www.worldscientific.com/doi/pdf/10.1142/S2010194514603822"),
    ]),
    ("Taylor 凍結湍流假說（怎麼從時間推算漩渦大小）", [
        ("The Taylor Hypothesis — 賓州州立大學公開課（圖文並茂）",
         "https://www.e-education.psu.edu/meteo300/node/737"),
        ("Taylor's hypothesis — 美國氣象學會術語表",
         "https://glossary.ametsoc.org/wiki/Taylor%27s_hypothesis"),
    ]),
    ("風速隨高度的冪次律外推（垂直外推）", [
        ("Estimation of Wind Energy Production — energypedia 教學 wiki",
         "https://energypedia.info/wiki/Estimation_of_Wind_Energy_Production"),
        ("The role of the power law exponent in wind energy assessment（2021 全球分析）",
         "https://onlinelibrary.wiley.com/doi/full/10.1002/er.6382"),
    ]),
    ("長期風資源評估 MCP 方法（延伸專案方向）", [
        ("Assessing Long-Term Wind Conditions — 美國 NREL 官方報告",
         "https://docs.nrel.gov/docs/fy13osti/57647.pdf"),
        ("A review of measure-correlate-predict (MCP) methods — 回顧論文",
         "https://www.sciencedirect.com/science/article/abs/pii/S1364032113004498"),
    ]),
    ("機器學習工具", [
        ("LightGBM 官方文件",
         "https://lightgbm.readthedocs.io/"),
        ("LightGBM: A Highly-Efficient Gradient Boosting Decision Tree — KDnuggets 導讀",
         "https://www.kdnuggets.com/2020/06/lightgbm-gradient-boosting-decision-tree.html"),
    ]),
]

CONCEPTS = [
    ("🗼", "測風塔在量什麼？",
     "把它想成一根一百公尺高、插滿感測器的巨大量尺，每秒記錄一次風速、風向、溫濕度。"
     "蓋風場前要先花好幾年量風，因為風況直接決定這塊地能發多少電、值不值得投資。"),
    ("〰️", "「平均」藏起來的東西",
     "同樣是「10 分鐘平均 8 公尺/秒」，可以是平穩的巡航，也可以是忽強忽弱的亂流。"
     "就像開車，平均時速 60 可以很順，也可以一路走走停停。<b>平均值把這種「顛簸」藏起來了</b>，"
     "而顛簸正是會把風機搖壞的元兇。"),
    ("🎯", "這個專案想幹嘛？",
     "大多數氣象資料只有「平均值」，但風機工程需要「顛簸程度」。這中間的落差，"
     "能不能用機器學習補起來？這叫<b>降尺度</b>——從粗資訊推算細節。這批 1 Hz 逐秒資料剛好有秒級真相可以當答案。"),
    ("🔁", "為什麼不做「預測明天的風」？",
     "那是最擁擠、最沒記憶點的題目。這批資料真正稀有的是<b>秒級解析度</b>和<b>雙備援感測器</b>，"
     "都跟時間軸無關。所以我們挑了一條更有意思、也更能展現資料價值的路。"),
]


# 哪些圖歸「先認識資料」、哪些歸「結果」
DATA_FIG_NUMS = {"01", "02", "03", "04", "05", "06", "16", "07", "08"}


def build(figures_dir: Path, out_path: Path) -> None:
    def render(fname, title, expl):
        p = figures_dir / fname
        if not p.exists():
            print(f"  ⚠ 找不到 {fname}，跳過")
            return ""
        num = fname[:2]
        return f"""
      <figure class="figblock" id="fig{num}">
        <div class="fignum">圖 {int(num)}</div>
        <h3>{title}</h3>
        <img src="{data_uri(p)}" alt="{title}" loading="lazy">
        <div class="zoomhint">🔍 點圖可放大</div>
        <div class="plain"><span class="plabel">白話解釋</span>{expl}</div>
      </figure>"""

    data_html, result_html = [], []
    for fname, title, expl in FIGURES:
        block = render(fname, title, expl)
        (data_html if fname[:2] in DATA_FIG_NUMS else result_html).append(block)

    concepts_html = "".join(f"""
        <div class="ccard">
          <div class="cicon">{icon}</div>
          <div><h3>{t}</h3><p>{d}</p></div>
        </div>""" for icon, t, d in CONCEPTS)

    glossary_html = "".join(f"""
        <div class="gitem"><dt>{term}</dt><dd>{d}</dd></div>""" for term, d in GLOSSARY)

    refs_html = ""
    for group, links in REFERENCES:
        items = "".join(f'<li><a href="{url}" target="_blank" rel="noopener">{txt}</a></li>'
                        for txt, url in links)
        refs_html += f'<div class="refgroup"><h4>{group}</h4><ul>{items}</ul></div>'

    html = TEMPLATE
    html = html.replace("__CONCEPTS__", concepts_html)
    html = html.replace("__DATA_FIGS__", "".join(data_html))
    html = html.replace("__RESULT_FIGS__", "".join(result_html))
    html = html.replace("__GLOSSARY__", glossary_html)
    html = html.replace("__REFS__", refs_html)

    out_path.write_text(html, encoding="utf-8")
    size = out_path.stat().st_size / 1e6
    print(f"✓ 已產出 {out_path}  ({size:.1f} MB)")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>看懂「湍流降尺度」— 從零開始的風能機器學習</title>
<style>
  :root{
    --blue:#2E5E8C; --blue2:#4E80B0; --red:#C1584B; --green:#4E9A6B;
    --warm:#D69A3C; --purple:#7B5AA6; --ink:#232A31; --muted:#5C6873;
    --bg:#F4F6F8; --card:#FFFFFF; --line:#E2E7EC;
  }
  *{box-sizing:border-box}
  html{scroll-behavior:smooth}
  body{margin:0;font-family:"Noto Sans TC","Microsoft JhengHei","PingFang TC",
       "Hiragino Sans GB",sans-serif;color:var(--ink);background:var(--bg);
       line-height:1.85;-webkit-font-smoothing:antialiased}
  a{color:var(--blue)}
  .wrap{max-width:920px;margin:0 auto;padding:0 22px}

  /* nav */
  nav{position:sticky;top:0;z-index:20;background:rgba(255,255,255,.92);
      backdrop-filter:blur(8px);border-bottom:1px solid var(--line)}
  nav .wrap{display:flex;gap:6px;align-items:center;height:52px;overflow-x:auto}
  nav a{color:var(--muted);text-decoration:none;font-size:14px;padding:6px 10px;
        border-radius:8px;white-space:nowrap}
  nav a:hover{background:var(--bg);color:var(--blue)}
  nav .brand{font-weight:700;color:var(--ink);margin-right:8px}

  /* hero */
  header.hero{background:linear-gradient(135deg,#1F3F5C 0%,#2E5E8C 55%,#3F72A0 100%);
       color:#fff;padding:64px 0 56px}
  header.hero .wrap{max-width:920px}
  .kick{display:inline-block;background:rgba(255,255,255,.16);padding:5px 14px;
        border-radius:999px;font-size:13px;letter-spacing:.06em;margin-bottom:18px}
  header.hero h1{font-size:34px;line-height:1.35;margin:.1em 0 .35em;font-weight:800}
  header.hero p.lead{font-size:18px;color:#E7EFF6;margin:0 0 26px;max-width:44em}
  .oneliner{background:rgba(255,255,255,.12);border-left:4px solid var(--warm);
       padding:16px 20px;border-radius:10px;font-size:17px;max-width:46em}
  .oneliner b{color:#FFE1B0}
  .herometa{margin-top:26px;display:flex;flex-wrap:wrap;gap:10px}
  .chip{background:rgba(255,255,255,.14);padding:7px 13px;border-radius:8px;font-size:13.5px}

  section{padding:46px 0}
  section:nth-child(even){background:var(--card)}
  h2.sec{font-size:26px;font-weight:800;margin:0 0 6px;display:flex;
         align-items:center;gap:12px}
  h2.sec .dot{width:12px;height:26px;border-radius:4px;background:var(--blue)}
  .sub{color:var(--muted);margin:0 0 26px;font-size:15.5px}

  /* concept cards */
  .ccard{display:flex;gap:16px;background:var(--card);border:1px solid var(--line);
         border-left:4px solid var(--green);border-radius:12px;padding:18px 20px;margin:14px 0;
         box-shadow:0 1px 3px rgba(20,40,60,.04)}
  section:nth-child(even) .ccard{background:var(--bg)}
  .cicon{font-size:30px;line-height:1.2;flex-shrink:0}
  .ccard h3{margin:.1em 0 .3em;font-size:18px}
  .ccard p{margin:0;color:#31414E}

  /* figure blocks */
  .figblock{margin:0 0 34px;background:var(--card);border:1px solid var(--line);
            border-radius:14px;padding:22px 22px 20px;box-shadow:0 2px 10px rgba(20,40,60,.05)}
  section:nth-child(even) .figblock{background:var(--bg)}
  .fignum{display:inline-block;background:var(--blue);color:#fff;font-size:12.5px;
          font-weight:700;padding:3px 11px;border-radius:999px;letter-spacing:.05em}
  .figblock h3{margin:.5em 0 .6em;font-size:20px}
  .figblock img{display:block;width:100%;max-width:520px;height:auto;margin:0 auto;
                border-radius:10px;border:1px solid var(--line);background:#fff;
                cursor:zoom-in;transition:box-shadow .15s}
  .figblock img:hover{box-shadow:0 4px 16px rgba(20,40,60,.16)}
  .zoomhint{text-align:center;color:var(--muted);font-size:12.5px;margin-top:8px}
  /* lightbox */
  #lb{position:fixed;inset:0;z-index:100;background:rgba(15,22,28,.9);
      display:none;align-items:center;justify-content:center;cursor:zoom-out;padding:24px}
  #lb.on{display:flex}
  #lb img{max-width:96vw;max-height:92vh;border-radius:8px;box-shadow:0 8px 40px rgba(0,0,0,.5)}
  .plain{margin-top:16px;background:linear-gradient(0deg,#FBF6EE,#FCFAF5);
         border:1px solid #EFE2CC;border-radius:10px;padding:14px 16px;font-size:15.5px}
  .plabel{display:inline-block;background:var(--warm);color:#3a2a08;font-weight:700;
          font-size:12.5px;padding:2px 9px;border-radius:6px;margin-right:9px;
          vertical-align:1.5px}
  .plain b{color:var(--red)}

  .divider-note{border-left:4px solid var(--red);background:#FBEEEC;border-radius:10px;
        padding:14px 18px;margin:22px 0;font-size:15.5px}

  /* glossary */
  .gwrap{columns:2;column-gap:26px}
  .gitem{break-inside:avoid;margin:0 0 14px;padding:0 0 12px;border-bottom:1px dashed var(--line)}
  .gitem dt{font-weight:700;color:var(--blue);font-size:15.5px}
  .gitem dd{margin:2px 0 0;color:#33424E;font-size:14.5px}

  /* references */
  .refgroup{margin:0 0 20px}
  .refgroup h4{margin:0 0 8px;font-size:16px;color:var(--ink)}
  .refgroup ul{margin:0;padding-left:20px}
  .refgroup li{margin:5px 0;font-size:15px}

  .callout{background:linear-gradient(135deg,#EAF1F7,#F3F7FA);border:1px solid #D3E0EC;
       border-radius:14px;padding:22px 24px;margin:8px 0}
  .callout h3{margin:0 0 8px;font-size:19px;color:var(--blue)}
  .big{font-size:20px;font-weight:800;color:var(--red)}

  footer{background:#1F2A33;color:#B9C4CE;padding:34px 0;font-size:14px}
  footer a{color:#9FC3E0}
  @media(max-width:680px){
    .gwrap{columns:1}
    header.hero h1{font-size:27px}
    section{padding:36px 0}
  }
</style>
</head>
<body>

<nav><div class="wrap">
  <span class="brand">🌀 湍流降尺度</span>
  <a href="#start">30 秒總覽</a>
  <a href="#concept">核心觀念</a>
  <a href="#data">先認識資料</a>
  <a href="#method">機器學習怎麼做</a>
  <a href="#result">結果</a>
  <a href="#honest">誠實的失敗</a>
  <a href="#glossary">名詞辭典</a>
  <a href="#refs">延伸閱讀</a>
</div></nav>

<header class="hero"><div class="wrap">
  <span class="kick">零基礎也能看懂 · 一頁入門</span>
  <h1>從一座測風塔，看懂什麼是<br>「湍流降尺度」與機器學習</h1>
  <p class="lead">這是一份用真實資料做出來的專案。你不需要任何風能或程式背景——
     這一頁會用生活比喻，帶你從「風是什麼」一路看到「AI 學到了什麼」。</p>
  <div class="oneliner">一句話總結：<b>把風速的影響拿掉之後，風平不平順，幾乎完全由「風從哪裡來」決定。</b></div>
  <div class="herometa">
    <span class="chip">📍 台灣西海岸 100 公尺測風塔</span>
    <span class="chip">⏱ 每秒一筆，2016–2021</span>
    <span class="chip">📊 1.6 億筆原始資料</span>
    <span class="chip">🤖 一般筆電就能跑</span>
  </div>
</div></header>

<section id="start"><div class="wrap">
  <h2 class="sec"><span class="dot"></span>30 秒總覽</h2>
  <p class="sub">如果你只有 30 秒，看這裡就夠了。</p>
  <div class="callout">
    <h3>這個專案在做什麼？</h3>
    <p>一般氣象資料只告訴你「平均風速」，但風機工程真正怕的是風的<b>顛簸</b>（工程上叫「湍流」）。
       這個專案用機器學習，<b>從 10 分鐘的平均狀態，反推出那 10 分鐘內每秒的顛簸程度</b>——
       這叫「降尺度」。</p>
    <p style="margin-bottom:0">結果發現：只用風速，模型準確度（R²）卡在 <b>0.12</b>；
       一旦加入<b>風向</b>，準確度直接跳到 <span class="big">0.66</span>。
       也就是說，<b>風平不平順，看的是方向不是速度</b>。這在物理上說得通——
       吹過平滑海面的風，就是比吹過崎嶇陸地的風平順。</p>
  </div>
  <p style="margin-top:22px">下面我們會慢慢把每個名詞拆開講清楚。遇到不懂的詞，
     隨時可以跳到最下面的<a href="#glossary">名詞小辭典</a>。</p>
</div></section>

<section id="concept"><div class="wrap">
  <h2 class="sec"><span class="dot"></span>核心觀念（先建立直覺）</h2>
  <p class="sub">四個生活比喻，幫你建立整個專案的直覺。</p>
  __CONCEPTS__
</div></section>

<section id="data"><div class="wrap">
  <h2 class="sec"><span class="dot"></span>第一部分：先認識這批風</h2>
  <p class="sub">在讓 AI 學習之前，要先用眼睛把資料看懂。以下每張圖都配一段白話解釋。</p>
  __DATA_FIGS__
</div></section>

<section id="method"><div class="wrap">
  <h2 class="sec"><span class="dot"></span>第二部分：機器學習到底怎麼做</h2>
  <p class="sub">四個關鍵觀念，也是這個專案最想教會你的方法學。</p>

  <div class="ccard" style="border-left-color:var(--blue)">
    <div class="cicon">📏</div>
    <div><h3>基準線：先設一個「笨對照組」</h3>
    <p>評價一個模型好不好，不能只看它的分數，要看它有沒有贏過一個刻意很笨的對照組——
       例如「永遠猜平均值」。很多看起來很厲害的模型，其實根本沒贏過這種笨方法。
       這個專案設了一整排由笨到聰明的基準線（圖 9），一步步證明 AI 真的有加值。</p></div>
  </div>

  <div class="ccard" style="border-left-color:var(--green)">
    <div class="cicon">📈</div>
    <div><h3>R² 是什麼：一個 0 到 1 的分數</h3>
    <p>R²（判定係數）衡量預測有多準。<b>1 = 完美命中，0 = 跟亂猜平均一樣爛，負數 = 比亂猜還差</b>。
       所以當你看到「R² 從 0.12 跳到 0.66」，意思是模型從「幾乎沒用」變成「解釋掉三分之二的變化」。</p></div>
  </div>

  <div class="ccard" style="border-left-color:var(--red)">
    <div class="cicon">🚨</div>
    <div><h3>資料洩漏：最容易踩的作弊陷阱</h3>
    <p>如果不小心把「答案的近親」放進題目，分數會虛高到爆，但模型其實什麼都沒學到。
       比如要預測顛簸程度，卻偷偷把「陣風大小」當線索——那幾乎等於直接看答案。
       這個專案特別寫了一份「禁用欄位清單」，每次訓練前都自動檢查，確保沒有作弊。</p></div>
  </div>

  <div class="ccard" style="border-left-color:var(--purple)">
    <div class="cicon">🌡️</div>
    <div><h3>噪聲天花板：知道「最多能多準」</h3>
    <p>塔上同高度有兩支風速計量同一團空氣。它們彼此有多一致，就是這個量「先天能被預測到多準」的上限——
       因為連兩支真儀器都不會完全一樣。有了天花板，才知道 0.66 是「滿分 0.99 裡的 0.66」，
       而不是一個看起來普通、其實已經接近極限的數字。</p></div>
  </div>
</div></section>

<section id="result"><div class="wrap">
  <h2 class="sec"><span class="dot"></span>第三部分：結果</h2>
  <p class="sub">把上面的觀念套用到真實資料，得到這些圖。</p>
  __RESULT_FIGS__
</div></section>

<section id="honest"><div class="wrap">
  <h2 class="sec"><span class="dot"></span>誠實的失敗（這才是好研究）</h2>
  <p class="sub">不是每個目標都能預測。把做不到的照實說出來，比只報告成功更值得信任。</p>
  <div class="divider-note">
    <b>有兩個目標，模型幾乎預測不了：</b>「一陣風有多長」（積分尺度）和「頻譜斜率」。
    但關鍵在於——透過兩支感測器互相對答案，我們知道「積分尺度」其實<b>量得很準</b>（一致性 0.98），
    只是它<b>不由塔上的 10 分鐘平均狀態決定</b>，比較可能取決於整個大氣邊界層的高度這種塔上量不到的東西。
    能區分「量不準」和「量得準但天生無法預測」，正是靠前面講的噪聲天花板。這也指出了下一步該引入衛星／再分析資料。
  </div>
</div></section>

<section id="glossary"><div class="wrap">
  <h2 class="sec"><span class="dot"></span>名詞小辭典</h2>
  <p class="sub">隨時回來查。每個詞一句話講清楚。</p>
  <div class="gwrap">__GLOSSARY__</div>
</div></section>

<section id="refs"><div class="wrap">
  <h2 class="sec"><span class="dot"></span>延伸閱讀與資源</h2>
  <p class="sub">想再深入的話，這些連結都是精選的權威或入門資源（2026 年 7 月查核可用）。</p>
  __REFS__
</div></section>

<footer><div class="wrap">
  <p>本頁由 BSMI 測風塔資料分析專案自動產生。所有圖表來自真實觀測與可重現的程式
     （<code>preprocess.py</code> · <code>extract_turbulence.py</code> · <code>downscale_turbulence.py</code>）。</p>
  <p style="margin-bottom:0;color:#8896A2">延伸閱讀連結指向外部網站，內容由各原作者維護。</p>
</div></footer>

<div id="lb"><img alt=""></div>
<script>
  (function(){
    var lb=document.getElementById('lb'), lbimg=lb.querySelector('img');
    document.querySelectorAll('.figblock img').forEach(function(im){
      im.addEventListener('click',function(){lbimg.src=im.src;lb.classList.add('on');});
    });
    lb.addEventListener('click',function(){lb.classList.remove('on');lbimg.src='';});
    document.addEventListener('keydown',function(e){
      if(e.key==='Escape'){lb.classList.remove('on');lbimg.src='';}
    });
  })();
</script>
</body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    build(Path(args.figures), Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
