"""自動車・物流の『公開統計』が、関連株の将来の値動きを予測するか。

使い方:
  python research/auto_logistics.py fetch     # 価格・統計をキャッシュ(research/data/ に保存)
  python research/auto_logistics.py analyze   # 検証して表を出力
  python research/auto_logistics.py           # fetch + analyze

【法令上の線引き】使うのは政府・業界団体が公表している統計(日銀短観、財務省貿易統計、
JADA 新車登録台数)と市場価格だけ。勤務先・取引先の非公開情報は使わない・勧めない。

データと公表日(公表遅れ。ここでは保守的に遅めに置く):
- 日銀短観(業況判断DI 大企業 自動車/運輸・郵便、実績と先行き予測): 日銀の時系列統計API。
  3月調査=4/1頃、6月=7/1頃、9月=10/1頃、12月=12/15頃(概算)。
- 財務省貿易統計(自動車の輸出 数量/金額): 税関の月次CSV。翌月20日頃(速報は中旬)。
  注意: CSVは『最新の改訂後』の値。速報値とのズレ(後知恵)が少しある。
- JADA 新車登録台数(登録車): 公式サイトの年別xlsは2022年以降しか残っていない。翌月3日頃。
- 取得不可/見送り: 豊田自動織機6201.T(上場廃止のためYahooに無い=生存者バイアスの例)、JAMA(アクセス拒否403のため回避せず)、経産省 鉱工業生産の時系列(同403)、
  貨物輸送量・トラック運送(国交省・全ト協: 機械で取れる長期の月次表が見つからず)、
  全軽自協(月別ファイルが直近しか辿れない)。
- 価格: Yahoo Finance 日足(調整後終値。株式分割の調整は『リターン』には影響しない)。
  日次±15%超の異常値を検査して警告する。

検証設計(ルックアヘッド対策):
- 公表日(遅め)の『翌営業日』の終値で建て、h営業日後(1/5/20/60)の終値で決済。
- 超過リターン = 対象のリターン − TOPIX連動ETF(1348.T)のリターン。
- 信号: (a)前年比(水準) (b)サプライズ。短観は 実績 − 前期の先行き予測、
  貿易/JADAは 前年比 − 直近12(JADAは6)か月の前年比平均(トレンドからの乖離)。市場予想は取れない。
- 傾き(超過リターン ~ 信号)のt値は Newey-West(ラグ = 重なり分+1)。前半/後半の符号も見る。
- 試した全テスト数を数え、Bonferroni補正の閾値を出す。
- 検出力: 信号1標準偏差あたりの『最小検出効果(80%検出力)』を併記。
  標本が小さい(短観~70点、貿易~200点、JADA~40点)ので、『効かない』ではなく『検出できない』を区別する。
"""
import csv
import io
import json
import math
import os
import re
import sys
import time
from datetime import date, datetime, timedelta
from statistics import NormalDist
from urllib.parse import quote, unquote

import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) research-script/1.0"}
HORIZONS = [1, 5, 20, 60]
BENCH = "1348.T"
STOCKS = {
    "7203.T": "トヨタ", "7267.T": "ホンダ", "7201.T": "日産", "7269.T": "スズキ",
    "7270.T": "SUBARU", "7261.T": "マツダ", "6902.T": "デンソー",
    "9064.T": "ヤマトHD", "9143.T": "SGHD", "9101.T": "日本郵船", "1622.T": "TOPIX-17自動車ETF",
    BENCH: "TOPIX連動ETF",
}
BASKETS = {
    "自動車完成車6社": ["7203.T", "7267.T", "7201.T", "7269.T", "7270.T", "7261.T"],
    "部品(デンソー)": ["6902.T"],
    "運輸(ヤマト・SG)": ["9064.T", "9143.T"],
    "1622.T(自動車ETF)": ["1622.T"],
}


def log(*a):
    print(*a, flush=True)


def http(url, **kw):
    r = requests.get(url, headers=UA, timeout=60, **kw)
    return r


# ------------------------------------------------------------------ fetch
def fetch_prices():
    for sym in STOCKS:
        p = os.path.join(DATA, f"autolog_px_{sym}.json")
        if os.path.exists(p):
            continue
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(sym)}"
               f"?interval=1d&period1=0&period2={int(time.time())}")
        r = http(url)
        if r.status_code != 200:
            log("  価格取得失敗", sym, r.status_code)
            continue
        open(p, "w").write(r.text)
        time.sleep(0.5)


def fetch_boj():
    codes = {"自動車_実績": "TK99F1180601GCQ01000", "自動車_予測": "TK99F1180601GCQ11000",
             "運輸郵便_実績": "TK99F2040601GCQ01000", "運輸郵便_予測": "TK99F2040601GCQ11000"}
    for k, c in codes.items():
        p = os.path.join(DATA, f"autolog_boj_{k}.csv")
        if os.path.exists(p):
            continue
        url = ("https://www.stat-search.boj.or.jp/api/v1/getDataCode?format=csv&lang=jp&db=CO"
               f"&code={c}&startDate=199001")
        r = http(url)
        if r.status_code != 200:
            log("  日銀API失敗", k, r.status_code)
            continue
        open(p, "wb").write(r.content)


def fetch_customs():
    p = os.path.join(DATA, "autolog_customs_d51ma.csv")
    if not os.path.exists(p):
        r = http("https://www.customs.go.jp/toukei/suii/html/data/d51ma.csv")
        if r.status_code != 200:
            log("  税関CSV失敗", r.status_code)
            return
        open(p, "wb").write(r.content)


def fetch_jada():
    page = "https://www.jada.or.jp/pages/336/"
    idx = os.path.join(DATA, "autolog_jada_index.html")
    if not os.path.exists(idx):
        r = http(page)
        if r.status_code != 200:
            log("  JADAページ失敗", r.status_code)
            return
        open(idx, "w").write(r.text)
    t = open(idx).read()
    paths = set()
    for m in re.findall(r'href="([^"]+)"', t):
        m = m.replace("&amp;", "&")
        if "relays/download" in m and ".xls" in m:
            m2 = re.search(r"file=(/files/libs/[^&]+\.xls)", m)
            if m2:
                paths.add(m2.group(1))
        elif re.search(r"/files/libs/\d+/\d+\.xls$", m):
            paths.add(re.sub(r"^https://www\.jada\.or\.jp", "", m))
    for pth in sorted(paths):
        fn = os.path.join(DATA, "autolog_jada_" + pth.strip("/").replace("/", "_"))
        if os.path.exists(fn):
            continue
        r = http("https://www.jada.or.jp" + pth)
        if r.status_code == 200:
            open(fn, "wb").write(r.content)
            time.sleep(0.5)


def do_fetch():
    for f in (fetch_prices, fetch_boj, fetch_customs, fetch_jada):
        try:
            f()
        except Exception as e:  # 取れないものは報告して続行
            log("  取得エラー", f.__name__, repr(e))


# ------------------------------------------------------------------ load
def load_prices():
    ser = {}
    for sym in STOCKS:
        p = os.path.join(DATA, f"autolog_px_{sym}.json")
        if not os.path.exists(p):
            continue
        j = json.load(open(p))["chart"]["result"][0]
        adj = j["indicators"].get("adjclose", [{}])[0].get("adjclose") or j["indicators"]["quote"][0]["close"]
        s = pd.Series(adj, index=pd.to_datetime(j["timestamp"], unit="s").tz_localize("UTC")
                      .tz_convert("Asia/Tokyo").tz_localize(None).normalize(), dtype=float)
        s = s[~s.index.duplicated(keep="last")].dropna()
        s = s[s > 0]
        ser[sym] = s
    px = pd.DataFrame(ser).sort_index()
    return px


def price_checks(px):
    for c in px.columns:
        r = px[c].dropna().pct_change().dropna()
        bad = r[r.abs() > 0.15]
        flag = f"日次±15%超 {len(bad)}件 {[d.strftime('%Y-%m-%d') for d in bad.index[:3]]}" if len(bad) else "異常なし"
        log(f"  {c:7s} {STOCKS[c][:12]:12s} {px[c].dropna().index[0].date()}〜 {len(px[c].dropna())}日  {flag}")


def load_tankan():
    out = {}
    for k in ["自動車_実績", "自動車_予測", "運輸郵便_実績", "運輸郵便_予測"]:
        p = os.path.join(DATA, f"autolog_boj_{k}.csv")
        if not os.path.exists(p):
            continue
        rows = [l for l in open(p, encoding="cp932", errors="replace").read().splitlines() if l.startswith("TK99")]
        d = {}
        for l in rows:
            c = l.split(",")
            lab, v = c[6], c[7]
            try:
                d[(int(lab[:4]), int(lab[4:]))] = float(v)  # (年, 四半期1..4)
            except ValueError:
                pass
        out[k] = d
    return out


def tankan_release(y, q):
    # 公表日は概算(遅め)。3月調査=4/1、6月=7/1、9月=10/1、12月=12/15
    if q == 1:
        return date(y, 4, 1)
    if q == 2:
        return date(y, 7, 1)
    if q == 3:
        return date(y, 10, 1)
    return date(y, 12, 15)


def tankan_signals(tk):
    sigs = {}
    for ind in ["自動車", "運輸郵便"]:
        a, f = tk.get(ind + "_実績", {}), tk.get(ind + "_予測", {})
        s_sur, s_yoy = {}, {}
        for (y, q), v in a.items():
            rel = tankan_release(y, q)
            # 実績: 同じ四半期ラベルの予測 = 前回調査時点の先行き予測
            if (y, q) in f:
                s_sur[rel] = v - f[(y, q)]
            if (y - 1, q) in a:
                s_yoy[rel] = v - a[(y - 1, q)]
        sigs[f"短観{ind}:サプライズ(実績-前回予測)"] = pd.Series(s_sur).sort_index()
        sigs[f"短観{ind}:前年差(DI)"] = pd.Series(s_yoy).sort_index()
    return sigs


def load_customs():
    p = os.path.join(DATA, "autolog_customs_d51ma.csv")
    if not os.path.exists(p):
        return {}
    rows = list(csv.reader(open(p, encoding="cp932", errors="replace")))
    qty, val = {}, {}
    for r in rows:
        if not re.match(r"^\d{4}/\d{2}$", r[0]):
            continue
        y, m = map(int, r[0].split("/"))
        try:
            qty[(y, m)] = float(r[94])
            val[(y, m)] = float(r[95])
        except (ValueError, IndexError):
            pass
    return {"数量": pd.Series(qty), "金額": pd.Series(val)}


def add_months(y, m, k):
    t = y * 12 + (m - 1) + k
    return t // 12, t % 12 + 1


def monthly_signals(level, name, rel_day, rel_lag_months, trend_n, start_year):
    """level: Series[(y,m)] -> 前年比、トレンド乖離。公表日=翌月(+rel_lag_months-1)の rel_day 日。"""
    keys = sorted(level.index)
    yoy = {}
    for (y, m) in keys:
        if (y - 1, m) in level.index and level[(y - 1, m)] > 0:
            yoy[(y, m)] = level[(y, m)] / level[(y - 1, m)] - 1
    yoy = pd.Series(yoy)
    s_yoy, s_sur = {}, {}
    for (y, m) in sorted(yoy.index):
        if y < start_year:
            continue
        ry, rm = add_months(y, m, rel_lag_months)
        rel = date(ry, rm, rel_day)
        s_yoy[rel] = yoy[(y, m)]
        prev = [yoy.get(add_months(y, m, -k)) for k in range(1, trend_n + 1)]
        prev = [x for x in prev if x is not None]
        if len(prev) == trend_n:
            s_sur[rel] = yoy[(y, m)] - float(np.mean(prev))
    return {f"{name}:前年比": pd.Series(s_yoy).sort_index(),
            f"{name}:サプライズ(対過去{trend_n}か月平均)": pd.Series(s_sur).sort_index()}


def load_jada():
    import glob
    try:
        import xlrd
    except ImportError:
        log("  xlrd が無いのでJADAは読めない")
        return pd.Series(dtype=float)
    tot = {}
    for fn in glob.glob(os.path.join(DATA, "autolog_jada_files_libs_*.xls")):
        try:
            b = xlrd.open_workbook(fn)
        except Exception:
            continue
        for s in b.sheets():
            m = re.match(r"(\d{4})年(\d{1,2})月$", s.name)  # 月ごとのシート(当月の合計)
            if not m:
                continue
            y, mo = int(m.group(1)), int(m.group(2))
            for i in range(s.nrows):
                row = s.row_values(i)
                labs = [k for k in range(min(3, len(row))) if re.sub(r"[ \u3000]", "", str(row[k])) == "合計"]
                if labs:  # ファイルにより列の位置が違う。ラベルの右隣の最初の数値=当月
                    nums = [v for v in row[labs[0] + 1:] if isinstance(v, float)]
                    if nums and nums[0] > 0:
                        tot[(y, mo)] = nums[0]
                        break
    return pd.Series(tot).sort_index()


# ------------------------------------------------------------------ stats
def nw_slope_t(x, y, L):
    """y = a + b x の b と Newey-West t値(ラグL)。"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    XtXi = np.linalg.inv(X.T @ X)
    beta = XtXi @ X.T @ y
    e = y - X @ beta
    u = X * e[:, None]
    S = u.T @ u
    for l in range(1, min(L, n - 1) + 1):
        w = 1 - l / (L + 1)
        G = u[l:].T @ u[:-l]
        S += w * (G + G.T)
    V = XtXi @ S @ XtXi
    se = math.sqrt(max(V[1, 1], 1e-300))
    return beta[1], beta[1] / se, se


def build_events(sig, pxdates, pxmat, targets, bench):
    """sig: Series[date]->値。各信号日の『翌営業日』の終値で建て。行=イベント。"""
    out = []
    for d, v in sig.items():
        ts = pd.Timestamp(d)
        i = pxdates.searchsorted(ts, side="right")  # ts より後の最初の営業日
        if i >= len(pxdates):
            continue
        out.append((ts, i, v))
    return out


def run_pair(sig, px, tname, members, bench, h, spacing_days):
    basket = px[members].dropna(how="all")
    dates = px.index
    rows = []
    for d, v in sig.items():
        i = dates.searchsorted(pd.Timestamp(d), side="right")
        j = i + h
        if j >= len(dates):
            continue
        p0, p1 = px.iloc[i], px.iloc[j]
        rets = []
        for m in members:
            if pd.notna(p0[m]) and pd.notna(p1[m]):
                rets.append(p1[m] / p0[m] - 1)
        if not rets or pd.isna(p0[bench]) or pd.isna(p1[bench]):
            continue
        ex = float(np.mean(rets)) - (p1[bench] / p0[bench] - 1)
        rows.append((dates[i], v, ex))
    if len(rows) < 12:
        return None
    df = pd.DataFrame(rows, columns=["d", "x", "ex"])
    df["z"] = (df.x - df.x.mean()) / df.x.std(ddof=1)
    L = int(math.ceil(h / spacing_days)) + 1
    b, t, se = nw_slope_t(df.z, df.ex, L)
    mid = len(df) // 2
    r = dict(n=len(df), b=b, t=t, se=se, mde=2.8 * se)
    for nm, part in (("h1", df.iloc[:mid]), ("h2", df.iloc[mid:])):
        if len(part) >= 8:
            bb, tt, _ = nw_slope_t(part.z, part.ex, L)
            r[nm] = (bb, tt)
        else:
            r[nm] = (np.nan, np.nan)
    # 循環シフト検定: 信号を時間方向にずらして傾きの分布を作る(自己相関・重なりを保つのでHACより保守的)。
    # ずらし方は n-6 通りしかないので、最小のp値は 1/(n-5) 程度。短い標本では有意になりにくい(=検出力の限界)
    zz, ee = df.z.values, df.ex.values
    sh = [abs(np.polyfit(np.roll(zz, k), ee, 1)[0]) for k in range(3, len(df) - 3)]
    r["p_shift"] = (1 + sum(v >= abs(b) - 1e-12 for v in sh)) / (1 + len(sh))
    q = df.z.rank(pct=True)
    hi, lo = df.ex[q > 2 / 3], df.ex[q <= 1 / 3]
    r["spread"] = hi.mean() - lo.mean() if len(hi) > 3 and len(lo) > 3 else np.nan
    return r


def main_analysis():
    px = load_prices()
    if BENCH not in px:
        log("ベンチマーク価格が無いので中止")
        return
    log("== 価格データの検査")
    price_checks(px)

    tk = load_tankan()
    sigs = {}
    spacing = {}
    if tk:
        for k, s in tankan_signals(tk).items():
            sigs[k] = s
            spacing[k] = 63
    cu = load_customs()
    for nm, lv in cu.items():
        # 公表日: 翌月20日(速報は中旬、保守的に)
        for k, s in monthly_signals(lv, f"貿易:自動車輸出{nm}", 20, 1, 12, 2005).items():
            sigs[k] = s
            spacing[k] = 21
    jd = load_jada()
    if len(jd):
        for k, s in monthly_signals(jd, "JADA新車登録合計", 3, 1, 6, 2022).items():
            sigs[k] = s
            spacing[k] = 21
    log("\n== 取得できた信号(公表日ベース)")
    for k, s in sigs.items():
        s2 = s[(pd.to_datetime(list(s.index)) >= px.index[0]) & (pd.to_datetime(list(s.index)) <= px.index[-1])]
        log(f"  {k:48s} 全{len(s)}点 / 株価期間内 {len(s2)}点  {min(s.index)}〜{max(s.index)}")

    # 対応づけ: 自動車系の信号 → 完成車/部品/自動車ETF、運輸の信号 → 運輸
    auto_t = ["自動車完成車6社", "部品(デンソー)", "1622.T(自動車ETF)"]
    trans_t = ["運輸(ヤマト・SG)"]
    results = []
    for k, s in sigs.items():
        tg = trans_t if "運輸" in k else auto_t
        for tname in tg:
            for h in HORIZONS:
                r = run_pair(s, px, tname, BASKETS[tname], BENCH, h, spacing[k])
                if r:
                    results.append(dict(signal=k, target=tname, h=h, **r))
    res = pd.DataFrame(results)
    ntests = len(res)
    nd = NormalDist()
    z_bonf = nd.inv_cdf(1 - 0.05 / ntests / 2)
    log(f"\n== 検証した全パターン数 = {ntests} (信号{len(sigs)} × 対象 × 地平 {HORIZONS})")
    log(f"   Bonferroni(両側5%)のt閾値 = {z_bonf:.2f}。偶然でも|t|>2 は約5%({ntests*0.05:.1f}件)出るはず")
    res["sig_bonf"] = res.t.abs() > z_bonf
    res["shift_ok"] = res.p_shift < 0.05 / ntests
    res["sign_same"] = [(a[0] * b_[0] > 0) if not (np.isnan(a[0]) or np.isnan(b_[0])) else False
                        for a, b_ in zip(res.h1, res.h2)]
    log(f"   循環シフト検定(自己相関に頑健)で p<0.05 は {int((res.p_shift < 0.05).sum())}件、"
        f"Bonferroni(p<{0.05/ntests:.4f})を満たすのは {int(res.shift_ok.sum())}件"
        f" (ずらし方が少なく、短い標本では最小p値がその水準に届かないことが多い)")
    n2 = int((res.t.abs() > 2).sum())
    log(f"   |t|>2 は {n2}件 / 補正後も有意 {int(res.sig_bonf.sum())}件 / "
        f"|t|>2 かつ前半後半で同符号 {int(((res.t.abs() > 2) & res.sign_same).sum())}件")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 40)
    show = res.reindex(res.t.abs().sort_values(ascending=False).index).head(15).copy()
    log("\n== |t| 上位15件(信号1σ上昇あたりの対TOPIX超過リターン b[%]、t=Newey-West、前半/後半のb)")
    for _, r in show.iterrows():
        log(f"  {r.signal[:40]:40s} {r.target[:12]:12s} h={int(r.h):2d} n={int(r.n):3d} "
            f"b={r.b*100:+.2f}% t={r.t:+.2f} 前半{r.h1[0]*100:+.2f}%(t{r.h1[1]:+.1f}) "
            f"後半{r.h2[0]*100:+.2f}%(t{r.h2[1]:+.1f}) 上位-下位={r.spread*100:+.2f}% シフトp={r.p_shift:.3f}")

    log("\n== 信号ごとの要約(その信号の全対象×地平の中の最大|t|と最小検出効果)")
    for k, g in res.groupby("signal"):
        top = g.reindex(g.t.abs().sort_values(ascending=False).index).iloc[0]
        log(f"  {k[:44]:44s} n≈{int(g.n.median()):3d} 最大|t|={abs(top.t):.2f}({top.target[:6]} h={int(top.h)}) "
            f"最小検出効果(60日,80%検出力)≈{g[g.h==60].mde.median()*100:.1f}%/1σ")

    log("\n== 検出力の目安: 60日地平で、信号1σあたり何%の超過リターンなら80%で検出できるか(中央値)")
    for k, g in res[res.h == 60].groupby("signal"):
        log(f"  {k[:44]:44s} n={int(g.n.median()):3d}  MDE≈{g.mde.median()*100:.1f}%")
    log("  (自動車株の対TOPIX超過リターンの60日の標準偏差は数%。現実的な効果は年に数%以下なので、"
        "この標本数では『ある程度大きい効果』しか検出できない)")

    res.drop(columns=["h1", "h2"]).to_csv(os.path.join(DATA, "autolog_results.csv"), index=False)
    log(f"\n結果を {os.path.join(DATA, 'autolog_results.csv')} に保存")
    log("\n解釈の注意:")
    log("- |t|>2 が数件あっても、試行数が多いので偶然の範囲か、前半/後半で符号が逆かを必ず確認する")
    log("- 『効かなかった』ではなく『この標本の大きさでは検出できなかった』が正確")
    log("- 貿易統計は改訂後の値、短観の公表日は概算(遅め)。JADAは2022年以降のみ(約45点)")
    log("- 使ったのは公表統計だけ。職務上知り得た非公開情報は使わない・勧めない(インサイダー取引規制)")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("fetch", "all"):
        do_fetch()
    if cmd in ("analyze", "all"):
        main_analysis()


if __name__ == "__main__":
    main()
