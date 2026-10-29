"""ポジショニング・センチメント指標が将来リターンを予測するかの検証。

データ(全て無料公開。research/data/ にキャッシュ):
  - CFTC Commitments of Traders (legacy, futures only): 円/ユーロ/金/E-mini S&P/ドル指数。投機筋=Non-commercial
  - AAII センチメント調査 (強気-弱気スプレッド)
  - CBOE Put/Call 比率 (公開CSVは 2006-11〜2019-10 まで)
  - JPX 投資部門別売買状況: 週次の過去データは公開ページから取得不可(直近5週のみ) → 検証せず、可用性だけ確認
  - 価格: Yahoo Finance 日足

ルックアヘッド対策:
  - COT は火曜時点・金曜公表。株/金は火曜+6日(翌月曜)終値、FX は火曜+7日(翌火曜ラベル=実質月曜夕)の終値で建てる
  - AAII は水曜付け・木曜公表。金曜終値(+2日)で建てる
  - P/C は金曜までの5営業日平均。翌月曜終値で建てる
検証: 各(指標, 窓, 分位, 側, 保有週)セルで、極端値の週の平均リターン - 全期間平均 (= 超過) を
  Newey-West(ラグ=保有週数)のt値で評価。全セル数でBonferroni。前半(t>=2のセルを選ぶ)→後半で確認。
実行: python research/positioning.py
"""
import io
import json
import math
import os
import re
import time
import urllib.parse
import urllib.request
import zipfile
from datetime import date, datetime, timedelta, timezone
from statistics import NormalDist

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
UA = {"User-Agent": "Mozilla/5.0"}
os.makedirs(DATA, exist_ok=True)


def http(url, tries=3):
    err = None
    for _ in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            return urllib.request.urlopen(req, timeout=60).read()
        except Exception as e:  # noqa
            err = e
            time.sleep(1)
    raise err


def cached(name, url):
    p = os.path.join(DATA, name)
    if not os.path.exists(p) or os.path.getsize(p) < 100:
        with open(p, "wb") as f:
            f.write(http(url))
    return p


# ---------------- 価格 ----------------
def yahoo(sym):
    p = os.path.join(DATA, "pos_px_%s.json" % sym.replace("^", "_").replace("=", "_").replace(".", "_"))
    if not os.path.exists(p):
        url = ("https://query1.finance.yahoo.com/v8/finance/chart/%s?interval=1d&period1=0&period2=%d"
               % (urllib.parse.quote(sym), int(time.time()) + 86400 * 2))
        with open(p, "wb") as f:
            f.write(http(url))
    j = json.load(open(p))
    r = j["chart"]["result"][0]
    ts = r["timestamp"]
    cl = r["indicators"]["quote"][0]["close"]
    idx = pd.to_datetime([datetime.fromtimestamp(t, timezone.utc).date() for t in ts])
    s = pd.Series(cl, index=idx, dtype=float)
    s = s[~s.index.duplicated(keep="last")].dropna()
    bad = s.pct_change().abs() > 0.15
    if bad.any():
        print("  [warn] %s: 日次±15%%超が %d 件 (%s)" % (sym, bad.sum(), ",".join(str(d.date()) for d in s.index[bad])))
    return s


def asof(px, d):
    """日付d以前の最後の終値(なければNaN)"""
    i = px.index.searchsorted(pd.Timestamp(d), side="right") - 1
    if i < 0:
        return np.nan
    return px.iloc[i]


# ---------------- COT ----------------
COT = {  # 名前: (CFTCコード, 価格シンボル, 符号(先物ロング=資産ロングか), FXか)
    "JPY": ("097741", "USDJPY=X", -1, True),   # 円ロング = USDJPY ショート
    "EUR": ("099741", "EURUSD=X", +1, True),
    "SPX": ("13874A", "^GSPC", +1, False),
    "GOLD": ("088691", "GLD", +1, False),
    "DXY": ("098662", "DX-Y.NYB", +1, True),
}


def load_cot():
    out = {k: [] for k in COT}
    for y in range(2006, date.today().year + 1):
        p = cached("pos_cot_%d.zip" % y, "https://www.cftc.gov/files/dea/history/deacot%d.zip" % y)
        z = zipfile.ZipFile(p)
        df = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), dtype={"CFTC Contract Market Code": str},
                         low_memory=False, encoding="latin1")
        df.columns = [c.strip() for c in df.columns]
        df["code"] = df["CFTC Contract Market Code"].str.strip()
        for k, (code, *_r) in COT.items():
            d = df[df["code"] == code].copy()
            d["asof"] = pd.to_datetime(d["As of Date in Form YYYY-MM-DD"])
            net = d["Noncommercial Positions-Long (All)"] - d["Noncommercial Positions-Short (All)"]
            out[k].append(pd.Series((net / d["Open Interest (All)"]).astype(float).values, index=d["asof"].values))
    res = {}
    for k, v in out.items():
        s = pd.concat(v).sort_index()
        res[k] = s[~s.index.duplicated()]
    return res


# ---------------- AAII / CBOE ----------------
def load_aaii():
    import xlrd
    p = cached("pos_aaii.xls", "https://www.aaii.com/files/surveys/sentiment.xls")
    s = xlrd.open_workbook(p).sheet_by_index(0)
    rows = []
    for r in range(5, s.nrows):
        v = s.row_values(r)
        if isinstance(v[0], float) and v[1] != "" and v[3] != "":
            d = xlrd.xldate_as_datetime(v[0], 0)
            rows.append((pd.Timestamp(d), float(v[1]) - float(v[3])))
    return pd.Series(dict(rows)).sort_index()


def load_pc(name):
    p = cached("pos_cboe_%s.csv" % name, "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/%s.csv" % name)
    txt = open(p, encoding="latin1").read().splitlines()
    i = next(k for k, l in enumerate(txt) if l.upper().startswith("DATE"))
    rows = []
    for l in txt[i + 1:]:
        c = [x.strip() for x in l.split(",")]
        try:
            rows.append((pd.to_datetime(c[0], format="%m/%d/%Y"), float(c[-1])))
        except Exception:  # noqa
            pass
    return pd.Series(dict(rows)).sort_index()


# ---------------- 統計 ----------------
def nw_t(y, d, lag):
    """y = a + b*d の b の Newey-West t値"""
    n = len(y)
    if d.sum() < 5 or (1 - d).sum() < 5:
        return np.nan, np.nan
    X = np.column_stack([np.ones(n), d.astype(float)])
    XtXi = np.linalg.inv(X.T @ X)
    beta = XtXi @ X.T @ y
    e = y - X @ beta
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for L in range(1, lag + 1):
        w = 1 - L / (lag + 1)
        G = Xe[L:].T @ Xe[:-L]
        S += w * (G + G.T)
    V = XtXi @ S @ XtXi
    return beta[1], beta[1] / math.sqrt(V[1, 1])


def rolling_pct(s, n):
    """直近n本(当期含む)の中での当期の順位(0〜1)。過去のみ使用"""
    v = s.values
    out = np.full(len(v), np.nan)
    for i in range(n - 1, len(v)):
        w = v[i - n + 1:i + 1]
        out[i] = (w < v[i]).sum() / (n - 1)
    return pd.Series(out, index=s.index)


HS = [1, 2, 4, 8]
QS = [0.10, 0.20]
COST_RT = {"fx": 0.0004, "eq": 0.002}  # 往復: FX 0.02%/片道, 株/金 0.1%/片道


def build_returns(sig_dates, px, entry_lag, h, sign):
    r = []
    last = px.index[-1]
    for d in sig_dates:
        e = pd.Timestamp(d) + timedelta(days=entry_lag)
        x = e + timedelta(days=7 * h)
        if x > last:
            r.append(np.nan)
            continue
        p0, p1 = asof(px, e), asof(px, x)
        r.append(sign * math.log(p1 / p0) if p0 > 0 and p1 > 0 else np.nan)
    return np.array(r)


def run_family(name, sig, px, entry_lag, sign, kind, lookbacks, results):
    for lb in lookbacks:
        pct = rolling_pct(sig, lb)
        for h in HS:
            r = build_returns(list(sig.index), px, entry_lag, h, sign)
            ok = ~np.isnan(r) & ~np.isnan(pct.values)
            if ok.sum() < 100:
                continue
            idx = sig.index[ok]
            rr, pp = r[ok], pct.values[ok]
            mid = idx[len(idx) // 2]
            for q in QS:
                for tail in ("high", "low"):
                    D = (pp >= 1 - q) if tail == "high" else (pp <= q)
                    b, t = nw_t(rr, D.astype(float), h)
                    if np.isnan(t):
                        continue
                    h1 = idx < mid
                    b1, t1 = nw_t(rr[h1], D[h1].astype(float), h)
                    b2, t2 = nw_t(rr[~h1], D[~h1].astype(float), h)
                    results.append(dict(fam=name, kind=kind, lb=lb, h=h, q=q, tail=tail, n=int(D.sum()),
                                        mean_ext=float(rr[D].mean()), mean_all=float(rr.mean()), b=b, t=t,
                                        b1=b1, t1=t1, b2=b2, t2=t2, idx=idx, rr=rr, D=D, mid=mid))


def oos_trade(c):
    """前半の符号で後半に売買(非重複・コスト後)。戻り値: (回数, 往復後1回平均, 合計)"""
    direction = 1 if c["b1"] > 0 else -1
    idx, rr, D, h = c["idx"], c["rr"], c["D"], c["h"]
    cost = COST_RT[c["kind"]]
    rets, nxt = [], None
    for i in range(len(idx)):
        if idx[i] < c["mid"] or not D[i]:
            continue
        if nxt is not None and idx[i] < nxt:
            continue
        rets.append(direction * rr[i] - cost)
        nxt = idx[i] + pd.Timedelta(days=7 * h)
    if not rets:
        return 0, np.nan, 0.0
    return len(rets), float(np.mean(rets)), float(np.sum(rets))


def summarize(title, res, ntests_total, zbonf):
    print("\n=== %s ===" % title)
    n = len(res)
    if not n:
        print("セルなし")
        return
    t = np.array([c["t"] for c in res])
    print("セル数 %d / |t|>2: %d (偶然の期待 約%.0f) / |t|>%.2f(Bonferroni, 全%d本): %d"
          % (n, (abs(t) > 2).sum(), 0.0455 * n, zbonf, ntests_total, (abs(t) > zbonf).sum()))
    top = sorted(res, key=lambda c: -abs(c["t"]))[:5]
    print("事後ベスト5(選別バイアスあり、参考):")
    for c in top:
        print("  %-8s lb=%3d h=%dw q=%.2f %-4s n=%3d 極端値平均%+.2f%% 全体平均%+.2f%% 超過t=%+.2f | 前半t=%+.2f 後半t=%+.2f"
              % (c["fam"], c["lb"], c["h"], c["q"], c["tail"], c["n"], 100 * c["mean_ext"], 100 * c["mean_all"],
                 c["t"], c["t1"], c["t2"]))
    sel = [c for c in res if not np.isnan(c["t1"]) and not np.isnan(c["t2"]) and abs(c["t1"]) >= 2]
    if not sel:
        print("前半で|t|>=2のセルなし")
        return
    same = [c for c in sel if np.sign(c["b2"]) == np.sign(c["b1"])]
    hit = [c for c in same if abs(c["t2"]) >= 2]
    print("前半で|t|>=2のセル %d 個 → 後半で同符号 %d 個、後半も|t|>=2 %d 個 (偶然なら同符号50%%)"
          % (len(sel), len(same), len(hit)))
    tot = [(*oos_trade(c), c) for c in sel]
    ks = sum(x[0] for x in tot)
    if ks:
        avg = sum(x[1] * x[0] for x in tot if x[0]) / ks
        print("  それらを後半で売買(非重複、コスト後): 計%d回、1回平均%+.3f%%" % (ks, 100 * avg))
    best = sorted([x for x in tot if x[0] >= 5], key=lambda x: -x[1])[:3]
    for k, m, s, c in best:
        print("    例 %-8s lb=%d h=%dw q=%.2f %s: 後半%d回 1回%+.3f%% → 建玉25万円で1回 %+.0f円(事後上位, 参考)"
              % (c["fam"], c["lb"], c["h"], c["q"], c["tail"], k, 100 * m, m * 250000))


def main():
    results = {}
    print("データ取得...")
    cot = load_cot()
    pxs = {k: yahoo(v[1]) for k, v in COT.items()}
    print("COT: 取得OK", {k: (s.index[0].date(), s.index[-1].date(), len(s)) for k, s in cot.items()})

    res = []
    for k, (code, sym, sign, isfx) in COT.items():
        run_family(k, cot[k], pxs[k], 7 if isfx else 6, sign, "fx" if isfx else "eq", [156, 260], res)
    results["COT"] = res

    resa = []
    try:
        aaii = load_aaii()
        print("AAII: 取得OK", aaii.index[0].date(), aaii.index[-1].date(), len(aaii))
        run_family("AAII", aaii, pxs["SPX"], 2, +1, "eq", [156, 260], resa)
    except Exception as e:  # noqa
        print("AAII: 取得不可", e)
    results["AAII"] = resa

    resp = []
    try:
        for nm in ("totalpc", "equitypc"):
            pc = load_pc(nm)
            wk = pc.groupby(pd.Grouper(freq="W-FRI")).mean().dropna()
            print("CBOE %s: 取得OK %s〜%s" % (nm, pc.index[0].date(), pc.index[-1].date()))
            run_family("PC_" + nm[:3], wk, pxs["SPX"], 3, +1, "eq", [52, 156], resp)
    except Exception as e:  # noqa
        print("CBOE: 取得不可", e)
    results["CBOE"] = resp

    try:
        html = http("https://www.jpx.co.jp/markets/statistics-equities/investor-type/index.html").decode("utf8", "ignore")
        wk = sorted(set(re.findall(r"stock_1_w_(\d{8})_(\d{8})\.xlsx", html)))
        print("JPX 週次ファイル: 公開ページで確認できたのは %d 週のみ (%s〜%s)。過去の週次系列は取得不可 → 検証せず"
              % (len(wk), wk[0][0] if wk else "-", wk[-1][1] if wk else "-"))
    except Exception as e:  # noqa
        print("JPX: 取得不可", e)

    total = sum(len(v) for v in results.values())
    zb = NormalDist().inv_cdf(1 - 0.05 / total / 2)
    print("\n試行した全セル数(指標×窓×分位×側×保有週) = %d、Bonferroni閾値 |t|>%.2f" % (total, zb))
    for k, v in results.items():
        if k == "COT":
            for nm in COT:
                summarize("COT " + nm, [c for c in v if c["fam"] == nm], total, zb)
        else:
            summarize(k, v, total, zb)
    allr = [c for v in results.values() for c in v]
    summarize("全体", allr, total, zb)
    print("\n注: セル同士は相関(重複する窓・保有期間)しているため、独立な検定数はもっと少ない。Bonferroniは保守的。")


if __name__ == "__main__":
    main()
