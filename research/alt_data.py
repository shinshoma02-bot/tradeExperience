"""代替データ(GDELTニュース・Wikipedia閲覧数)が翌日〜5日後の値動きを予測するか。

使い方:
  python research/alt_data.py fetch     # 価格・GDELT日次・Wikipediaをキャッシュ(research/data/)
  python research/alt_data.py analyze   # 検証して表を出力
  python research/alt_data.py           # fetch + analyze

設計(ルックアヘッド対策):
- GDELT/Wikipediaの「日D」はUTCの0:00〜24:00の集計。Dの終わりまで確定しないので、
  米国の終値(21:00UTC)にも日本の終値(06:00UTC)にも間に合わない日がある。
  そこで、シグナル日Dの翌取引日の終値で建て、そこから h 営業日後の終値で決済する(保守的)。
- FX(Yahoo日足)は日付tの終値が実質0:00UTC頃なので、さらに1行ずらして 'その日の終わり' に揃える。
- 極端値の判定は過去250〜500日の分布のみ(上位/下位10%)。z値は過去60日の平均・標準偏差から。
- 超過リターン=条件付き平均 − 全期間平均。t値は Newey-West(ラグ=h)。
- 試した全テスト数を数え、Bonferroni補正の閾値を出す。前半で決めて後半で確かめる。
"""
import concurrent.futures as cf
import io
import json
import math
import os
import sys
import time
import zipfile
from datetime import date, datetime, timedelta
from statistics import NormalDist
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 research-script/1.0"}
START = date(2016, 1, 1)
END = date(2026, 10, 27)
ASSETS = {"1348.T": "TOPIX連動ETF", "SPY": "SPY", "USDJPY=X": "ドル円"}
HORIZONS = [1, 2, 3, 5]
COST_ONEWAY = {"1348.T": 0.001, "SPY": 0.001, "USDJPY=X": 0.0001}  # 片道(要確認の想定)
WIKI_PAGES = [
    ("en", "Recession"), ("en", "Inflation"), ("en", "Stock_market_crash"),
    ("en", "Bear_market"), ("en", "Stagflation"), ("en", "Unemployment"),
    ("en", "S&P_500"), ("en", "Nikkei_225"), ("en", "TOPIX"),
    ("en", "Yen"), ("ja", "景気後退"), ("ja", "インフレーション"), ("ja", "日経平均株価"),
]


# ---------------------------------------------------------------- fetch
def fetch_price(sym):
    p = os.path.join(DATA, f"alt_px_{sym.replace('=', '_')}.json")
    if not os.path.exists(p):
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(sym)}"
               f"?interval=1d&period1=0&period2={int(time.time())}")
        r = requests.get(url, headers=UA, timeout=60)
        r.raise_for_status()
        open(p, "w").write(r.text)
    j = json.load(open(p))["chart"]["result"][0]
    s = pd.Series(j["indicators"]["quote"][0]["close"],
                  index=pd.to_datetime(j["timestamp"], unit="s").normalize(), name=sym)
    s = s[~s.index.duplicated(keep="last")].dropna()
    return s


def gdelt_day(d):
    """GDELT 1.0 日次イベントファイルを集計(日本/米国/世界)。"""
    cache = os.path.join(DATA, "gdelt", f"{d:%Y%m%d}.json")
    if os.path.exists(cache):
        return d, json.load(open(cache))
    url = f"http://data.gdeltproject.org/events/{d:%Y%m%d}.export.CSV.zip"
    for attempt in range(4):
        try:
            r = requests.get(url, headers=UA, timeout=120)
            if r.status_code == 404:
                return d, None
            r.raise_for_status()
            z = zipfile.ZipFile(io.BytesIO(r.content))
            raw = z.read(z.namelist()[0])
            nf = raw[:200000].split(b"\n")[0].count(b"\t") + 1
            cc = nf - 7 if nf == 58 else nf - 8  # ActionGeo_CountryCode
            df = pd.read_csv(io.BytesIO(raw), sep="\t", header=None, quoting=3,
                             usecols=[30, 31, 34, cc], names=None, dtype=str,
                             on_bad_lines="skip")
            df.columns = ["gold", "ment", "tone", "cc"]
            for c in ("gold", "ment", "tone"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["ment", "tone"])
            out = {}
            for name, sub in (("JA", df[df.cc == "JA"]), ("US", df[df.cc == "US"]), ("W", df)):
                m = sub.ment.sum()
                out[name] = {"n": int(len(sub)), "ment": float(m),
                             "tone": float((sub.tone * sub.ment).sum() / m) if m > 0 else None}
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            json.dump(out, open(cache, "w"))
            return d, out
        except Exception as e:  # noqa
            time.sleep(2 + attempt * 3)
            err = e
    print("gdelt fail", d, err, file=sys.stderr)
    return d, None


def fetch_gdelt():
    days = [START + timedelta(i) for i in range((END - START).days + 1)]
    todo = [d for d in days if not os.path.exists(os.path.join(DATA, "gdelt", f"{d:%Y%m%d}.json"))]
    print(f"GDELT: {len(days)} days, {len(todo)} to fetch", flush=True)
    done = 0
    with cf.ThreadPoolExecutor(4) as ex:
        for d, _ in ex.map(gdelt_day, todo):
            done += 1
            if done % 200 == 0:
                print("  gdelt", done, "/", len(todo), flush=True)
    rows = {}
    for d in days:
        p = os.path.join(DATA, "gdelt", f"{d:%Y%m%d}.json")
        if os.path.exists(p):
            o = json.load(open(p))
            rows[pd.Timestamp(d)] = {f"{g}_{k}": o[g][k] for g in o for k in ("n", "ment", "tone")}
    df = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    df.to_csv(os.path.join(DATA, "alt_gdelt_daily.csv"))
    print("GDELT daily rows:", len(df), "missing:", len(days) - len(df))


def fetch_wiki():
    out = {}
    for lang, page in WIKI_PAGES:
        p = os.path.join(DATA, f"alt_wiki_{lang}_{quote(page, safe='')}.json")
        if not os.path.exists(p):
            url = (f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{lang}.wikipedia/"
                   f"all-access/user/{quote(page, safe='')}/daily/{START:%Y%m%d}/{END:%Y%m%d}")
            ok = False
            for attempt in range(8):
                r = requests.get(url, headers=UA, timeout=60)
                if r.status_code == 200:
                    open(p, "w").write(r.text)
                    ok = True
                    break
                time.sleep(5 + attempt * 5)
            if not ok:
                print("wiki 取得不可:", lang, page, r.status_code, file=sys.stderr)
                continue
            time.sleep(2)
        j = json.load(open(p))["items"]
        out[f"{lang}:{page}"] = pd.Series(
            [x["views"] for x in j],
            index=pd.to_datetime([x["timestamp"][:8] for x in j]))
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(DATA, "alt_wiki_daily.csv"))
    print("Wiki pages:", list(df.columns), "rows", len(df))


def fetch_trends():
    """Google Trends: 公式APIなし。非公式エンドポイントに1回だけ試す。"""
    try:
        r = requests.get("https://trends.google.com/trends/api/explore?hl=en-US&tz=0&req="
                         + quote(json.dumps({"comparisonItem": [{"keyword": "recession", "geo": "US",
                                  "time": "2016-01-01 2026-10-01"}], "category": 0, "property": ""})),
                         headers=UA, timeout=30)
        print("Google Trends HTTP", r.status_code, "-> 取得不可" if r.status_code != 200 else "")
    except Exception as e:  # noqa
        print("Google Trends 取得不可:", e)


# ---------------------------------------------------------------- analysis
def nw_t(y, x, lag):
    """y を定数とダミー x に回帰、x係数のNewey-West t値と係数。"""
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for l in range(1, lag + 1):
        w = 1 - l / (lag + 1)
        G = Xe[l:].T @ Xe[:-l]
        S += w * (G + G.T)
    V = XtXi @ S @ XtXi
    return b[1], b[1] / math.sqrt(V[1, 1])


def zscore(s, win=60):
    m = s.rolling(win, min_periods=40).mean()
    sd = s.rolling(win, min_periods=40).std()
    return (s - m) / sd


def tail_flags(z, win=500, minp=250, q=0.10):
    """過去(当日含む)win日の分布での上位/下位q。未来は使わない。"""
    hi = z.rolling(win, min_periods=minp).quantile(1 - q)
    lo = z.rolling(win, min_periods=minp).quantile(q)
    return z >= hi, z <= lo


def build_features():
    g = pd.read_csv(os.path.join(DATA, "alt_gdelt_daily.csv"), index_col=0, parse_dates=True)
    full = pd.date_range(g.index.min(), g.index.max())
    g = g.reindex(full)
    feats = {}
    for geo in ("JA", "US", "W"):
        feats[f"gdelt_{geo}_tone"] = zscore(g[f"{geo}_tone"].ffill(limit=3))
        feats[f"gdelt_{geo}_vol"] = zscore(np.log(g[f"{geo}_ment"].where(g[f"{geo}_ment"] > 0)).ffill(limit=3))
    w = pd.read_csv(os.path.join(DATA, "alt_wiki_daily.csv"), index_col=0, parse_dates=True)
    for c in w.columns:
        feats[f"wiki_{c}"] = zscore(np.log1p(w[c]))
    return feats


def asset_prices(sym):
    s = fetch_price(sym)
    s = s[(s.index >= pd.Timestamp(START)) & (s.index <= pd.Timestamp(END) + pd.Timedelta(days=15))]
    if sym == "USDJPY=X":
        s = s.shift(-1).dropna()  # 日付tの値を'tの終わり'に揃える(1日ラグ)
    jumps = (s.pct_change().abs() > 0.15).sum()
    if jumps:
        print(f"  [注意] {sym} に日次±15%超が {jumps} 回(データ異常の疑い)")
    return s


def run_tests():
    feats = build_features()
    results = []
    for sym in ASSETS:
        px = asset_prices(sym)
        lp = np.log(px)
        dates = px.index
        for fname, z in feats.items():
            hi, lo = tail_flags(z)
            hi = hi.reindex(dates, method="ffill").fillna(False).values
            lo = lo.reindex(dates, method="ffill").fillna(False).values
            valid = z.reindex(dates, method="ffill").notna().values
            for h in HORIZONS:
                # シグナル日t → 建て t+1 の終値、決済 t+1+h の終値
                fwd = (lp.shift(-(1 + h)) - lp.shift(-1)).values
                ok = valid & ~np.isnan(fwd) & (dates >= pd.Timestamp(START))
                # 本物のシグナル日のみ(日付が後半に偏らないよう全期間)
                idx = np.where(ok)[0]
                y = fwd[idx]
                mid = idx[len(idx) // 2]
                for tail, flag in (("上位10%", hi), ("下位10%", lo)):
                    x = flag[idx].astype(float)
                    if x.sum() < 20:
                        continue
                    mon = pd.Series(dates[idx][x > 0].month).value_counts(normalize=True)
                    rec = {"asset": sym, "feat": fname, "h": h, "tail": tail, "n_sig": int(x.sum()),
                           "top_months": ",".join(f"{m}月{v:.0%}" for m, v in mon.head(3).items())}
                    for name, sl in (("all", slice(None)), ("h1", idx < mid), ("h2", idx >= mid)):
                        yy, xx = (y[sl], x[sl]) if not isinstance(sl, slice) else (y, x)
                        if xx.sum() < 8 or (1 - xx).sum() < 8:
                            rec[f"b_{name}"], rec[f"t_{name}"] = np.nan, np.nan
                            continue
                        b, t = nw_t(yy, xx, h)
                        rec[f"b_{name}"], rec[f"t_{name}"] = b, t
                        rec[f"n_{name}"] = int(xx.sum())
                    results.append(rec)
    return pd.DataFrame(results)


def analyze():
    df = run_tests()
    N = len(df)
    zthr = NormalDist().inv_cdf(1 - 0.05 / 2 / N)
    print(f"\n試した全テスト数 N = {N}(資産3 × 特徴量{df.feat.nunique()} × 期間4 × 上下2。シグナル20日未満は除外)")
    print(f"Bonferroni補正後の |t| 閾値(両側5%) = {zthr:.2f}。補正なし |t|>2 の偶然期待数 ≈ {N*0.0455:.0f}件")
    print(f"補正なしで |t_all|>2: {(df.t_all.abs()>2).sum()}件 / 補正後を通る: {(df.t_all.abs()>=zthr).sum()}件")
    print("\n|t_all| 上位10:")
    cols = ["asset", "feat", "h", "tail", "n_sig", "b_all", "t_all", "t_h1", "t_h2"]
    top = df.reindex(df.t_all.abs().sort_values(ascending=False).index).head(10)
    t = top[cols].copy()
    t["b_all"] = (t.b_all * 1e4).round(1)
    print(t.round(2).to_string(index=False), "\n(b_all は超過リターン bp)")

    # 前半で決めて後半で確かめる: 前半 |t|>=2 かつ 向きが同じ後半 |t|>=2
    cand = df[df.t_h1.abs() >= 2]
    print(f"\n前半で |t|>=2 だったもの: {len(cand)}件(偶然期待 ≈ {N*0.0455:.0f})")
    k = len(cand)
    thr2 = NormalDist().inv_cdf(1 - 0.05 / 2 / max(k, 1))
    conf = cand[(np.sign(cand.b_h1) == np.sign(cand.b_h2)) & (cand.t_h2.abs() >= 2)]
    conf_b = cand[(np.sign(cand.b_h1) == np.sign(cand.b_h2)) & (cand.t_h2.abs() >= thr2)]
    print(f"後半で同じ向きかつ |t|>=2: {len(conf)}件 / 候補数{k}での補正閾値 {thr2:.2f} を通る: {len(conf_b)}件")
    if len(conf):
        c = conf.copy()
        c["dir"] = np.sign(c.b_h1)
        # コスト後: 後半、前半の向きでポジションを取ったときの1回あたり平均(bp) − 往復コスト
        c["gross_bp_h2"] = c.b_h2 * c.dir * 1e4
        c["rt_cost_bp"] = c.asset.map(lambda a: COST_ONEWAY[a] * 2 * 1e4)
        c["net_bp_h2"] = c.gross_bp_h2 - c.rt_cost_bp
        c["trades_h2"] = c.n_h2
        print(c[["asset", "feat", "h", "tail", "b_h1", "t_h1", "t_h2", "gross_bp_h2", "rt_cost_bp", "net_bp_h2", "trades_h2", "top_months"]]
              .round(2).to_string(index=False))
    # 全体の符号の偏り: 前半と後半の相関(再現性の目安)
    m = df.dropna(subset=["b_h1", "b_h2"])
    print(f"\n前半bと後半bの相関(全{len(m)}件): {np.corrcoef(m.b_h1, m.b_h2)[0,1]:.3f}  (0なら再現性なし)")
    df.to_csv(os.path.join(DATA, "alt_results.csv"), index=False)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("fetch", "all"):
        for s in ASSETS:
            fetch_price(s)
        fetch_trends()
        fetch_wiki()
        fetch_gdelt()
    if cmd in ("analyze", "all"):
        analyze()
