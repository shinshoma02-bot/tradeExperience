"""内部者買い(米国, SEC Insider Transactions Data Sets) の検証。
使い方: python research/insider_buyback.py [開始年 終了年]
SECがアクセスを拒否(403)した場合は、その旨を表示して終了する(推測で埋めない)。
注意: 2026-10-29 の実行環境では SEC が 403 を返し、分析部分は実データで未検証。
UAは一般的な識別文字列のみ。403を回避するためのUA差し替え等はしない。"""
import io, os, sys, json, time, zipfile, urllib.request, urllib.error
import numpy as np, pandas as pd

D = os.path.join(os.path.dirname(__file__), "data", "sec"); os.makedirs(D, exist_ok=True)
UA = "research-script"
SEC = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{y}q{q}_form345.zip"

def get(url, ua=UA):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": ua}), timeout=60).read()

def fetch_quarter(y, q):
    p = os.path.join(D, f"{y}q{q}.zip")
    if not os.path.exists(p):
        open(p, "wb").write(get(SEC.format(y=y, q=q))); time.sleep(0.3)
    return p

def load_buys(path):
    z = zipfile.ZipFile(path)
    sub = pd.read_csv(z.open("SUBMISSION.tsv"), sep="\t", dtype=str)
    tr = pd.read_csv(z.open("NONDERIV_TRANS.tsv"), sep="\t", dtype=str)
    ow = pd.read_csv(z.open("REPORTINGOWNER.tsv"), sep="\t", dtype=str)
    tr = tr[tr.TRANS_CODE == "P"].copy()
    tr["val"] = pd.to_numeric(tr.TRANS_SHARES) * pd.to_numeric(tr.TRANS_PRICEPERSHARE)
    s = sub[sub.DOCUMENT_TYPE.isin(["4", "4/A"])][["ACCESSION_NUMBER", "FILING_DATE", "ISSUERTRADINGSYMBOL"]]
    o = ow.groupby("ACCESSION_NUMBER").agg(owner=("RPTOWNERCIK", "first"), title=("RPTOWNER_TITLE", "first")).reset_index()
    return tr.merge(s, on="ACCESSION_NUMBER").merge(o, on="ACCESSION_NUMBER")

def yahoo(sym):
    p = os.path.join(D, f"px_{sym}.json")
    if not os.path.exists(p):
        time.sleep(0.08)
        u = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval=1d&period1=0&period2={int(time.time())}"
        try: open(p, "wb").write(get(u, "Mozilla/5.0"))
        except Exception: open(p, "w").write("{}")
    try:
        r = json.load(open(p))["chart"]["result"][0]
        return pd.Series(r["indicators"]["quote"][0]["open"], index=pd.to_datetime(r["timestamp"], unit="s").normalize()).dropna()
    except Exception: return None

def hold_ret(px, d0, months):
    """提出日d0の翌営業日始値で買い、months後(以降最初の営業日)の始値で売る"""
    i = px.index.searchsorted(d0 + pd.Timedelta(days=1))
    j = px.index.searchsorted(px.index[min(i, len(px)-1)] + pd.DateOffset(months=months)) if i < len(px) else len(px)
    return px.iloc[j] / px.iloc[i] - 1 if j < len(px) else np.nan

def main():
    import glob
    local = sorted(glob.glob(os.path.join(D, "[0-9][0-9][0-9][0-9]q[1-4].zip")))   # 手元にある四半期だけ使う(欠けた四半期は飛ばす)
    if not local:
        y0, y1 = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (2010, 2025)
        try: fetch_quarter(y0, 1)
        except urllib.error.HTTPError as e:
            print(f"SEC取得失敗: HTTP {e.code} (UA='{UA}')。ZIPを research/data/sec/ に YYYYqN.zip の名前で置けば検証できる。"); return
        local = [fetch_quarter(y, q) for y in range(y0, y1 + 1) for q in (1, 2, 3, 4) if not (y == 2026 and q > 3)]
    print("使う四半期:", [os.path.basename(x)[:6] for x in local])
    frames = [load_buys(x) for x in local]
    b = pd.concat(frames); b["fd"] = pd.to_datetime(b.FILING_DATE, format="%d-%b-%Y")
    b = b.dropna(subset=["ISSUERTRADINGSYMBOL"]); b["sym"] = b.ISSUERTRADINGSYMBOL.str.upper()
    # 銘柄ごと: 30日以内に3人以上 → クラスターの最初のトリガー日のみイベント(重複抑制)
    ev = []
    for sym, g in b.groupby("sym"):
        g = g.sort_values("fd"); last = None
        for d in g.fd.unique():
            w = g[(g.fd <= d) & (g.fd > d - pd.Timedelta(days=30))]
            if w.owner.nunique() >= 3 and (last is None or (d - last).days > 90):
                ev.append(dict(sym=sym, fd=pd.Timestamp(d), kind="cluster3")); last = d
        for _, r in g[g.title.fillna("").str.contains(r"CEO|CFO|Chief (Exec|Fin)", case=False)].iterrows():
            ev.append(dict(sym=sym, fd=r.fd, kind="ceo_cfo"))
        for _, r in g[g.val >= 500_000].iterrows():
            ev.append(dict(sym=sym, fd=r.fd, kind="big500k"))
    ev = pd.DataFrame(ev); spy = yahoo("SPY"); out = []
    got = 0; tot = ev.sym.nunique()
    px = {s: yahoo(s) for s in ev.sym.unique()}; got = sum(v is not None for v in px.values())
    print(f"価格が取れた銘柄: {got}/{tot} = {got/tot:.1%} (取れない=上場廃止等 → 生存者バイアス)")
    for kind, g in ev.groupby("kind"):
        for m in (3, 6, 12):
            r = [(e.fd, hold_ret(px[e.sym], e.fd, m) - hold_ret(spy, e.fd, m)) for e in g.itertuples() if px[e.sym] is not None]
            r = pd.DataFrame(r, columns=["fd", "x"]).dropna()
            if len(r) < 20:
                print(f"{kind:9s} {m:2d}M n={len(r)} (少なすぎて評価しない)"); continue
            # 月単位でクラスタ化したt値(同時期イベントの相関を考慮)
            mm = r.groupby(r.fd.dt.to_period("M")).x.mean(); t = mm.mean() / (mm.std() / np.sqrt(len(mm)))
            h = r[r.fd < "2026-01-01"].x.mean(), r[r.fd >= "2026-01-01"].x.mean()   # 標本が短いので2025年/2026年で分ける
            print(f"{kind:9s} {m:2d}M n={len(r):5d} 超過{r.x.mean():+.2%} t={t:.2f} 2025年{h[0]:+.2%} 2026年{h[1]:+.2%} コスト0.2%往復後{r.x.mean()-0.002:+.2%}")
    print("試したパターン数: 3種類のイベント x 3保有期間 = 9 (多重比較に注意)")

if __name__ == "__main__": main()
