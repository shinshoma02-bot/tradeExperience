"""資産の組み合わせ(分散・リバランス効果)の検証。円ベースの投資家として。

  python research/asset_mix.py            # キャッシュ(research/data/asset_mix_raw.json)を使う
  python research/asset_mix.py --refresh  # Yahoo から取り直す

問い: 「SPY持ちっぱなし」より、(a)同じ収益で落ち込みが小さい か (b)落ち込みが同じで収益が高い 組み合わせがあるか。
- ドル建て資産は USDJPY を掛けて円建てリターンに。ETF/投信は配当込みの調整後終値。金の長期は先物 GC=F(価格のみ)。
- 比較の公平さのため、ポートフォリオごとに「SPYと現金(円・利回り0)を混ぜて同じボラ/同じ最大下落にした」収益と比べる。
- 日次±15%超は異常値の疑いとして一覧表示する(2008年10月などの本物の暴落は目視で確認)。
- 試した構成の総数を最後に表示する(多重比較の注意)。前半で選び後半で確かめる検定も入れた。
- 1306/1321/1489 は Yahoo の調整後終値が壊れているので使わない(日本株は 1348.T)。
"""
import json, os, sys, time, urllib.request
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "asset_mix_raw.json")
SYMS = ["SPY", "VT", "1348.T", "TLT", "IEF", "SHY", "GLD", "GC=F", "DBC", "VNQ", "USDJPY=X",
        "VFITX", "VGSIX", "^N225"]
TAX = 0.20315          # 課税口座の譲渡益税(NISA外。要確認)
YEAR = 252


def fetch(refresh=False):
    if os.path.exists(CACHE) and not refresh:
        return json.load(open(CACHE))
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    out = {}
    now = int(time.time())
    for s in SYMS:
        # range=max だと長い履歴は月次に落ちるので period1/period2 で日次を取る
        u = f"https://query1.finance.yahoo.com/v8/finance/chart/{s}?period1=0&period2={now}&interval=1d"
        r = json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=40))["chart"]["result"][0]
        a = r["indicators"].get("adjclose", [{}])[0].get("adjclose") or r["indicators"]["quote"][0]["close"]
        out[s] = list(zip(r["timestamp"], a))
    json.dump(out, open(CACHE, "w"))
    return out


def despike(x, thr=0.035):
    """1日だけ飛んで翌日戻る為替のデータ異常(Yahooの2008-10/12など)を除く。往復で打ち消し合う±3.5%超の点を削除。"""
    for _ in range(3):
        r = x.pct_change()
        bad = (r.abs() > thr) & (((1 + r) * (1 + r.shift(-1)) - 1).abs() < 0.03) & (np.sign(r) != np.sign(r.shift(-1)))
        if not bad.any(): break
        print("  [除外した為替の異常点] " + ", ".join(f"{d.date()}({x[d]:.1f})" for d in x.index[bad]))
        x = x[~bad]
    return x


def build(raw):
    """円建て価格 DataFrame(SPYの営業日に揃え、休場日は前日値)。"""
    ser = {}
    for s, rows in raw.items():
        # 為替は前日23:00(UTC)の時刻印が付くことがあるので、+3時間してから日付に丸める
        x = pd.Series([v for _, v in rows], index=(pd.to_datetime([t for t, _ in rows], unit="s") + pd.Timedelta(hours=3)).normalize(), dtype=float)
        x = x[~x.index.duplicated(keep="last")].dropna().sort_index()
        if s == "USDJPY=X":
            x = despike(x)
        ser[s] = x
    cal = ser["SPY"].index
    fx = ser["USDJPY=X"].reindex(cal.union(ser["USDJPY=X"].index)).ffill().reindex(cal)
    J = {}
    for s, x in ser.items():
        if s == "USDJPY=X":
            continue
        xx = x.reindex(x.index.union(cal)).ffill().reindex(cal)
        xx = xx[x.index[0]:]
        J[s] = xx if s in ("1348.T", "^N225") else xx * fx.reindex(xx.index)
    P = pd.DataFrame(J)
    P["CASH"] = 1.0
    return P, ser


def anomalies(P):
    R = P.pct_change()
    print("【データ異常チェック: 円建て日次±15%超】")
    for c in P.columns:
        big = R[c][R[c].abs() > 0.15]
        if len(big):
            print(f"  {c}: " + ", ".join(f"{d.date()} {v*100:+.1f}%" for d, v in big.items()))
    print("  (上記の2008-10 や 2020-03 などは実際の暴落。それ以外が出たら要調査)\n")


# ---------------- シミュレーション ----------------
def month_end_flags(idx):
    s = pd.Series(np.arange(len(idx)), index=idx)
    m = np.zeros(len(idx), bool); m[s.groupby([idx.year, idx.month]).max().values] = True
    return m


def period_end_flags(idx, rule):
    if rule == "month": return month_end_flags(idx)
    s = pd.Series(np.arange(len(idx)), index=idx)
    key = [idx.year, (idx.month - 1) // 3] if rule == "quarter" else [idx.year]
    m = np.zeros(len(idx), bool); m[s.groupby(key).max().values] = True
    return m


def simulate(R, assets, wts, rule="year", cost=0.001, tax=False, thresh=0.05, vol_n=60):
    """日次リターン表 R から、目標配分 wts(または 'invvol')を rule(none/year/quarter/month/thresh)で保つ。
    cost=売買額に対する片道コスト。tax=True なら売却益に課税し、最後に全額売却した税も引く。
    戻り値: (日次リターン Series, 年あたり回転率, 最終資産(税引後), 最終資産(税前))"""
    X = R[assets].fillna(0.0).values
    n, k = X.shape
    inv = wts == "invvol"
    tw = np.full(k, 1 / k) if inv else np.asarray(wts, float)
    me = month_end_flags(R.index)
    flag = {"none": np.zeros(n, bool), "thresh": me, "month": me}.get(rule)
    if flag is None: flag = period_end_flags(R.index, rule)
    if inv: flag = me
    vol = pd.DataFrame(X).rolling(vol_n).std().shift(1).values   # 前日までのボラ(先読みなし)

    def target(i):
        if not inv: return tw
        v = vol[i]
        if np.isnan(v).any() or (v <= 0).any(): return np.full(k, 1 / k)
        w = 1 / v; return w / w.sum()

    v = target(0).copy() * (1 - cost)          # 初回購入コスト
    basis = target(0).copy()
    tot0 = 1.0
    out = np.empty(n); turn = 0.0
    prev = v.sum()
    out[0] = prev - 1.0
    for i in range(1, n):
        v = v * (1 + X[i])
        t = v.sum()
        if flag[i] and rule != "none":
            w = v / t; tgt = target(i)
            if rule != "thresh" or np.abs(w - tgt).max() > thresh:
                new = tgt * t
                tr = np.abs(new - v).sum()
                c = tr * cost
                tx = 0.0
                if tax:
                    sell = np.maximum(v - new, 0.0)
                    frac = np.where(v > 0, 1 - basis / np.where(v > 0, v, 1), 0.0)
                    tx = TAX * max((sell * frac).sum(), 0.0)
                    basis = basis * np.where(v > 0, 1 - sell / np.where(v > 0, v, 1), 1.0) + np.maximum(new - v, 0.0)
                scale = (t - c - tx) / t
                v = new * scale; basis = basis * 1.0
                turn += tr
                t = v.sum()
        out[i] = t / prev - 1
        prev = t
    final_pre = prev
    final_after = prev - TAX * max((v - basis).sum(), 0.0) if tax else prev
    return pd.Series(out, index=R.index), turn / (n / YEAR), final_after, final_pre


# ---------------- 指標 ----------------
def metrics(r):
    r = r.dropna(); eq = (1 + r).cumprod(); yrs = len(r) / YEAR
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(YEAR)
    dd = eq / eq.cummax() - 1
    mdd = dd.min()
    # 最大下落の回復期間: 最大下落の山から、山を回復するまで(営業日→月)
    trough = dd.idxmin(); peak = eq[:trough].idxmax()
    rec = eq[trough:][eq[trough:] >= eq[peak]]
    rec_m = (rec.index[0] - peak).days / 30.4 if len(rec) else None
    under = (eq < eq.cummax()).values           # 最長の水中期間
    best = cur = 0
    for u in under:
        cur = cur + 1 if u else 0; best = max(best, cur)
    w12 = (eq / eq.shift(YEAR) - 1).min()
    return dict(cagr=cagr * 100, vol=vol * 100, sharpe=(r.mean() * YEAR) / (r.std() * np.sqrt(YEAR)) if r.std() else 0,
                dd=mdd * 100, calmar=cagr / abs(mdd) if mdd else np.nan, w12=w12 * 100,
                rec=rec_m, uw=best / YEAR, peak=peak, trough=trough)


def matched_spy(spy, p):
    """SPY(円)と現金0%を混ぜて、pと同じボラ / 同じ最大下落にしたときのCAGR(%)。"""
    kv = p.std() / spy.std()
    cv = metrics(spy * kv)["cagr"] if kv <= 1.0 else np.nan
    target = metrics(p)["dd"]
    lo, hi = 0.0, 1.0
    if metrics(spy)["dd"] < target:      # SPY自身の下落の方が深い(普通)
        for _ in range(30):
            mid = (lo + hi) / 2
            if metrics(spy * mid)["dd"] > target: lo = mid
            else: hi = mid
        cd = metrics(spy * lo)["cagr"]
    else:
        cd = np.nan                       # SPYより深い下落なら比較対象外
    return cv, cd


def edge(spy, p):
    """同ボラのSPY+現金に対する年率の上乗せ(pt)。SPYよりボラが大きい構成はnan。"""
    cv, _ = matched_spy(spy, p)
    return metrics(p)["cagr"] - cv


def alpha_t(spy, p):
    """月次リターンの回帰 p = a + b*SPY。年率アルファ(%)とt値。"""
    m = pd.concat([spy, p], axis=1).dropna()
    m = (1 + m).groupby([m.index.year, m.index.month]).prod() - 1
    x, y = m.iloc[:, 0].values, m.iloc[:, 1].values
    b = np.cov(x, y)[0, 1] / x.var(ddof=1); a = y.mean() - b * x.mean()
    res = y - a - b * x; se = res.std(ddof=2) / np.sqrt(len(x)) if len(x) > 2 else np.nan
    return a * 12 * 100, (a / se if se else np.nan), b


# ---------------- 構成 ----------------
def S(name, assets, wts, rule="year"):
    return dict(name=name, assets=assets, wts=wts, rule=rule)


E4 = ["SPY", "IEF", "GLD", "VNQ"]
SET_E = [  # 2004-12〜 (GLD上場から)。ETFだけ
    S("SPY100", ["SPY"], [1]),
    S("株80/債20(IEF)", ["SPY", "IEF"], [.8, .2]),
    S("株70/債30(IEF)", ["SPY", "IEF"], [.7, .3]),
    S("株60/債40(IEF)", ["SPY", "IEF"], [.6, .4]),
    S("株60/債40(TLT)", ["SPY", "TLT"], [.6, .4]),
    S("株70/金15/債15(IEF)", ["SPY", "GLD", "IEF"], [.7, .15, .15]),
    S("株70/金15/債15(TLT)", ["SPY", "GLD", "TLT"], [.7, .15, .15]),
    S("株90/金10", ["SPY", "GLD"], [.9, .1]),
    S("株80/金20", ["SPY", "GLD"], [.8, .2]),
    S("株50/金50", ["SPY", "GLD"], [.5, .5]),
    S("4資産等分(株債金REIT)", E4, [.25] * 4),
    S("逆ボラ(株債金REIT)", E4, "invvol", "month"),
    S("逆ボラ(株債金)", ["SPY", "IEF", "GLD"], "invvol", "month"),
    S("3資産等分(株債金)", ["SPY", "IEF", "GLD"], [1 / 3] * 3),
]
VARIANTS = [  # リバランス頻度の比較(基準は年1回)
    ("株60/債40(IEF)", ["SPY", "IEF"], [.6, .4]),
    ("株70/金15/債15(IEF)", ["SPY", "GLD", "IEF"], [.7, .15, .15]),
    ("4資産等分(株債金REIT)", E4, [.25] * 4),
]
RULES = {"none": "放置", "year": "年1回", "quarter": "四半期", "month": "月次", "thresh": "しきい値5pp"}
SET_D = [  # 2006-03〜 (DBC上場から)
    S("SPY100", ["SPY"], [1]),
    S("株90/DBC10", ["SPY", "DBC"], [.9, .1]),
    S("5資産等分(株債金REIT+DBC)", E4 + ["DBC"], [.2] * 5),
    S("株70/金15/債15(IEF)", ["SPY", "GLD", "IEF"], [.7, .15, .15]),
]
SET_J = [  # 2009-06〜 (1348.T上場から)
    S("日本株100(1348)", ["1348.T"], [1]),
    S("米国株100(SPY)", ["SPY"], [1]),
    S("日米50/50", ["1348.T", "SPY"], [.5, .5]),
    S("世界株VT100", ["VT"], [1]),
    S("日本70/米国30", ["1348.T", "SPY"], [.7, .3]),
    S("日本株60/債40(IEF)", ["1348.T", "IEF"], [.6, .4]),
]
L4 = ["SPY", "VFITX", "GC=F", "VGSIX"]
SET_L = [  # 2000-09〜 (GC=F先物, 債は投信VFITX, REITは投信VGSIX)。長期用
    S("SPY100", ["SPY"], [1]),
    S("株80/債20(VFITX)", ["SPY", "VFITX"], [.8, .2]),
    S("株60/債40(VFITX)", ["SPY", "VFITX"], [.6, .4]),
    S("株70/金15/債15", ["SPY", "GC=F", "VFITX"], [.7, .15, .15]),
    S("株80/金20(GC=F)", ["SPY", "GC=F"], [.8, .2]),
    S("株50/金50(GC=F)", ["SPY", "GC=F"], [.5, .5]),
    S("4資産等分(株債金REIT)", L4, [.25] * 4),
    S("逆ボラ(株債金REIT)", L4, "invvol", "month"),
    S("3資産等分(株債金)", ["SPY", "VFITX", "GC=F"], [1 / 3] * 3),
    S("日経225(価格のみ)100", ["^N225"], [1]),
    S("日米50/50(日経+SPY)", ["^N225", "SPY"], [.5, .5]),
]

N_TRIED = 0


def run_set(R, specs, start, end=None, **kw):
    global N_TRIED
    Rs = R[start:end] if end else R[start:]
    out = {}
    for s in specs:
        r, t, fa, fp = simulate(Rs, s["assets"], s["wts"], s["rule"], **kw)
        out[s["name"]] = (r, t, fa, fp)
    return out


def fmt(m):
    rec = "未回復" if m["rec"] is None else f"{m['rec']:.0f}か月"
    return (f"{m['cagr']:5.1f} {m['vol']:5.1f} {m['sharpe']:5.2f} {m['dd']:6.1f} {m['calmar']:5.2f} {m['w12']:6.1f} "
            f"{rec:>7} {m['uw']:4.1f}年")


def table(title, res, bench="SPY100", bench_ret=None):
    print(f"■ {title}")
    print(f"{'構成':26}{'年率%':>6}{'ボラ%':>6}{'ｼｬｰﾌﾟ':>6}{'最大下落':>7}{'ｶﾙﾏｰ':>6}{'最悪12m':>7}{'回復':>8}{'最長水中':>8} | 同ボラSPY混合 同DD SPY混合 ｱﾙﾌｧ(t) 回転/年")
    spy = bench_ret if bench_ret is not None else res[bench][0]
    for name, (r, t, fa, fp) in res.items():
        m = metrics(r)
        if name == bench:
            print(f"{name:24}" + fmt(m) + "   (基準)"); continue
        cv, cd = matched_spy(spy.loc[r.index], r)
        a, tt, b = alpha_t(spy.loc[r.index], r)
        f = lambda x: "  n/a" if np.isnan(x) else f"{x:5.1f}"
        print(f"{name:24}" + fmt(m) + f" |   {f(cv)}        {f(cd)}      {a:+5.1f}({tt:+.1f})  {t*100:4.0f}%")
    print()


def sub(res, a, b=None):
    return {k: (v[0][a:b] if b else v[0][a:], v[1], v[2], v[3]) for k, v in res.items()}


def main():
    global N_TRIED
    raw = fetch("--refresh" in sys.argv)
    P, ser = build(raw)
    anomalies(P)
    R = P.pct_change()
    cfgs = 0

    print("【単体資産の円建て成績 2006-03〜(全資産が揃う期間)】")
    print(f"{'資産':8}{'年率%':>7}{'ボラ%':>7}{'最大下落':>8}{'前半年率(〜2014)':>16}{'後半年率(2015〜)':>16}")
    for c in ["SPY", "VT", "1348.T", "IEF", "TLT", "GLD", "VNQ", "DBC"]:
        r = R[c]["2006-03-10":].dropna(); m = metrics(r)
        h1 = metrics(r[:"2014-12-31"])["cagr"]; h2 = metrics(r["2015-01-01":])["cagr"]
        print(f"{c:8}{m['cagr']:7.1f}{m['vol']:7.1f}{m['dd']:8.1f}{h1:16.1f}{h2:16.1f}")
    print("  (金GLDの年率が高い=2005〜2026の金高騰が、金を含む構成の好成績の主因。再現する保証はない)\n")

    # ============ A. ETFのみ 2004-12〜 ============
    startE = "2004-12-02"
    resE = run_set(R, SET_E, startE)
    cfgs += len(SET_E) - 1
    spyE = resE["SPY100"][0]
    print("=" * 30, "A. 円建て・ETF 2004-12〜2026-10 (コスト0.1%/売買, 税なし)", "=" * 30)
    table("全期間", resE)
    splitE = "2015-01-01"
    table("前半 2004-12〜2014-12 (リーマン含む)", sub(resE, startE, "2014-12-31"))
    table("後半 2015-01〜2026-10 (2022の株債同時安含む)", sub(resE, splitE))

    # 2022: 株と債が同時に下がった年
    print("■ 2022年(暦年)と2022ピーク→谷の挙動(円建て)")
    for name, (r, *_ ) in resE.items():
        y = r["2022-01-01":"2022-12-31"]; e = (1 + y).cumprod()
        print(f"  {name:26} 2022年 {((1+y).prod()-1)*100:+6.1f}%  年内最大下落 {((e/e.cummax()-1).min())*100:6.1f}%")
    print("  (注: 2022は円安が進んだので、円建てではドル資産の下落が緩和されている)\n")

    # ドルベース(為替なし)の2022も参考に
    Rusd = R.copy()
    fx = ser["USDJPY=X"].reindex(P.index).ffill().pct_change()
    for c in ["SPY", "IEF", "TLT", "GLD", "VNQ", "DBC", "GC=F", "VT", "VFITX", "VGSIX"]:
        Rusd[c] = (1 + R[c]) / (1 + fx) - 1
    print("■ 2022年 ドル建て(為替の影響を除く)")
    for name in ["SPY100", "株60/債40(IEF)", "株70/金15/債15(IEF)", "4資産等分(株債金REIT)"]:
        s = [x for x in SET_E if x["name"] == name][0]
        r, *_ = simulate(Rusd["2004-12-02":], s["assets"], s["wts"], s["rule"])
        y = r["2022-01-01":"2022-12-31"]; e = (1 + y).cumprod()
        print(f"  {name:26} 2022年 {((1+y).prod()-1)*100:+6.1f}%  年内最大下落 {((e/e.cummax()-1).min())*100:6.1f}%")
    print()

    # リバランス頻度
    print("■ リバランス頻度の比較(全期間, コスト0.1%)")
    print(f"{'構成':24}{'ルール':10}{'年率%':>6}{'最大下落':>8}{'ｶﾙﾏｰ':>7}{'ｼｬｰﾌﾟ':>7}{'回転/年':>8}")
    for nm, a, w in VARIANTS:
        for rule, lab in RULES.items():
            r, t, *_ = simulate(R[startE:], a, w, rule)
            m = metrics(r); cfgs += 1
            print(f"{nm:24}{lab:10}{m['cagr']:6.2f}{m['dd']:8.1f}{m['calmar']:7.2f}{m['sharpe']:7.2f}{t*100:7.0f}%")
    print()

    # ============ B. コスト・税 ============
    print("=" * 30, "B. 手数料・税を引いた最終資産(年率換算、全期間A)", "=" * 30)
    print("  税=課税口座で20.315%、最後に全部売った税も全構成で引く(SPY持ちっぱなしも同条件)。NISAなら税0。")
    print(f"{'構成':26}" + "".join(f"{lab:>16}" for lab in ["ｺｽﾄ0/税0", "ｺｽﾄ0.1%/税0", "ｺｽﾄ0.5%/税0", "0.5%+税"]))
    yrs = len(R[startE:]) / YEAR
    pick = ["SPY100", "株60/債40(IEF)", "株70/金15/債15(IEF)", "4資産等分(株債金REIT)", "株90/金10", "逆ボラ(株債金REIT)"]
    cost_tab = {}
    for nm in pick:
        s = [x for x in SET_E if x["name"] == nm][0]; row = []
        for cost, tax in [(0, False), (0.001, False), (0.005, False), (0.005, True)]:
            _, _, fa, fp = simulate(R[startE:], s["assets"], s["wts"], s["rule"], cost=cost, tax=tax)
            row.append((fa ** (1 / yrs) - 1) * 100)
        cost_tab[nm] = row
        print(f"{nm:26}" + "".join(f"{x:15.2f}%" for x in row))
    print("  差(構成 - SPY100):")
    for nm in pick[1:]:
        print(f"{nm:26}" + "".join(f"{a-b:+15.2f}%" for a, b in zip(cost_tab[nm], cost_tab['SPY100'])))
    print()

    # ============ C. 開始時期をずらす ============
    print("=" * 30, "C. 開始時期ずらし(2026-10まで)。同ボラSPY混合に対する年率の上乗せ(pt)", "=" * 30)
    cand = [s for s in SET_E if s["name"] != "SPY100"]
    starts = [f"{y}-01-02" for y in range(2005, 2017)]
    sp = {}
    for st in starts:
        sp[st] = simulate(R[st:], ["SPY"], [1])[0]
    print(f"{'構成':26}" + "".join(f"{st[:4]:>6}" for st in starts) + "  ｶﾙﾏｰ勝ち  CAGR勝ち  DD浅い")  # noqa
    for s in cand:
        cal, cg, dd, ed = [], [], [], []
        line = ""
        for st in starts:
            r = simulate(R[st:], s["assets"], s["wts"], s["rule"])[0]
            m, b = metrics(r), metrics(sp[st])
            cal.append(m["calmar"] > b["calmar"]); cg.append(m["cagr"] > b["cagr"]); dd.append(m["dd"] > b["dd"])
            e = edge(sp[st], r); ed.append(e > 0)
            line += f"{e:+6.1f}"
        print(f"{s['name']:24}" + line + f"  {sum(cal):2d}/{len(cal)}    {sum(cg):2d}/{len(cg)}    {sum(dd):2d}/{len(dd)}   上乗せ>0: {sum(ed)}/{len(ed)}")
    print("  (数値は 同ボラSPY混合との年率差pt。マイナスなら『SPYを少し減らして現金を持つ』方が良かった。右の勝敗はSPY100と直接比較)\n")

    # ローリング10年(月次開始)
    print("■ ローリング10年窓(全期間シミュレーションを切り出し)で SPY100 に勝った割合")
    print(f"{'構成':26}{'ｶﾙﾏｰ':>8}{'年率':>8}{'最大下落':>9}{'ｼｬｰﾌﾟ':>8}")
    starts_m = pd.date_range(startE, "2016-10-01", freq="MS")
    for s in cand:
        rp = resE[s["name"]][0]; w = {"c": 0, "g": 0, "d": 0, "s": 0}; n = 0
        for st in starts_m:
            en = st + pd.DateOffset(years=10)
            a, b = metrics(rp[st:en]), metrics(spyE[st:en]); n += 1
            w["c"] += a["calmar"] > b["calmar"]; w["g"] += a["cagr"] > b["cagr"]
            w["d"] += a["dd"] > b["dd"]; w["s"] += a["sharpe"] > b["sharpe"]
        print(f"{s['name']:24}{w['c']/n*100:7.0f}%{w['g']/n*100:7.0f}%{w['d']/n*100:8.0f}%{w['s']/n*100:7.0f}%")
    print()

    # ============ D. 前半で選び後半で確かめる ============
    print("=" * 30, "D. 前半(2004-12〜2014-12)で選び、後半(2015〜)で確かめる", "=" * 30)
    f_, s_ = [], []
    for s in cand:
        r = resE[s["name"]][0]
        a = metrics(r[startE:"2014-12-31"]); b = metrics(r[splitE:])
        a0 = metrics(spyE[startE:"2014-12-31"]); b0 = metrics(spyE[splitE:])
        ea = edge(spyE[startE:"2014-12-31"], r[startE:"2014-12-31"]); eb = edge(spyE[splitE:], r[splitE:])
        f_.append((s["name"], a["calmar"] - a0["calmar"], a["sharpe"] - a0["sharpe"], a["cagr"] - a0["cagr"], ea,
                   b["calmar"] - b0["calmar"], b["sharpe"] - b0["sharpe"], b["cagr"] - b0["cagr"], eb))
    df = pd.DataFrame(f_, columns=["name", "前ｶﾙﾏｰ差", "前ｼｬｰﾌﾟ差", "前年率差", "前上乗せ", "後ｶﾙﾏｰ差", "後ｼｬｰﾌﾟ差", "後年率差", "後上乗せ"]).set_index("name")
    print(df.round(2).to_string())
    print(f"  前半ｶﾙﾏｰ差で1位 = {df['前ｶﾙﾏｰ差'].idxmax()} -> 後半ｶﾙﾏｰ差 {df.loc[df['前ｶﾙﾏｰ差'].idxmax(), '後ｶﾙﾏｰ差']:+.2f}, 後半年率差 {df.loc[df['前ｶﾙﾏｰ差'].idxmax(), '後年率差']:+.2f}pt")
    print(f"  前半でSPYよりカルマーが良かった構成: {(df['前ｶﾙﾏｰ差']>0).sum()}/{len(df)}、うち後半も良かった: {((df['前ｶﾙﾏｰ差']>0)&(df['後ｶﾙﾏｰ差']>0)).sum()}")
    print(f"  順位相関(前半カルマー差 vs 後半カルマー差, Spearman) = {df['前ｶﾙﾏｰ差'].rank().corr(df['後ｶﾙﾏｰ差'].rank()):+.2f}")
    print(f"  順位相関(前半シャープ差 vs 後半シャープ差) = {df['前ｼｬｰﾌﾟ差'].rank().corr(df['後ｼｬｰﾌﾟ差'].rank()):+.2f}")
    print(f"  順位相関(前半上乗せ vs 後半上乗せ) = {df['前上乗せ'].rank().corr(df['後上乗せ'].rank()):+.2f}")
    top = df['前上乗せ'].idxmax()
    print(f"  前半の上乗せで1位 = {top} -> 前 {df.loc[top,'前上乗せ']:+.1f}pt, 後 {df.loc[top,'後上乗せ']:+.1f}pt")
    print(f"  前半で上乗せ>0: {(df['前上乗せ']>0).sum()}/{len(df)}、うち後半も>0: {((df['前上乗せ']>0)&(df['後上乗せ']>0)).sum()}、後半の上乗せの平均 {df['後上乗せ'].mean():+.2f}pt (前半 {df['前上乗せ'].mean():+.2f}pt)")
    nog = df[~df.index.str.contains("金")]
    print(f"  金を含まない構成だけ({len(nog)}個): 前半の上乗せ平均 {nog['前上乗せ'].mean():+.2f}pt、後半 {nog['後上乗せ'].mean():+.2f}pt\n")

    # ============ E. DBC を含む(2006-03〜) ============
    print("=" * 30, "E. コモディティDBCを含む 2006-03〜", "=" * 30)
    resD = run_set(R, SET_D, "2006-03-10"); cfgs += len(SET_D) - 1
    table("全期間", resD)

    # ============ F. 日本株 vs 米国株 ============
    print("=" * 30, "F. 日本株/米国株/世界株 (円建て) 2009-06〜", "=" * 30)
    resJ = run_set(R, SET_J, "2009-06-01"); cfgs += len(SET_J)
    table("全期間(基準=米国株SPY)", resJ, bench="米国株100(SPY)")
    table("前半 〜2017-12", sub(resJ, "2009-06-01", "2017-12-31"), bench="米国株100(SPY)")
    table("後半 2018〜", sub(resJ, "2018-01-01"), bench="米国株100(SPY)")

    # ============ G. 長期 2000-09〜 ============
    print("=" * 30, "G. 長期 2000-09〜 (金=GC=F先物, 債=投信VFITX, REIT=投信VGSIX)", "=" * 30)
    startL = "2000-09-05"
    resL = run_set(R, SET_L, startL); cfgs += len(SET_L) - 1
    spyL = resL["SPY100"][0]
    table("全期間 2000-09〜", resL)
    table("前半 2000-09〜2013-12 (ITバブル崩壊・リーマン含む)", sub(resL, startL, "2013-12-31"))
    table("後半 2014-01〜", sub(resL, "2014-01-01"))
    # GC=F と GLD の整合
    j = pd.concat([R["GC=F"], R["GLD"]], axis=1).dropna()
    print(f"  参考: GC=F と GLD の日次リターン相関 = {j.corr().iloc[0,1]:.3f}\n")

    # 2000年開始の開始時期ずらし(2000〜2008開始)
    print("■ 長期セット: 開始年ずらし(カルマー差。マイナス=SPY持ちっぱなしが上)")
    stL = [f"{y}-01-02" for y in range(2001, 2013)]
    print(f"{'構成':26}" + "".join(f"{s[:4]:>6}" for s in stL) + "  勝ち")
    spL = {st: simulate(R[st:], ["SPY"], [1])[0] for st in stL}
    for s in SET_L[1:9]:
        line = ""; w = 0
        for st in stL:
            m = metrics(simulate(R[st:], s["assets"], s["wts"], s["rule"])[0]); b = metrics(spL[st])
            line += f"{m['calmar']-b['calmar']:+6.2f}"; w += m["calmar"] > b["calmar"]
        print(f"{s['name']:24}{line}  {w}/{len(stL)}")
    print()

    # ============ 総括 ============
    n_specs = len(SET_E) - 1 + len(SET_D) - 1 + len(SET_J) + len(SET_L) - 1 + 3 * len(RULES)
    print("=" * 30, "試行数と多重比較", "=" * 30)
    print(f"  別々に検証した構成の数(基準を除く) = 約{n_specs} (A:{len(SET_E)-1}, リバランス変種:{3*len(RULES)}, E:{len(SET_D)-1}, F:{len(SET_J)}, G:{len(SET_L)-1})")
    print(f"  それぞれを 全期間/前半/後半 や 開始年ずらしで何度も見ているので、実質的な比較回数はさらに多い。")
    print(f"  5%水準でも約{n_specs}回なら偶然に1〜{max(2, int(n_specs*0.05)+1)}個は 'ｱﾙﾌｧ|t|>2' が出る。ボンフェローニ補正ならt>{ {20:2.9,30:3.0,50:3.1,80:3.3}[min([20,30,50,80],key=lambda z:abs(z-n_specs))]:.1f}程度が必要。")


if __name__ == "__main__":
    main()
