"""ボラティリティ・リスクプレミアム & オルタナティブ資産の検証(2026-10)。
  python research/vol_premia.py
データ: Yahoo Finance(日足, period1=0, 配当込みは adjclose) と CBOE の公開CSV(BXM/PUT/BXMD/BXY/WPUT)。
キャッシュは環境変数 VP_CACHE(既定 /tmp/vol_premia_cache)。シャープは無リスク金利0%で統一(相対比較用)。
ルール: 前半/後半は共通標本の中点で分割。シグナルは t の終値で決め、t+1 のリターンを得る(ルックアヘッドなし)。
税: 分配金は毎年20.315%(NISAなら0)、売却益は最後に20.315%。米国源泉10%は無視(要確認)。
試行数は最後に表示(多重比較の注意)。
"""
import io, json, os, sys, time, urllib.request
import numpy as np
import pandas as pd

CACHE = os.environ.get("VP_CACHE", "/tmp/vol_premia_cache")
os.makedirs(CACHE, exist_ok=True)
H = {"User-Agent": "Mozilla/5.0"}
TAX = 0.20315
TRIALS = {}


def count(part, n):
    TRIALS[part] = TRIALS.get(part, 0) + n


def yahoo(sym):
    """(close, adjclose) の DataFrame。極端な日次(±15%超)は警告のみ出して残す。"""
    fn = os.path.join(CACHE, "y_" + sym.replace("^", "_") + ".json")
    if os.path.exists(fn):
        d = json.load(open(fn))
    else:
        u = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval=1d&period1=0&period2={int(time.time())}"
        r = json.load(urllib.request.urlopen(urllib.request.Request(u, headers=H), timeout=40))["chart"]["result"][0]
        d = {"t": r["timestamp"], "c": r["indicators"]["quote"][0]["close"],
             "a": (r["indicators"].get("adjclose") or [{}])[0].get("adjclose") or r["indicators"]["quote"][0]["close"]}
        json.dump(d, open(fn, "w"))
    idx = pd.to_datetime(d["t"], unit="s").normalize()
    df = pd.DataFrame({"c": d["c"], "a": d["a"]}, index=idx).dropna()
    return df[~df.index.duplicated(keep="last")]


def cboe(name):
    fn = os.path.join(CACHE, f"cboe_{name}.csv")
    if not os.path.exists(fn):
        u = f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{name}_History.csv"
        open(fn, "wb").write(urllib.request.urlopen(urllib.request.Request(u, headers=H), timeout=60).read())
    df = pd.read_csv(fn)
    df.index = pd.to_datetime(df.iloc[:, 0], format="%m/%d/%Y")
    return df.iloc[:, 1 if name != "VIX3M" else 4].astype(float)


def tr(sym):  # 配当込み日次リターン
    r = yahoo(sym)["a"].pct_change().dropna()
    # 1日で大きく動いて翌日ほぼ元に戻る=データ異常(例: BTAL 2015-04)として両日を除外
    nxt = r.shift(-1)
    glitch = (r.abs() > 0.15) & (nxt.abs() > 0.15) & (np.sign(r) != np.sign(nxt)) & (((1 + r) * (1 + nxt) - 1).abs() < 0.03)
    if glitch.any():
        drop = glitch | glitch.shift(1, fill_value=False)
        print(f"  [データ異常除外] {sym}: " + ", ".join(str(d.date()) for d in r[drop].index))
        r = r[~drop]
    bad = r[r.abs() > 0.15]
    if len(bad):
        print(f"  [警告] {sym} 日次±15%超 {len(bad)}日: " + ", ".join(f"{d.date()}:{v:+.0%}" for d, v in bad.items()))
    return r


def pr(sym):  # 価格のみ日次リターン
    return yahoo(sym)["c"].pct_change().dropna()


def stats(r, ppy=252):
    r = r.dropna()
    if len(r) < 20:
        return dict(cagr=np.nan, mdd=np.nan, sh=np.nan, vol=np.nan)
    w = (1 + r).cumprod()
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    cagr = w.iloc[-1] ** (1 / yrs) - 1
    mdd = (w / w.cummax() - 1).min()
    return dict(cagr=cagr, mdd=mdd, sh=r.mean() / r.std() * np.sqrt(ppy), vol=r.std() * np.sqrt(ppy))


def line(name, r, ppy=252, extra=""):
    s = stats(r, ppy)
    a = r.index[0].date()
    print(f"  {name:<22}{s['cagr']:>7.1%}{s['mdd']:>8.0%}{s['sh']:>7.2f}{s['vol']:>7.1%}  ({a}〜){extra}")


def halves(df, ppy=252, ref=None):
    """共通標本を中点で分割して年率/シャープを表示"""
    df = df.dropna()
    mid = df.index[len(df) // 2]
    print(f"  前半〜{mid.date()} / 後半  (年率, シャープ)")
    for c in df.columns:
        a, b = stats(df[c][:mid], ppy), stats(df[c][mid:], ppy)
        print(f"    {c:<20} 前 {a['cagr']:>6.1%} {a['sh']:>5.2f} | 後 {b['cagr']:>6.1%} {b['sh']:>5.2f}")


def window(name, df, a, b):
    out = []
    for c in df.columns:
        x = df[c][a:b].dropna()
        if len(x) < 2:
            out.append(f"{c}:NA")
            continue
        w = (1 + x).cumprod()
        out.append(f"{c}:{w.iloc[-1]-1:+.0%}/DD{(w/w.cummax()-1).min():.0%}")
    print(f"  {name}: " + "  ".join(out))


def crises(df, daily=True):
    window("2008-09 GFC(2007-10〜2009-03)", df, "2007-10-31" if not daily else "2007-10-09", "2009-03-09" if daily else "2009-03-31")
    window("2020 コロナ(2月〜3月)", df, "2020-02-19" if daily else "2020-02-01", "2020-03-23" if daily else "2020-03-31")
    window("2022 年間", df, "2022-01-01", "2022-12-31")


def aftertax(r, div=None):
    """日次リターン r(配当込み)。div=配当分の日次リターン(無ければ 0)。
    分配金課税を毎日の複利に織り込み、最後に売却益20.315%。最終的な年率を返す。"""
    d = div if div is not None else pd.Series(0.0, index=r.index)
    rr = r - TAX * d.clip(lower=0)
    w = (1 + rr).cumprod().iloc[-1]
    w_net = w - TAX * max(w - 1, 0)
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    return w_net ** (1 / yrs) - 1


def aftertax_realized(r):
    """配当のない戦略(SVXY等): 暦年ごとの純利益に20.315%、損失は3年繰越。"""
    y = (1 + r).groupby(r.index.year).prod() - 1
    w, losses = 1.0, []
    for yr, g in y.items():
        pre = w * (1 + g)
        gain = pre - w
        losses = [(yy, l) for yy, l in losses if yr - yy <= 3]
        if gain > 0:
            use = min(gain, sum(l for _, l in losses))
            tax = TAX * (gain - use)
            rem = use
            nl = []
            for yy, l in losses:
                t = min(l, rem)
                rem -= t
                if l - t > 1e-12:
                    nl.append((yy, l - t))
            losses = nl
        else:
            tax = 0.0
            losses.append((yr, -gain))
        w = pre - tax
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    return w ** (1 / yrs) - 1


# ----------------------------------------------------------------------------------------
def part1():
    print("\n===== 1. オプション売りの指数(月次, CBOE公式CSV) =====")
    raw = {k: cboe(k) for k in ["BXM", "BXMD", "BXY", "PUT", "WPUT"]}
    raw["SPX_TR"] = yahoo("^SP500TR")["c"]  # S&P500 配当込み(1988〜)
    raw["SPX_PR"] = yahoo("^GSPC")["c"]
    m = pd.DataFrame({k: v.resample("ME").last() for k, v in raw.items()}).dropna(how="all")
    print("  開始日: " + ", ".join(f"{k} {v.dropna().index[0].date()}" for k, v in raw.items()))
    print("  注: CBOEのBXMファイルは2002開始(公式は1986〜だがこのCSVは2002〜)。BXMDは1986〜(30デルタ・バイライト)")
    count(1, 6)
    print("  PUTのCSVは2007以前が歯抜け(月次相当)なので、PUT/WPUTは2007-02〜の表のみで比較する")
    for lab, a, cols in [("1988-07〜(BXMD,BXY)", "1988-06-30", ["SPX_TR", "BXMD", "BXY"]),
                         ("2002-04〜(+BXM)", "2002-03-31", ["SPX_TR", "BXM", "BXMD", "BXY"]),
                         ("2007-02〜(+PUT,WPUT)", "2007-01-31", ["SPX_TR", "BXM", "BXMD", "BXY", "PUT", "WPUT"])]:
        r = m.loc[a:, cols].pct_change().dropna()
        print(f"\n -- {lab}  (月次リターンから。年率/最大下落/シャープ/年率ボラ)")
        for c in r.columns:
            line(c, r[c], 12)
        halves(r, 12)
        crises(r, daily=False)
    p = raw["PUT"]
    x, y = p.loc["1991-03-01":"2007-01-31"], raw["SPX_TR"].loc["1991-03-01":"2007-01-31"]
    yrs = (x.index[-1] - x.index[0]).days / 365.25
    print(f"\n  PUT 1991-03〜2007-01(端点のみ): {(x.iloc[-1]/x.iloc[0])**(1/yrs)-1:.1%}/年 vs SPX_TR(同期間・端点) {(y.iloc[-1]/y.iloc[0])**(1/yrs)-1:.1%}/年")
    print("\n  [WPUT(週次プット)は2006〜のため上の2002〜表に含む。BXY=2%OTMコール売り]")

    print("\n -- 実在のカバードコールETF(日次, 配当込み) vs SPY / 指数")
    etfs = ["JEPI", "JEPQ", "QYLD", "XYLD", "DIVO", "SPY", "QQQ"]
    count(1, 7)
    R = {e: tr(e) for e in etfs}
    P = {e: pr(e) for e in etfs}
    for e in etfs:
        R[e].name = e
    # 各ETFと同期間のSPY比較
    print("  (各ETFの上場後で SPY と比較。税後=分配金課税込み・売却益課税込み、NISAなら=税前)")
    print(f"  {'':<14}{'年率':>7}{'最大下落':>8}{'シャープ':>7}{'ボラ':>7} | {'同期間SPY年率':>12}{'SPY下落':>8}{'SPYシャープ':>9} | 税後ETF/税後SPY | 分配利回り概算")
    for e in ["JEPI", "JEPQ", "QYLD", "XYLD", "DIVO"]:
        r = R[e]
        spy = R["SPY"].loc[r.index[0]:]
        s, b = stats(r), stats(spy)
        dv = (r - P[e]).dropna()
        dvs = (spy - P["SPY"].loc[spy.index[0]:]).dropna()
        yld = dv.clip(lower=0).sum() / ((r.index[-1] - r.index[0]).days / 365.25)
        print(f"  {e:<14}{s['cagr']:>7.1%}{s['mdd']:>8.0%}{s['sh']:>7.2f}{s['vol']:>7.1%} | {b['cagr']:>12.1%}{b['mdd']:>8.0%}{b['sh']:>9.2f} |"
              f" {aftertax(r, dv):>6.1%}/{aftertax(spy, dvs):<6.1%} | {yld:>5.1%}")
    # QYLD は NDX ベースなので QQQ と、XYLD は BXM と
    q = R["QYLD"].loc["2013-12-13":]
    print(f"  QYLD vs QQQ(同期間): QYLD {stats(q)['cagr']:.1%} / QQQ {stats(R['QQQ'].loc[q.index[0]:])['cagr']:.1%}"
          f"  下落 {stats(q)['mdd']:.0%} / {stats(R['QQQ'].loc[q.index[0]:])['mdd']:.0%}")
    bx = raw["BXM"].resample("ME").last().pct_change()
    bxd = raw["BXMD"].resample("ME").last().pct_change()
    x = R["XYLD"].resample("ME").apply(lambda v: (1 + v).prod() - 1)
    j = pd.DataFrame({"XYLD": x, "BXM": bx, "BXMD": bxd}).dropna().loc["2013-07-31":]
    n = len(j)
    yrs = n / 12
    for c in j.columns:
        print(f"  XYLD期間の {c:<5} 年率 {(1+j[c]).prod()**(1/yrs)-1:.2%}")
    print(f"  XYLD-BXM 月次差の平均 年換算 {(j['XYLD']-j['BXM']).mean()*12:+.2%}(経費率0.60%が含まれる。要確認)")
    # ETF同士の前半/後半(共通期間: JEPQは2022〜なので、JEPIはJEPI期間、他は2014〜)
    d = pd.DataFrame({e: R[e] for e in ["QYLD", "XYLD", "DIVO", "SPY"]}).dropna()
    print("  2017〜共通(QYLD,XYLD,DIVO,SPY):")
    halves(d)
    d2 = pd.DataFrame({e: R[e] for e in ["JEPI", "JEPQ", "SPY", "QQQ"]}).dropna()
    print("  JEPQ期間共通:")
    halves(d2)
    crises(pd.DataFrame({e: R[e] for e in ["JEPI", "JEPQ", "QYLD", "XYLD", "DIVO", "SPY"]}))
    return R


# ----------------------------------------------------------------------------------------
def part2(R):
    print("\n===== 2. 短期ボラ売り + 期間構造フィルタ(SVXY) =====")
    svxy = yahoo("SVXY")
    sr = svxy["a"].pct_change().dropna()
    print(f"  SVXY 最悪日: {sr.idxmin().date()} {sr.min():+.1%}、次点 {sr.nsmallest(2).index[1].date()} {sr.nsmallest(2).iloc[1]:+.1%}")
    print("  [注意] 2018-02-05/06 に-80〜-90%。2018-02-28から倍率が -1x → -0.5x に変更(Yahooの価格はこの変更を補正しない=2018前の成績は-1x, 後は-0.5x)")
    v, v3 = yahoo("^VIX")["c"], yahoo("^VIX3M")["c"]
    ratio = (v / v3).dropna()
    print(f"  VIX/VIX3M 開始 {ratio.index[0].date()}、コンタンゴ(<1)の日の割合 {(ratio<1).mean():.0%}")
    spy = R["SPY"]
    ix = sr.index.intersection(ratio.index)
    sr_, ra = sr.loc[ix], ratio.loc[ix]
    variants = {"SVXY買い持ち": pd.Series(1.0, index=ix),
                "比<1.0": (ra < 1.0).astype(float),
                "比<0.95": (ra < 0.95).astype(float),
                "比<0.90": (ra < 0.90).astype(float)}
    count(2, 4)
    out = {}
    for cost_name, c in [("コスト0.1%/片道", 0.001), ("コスト0.5%/片道", 0.005)]:
        rows = {}
        for k, sig in variants.items():
            pos = sig.shift(1).fillna(0.0)   # t の終値で決め、t+1 に保有
            turn = pos.diff().abs().fillna(0)
            rows[k] = pos * sr_ - turn * c
        out[cost_name] = pd.DataFrame(rows)
    for cn, df in out.items():
        print(f"\n -- {cn} (現金0%)  年率/最大下落/シャープ/ボラ")
        for k in df.columns:
            line(k, df[k])
        w = df
    df = out["コスト0.1%/片道"]
    # 2018-02 を除く: 倍率変更後だけ
    print("\n -- 2018-03以降(-0.5x版のみ) コスト0.1%")
    d2 = df.loc["2018-03-01":]
    for k in d2.columns:
        line(k, d2[k])
    line("SPY(同期間)", spy.reindex(d2.index).dropna())
    print("\n -- 前半/後半(コスト0.1%, 中点で分割)")
    halves(df)
    print("\n -- 最悪日の損失と保有日数")
    for k in df.columns:
        pos = variants[k].shift(1).fillna(0)
        print(f"  {k:<12} 最悪日 {df[k].min():+.1%}({df[k].idxmin().date()})  保有比率 {pos.mean():.0%}  売買回数/年 {pos.diff().abs().sum()/ ((ix[-1]-ix[0]).days/365.25):.1f}")
    crises_df = pd.DataFrame({"比<1.0": df["比<1.0"], "買持": df["SVXY買い持ち"], "SPY": spy.reindex(df.index)})
    window("2018-02 暴落", crises_df, "2018-01-26", "2018-02-28")
    window("2020 コロナ", crises_df, "2020-02-19", "2020-03-23")
    window("2022 年間", crises_df, "2022-01-01", "2022-12-31")
    f = df["比<1.0"]
    print(f"  フィルタ(比<1)税後(売買益課税, 3年繰越): {aftertax_realized(f):.1%}  vs 税前 {stats(f)['cagr']:.1%}。SPY税後 {aftertax(spy.loc[f.index[0]:]):.1%}")
    print(f"  フィルタの2018-02-05/06の損失: " + ", ".join(f"{d.date()} {f.loc[d]:+.1%}" for d in f.loc['2018-02-02':'2018-02-08'].index))
    print("\n -- VIXY / UVXY(長期保有の目減り)")
    count(2, 2)
    for s in ["VIXY", "UVXY"]:
        y = yahoo(s)
        r = y["a"].pct_change().dropna()
        w = (1 + r).cumprod()
        print(f"  {s}: 上場{y.index[0].date()}〜 年率 {stats(r)['cagr']:.1%}、累積 {w.iloc[-1]-1:.2%}(元の{w.iloc[-1]:.4f}倍)。※Yahooは逆分割補正済みの価格")
    return out


# ----------------------------------------------------------------------------------------
def monthly(r):
    return (1 + r).groupby(pd.Grouper(freq="ME")).prod() - 1


def rebal_mix(rets, w):
    """月次リバランスの固定比率(日次リターンから月内ドリフト)"""
    df = rets.dropna()
    mi = df.index.to_period("M")
    out = []
    cur = None
    for p in pd.unique(mi):
        sub = df[mi == p]
        v = np.array([w[c] for c in df.columns], dtype=float)
        vals = v.copy()
        series = []
        for _, row in sub.iterrows():
            new = vals * (1 + row.values)
            series.append(new.sum() / vals.sum() - 1)
            vals = new
        out.append(pd.Series(series, index=sub.index))
    return pd.concat(out)


def part3(R):
    print("\n===== 3. 管理先物(CTA)・オルタナ =====")
    names = ["DBMF", "KMLM", "CTA", "BTAL", "TAIL", "SPY", "AGG"]
    count(3, 5 + 6)
    R3 = {n: tr(n) for n in names}
    alts = ["DBMF", "KMLM", "CTA", "BTAL", "TAIL"]
    df = pd.DataFrame(R3)
    print("  単体 + SPY との相関(各自の上場後)  年率/最大下落/シャープ/ボラ")
    for n in names:
        r = df[n].dropna()
        corr = df[[n, "SPY"]].dropna().corr().iloc[0, 1]
        line(n, r, extra=f" 相関SPY {corr:.2f}")
    common = df[["DBMF", "SPY", "AGG"]].dropna()
    print("\n  2019-05〜 共通(DBMF,SPY,AGG)の2022年: ", end="")
    window("", common, "2022-01-01", "2022-12-31")
    print("\n -- 60/40(SPY/AGG)に20%足したときの比較(各alt上場後の同期間, 月次リバランス)")
    print("    構成: A=60/40, B=48/32/20alt (株債を各8割に縮小して20%をalt), C=42/28/30は試さない")
    print(f"  {'':<26}{'年率':>7}{'最大下落':>8}{'シャープ':>7}{'ボラ':>7}")
    for a in alts:
        d = df[["SPY", "AGG", a]].dropna()
        base = rebal_mix(d[["SPY", "AGG"]], {"SPY": 0.6, "AGG": 0.4})
        mix = rebal_mix(d, {"SPY": 0.48, "AGG": 0.32, a: 0.2})
        print(f"  [{a} 期間 {d.index[0].date()}〜]")
        line("60/40", base)
        line(f"48/32/20 {a}", mix)
        # 2022
        w = lambda x: ((1 + x["2022-01-01":"2022-12-31"]).prod() - 1) if len(x["2022-01-01":"2022-12-31"]) > 100 else np.nan
        print(f"     2022年: 60/40 {w(base):+.1%} / +{a} {w(mix):+.1%}")
        h = pd.DataFrame({"60/40": base, f"+{a}": mix})
        halves(h)
    return R3


# ----------------------------------------------------------------------------------------
def part4(R):
    print("\n===== 4. クレジット・インカム系(日次, 配当込み) =====")
    names = ["HYG", "JNK", "PFF", "EMB", "BIZD", "VNQ", "TIP", "AGG", "SPY"]
    count(4, 7)
    R4 = {n: tr(n) for n in names}
    P4 = {n: pr(n) for n in names}
    df = pd.DataFrame(R4)
    print("  単体(各自の上場後) 年率/最大下落/シャープ/ボラ  | 相関SPY・β | 分配利回り概算 | 税後年率(分配課税+売却益)")
    for n in names:
        r = df[n].dropna()
        j = df[[n, "SPY"]].dropna() if n != "SPY" else pd.DataFrame({"x": r, "SPY": r})
        beta = j.cov().iloc[0, 1] / j["SPY"].var()
        dv = (r - P4[n]).dropna()
        yld = dv.clip(lower=0).sum() / ((r.index[-1] - r.index[0]).days / 365.25)
        line(n, r, extra=f" 相関 {j.corr().iloc[0,1]:.2f} β{beta:.2f} | 利回り {yld:.1%} | 税後 {aftertax(r, dv):.1%}")
    d = df.dropna()  # 2013-02〜(BIZD)
    print(f"\n  共通期間 {d.index[0].date()}〜 で前半/後半:")
    halves(d)
    print("  危機(上場している銘柄のみ。下は各期間の累積/最大下落):")
    crises(df[["HYG", "JNK", "PFF", "EMB", "VNQ", "TIP", "AGG", "SPY", "BIZD"]])
    # 「高利回り=危険の対価」: 利回りとリスク調整後リターン
    print("\n  リスク調整後(シャープ)がSPY・AGGより高いか: 共通期間")
    for n in d.columns:
        s = stats(d[n])
        print(f"    {n:<5} 年率 {s['cagr']:>6.1%} シャープ {s['sh']:.2f}")
    # HY・REIT を株ベータで補正した超過(α)
    print("\n  SPYに対するα(日次回帰, 年率)と t値(全期間):")
    for n in ["HYG", "JNK", "PFF", "EMB", "BIZD", "VNQ", "TIP", "AGG"]:
        j = df[[n, "SPY"]].dropna()
        X = np.c_[np.ones(len(j)), j["SPY"].values]
        b, *_ = np.linalg.lstsq(X, j[n].values, rcond=None)
        res = j[n].values - X @ b
        se = np.sqrt(res.var() * np.linalg.inv(X.T @ X)[0, 0])
        print(f"    {n:<5} α {b[0]*252:+.1%}/年 (t={b[0]/se:.1f}) β={b[1]:.2f}")
    return R4


def practical():
    print("\n===== 5. 少額での実現性(最新終値、円換算は概算。為替は USDJPY=X 最新) =====")
    fx = yahoo("USDJPY=X")["c"].iloc[-1]
    print(f"  USDJPY {fx:.1f}")
    for s in ["JEPI", "JEPQ", "QYLD", "XYLD", "DIVO", "DBMF", "KMLM", "CTA", "BTAL", "TAIL", "SVXY", "HYG", "JNK", "PFF", "EMB", "BIZD", "VNQ", "TIP", "SPY", "AGG"]:
        p = yahoo(s)["c"].iloc[-1]
        print(f"  {s:<5} 1口 ${p:7.2f} = 約{p*fx:>8,.0f}円")


if __name__ == "__main__":
    R = part1()
    part2(R)
    part3(R)
    part4(R)
    practical()
    print("\n===== 試行数(多重比較の注意) =====")
    print("  ", TRIALS, "合計", sum(TRIALS.values()), "(+期間分割・シグナル閾値などの見方違いは別)")
    print("  生存者バイアス: 今も上場しているETFだけを見ている。閉鎖・合併されたカバードコール/CTA/ボラETF(例 XIV)は含まれない")
