"""レバレッジ + 200日線フィルタの検証(合成レバレッジ vs 実在ETF、各種戦略、期間別、ローリング10年、日本の税)。
  python research/leverage_trend.py            → 全結果を標準出力へ(初回はYahooから取得し research/data/lev_*.json にキャッシュ)
  python research/leverage_trend.py --refresh  → キャッシュを捨てて取り直す

設計(多重比較を避けるため、定番値だけを固定して全件を表示する):
  SMA期間は200日のみ、Lは 1/2/3、バンドは 0/1/2/3%、ボラ目標は15%・上限2倍/3倍。試した戦略数は main の末尾に表示。
  シグナルは当日(t)終値。約定は翌営業日(t+1)終値、収益が乗るのは t+2 から(LAG=2、保守的)。感度として LAG=1 も表示。
  合成レバ: 日次リターン = e*r - max(e-1,0)*(短期金利+0.5%)/252 - 経費率/252 (+ 現金部分は短期金利)。日次リバランス型なのでボラ・ドラッグは自動で含まれる。
  指数は価格指数なので、配当利回りを定数で足す(^GSPC 2.0%、^NDX 0.9%、^N225 0%)。古い年代のS&P配当(3〜4%)は過小。
"""
import json, os, sys, urllib.request
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)
H = {"User-Agent": "Mozilla/5.0"}
LAG = 2
COST = 0.001          # 売買コスト: 片道、取引額の0.1%
SPREAD = 0.005        # 金利に上乗せされる調達コスト(年)
EXPENSE = {1: 0.001, 2: 0.009, 3: 0.0095}
# ^N225は価格指数だが、先物で複製する1570は配当相当(約1.5%/年)を実質受け取る(配当0だと1570実物に対し合成が年-3.9%低かった)
DIV = {"^GSPC": 0.020, "^NDX": 0.009, "^N225": 0.015}
TAX = 0.20315


def fetch(sym, refresh=False):
    path = os.path.join(DATA, "lev_" + sym.replace("^", "_").replace("=", "_") + ".json")
    if os.path.exists(path) and not refresh:
        d = json.load(open(path))
    else:
        u = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.request.quote(sym)}?interval=1d&period1=0&period2=1900000000"
        res = json.load(urllib.request.urlopen(urllib.request.Request(u, headers=H), timeout=40))["chart"]["result"][0]
        adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose")
        d = {"t": res["timestamp"], "c": res["indicators"]["quote"][0]["close"], "a": adj}
        json.dump(d, open(path, "w"))
    idx = pd.to_datetime(d["t"], unit="s").normalize()
    df = pd.DataFrame({"c": d["c"], "a": d["a"] if d["a"] else d["c"]}, index=idx)
    df = df[~df.index.duplicated(keep="last")].dropna()
    return df


# ---------------- 合成レバレッジのエンジン ----------------
def veh_rate(e):
    """エクスポージャー e を実現するETFの経費率(段階)。"""
    if e <= 0: return 0.0
    v = 1 if e <= 1 else (2 if e <= 2 else 3)
    return EXPENSE[v] * min(1.0, e / v)


def daily_ret(e, r, rf):
    """e: エクスポージャー(配列)、r: 原資産の日次トータルリターン、rf: 日次短期金利。"""
    exp_ = np.array([veh_rate(x) for x in e]) / 252 if hasattr(e, "__len__") else veh_rate(e) / 252
    return e * r - np.maximum(e - 1, 0) * (rf + SPREAD / 252) + (1 - np.minimum(e, 1)) * rf - exp_


def run(e_target, r, rf, lag=LAG, cost=COST):
    """e_target: シグナル(各日終値で決めた目標エクスポージャー)。lag日後から効く。"""
    e = pd.Series(e_target, index=r.index).shift(lag).fillna(0.0).values
    gross = daily_ret(e, r.values, rf.values)
    turn = np.abs(np.diff(e, prepend=0.0))
    net = gross - cost * turn
    return pd.Series(net, index=r.index), pd.Series(e, index=r.index)


def sig_trend(px, L, band=0.0, n=200):
    sma = px.rolling(n).mean().values
    p = px.values
    on, out = False, np.zeros(len(p))
    for i in range(len(p)):
        if np.isnan(sma[i]): continue
        if not on and p[i] > sma[i] * (1 + band): on = True
        elif on and p[i] < sma[i] * (1 - band): on = False
        out[i] = L if on else 0.0
    return out


def sig_vol(r_idx, cap, target=0.15, gate=None, rebal=0.25):
    vol = (r_idx.rolling(20).std() * np.sqrt(252)).values
    tgt = np.where(np.isnan(vol), 0.0, np.minimum(cap, target / np.maximum(vol, 1e-6)))
    if gate is not None: tgt = tgt * (gate > 0)
    out, cur = np.zeros(len(tgt)), 0.0
    for i, t in enumerate(tgt):
        if t == 0 or cur == 0 or abs(t - cur) > rebal: cur = t   # 目標が0/新規/0.25以上ずれたときだけ売買
        out[i] = cur
    return out


# ---------------- 指標 ----------------
def stats(net, rf, e=None):
    eq = (1 + net).cumprod()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    dd = eq / eq.cummax() - 1
    ex = net - rf
    sh = ex.mean() / net.std() * np.sqrt(252) if net.std() > 0 else np.nan
    w12 = (eq / eq.shift(252) - 1).min()
    # 回復までの最長期間(高値更新まで)
    top = eq.cummax(); under = eq < top
    longest, start = 0, None
    for d, u in zip(eq.index, under):
        if u and start is None: start = d
        if not u and start is not None:
            longest = max(longest, (d - start).days); start = None
    if start is not None: longest = max(longest, (eq.index[-1] - start).days)
    out = dict(cagr=cagr, mdd=dd.min(), sharpe=sh, calmar=cagr / -dd.min() if dd.min() < 0 else np.nan,
               w12=w12, rec=longest / 365.25, final=eq.iloc[-1])
    if e is not None:
        on = (e > 0).astype(int)
        ch = on.diff().fillna(on.iloc[0])
        out["trades"] = int((ch != 0).sum())
        # 往復ごとの損益(だましの損失)
        ent, rts = None, []
        for d, c in ch.items():
            if c == 1: ent = eq.loc[d]
            elif c == -1 and ent is not None: rts.append(eq.loc[d] / ent - 1); ent = None
        rts = np.array(rts)
        out["rt"] = len(rts)
        out["lose_n"] = int((rts < 0).sum()) if len(rts) else 0
        out["lose_sum"] = float(rts[rts < 0].sum()) if len(rts) else 0.0
    return out


def window(net, a, b):
    s = net.loc[a:b]
    if len(s) < 5: return np.nan, np.nan
    eq = (1 + s).cumprod()
    return eq.iloc[-1] - 1, (eq / eq.cummax() - 1).min()


def rolling10(nets, base_key, h=2520, step=21):
    eqs = {k: (1 + v).cumprod().values for k, v in nets.items()}
    n = len(next(iter(eqs.values())))
    starts = list(range(0, n - h, step))
    res = {}
    for k, eq in eqs.items():
        res[k] = np.array([(eq[s + h] / eq[s]) ** (1 / 10) - 1 for s in starts])
    out = {}
    for k, v in res.items():
        out[k] = dict(min=v.min(), p10=np.percentile(v, 10), med=np.median(v), p90=np.percentile(v, 90),
                      neg=(v < 0).mean(), beat=(v > res[base_key]).mean())
    return out, len(starts)


# ---------------- 税(日本・特定口座の簡易モデル) ----------------
def after_tax_final(net, e):
    """売却(エクスポージャー→0)ごとに実現損益。年末に年間損益を通算(損失は3年繰越)して20.315%を資産から引く。
    最後は全部売却したとして課税。持ちっぱなしは最後だけ課税(課税繰延が効く)。"""
    eq = 1.0; basis = None; yr_pnl = {}; carry = []; cur_year = net.index[0].year
    ev = (e > 0).values
    prev = False
    for i, (d, rr) in enumerate(net.items()):
        if d.year != cur_year:                         # 年末精算
            eq = settle(eq, yr_pnl, carry, cur_year); cur_year = d.year
        eq *= 1 + rr
        if ev[i] and not prev: basis = eq
        if prev and not ev[i] and basis is not None:
            yr_pnl[d.year] = yr_pnl.get(d.year, 0.0) + (eq - basis); basis = None
        prev = ev[i]
    if basis is not None:
        yr_pnl[cur_year] = yr_pnl.get(cur_year, 0.0) + (eq - basis)
    eq = settle(eq, yr_pnl, carry, cur_year)
    return eq


def settle(eq, yr_pnl, carry, y):
    p = yr_pnl.pop(y, 0.0)
    carry[:] = [(yy, c) for yy, c in carry if y - yy <= 3]
    if p < 0:
        carry.append((y, -p)); return eq
    use = 0.0
    for k, (yy, c) in enumerate(carry):
        take = min(c, p - use); carry[k] = (yy, c - take); use += take
    taxable = max(p - use, 0.0)
    return eq - TAX * taxable


def at_cagr(net, e, bh=False):
    yrs = (net.index[-1] - net.index[0]).days / 365.25
    if bh:
        eq = (1 + net).cumprod().iloc[-1]
        eq = eq - TAX * max(eq - 1, 0)
    else:
        eq = after_tax_final(net, e)
    return eq ** (1 / yrs) - 1


# ---------------- 検証パート ----------------
def pct(x, d=1): return "  n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x*100:6.{d}f}%"


def build_rf(irx, idx, jp=False):
    if not jp:
        s = (irx["c"] / 100).reindex(idx, method="ffill").ffill().bfill()
    else:  # 日本の短期金利(円): 概算の年次テーブル。1991-94: 4%, 1995以降: 0.1%(2007前後は0.5%)。1965-90は5%
        y = idx.year
        s = pd.Series(np.where(y <= 1990, 0.05, np.where(y <= 1994, 0.04, np.where((y >= 2006) & (y <= 2008), 0.005, 0.001))), index=idx)
    return s / 252


def validate(refresh):
    print("\n=== 1. 合成レバレッジ vs 実在ETF(日次リターン。ETFは配当込み調整後終値。1570.Tは終値) ===")
    pairs = [("SSO", "^GSPC", 2), ("SPXL", "^GSPC", 3), ("UPRO", "^GSPC", 3), ("QLD", "^NDX", 2), ("TQQQ", "^NDX", 3), ("1570.T", "^N225", 2)]
    irx = fetch("^IRX", refresh)
    print("ETF     L  期間                 日次相関  追跡誤差/年  ETF年率  合成年率  差(合成-ETF)  累積比(合成/ETF)")
    for etf, idxs, L in pairs:
        e = fetch(etf, refresh); ix = fetch(idxs, refresh)
        jp = idxs == "^N225"
        col = "c" if jp else "a"
        d = pd.concat([e[col].pct_change().rename("etf"), ix["c"].pct_change().rename("ix")], axis=1).dropna()
        if jp:   # 1570.Tに壊れた日(分割など)がないか
            bad = d["etf"].abs() > 0.15
            d = d[~bad]
        rf = build_rf(irx, d.index, jp)
        r = d["ix"] + DIV[idxs] / 252
        syn = pd.Series(daily_ret(np.full(len(d), float(L)), r.values, rf.values), index=d.index)
        corr = np.corrcoef(d["etf"], syn)[0, 1]
        te = (d["etf"] - syn).std() * np.sqrt(252)
        yrs = (d.index[-1] - d.index[0]).days / 365.25
        ce = (1 + d["etf"]).prod() ** (1 / yrs) - 1; cs = (1 + syn).prod() ** (1 / yrs) - 1
        ratio = (1 + syn).prod() / (1 + d["etf"]).prod()
        print(f"{etf:7s} {L}  {d.index[0].date()}~{d.index[-1].date()}  {corr:6.4f}  {pct(te)}   {pct(ce)}  {pct(cs)}   {pct(cs-ce)}     {ratio:6.3f}")
    print("  (^GSPC/^NDXの配当は定数近似、^N225は配当ゼロ。日本の短期金利は粗い年次テーブル)")


def analyze_market(name, sym, refresh, start=None, jp=False, show_tax=False):
    irx = fetch("^IRX", refresh)
    ix = fetch(sym, refresh)
    rf_all = build_rf(irx, ix.index, jp)
    first = max(ix.index[0], irx.index[0]) if not jp else ix.index[0]
    ix = ix[ix.index >= first]
    px = ix["c"]
    rpx = px.pct_change().fillna(0.0)
    r = rpx + DIV[sym] / 252
    rf = rf_all.reindex(ix.index)
    # SMAのウォームアップ後(200日)から評価
    warm = ix.index[200]
    print(f"\n\n################ {name}  ({ix.index[0].date()} ~ {ix.index[-1].date()}、評価開始 {warm.date()}、{len(ix)}日) ################")
    S = {}   # name -> (net, e)
    def add(k, e_sig, lag=LAG):
        net, e = run(e_sig, r, rf, lag)
        S[k] = (net.loc[warm:], e.loc[warm:])
    for L in (1, 2, 3):
        add(f"持ちっぱなし {L}x", np.full(len(r), float(L)))
    for L in (1, 2, 3):
        add(f"SMA200 {L}x", sig_trend(px, L))
    for b in (0.01, 0.02, 0.03):
        for L in (2, 3):
            add(f"SMA200±{int(b*100)}% {L}x", sig_trend(px, L, b))
    gate1 = sig_trend(px, 1)
    for cap in (2, 3):
        add(f"ボラ目標15% 上限{cap}x", sig_vol(rpx, cap))
        add(f"SMA200+ボラ目標 上限{cap}x", sig_vol(rpx, cap, gate=gate1))
    base = "持ちっぱなし 1x"

    print("\n--- 全期間(評価開始以降。コスト込み、LAG=2) ---")
    print(f"{'戦略':26s} 年率    最大下落  シャープ カルマー 最悪12M  最長水面下 売買 往復 負け往復(合計)")
    allst = {}
    for k, (net, e) in S.items():
        st = stats(net, rf.loc[net.index], e); allst[k] = st
        extra = f"{st['trades']:4d} {st['rt']:3d} {st['lose_n']:3d}({st['lose_sum']*100:6.0f}%)" if k.split()[0] not in ("持ちっぱなし",) else "   -"
        print(f"{k:26s}{pct(st['cagr'])} {pct(st['mdd'])}  {st['sharpe']:6.2f}  {st['calmar']:6.2f} {pct(st['w12'])}  {st['rec']:5.1f}年   {extra}")

    # 前半/後半
    mid = S[base][0].index[len(S[base][0]) // 2]
    print(f"\n--- 前半/後半(分割 {mid.date()}) 年率 / 最大下落 ---")
    for k, (net, e) in S.items():
        a = stats(net.loc[:mid], rf.loc[net.loc[:mid].index]); b = stats(net.loc[mid:], rf.loc[net.loc[mid:].index])
        print(f"{k:26s} 前半 {pct(a['cagr'])} {pct(a['mdd'])} | 後半 {pct(b['cagr'])} {pct(b['mdd'])}")

    # 個別局面
    wins = [("2000-2002(ドットコム)", "2000-01-01", "2002-12-31"), ("2008", "2008-01-01", "2008-12-31"),
            ("2020", "2020-01-01", "2020-12-31"), ("2022", "2022-01-01", "2022-12-31")]
    if sym == "^N225": wins = [("1990-1992(バブル崩壊)", "1990-01-01", "1992-12-31")] + wins
    print("\n--- 個別局面の騰落(期間リターン / 期間内最大下落) ---")
    print(f"{'戦略':26s} " + " | ".join(w[0] for w in wins))
    for k, (net, e) in S.items():
        cells = []
        for _, a, b in wins:
            rr, dd = window(net, a, b); cells.append(f"{pct(rr,0)}/{pct(dd,0)}")
        print(f"{k:26s} " + " | ".join(cells))

    # ローリング10年
    nets = {k: v[0] for k, v in S.items()}
    ro, n = rolling10(nets, base)
    print(f"\n--- 開始時期をずらしたローリング10年CAGR(月次開始、{n}窓)。beat=持ちっぱなし1xに勝った窓の割合 ---")
    print(f"{'戦略':26s}  最悪    10%点   中央値   90%点   マイナス窓 beat")
    for k, v in ro.items():
        print(f"{k:26s} {pct(v['min'])} {pct(v['p10'])} {pct(v['med'])} {pct(v['p90'])}   {v['neg']*100:5.1f}%  {v['beat']*100:5.1f}%")

    # LAG=1感度
    print("\n--- 感度: LAG=1(当日引けで約定=ルックアヘッドではないが楽観) 年率 / 最大下落 ---")
    for k, e_sig in [("SMA200 2x", sig_trend(px, 2)), ("SMA200 3x", sig_trend(px, 3)), ("SMA200±2% 2x", sig_trend(px, 2, 0.02))]:
        net, _ = run(e_sig, r, rf, 1); net = net.loc[warm:]
        st = stats(net, rf.loc[net.index]); print(f"{k:26s} {pct(st['cagr'])} {pct(st['mdd'])}")

    # 10年刻み
    print("\n--- 年代別の年率(上がった時代に乗っただけか) ---")
    ys = sorted(set(S[base][0].index.year))
    edges = [(1960, 1979), (1980, 1989), (1990, 1999), (2000, 2009), (2010, 2019), (2020, 2026)]
    keys = [base, "持ちっぱなし 2x", "持ちっぱなし 3x", "SMA200 2x", "SMA200 3x", "SMA200±2% 2x", "ボラ目標15% 上限2x", "SMA200+ボラ目標 上限2x"]
    print(f"{'戦略':26s} " + " ".join(f"{a}-{b%100:02d}".rjust(9) for a, b in edges))
    for k in keys:
        net = S[k][0]; cells = []
        for a, b in edges:
            s = net[(net.index.year >= a) & (net.index.year <= b)]
            if len(s) < 200: cells.append("      -  "); continue
            yrs = (s.index[-1] - s.index[0]).days / 365.25
            cells.append(f"{((1+s).prod()**(1/yrs)-1)*100:8.1f}%")
        print(f"{k:26s} " + " ".join(cells))

    # 最悪日
    wd = rpx.loc[warm:].nsmallest(3)
    print("\n--- 指数の最悪日 → 3倍NAVの1日損失(日次リバランス型は1日で -100% になるのは指数 -33.3% のとき) ---")
    for d, v in wd.items():
        print(f"{d.date()} 指数 {v*100:6.1f}% → 2x {2*v*100:6.1f}%  3x {3*v*100:6.1f}%")
    n3 = S["持ちっぱなし 3x"][0]; eq3 = (1 + n3).cumprod()
    print(f"3x持ちっぱなしの最大下落 {pct(stats(n3, rf.loc[n3.index])['mdd'])}、資産が最初の10%を割った日: {(eq3 < 0.1).sum()}日 / 最小値 {eq3.min():.3f}")
    n23 = S["SMA200 3x"][0]; print(f"SMA200 3xの最大下落 {pct(stats(n23, rf.loc[n23.index])['mdd'])}")

    if show_tax:
        yrs_ = lambda n: (n.index[-1] - n.index[0]).days / 365.25
        print("\n--- 課税口座(20.315%、年内損益通算・3年繰越)の簡易モデル: 税前年率 → 税引後年率 ---")
        for k in [base, "持ちっぱなし 2x", "持ちっぱなし 3x", "SMA200 2x", "SMA200±2% 2x", "ボラ目標15% 上限2x", "SMA200+ボラ目標 上限2x"]:
            net, e = S[k]
            pre = stats(net, rf.loc[net.index])["cagr"]
            post = at_cagr(net, e, bh=k.startswith("持ちっぱなし"))
            print(f"{k:26s} {pct(pre)} → {pct(post)}   (差 {pct(post-pre)})")
    return S, len(S)


def main():
    refresh = "--refresh" in sys.argv
    validate(refresh)
    total = 0
    for name, sym, jp, tax in [("S&P500 (^GSPC)", "^GSPC", False, True), ("NASDAQ100 (^NDX)", "^NDX", False, False),
                               ("日経225 (^N225、円建て、配当なし)", "^N225", True, True)]:
        _, n = analyze_market(name, sym, refresh, jp=jp, show_tax=tax)
        total += n
    print(f"\n試した戦略数: 市場3 x {n}戦略 = {total}通り(SMA期間は200日のみ、Lは1/2/3、バンド1/2/3%、ボラ目標15%固定)。"
          "多重比較: これらは全件を表示しており、良い設定だけを選んでいない。")
    one = fetch("1570.T", refresh)
    px = one["c"].iloc[-1]
    print(f"\n1570.T 最新終値 {px:,.0f}円({one.index[-1].date()})。1口={px:,.0f}円 → 1万円で{int(10000//px)}口買える(売買単位1口の場合)")
    for s in ("2558.T", "1321.T", "1568.T", "1579.T"):
        try:
            o = fetch(s, refresh); print(f"{s} 最新終値 {o['c'].iloc[-1]:,.0f}円")
        except Exception as ex:
            print(s, "取得失敗", ex)


if __name__ == "__main__":
    main()
