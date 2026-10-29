import json
import numpy as np, pandas as pd

import os

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data") + os.sep
raw = json.load(open(D + "daily15.json"))
px = {}
for s, rows in raw.items():
    d = pd.DataFrame(rows, columns=["t", "h", "l", "c", "a"])
    d.index = pd.to_datetime(d.t, unit="s").dt.normalize()
    px[s] = d[~d.index.duplicated()].a
P = pd.DataFrame(px).sort_index()
R = P.pct_change()
COST = 0.001   # 売買回転1あたり(往復ではなく、売買した金額に対して)の手数料+スプレッド
CASH = "SHY"

def month_ends(idx):
    s = pd.Series(idx, index=idx)
    return set(s.groupby([idx.year, idx.month]).max().values)

def backtest(weights_fn, start="2012-01-01", cost=COST, universe=None):
    """月末に weights_fn(日付) で目標配分を決め、翌営業日から適用。回転率に手数料。"""
    idx = P.index[P.index >= start]
    me = month_ends(idx)
    w = pd.Series(0.0, index=P.columns)
    rets, turn_tot = [], 0.0
    for i, d in enumerate(idx[:-1]):
        nxt = idx[i + 1]
        if d in me:
            nw = weights_fn(d).reindex(P.columns).fillna(0.0)
            turn = (nw - w).abs().sum()
            turn_tot += turn
            w = nw
            c = turn * cost
        else:
            c = 0.0
        r = (w * R.loc[nxt].fillna(0)).sum() - c
        rets.append((nxt, r))
        # ドリフト(配分のずれ)は簡略化して無視せず反映
        g = w * (1 + R.loc[nxt].fillna(0)); tot = g.sum()
        w = g / tot if tot > 0 else w
    return pd.Series(dict(rets)), turn_tot

def stats(r):
    r = r.dropna()
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(252)
    dd = (eq / eq.cummax() - 1).min()
    return dict(cagr=cagr * 100, vol=vol * 100, sharpe=(r.mean() * 252) / (r.std() * np.sqrt(252)) if r.std() else 0, dd=dd * 100)

def mom(d, n_days):
    i = P.index.get_loc(d)
    if i < n_days: return None
    return P.iloc[i] / P.iloc[i - n_days] - 1

def sma_above(d, n):
    i = P.index.get_loc(d)
    return P.iloc[i] > P.iloc[i - n + 1 : i + 1].mean()

U1 = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "VNQ"]
SECT = ["XLK", "XLV", "XLE", "XLF", "XLU", "XLP"]
def only(names, wts=None):
    s = pd.Series(0.0, index=P.columns)
    for k, n in enumerate(names): s[n] = (wts[k] if wts is not None else 1 / len(names))
    return s

def bh(sym): return lambda d: only([sym])
def eqw(names): return lambda d: only(names)
def trend(sym, n=210): return lambda d: only([sym]) if sma_above(d, n)[sym] else only([CASH])
def gtaa(names, n=210):
    def f(d):
        ab = sma_above(d, n); w = pd.Series(0.0, index=P.columns)
        for x in names: w[x if ab[x] else CASH] += 1 / len(names)
        return w
    return f
def dual_mom(names, k=3, lb=252):
    def f(d):
        m = mom(d, lb)
        if m is None: return only([CASH])
        m = m[names].dropna().sort_values(ascending=False)
        pick = [x for x in m.index[:k] if m[x] > 0]
        w = pd.Series(0.0, index=P.columns)
        for x in pick: w[x] += 1 / k
        w[CASH] += 1 - len(pick) / k
        return w
    return f
def mom_blend(names, k=3):
    def f(d):
        ms = [mom(d, n) for n in (63, 126, 252)]
        if any(m is None for m in ms): return only([CASH])
        m = sum(ms[j][names] for j in range(3)) / 3
        m = m.dropna().sort_values(ascending=False)
        pick = [x for x in m.index[:k] if m[x] > 0]
        w = pd.Series(0.0, index=P.columns)
        for x in pick: w[x] += 1 / k
        w[CASH] += 1 - len(pick) / k
        return w
    return f
def inv_vol(names, n=60):
    def f(d):
        i = P.index.get_loc(d)
        if i < n: return only([CASH])
        v = R.iloc[i - n + 1 : i + 1][names].std()
        w = (1 / v) / (1 / v).sum()
        return only(list(w.index), list(w.values))
    return f
def voltarget(sym, tgt=0.10, n=20):
    def f(d):
        i = P.index.get_loc(d)
        if i < n: return only([CASH])
        v = R.iloc[i - n + 1 : i + 1][sym].std() * np.sqrt(252)
        x = min(1.0, tgt / v) if v > 0 else 1.0
        w = pd.Series(0.0, index=P.columns); w[sym] = x; w[CASH] = 1 - x
        return w
    return f
def crypto_trend(n=200):
    def f(d):
        ab = sma_above(d, n); names = [x for x in ["BTC-USD", "ETH-USD"] if not np.isnan(P.loc[d, x])]
        w = pd.Series(0.0, index=P.columns)
        for x in names: w[x if ab[x] else CASH] += 1 / len(names)
        return w
    return f

SPLIT = pd.Timestamp("2019-01-01")
CANDS = {
    "SPY 持ちっぱなし": bh("SPY"),
    "米国+世界株 VT 持ちっぱなし": bh("VT"),
    "10資産 等分(月次再配分)": eqw(U1),
    "10資産 逆ボラ配分": inv_vol(U1),
    "SPY 10か月線フィルタ": trend("SPY"),
    "QQQ 10か月線フィルタ": trend("QQQ"),
    "GTAA(10資産、各々10か月線)": gtaa(U1),
    "デュアルモメンタム(12か月,上位3)": dual_mom(U1, 3, 252),
    "モメンタム3-6-12か月平均(上位3)": mom_blend(U1, 3),
    "セクター上位3(6業種,12か月)": dual_mom(SECT, 3, 252),
    "SPY ボラ目標10%": voltarget("SPY"),
    "暗号資産BTC+ETH 200日線": crypto_trend(200),
}
if __name__ == "__main__":
    print(f"{'戦略':32}{'全期間':>20}{'前半(~2018)':>22}{'後半(2019~)':>22}   回転")
    print(f"{'':32}{'CAGR%  DD%  Sharpe':>20}{'CAGR%  DD%  Sharpe':>22}{'CAGR%  DD%  Sharpe':>22}")
    for name, fn in CANDS.items():
        start = "2016-01-01" if "暗号" in name else "2012-01-01"
        r, t = backtest(fn, start=start)
        a, b, c = stats(r), stats(r[r.index < SPLIT]) if (r.index < SPLIT).sum() > 250 else None, stats(r[r.index >= SPLIT])
        f = lambda x: "      -      " if x is None else f"{x['cagr']:5.1f} {x['dd']:5.0f} {x['sharpe']:5.2f}"
        print(f"{name:30}{f(a):>22}{f(b):>22}{f(c):>22}  {t/(len(r)/252):4.1f}/年")
