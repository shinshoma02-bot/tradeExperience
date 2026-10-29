import json, itertools
import numpy as np, pandas as pd

import os

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data") + os.sep
raw = json.load(open(D + "daily.json"))
COST = {"BTC-USD": .002, "ETH-USD": .002, "USDJPY=X": .0002, "EURUSD=X": .0002,
        "AAPL": .001, "NVDA": .001, "SPY": .001, "7203.T": .001, "1306.T": .001, "6758.T": .001}
SHORTABLE = {"USDJPY=X", "EURUSD=X"}
df = {}
for s, rows in raw.items():
    d = pd.DataFrame(rows, columns=["t", "h", "l", "c"])
    d.index = pd.to_datetime(d.t, unit="s").dt.normalize()
    df[s] = d[~d.index.duplicated()].drop(columns="t")

def rsi(c, n):
    ch = c.diff(); g = ch.clip(lower=0); l = -ch.clip(upper=0)
    ag = g.ewm(alpha=1/n, adjust=False).mean(); al = l.ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100/(1 + ag/al)

def state_machine(entry, exit_, short_entry=None, short_exit=None):
    pos = np.zeros(len(entry)); p = 0
    for i in range(len(entry)):
        if p == 0:
            if entry[i]: p = 1
            elif short_entry is not None and short_entry[i]: p = -1
        elif p == 1 and exit_[i]: p = 0
        elif p == -1 and short_exit is not None and short_exit[i]: p = 0
        pos[i] = p
    return pos

def strat(name, d, sym, **k):
    c, h, l = d.c, d.h, d.l; sh = sym in SHORTABLE
    if name == "sma":
        f, s = c.rolling(k["f"]).mean(), c.rolling(k["s"]).mean()
        up, dn = (f > s).values, (f < s).values
        return state_machine(up, dn, dn if sh else None, up if sh else None)
    if name == "donch":
        n, m = k["n"], k["m"]
        hi, lo = h.rolling(n).max().shift(1), l.rolling(n).min().shift(1)
        xl, xh = l.rolling(m).min().shift(1), h.rolling(m).max().shift(1)
        e, x = (c > hi).values, (c < xl).values
        se, sx = (c < lo).values, (c > xh).values
        return state_machine(e, x, se if sh else None, sx if sh else None)
    if name == "rsi":
        r = rsi(c, k["n"]); trend = (c > c.rolling(k["tf"]).mean()).values if k.get("tf") else np.ones(len(c), bool)
        e = ((r < k["lo"]).values) & trend
        x = (r > k["ex"]).values
        return state_machine(e, x)
    if name == "regime":
        m = c.rolling(k["n"]).mean()
        up = (c > m).values
        return state_machine(up, ~up)
    raise ValueError(name)

def returns(sym, pos, d):
    r = d.c.pct_change().fillna(0).values
    p = np.roll(pos, 1); p[0] = 0                     # 当日終値で決めた建玉は翌日から効く
    turn = np.abs(np.diff(np.r_[0, p]))
    return pd.Series(p * r - turn * COST[sym], index=d.index), int((turn > 0).sum())

def stats(r):
    eq = (1 + r).cumprod()
    return dict(ret=(eq.iloc[-1] - 1) * 100, dd=((eq / eq.cummax()) - 1).min() * 100)

SPLIT = pd.Timestamp("2022-10-29")
def portfolio(name, syms, **k):
    rs, trades = {}, 0
    for s in syms:
        d = df[s]
        r, n = returns(s, strat(name, d, s, **k), d); rs[s] = r; trades += n
    return pd.DataFrame(rs).fillna(0), trades

def report(name, syms, **k):
    P, trades = portfolio(name, syms, **k)
    port = P.mean(axis=1)
    a, b = port[port.index < SPLIT], port[port.index >= SPLIT]
    pos_oos = sum(stats(P[s][P.index >= SPLIT])["ret"] > 0 for s in syms)
    return stats(a), stats(b), trades, pos_oos

SYMS = ["BTC-USD", "ETH-USD", "USDJPY=X", "AAPL", "NVDA", "7203.T"]   # 1万円で現実に分散できる候補
if __name__ == "__main__":
    # 比較用: 持ちっぱなし
    bh = pd.DataFrame({s: df[s].c.pct_change().fillna(0) for s in SYMS}).fillna(0).mean(axis=1)
    print("buy&hold   IS ret %.0f dd %.0f | OOS ret %.0f dd %.0f" % (*stats(bh[bh.index < SPLIT]).values(), *stats(bh[bh.index >= SPLIT]).values()))
    grids = {
        "sma": [dict(f=20, s=50), dict(f=50, s=100), dict(f=50, s=200), dict(f=100, s=200)],
        "donch": [dict(n=20, m=10), dict(n=55, m=20), dict(n=100, m=50)],
        "rsi": [dict(n=14, lo=30, ex=50, tf=0), dict(n=14, lo=30, ex=50, tf=200), dict(n=2, lo=10, ex=60, tf=200), dict(n=14, lo=40, ex=60, tf=200)],
        "regime": [dict(n=100), dict(n=150), dict(n=200)],
    }
    print(f"{'strategy':28}{'IS ret':>8}{'IS dd':>7}{'OOS ret':>9}{'OOS dd':>8}{'trades':>7}{'OOS+':>6}")
    for name, ks in grids.items():
        for k in ks:
            a, b, t, p = report(name, SYMS, **k)
            print(f"{name+str(list(k.values())):28}{a['ret']:8.0f}{a['dd']:7.0f}{b['ret']:9.0f}{b['dd']:8.0f}{t:7}{p:>4}/{len(SYMS)}")
