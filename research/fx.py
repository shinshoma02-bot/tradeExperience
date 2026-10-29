import json
import os

import numpy as np
import pandas as pd

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data") + os.sep
raw = json.load(open(D + "fx15.json"))
dfs = {}
for s, rows in raw.items():
    d = pd.DataFrame(rows, columns=["t", "h", "l", "c"]); d.index = pd.to_datetime(d.t, unit="s").dt.normalize()
    dfs[s] = d[~d.index.duplicated()].drop(columns="t")
MAJ = ["USDJPY=X","EURJPY=X","GBPJPY=X","AUDJPY=X","NZDJPY=X","CADJPY=X","CHFJPY=X","EURUSD=X","GBPUSD=X","AUDUSD=X","NZDUSD=X","USDCAD=X","USDCHF=X"]
HY = ["MXNJPY=X","ZARJPY=X","TRYJPY=X"]
COST = 0.0003   # 売買ごとの片道コスト(スプレッド想定、価格の0.03%)
SPLIT = pd.Timestamp("2019-01-01")

def positions(sym, kind):
    d = dfs[sym]; c, h, l = d.c, d.h, d.l; n = len(c); pos = np.zeros(n)
    if kind.startswith("sma"):
        f, s = (50, 100) if kind == "sma" else (20, 50)
        up = (c.rolling(f).mean() > c.rolling(s).mean()).values; ok = c.rolling(s).mean().notna().values
        pos = np.where(ok, np.where(up, 1, -1), 0).astype(float)
    elif kind == "donch":
        hi, lo = h.rolling(55).max().shift(1), l.rolling(55).min().shift(1)
        xh, xl = h.rolling(20).max().shift(1), l.rolling(20).min().shift(1)
        p = 0
        for i in range(n):
            if p == 0:
                if c.iloc[i] > hi.iloc[i]: p = 1
                elif c.iloc[i] < lo.iloc[i]: p = -1
            elif p == 1 and c.iloc[i] < xl.iloc[i]: p = 0
            elif p == -1 and c.iloc[i] > xh.iloc[i]: p = 0
            pos[i] = p
    elif kind == "tsmom":   # 12か月リターンの向きに、月次で入れ替え
        r = c / c.shift(252) - 1
        m = pd.Series(np.sign(r.values), index=c.index)
        me = c.groupby([c.index.year, c.index.month]).transform("size")  # dummy
        last = c.index.to_series().groupby([c.index.year, c.index.month]).transform("max")
        sig = m.where(c.index == last).ffill().fillna(0)
        pos = sig.values.astype(float)
    elif kind == "rsi":     # 逆張り(両方向): RSI<30で買い→50で決済、RSI>70で売り→50で決済
        ch = c.diff(); g = ch.clip(lower=0).ewm(alpha=1/14, adjust=False).mean(); ls = (-ch.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        r = (100 - 100 / (1 + g / ls)).values; p = 0
        for i in range(n):
            if np.isnan(r[i]): pos[i] = 0; continue
            if p == 0:
                p = 1 if r[i] < 30 else (-1 if r[i] > 70 else 0)
            elif p == 1 and r[i] > 50: p = 0
            elif p == -1 and r[i] < 50: p = 0
            pos[i] = p
    elif kind == "long":
        pos = np.ones(n)
    return pd.Series(pos, index=c.index)

def pair_returns(sym, kind):
    d = dfs[sym]; pos = positions(sym, kind).shift(1).fillna(0)
    r = d.c.pct_change().fillna(0)
    return pos * r - pos.diff().abs().fillna(0) * COST

def stats(r):
    r = r.dropna(); eq = (1 + r).cumprod(); yrs = len(r) / 252
    return dict(cagr=(eq.iloc[-1] ** (1 / yrs) - 1) * 100, dd=(eq / eq.cummax() - 1).min() * 100,
                sharpe=(r.mean() * 252) / (r.std() * np.sqrt(252)) if r.std() else 0)

def run(universe, kind, start="2012-06-01"):
    P = pd.DataFrame({s: pair_returns(s, kind) for s in universe}).fillna(0)
    port = P.mean(axis=1); port = port[port.index >= start]
    return port, P[P.index >= start]

if __name__ == "__main__":
    for label, uni in (("主要13通貨ペア(1倍・レバレッジなし)", MAJ), ("円との高金利通貨3ペア(MXN/ZAR/TRY, 価格要因のみ)", HY)):
        print("\n##", label)
        print(f"{'戦略':26}{'全期間 CAGR/DD/Sharpe':>26}{'前半':>22}{'後半':>22}  後半プラスの通貨数")
        for name, kind in [("ロングのみ持ちっぱなし","long"),("SMA 50/100(売買両方向)","sma"),("SMA 20/50","sma2"),("ドンチャン55/20","donch"),("12か月モメンタム月次","tsmom"),("RSI逆張り(両方向)","rsi")]:
            port, P = run(uni, kind)
            a, b, c = stats(port), stats(port[port.index < SPLIT]), stats(port[port.index >= SPLIT])
            pos = sum(stats(P[s][P.index >= SPLIT])["cagr"] > 0 for s in uni)
            f = lambda x: f"{x['cagr']:6.1f} {x['dd']:5.0f} {x['sharpe']:5.2f}"
            print(f"{name:24}{f(a):>26}{f(b):>22}{f(c):>22}  {pos}/{len(uni)}")
    # 個別: USDJPYだけ
    print("\n## USDJPY だけ")
    for name, kind in [("持ちっぱなし","long"),("SMA 50/100","sma"),("ドンチャン55/20","donch")]:
        r = pair_returns("USDJPY=X", kind); r = r[r.index >= "2012-06-01"]
        a, b, c = stats(r), stats(r[r.index < SPLIT]), stats(r[r.index >= SPLIT])
        print(f"{name:14} 全期間 {a['cagr']:5.1f}%/{a['dd']:4.0f}%/{a['sharpe']:.2f}  前半 {b['cagr']:5.1f}%  後半 {c['cagr']:5.1f}%")
