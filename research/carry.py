import numpy as np, pandas as pd
from fx import dfs, SPLIT, stats, COST

import os

rates = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "bis_policy.csv"))
rates["d"] = pd.to_datetime(rates.TIME_PERIOD + "-01")
RT = rates.pivot(index="d", columns="REF_AREA", values="OBS_VALUE").sort_index()
AREA = {"USD": "US", "EUR": "XM", "GBP": "GB", "AUD": "AU", "NZD": "NZ", "CAD": "CA", "CHF": "CH", "MXN": "MX", "ZAR": "ZA", "TRY": "TR", "JPY": "JP"}
idx = dfs["USDJPY=X"].index
R = RT.reindex(idx.union(RT.index)).ffill().reindex(idx) / 100.0   # 日次の年率金利

def rate(ccy): return R[AREA[ccy]]
JPY_CROSS = {"USD": "USDJPY=X", "EUR": "EURJPY=X", "GBP": "GBPJPY=X", "AUD": "AUDJPY=X", "NZD": "NZDJPY=X", "CAD": "CADJPY=X", "CHF": "CHFJPY=X",
             "MXN": "MXNJPY=X", "ZAR": "ZARJPY=X", "TRY": "TRYJPY=X"}
SWAP_HAIRCUT = 0.005   # 業者のスワップ上乗せ分(年率0.5%を差し引く想定)

def price_ret(ccy): return dfs[JPY_CROSS[ccy]].c.pct_change().reindex(idx).fillna(0)
def carry_daily(ccy): return (rate(ccy) - rate("JPY")) / 365.0          # 買いで受け取る(売りなら逆)

def month_last(index):
    s = index.to_series(); return set(s.groupby([index.year, index.month]).max().values)

def run(universe, n_long=3, n_short=0, trend_filter=None, start="2012-06-01", cost=COST):
    me = month_last(idx); w = pd.Series(0.0, index=universe); rets = []
    PR = pd.DataFrame({c: price_ret(c) for c in universe}); CR = pd.DataFrame({c: carry_daily(c) for c in universe})
    C = {c: dfs[JPY_CROSS[c]].c for c in universe}
    ds = [d for d in idx if d >= pd.Timestamp(start)]
    for i, d in enumerate(ds[:-1]):
        nxt = ds[i + 1]
        if d in me:
            car = pd.Series({c: rate(c).loc[d] - rate("JPY").loc[d] for c in universe}).sort_values(ascending=False)
            nw = pd.Series(0.0, index=universe)
            longs = list(car.index[:n_long]); shorts = list(car.index[len(car) - n_short:]) if n_short else []
            if trend_filter:
                longs = [c for c in longs if C[c].loc[:d].iloc[-trend_filter:].mean() < C[c].loc[d]]   # 平均線より上の通貨だけ
            for c in longs: nw[c] += 1.0 / max(n_long, 1)
            for c in shorts: nw[c] -= 1.0 / n_short
            turn = (nw - w).abs().sum(); w = nw; c_ = turn * cost
        else: c_ = 0.0
        pr = (w * PR.loc[nxt]).sum()
        cr = (w * CR.loc[nxt]).sum() - (w.abs().sum() * SWAP_HAIRCUT / 365.0)
        rets.append((nxt, pr + cr - c_))
    s = pd.Series(dict(rets)); return s

def summarize(name, r):
    a, b, c = stats(r), stats(r[r.index < SPLIT]), stats(r[r.index >= SPLIT])
    f = lambda x: f"{x['cagr']:6.1f} {x['dd']:5.0f} {x['sharpe']:5.2f}"
    print(f"{name:38}{f(a):>22}{f(b):>20}{f(c):>20}")

G10 = ["USD", "EUR", "GBP", "AUD", "NZD", "CAD", "CHF"]
ALL = G10 + ["MXN", "ZAR", "TRY"]
if __name__ == "__main__":
    print(f"{'戦略(円に対して、1倍)':36}{'全期間 CAGR/DD/Sh':>24}{'前半':>20}{'後半':>20}")
    # 比較用: 価格だけ/スワップだけの内訳
    allp = pd.DataFrame({c: price_ret(c) for c in G10}).mean(axis=1); allp = allp[allp.index >= "2012-06-01"]
    summarize("G7通貨を円で等分(価格のみ)", allp)
    alls = pd.DataFrame({c: carry_daily(c) for c in G10}).mean(axis=1); alls = alls[alls.index >= "2012-06-01"]
    summarize("  同(スワップのみ・参考)", alls)
    summarize("G7等分 保有(価格+スワップ)", allp + alls - SWAP_HAIRCUT / 365)
    summarize("キャリー上位3(G7)", run(G10, 3))
    summarize("キャリー上位2(G7)", run(G10, 2))
    summarize("キャリー上位3+100日線フィルタ(G7)", run(G10, 3, trend_filter=100))
    summarize("キャリー上位1(G7)", run(G10, 1))
    summarize("キャリー上位3(新興含む10通貨)", run(ALL, 3))
    summarize("キャリー上位3+100日線(新興含む)", run(ALL, 3, trend_filter=100))
    summarize("買い上位3/売り下位3(G7)", run(G10, 3, 3))
    summarize("買い上位2/売り下位2(G7)", run(G10, 2, 2))
