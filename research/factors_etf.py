"""因子ETF(米国・日本)の実績比較。Yahoo Finance の調整後終値(配当込み)で月次リターンを作り、
基準(SPY / MAXIS TOPIX)に対する超過リターンと、回帰のα(t値)を見る。実行: python research/factors_etf.py
注意: Yahoo の調整後終値は、1306.T / 1321.T / 1489.T で分割処理が壊れている(日次-90%や負の値)。
      日本の基準には MAXIS TOPIX(1348.T)を使い、壊れている銘柄は使わない。
"""
import urllib.request, json

import numpy as np
import pandas as pd

H = {"User-Agent": "Mozilla/5.0"}
US = ["SPY", "RSP", "USMV", "SPLV", "MTUM", "QUAL", "VLUE", "VTV", "IWD", "SCHD", "VIG", "DGRO", "VYM", "IJR", "IWN", "VBR", "IJS", "VB", "EFV", "VEA", "VWO"]
JP = ["1348.T", "1308.T", "1577.T", "1478.T", "1698.T", "1494.T", "1651.T", "2529.T", "1593.T", "1599.T", "2564.T"]


def adj(sym):
    u = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval=1d&range=15y"
    res = json.load(urllib.request.urlopen(urllib.request.Request(u, headers=H), timeout=20))["chart"]["result"][0]
    a = res["indicators"]["adjclose"][0]["adjclose"]
    return pd.Series({pd.Timestamp(t, unit="s").normalize(): v for t, v in zip(res["timestamp"], a) if v})


def stats(r, b):
    x = pd.concat([r, b], axis=1).dropna(); x.columns = ["r", "b"]; n = len(x)
    cagr = (1 + x.r).prod() ** (12 / n) - 1; bc = (1 + x.b).prod() ** (12 / n) - 1
    eq = (1 + x.r).cumprod(); beta = np.cov(x.r, x.b)[0, 1] / x.b.var(); res = x.r - beta * x.b
    return dict(cagr=cagr * 100, excess=(cagr - bc) * 100, dd=((eq / eq.cummax()) - 1).min() * 100, beta=beta,
                alpha=res.mean() * 12 * 100, t=res.mean() / (res.std() / np.sqrt(n)) if res.std() > 0 else float("nan"), n=n)


for title, names, bench in (("米国", US, "SPY"), ("日本", JP, "1348.T")):
    M = pd.DataFrame({s: adj(s) for s in names}).sort_index().resample("ME").last().pct_change()
    print(f"\n## {title}(基準 {bench})")
    for s in names:
        r = M[s].loc["2012-01":]
        if r.dropna().shape[0] >= 36:
            st = stats(r, M[bench].loc["2012-01":])
            print(f"{s:8} 年率{st['cagr']:5.1f}% 対基準{st['excess']:+5.1f}% 最大DD{st['dd']:4.0f}% β{st['beta']:.2f} α{st['alpha']:+5.1f}%(t={st['t']:.1f}) {st['n']}か月")
