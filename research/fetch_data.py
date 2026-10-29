"""Yahoo Finance から日足を取って research/data/ に保存する(研究用。リポジトリには含めない)。
  python research/fetch_data.py      → daily.json(10年・個別銘柄)と daily15.json(15年・ETF/調整後終値)
"""
import json, os, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
H = {"User-Agent": "Mozilla/5.0"}


def get(sym, rng):
    u = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval=1d&range={rng}"
    res = json.load(urllib.request.urlopen(urllib.request.Request(u, headers=H), timeout=25))["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose") or q["close"]
    return res["timestamp"], q, adj


def save(name, syms, rng, with_adj):
    out = {}
    for s in syms:
        ts, q, adj = get(s, rng)
        rows = [(t, h, l, c, a) if with_adj else (t, h, l, c)
                for t, h, l, c, a in zip(ts, q["high"], q["low"], q["close"], adj) if None not in (h, l, c)]
        out[s] = rows
    json.dump(out, open(os.path.join(HERE, "data", name), "w"))


save("daily.json", ["BTC-USD", "ETH-USD", "USDJPY=X", "EURUSD=X", "AAPL", "NVDA", "SPY", "7203.T", "1306.T", "6758.T"], "10y", False)
save("daily15.json", ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "VNQ", "XLK", "XLV", "XLE", "XLF",
                      "XLU", "XLP", "BTC-USD", "ETH-USD", "1306.T", "1321.T", "USDJPY=X", "SHY", "VT", "AGG"], "15y", True)
