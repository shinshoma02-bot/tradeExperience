import json
import time
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class Candle:
    ts: int
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class Instrument:
    market: str    # crypto / fx / us_stock / jp_stock / futures
    provider: str  # kraken / yahoo
    symbol: str    # プロバイダ側の銘柄コード
    quote: str = "USD"  # 価格の通貨(損益の換算に使う)

    @property
    def name(self) -> str:
        return self.symbol


def parse_instruments(spec: str) -> list[Instrument]:
    """'market:provider:symbol:quote' をカンマ区切りで並べた文字列を読む。quote は省略で USD。"""
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) not in (3, 4):
            raise ValueError(f"RUNNER_INSTRUMENTS の書式が不正です: {item}")
        out.append(Instrument(parts[0], parts[1], parts[2], parts[3] if len(parts) == 4 else "USD"))
    return out


def _drop_incomplete(candles: list[Candle], interval_min: int, now: float | None = None) -> list[Candle]:
    """まだ確定していない足(開始+足の長さが現在より先)を捨てる。"""
    now = time.time() if now is None else now
    return [c for c in candles if c.ts + interval_min * 60 <= now]


def fetch_kraken(symbol: str, interval_min: int) -> list[Candle]:
    url = f"https://api.kraken.com/0/public/OHLC?pair={symbol}&interval={interval_min}"
    with urllib.request.urlopen(url, timeout=15) as r:
        body = json.load(r)
    if body.get("error"):
        raise RuntimeError(f"Kraken error: {body['error']}")
    rows = next(v for k, v in body["result"].items() if k != "last")
    return [Candle(int(r[0]), float(r[2]), float(r[3]), float(r[4])) for r in rows]


def fetch_yahoo(symbol: str, interval_min: int, days: int = 30) -> list[Candle]:
    """Yahoo Finance(非公式・キー不要)。FX は USDJPY=X、日本株は 7203.T、先物は ES=F の形式。"""
    interval = "1d" if interval_min >= 1440 else f"{interval_min}m"
    if interval == "1d":
        days = max(days, 1000)  # 日足は判定に100本以上要るので長めに取る
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={interval}&range={days}d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        body = json.load(r)
    res = (body.get("chart", {}).get("result") or [None])[0]
    if not res:
        raise RuntimeError(f"Yahoo error: {body.get('chart', {}).get('error')}")
    q = res["indicators"]["quote"][0]
    out = []
    for ts, h, l, c in zip(res["timestamp"], q["high"], q["low"], q["close"]):
        if None in (h, l, c):  # 取引のない時間帯の欠損
            continue
        out.append(Candle(int(ts), float(h), float(l), float(c)))
    return out


FETCHERS = {"kraken": fetch_kraken, "yahoo": fetch_yahoo}


def fetch_candles(inst: Instrument, interval_min: int, days: int | None = None) -> list[Candle]:
    """days は Yahoo のときだけ効く(バックテストで長く取る用)。Kraken は直近720本が上限。"""
    fn = FETCHERS[inst.provider]
    raw = fn(inst.symbol, interval_min, days) if (days and inst.provider == "yahoo") else fn(inst.symbol, interval_min)
    return _drop_incomplete(raw, interval_min)
