import json
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class Candle:
    ts: int
    high: float
    low: float
    close: float


def fetch_kraken(pair: str, interval_min: int) -> list[Candle]:
    """Kraken公開API(キー不要)から足データを取る。最後の1本は未確定なので捨てる。"""
    url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval={interval_min}"
    with urllib.request.urlopen(url, timeout=15) as r:
        body = json.load(r)
    if body.get("error"):
        raise RuntimeError(f"Kraken error: {body['error']}")
    rows = next(v for k, v in body["result"].items() if k != "last")
    return [Candle(int(r[0]), float(r[2]), float(r[3]), float(r[4])) for r in rows[:-1]]
