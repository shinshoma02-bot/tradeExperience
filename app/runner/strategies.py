"""Pine版(pine/*.pine)と同じ3戦略。確定足だけを見て、最後の足で出たシグナルを返す。

返り値: "buy" / "sell" / "close" / None
 - buy/sell は反対ポジションを持っていれば呼び出し側が先に決済する(Pineのstrategy.entryと同じ)
"""
from app.runner.data import Candle


def sma(vals: list[float], n: int, end: int) -> float:
    return sum(vals[end - n + 1 : end + 1]) / n


def rsi_series(closes: list[float], n: int) -> list[float | None]:
    """Wilder方式のRSI。"""
    out: list[float | None] = [None] * len(closes)
    if len(closes) <= n:
        return out
    gains = [max(closes[i] - closes[i - 1], 0) for i in range(1, len(closes))]
    losses = [max(closes[i - 1] - closes[i], 0) for i in range(1, len(closes))]
    ag, al = sum(gains[:n]) / n, sum(losses[:n]) / n
    for i in range(n, len(closes)):
        if i > n:
            ag = (ag * (n - 1) + gains[i - 1]) / n
            al = (al * (n - 1) + losses[i - 1]) / n
        out[i] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    return out


def sma_cross(c: list[Candle], position: float, fast: int = 20, slow: int = 50) -> str | None:
    closes = [x.close for x in c]
    if len(closes) < slow + 1:
        return None
    i = len(closes) - 1
    f0, s0 = sma(closes, fast, i - 1), sma(closes, slow, i - 1)
    f1, s1 = sma(closes, fast, i), sma(closes, slow, i)
    if f0 <= s0 and f1 > s1:
        return "buy"
    if f0 >= s0 and f1 < s1:
        return "sell"
    return None


def rsi_reversion(c: list[Candle], position: float, n: int = 14, lo: int = 30, hi: int = 70) -> str | None:
    r = rsi_series([x.close for x in c], n)
    if len(r) < 2 or r[-1] is None or r[-2] is None:
        return None
    prev, cur = r[-2], r[-1]
    if prev <= lo < cur:
        return "buy"
    if prev >= hi > cur:
        return "sell"
    if position > 0 and prev <= 50 < cur:
        return "close"
    if position < 0 and prev >= 50 > cur:
        return "close"
    return None


def donchian_breakout(c: list[Candle], position: float, n: int = 20) -> str | None:
    if len(c) < n + 1:
        return None
    window = c[-n - 1 : -1]  # 直前n本(最新足は含めない)
    if c[-1].close > max(x.high for x in window):
        return "buy"
    if c[-1].close < min(x.low for x in window):
        return "sell"
    return None


STRATEGIES = {
    "sma_cross": sma_cross,
    "rsi_reversion": rsi_reversion,
    "donchian_breakout": donchian_breakout,
}
