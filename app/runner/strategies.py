"""日足向けの3戦略(pine/*.pine は旧1時間足版のまま)。確定足だけを見て、最後の足で出たシグナルを返す。

返り値: "buy" / "sell" / "close" / None。保有(position)も見て判断する。
 - 現物の市場(暗号資産・株)では、売りシグナルは「持っていれば決済」だけになる(plan_orders 側で処理)
 - パラメータは過去10年の日足で検証した一般的な値(SMA 50/100、タートル 55/20、RSI 14/30)。
   検証結果の中で一番良く見えた値を選ぶと過去に合わせただけになるので、あえて定番の値にしている
"""
from app.runner.data import Candle

SMA_FAST, SMA_SLOW = 50, 100
DONCH_ENTRY, DONCH_EXIT = 55, 20
RSI_N, RSI_LO, RSI_EXIT = 14, 30, 50


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


def sma_cross(c: list[Candle], position: float) -> str | None:
    """短期線が長期線の上なら買い、下なら売り(トレンドに乗り、反転で降りる)。"""
    closes = [x.close for x in c]
    if len(closes) < SMA_SLOW:
        return None
    i = len(closes) - 1
    fast, slow = sma(closes, SMA_FAST, i), sma(closes, SMA_SLOW, i)
    if fast > slow:
        return "buy" if position <= 0 else None
    if fast < slow:
        return "sell" if position >= 0 else None
    return None


def donchian_breakout(c: list[Candle], position: float) -> str | None:
    """55日高値を抜けたら買い、20日安値を割ったら降りる(タートル流)。下抜けは売りから入れる市場だけ売り。"""
    if len(c) < DONCH_ENTRY + 1:
        return None
    close = c[-1].close
    entry, exit_ = c[-DONCH_ENTRY - 1 : -1], c[-DONCH_EXIT - 1 : -1]
    if position == 0:
        if close > max(x.high for x in entry):
            return "buy"
        if close < min(x.low for x in entry):
            return "sell"
    elif position > 0 and close < min(x.low for x in exit_):
        return "close"
    elif position < 0 and close > max(x.high for x in exit_):
        return "close"
    return None


def rsi_reversion(c: list[Candle], position: float) -> str | None:
    """売られすぎ(RSI<30)で買って、中立(RSI>50)まで戻したら決済する逆張り。買いのみ。"""
    r = rsi_series([x.close for x in c], RSI_N)[-1]
    if r is None:
        return None
    if position == 0 and r < RSI_LO:
        return "buy"
    if position > 0 and r > RSI_EXIT:
        return "close"
    return None


STRATEGIES = {
    "sma_cross": sma_cross,
    "rsi_reversion": rsi_reversion,
    "donchian_breakout": donchian_breakout,
}
