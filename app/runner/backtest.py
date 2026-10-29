"""過去の足に3戦略を当てはめて、もし動かしていたらどうだったかを計算する。

売買ルールはランナーと同じ(戦略関数と plan_orders を共用)。ただし簡略化している:
 - 約定はシグナルが出た足の終値(スリッページなし)。手数料だけ fee_bps で引く
 - 1注文は order_notional_usd 固定
"""
from dataclasses import dataclass
from typing import Callable

from app.brokers.paper import PaperBroker
from app.models import Signal
from app.runner.data import Candle, Instrument
from app.runner.runner import plan_orders
from app.runner.strategies import STRATEGIES

WARMUP = 60    # 戦略が判定に必要な最低本数
WINDOW = 300   # 判定に渡す直近の本数(ランナーが取る本数と大きくは変わらない)


@dataclass
class Result:
    market: str
    symbol: str
    strategy: str
    bars: int
    trades: int          # 決済した回数
    win_rate: float | None
    pnl_usd: float       # 手数料込みの合計(含み損益を含む)
    fees_usd: float
    buy_hold_pct: float  # 同期間に持ちっぱなしだった場合の値動き(%)


def run_backtest(
    inst: Instrument,
    candles: list[Candle],
    strategy: str,
    order_notional_usd: float = 100.0,
    fee_bps: float = 5.0,
    to_usd: Callable[[float, str], float] = lambda x, q: x,
    to_quote: Callable[[float, str], float] = lambda x, q: x,
) -> Result | None:
    if len(candles) <= WARMUP:
        return None
    fn = STRATEGIES[strategy]
    broker = PaperBroker()
    key = f"{strategy}:{inst.symbol}"
    realized_usd = fees_usd = 0.0
    wins = trades = 0

    for i in range(WARMUP, len(candles)):
        window = candles[max(0, i - WINDOW + 1) : i + 1]
        price = window[-1].close
        pos_row = broker.positions().get(key)
        pos = pos_row["qty"] if pos_row else 0.0
        action = fn(window, pos)
        if not action:
            continue
        for o in plan_orders(inst.market, action, pos):
            qty = 0.0 if o == "close" else to_quote(order_notional_usd, inst.quote) / price
            fill = broker.execute(Signal(secret="", market=inst.market, strategy=strategy,
                                         symbol=inst.symbol, action=o, price=price, qty=qty, quote=inst.quote))
            if fill is None:
                continue
            fee = to_usd(fill.qty * fill.price, inst.quote) * fee_bps / 1e4
            fees_usd += fee
            pnl = to_usd(fill.realized_pnl, inst.quote)
            realized_usd += pnl
            if o == "close" or fill.realized_pnl != 0:
                trades += 1
                wins += 1 if pnl > 0 else 0

    unreal = 0.0
    row = broker.positions().get(key)
    if row:
        unreal = to_usd((candles[-1].close - row["avg_price"]) * row["qty"], inst.quote)
    first = candles[WARMUP].close
    return Result(
        market=inst.market, symbol=inst.symbol, strategy=strategy, bars=len(candles) - WARMUP,
        trades=trades, win_rate=round(wins / trades * 100, 1) if trades else None,
        pnl_usd=round(realized_usd + unreal - fees_usd, 4), fees_usd=round(fees_usd, 4),
        buy_hold_pct=round((candles[-1].close / first - 1) * 100, 2),
    )


def run_all(instruments: list[Instrument], fetch: Callable[[Instrument], list[Candle]], **kw) -> dict:
    results, errors = [], []
    for inst in instruments:
        try:
            candles = fetch(inst)
        except Exception as e:
            errors.append({"symbol": inst.symbol, "error": str(e)})
            continue
        for name in STRATEGIES:
            r = run_backtest(inst, candles, name, **kw)
            if r:
                results.append(r.__dict__)
    by_strategy: dict[str, dict] = {}
    for r in results:
        b = by_strategy.setdefault(r["strategy"], {"pnl_usd": 0.0, "trades": 0, "runs": 0})
        b["pnl_usd"] = round(b["pnl_usd"] + r["pnl_usd"], 4)
        b["trades"] += r["trades"]
        b["runs"] += 1
    results.sort(key=lambda r: r["pnl_usd"], reverse=True)
    return {"results": results, "by_strategy": by_strategy, "errors": errors}
