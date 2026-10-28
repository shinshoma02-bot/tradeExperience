import asyncio
import logging
from typing import Callable

from app.models import Signal
from app.runner.data import Candle, Instrument, fetch_candles
from app.runner.strategies import STRATEGIES

log = logging.getLogger("runner")

# 現物株は空売りしない(売りシグナルは「持っていれば決済」だけにする)
NO_SHORT_MARKETS = {"us_stock", "jp_stock"}


class StrategyRunner:
    """定期的に価格を取って戦略を計算し、シグナルを handle(webhookと同じ経路)に流す。"""

    def __init__(
        self,
        handle: Callable[[Signal], dict],
        get_position: Callable[[str, str, str], float],
        instruments: list[Instrument],
        interval_min: int,
        order_notional_usd: float,
        to_quote: Callable[[float, str], float] = lambda usd, quote: usd,
        fetch: Callable[[Instrument, int], list[Candle]] = fetch_candles,
        secret: str = "",
        alert: Callable[[str], None] = lambda msg: None,
    ) -> None:
        self.handle, self.get_position = handle, get_position
        self.instruments, self.interval_min = instruments, interval_min
        self.order_notional_usd, self.to_quote = order_notional_usd, to_quote
        self.fetch, self.secret, self.alert = fetch, secret, alert
        self._last_ts: dict[tuple[str, str, str], int] = {}
        self._fail_cycles = 0

    def run_once(self) -> list[dict]:
        results = []
        failed = 0
        for inst in self.instruments:
            try:
                candles = self.fetch(inst, self.interval_min)
            except Exception as e:  # ネットワーク不調で全体を止めない
                log.warning("fetch failed %s: %s", inst.symbol, e)
                failed += 1
                continue
            if not candles:
                continue
            last = candles[-1]
            for name, fn in STRATEGIES.items():
                key = (inst.market, name, inst.symbol)
                if key not in self._last_ts:
                    # 起動直後は最新足を処理済みにするだけ(再起動での二重発注を防ぐ)
                    self._last_ts[key] = last.ts
                    continue
                if last.ts <= self._last_ts[key]:
                    continue  # 新しい確定足がまだ無い(市場が閉まっている間もここ)
                self._last_ts[key] = last.ts
                pos = self.get_position(inst.market, name, inst.symbol)
                action = fn(candles, pos)
                if action:
                    results += self._send(inst, name, action, pos, last.close)
        # 全銘柄の取得が5サイクル連続で失敗したら1回だけ知らせる(回復したらリセット)
        self._fail_cycles = self._fail_cycles + 1 if self.instruments and failed == len(self.instruments) else 0
        if self._fail_cycles == 5:
            self.alert("価格データを取得できません(5回連続)。取引所APIの不調か、サーバーのネットワーク問題の可能性")
        return results

    def _send(self, inst: Instrument, name: str, action: str, pos: float, price: float) -> list[dict]:
        def sig(a: str, qty: float = 0.0) -> Signal:
            return Signal(secret=self.secret, market=inst.market, strategy=name, symbol=inst.symbol,
                          action=a, price=price, qty=qty, quote=inst.quote)

        if action == "close":
            return [self.handle(sig("close"))]
        out = []
        # 反対ポジションがあれば先に決済(ドテン)
        if (action == "buy" and pos < 0) or (action == "sell" and pos > 0):
            out.append(self.handle(sig("close")))
        if (action == "buy" and pos > 0) or (action == "sell" and pos < 0):
            return out  # 同方向を既に持っている
        if action == "sell" and inst.market in NO_SHORT_MARKETS:
            return out  # 株は新規の売り(空売り)をしない
        qty = self.to_quote(self.order_notional_usd, inst.quote) / price
        out.append(self.handle(sig(action, qty)))
        return out

    async def loop(self, every_sec: int) -> None:
        while True:
            try:
                for r in await asyncio.to_thread(self.run_once):
                    log.info("runner: %s", r)
            except Exception:
                log.exception("runner cycle failed")
            await asyncio.sleep(every_sec)
