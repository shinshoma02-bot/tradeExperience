import asyncio
import logging
from typing import Callable

from app.models import Signal
from app.runner.data import Candle, fetch_kraken
from app.runner.strategies import STRATEGIES

log = logging.getLogger("runner")


class StrategyRunner:
    """定期的に価格を取って戦略を計算し、シグナルを handle(webhookと同じ経路)に流す。"""

    def __init__(
        self,
        handle: Callable[[Signal], dict],
        get_position: Callable[[str, str], float],
        symbols: list[str],
        interval_min: int,
        order_notional: float,
        fetch: Callable[[str, int], list[Candle]] = fetch_kraken,
        secret: str = "",
        alert: Callable[[str], None] = lambda msg: None,
    ) -> None:
        self.handle, self.get_position = handle, get_position
        self.symbols, self.interval_min = symbols, interval_min
        self.order_notional, self.fetch, self.secret = order_notional, fetch, secret
        self._last_ts: dict[tuple[str, str], int] = {}
        self.alert = alert
        self._fail_cycles = 0

    def run_once(self) -> list[dict]:
        results = []
        failed = 0
        for symbol in self.symbols:
            try:
                candles = self.fetch(symbol, self.interval_min)
            except Exception as e:  # ネットワーク不調で全体を止めない
                log.warning("fetch failed %s: %s", symbol, e)
                failed += 1
                continue
            if not candles:
                continue
            last = candles[-1]
            for name, fn in STRATEGIES.items():
                key = (name, symbol)
                if key not in self._last_ts:
                    # 起動直後は最新足を処理済みにするだけ(再起動での二重発注を防ぐ)
                    self._last_ts[key] = last.ts
                    continue
                if last.ts <= self._last_ts[key]:
                    continue  # 新しい確定足がまだ無い
                self._last_ts[key] = last.ts
                pos = self.get_position(name, symbol)
                action = fn(candles, pos)
                if action:
                    results += self._send(name, symbol, action, pos, last.close)
        # 全銘柄の取得が5サイクル連続で失敗したら1回だけ知らせる(回復したらリセット)
        self._fail_cycles = self._fail_cycles + 1 if self.symbols and failed == len(self.symbols) else 0
        if self._fail_cycles == 5:
            self.alert("価格データを取得できません(5回連続)。取引所APIの不調か、サーバーのネットワーク問題の可能性")
        return results

    def _send(self, name: str, symbol: str, action: str, pos: float, price: float) -> list[dict]:
        def sig(a: str, qty: float = 0.0) -> Signal:
            return Signal(secret=self.secret, market="crypto", strategy=name, symbol=symbol,
                          action=a, price=price, qty=qty)

        out = []
        if action == "close":
            return [self.handle(sig("close"))]
        # 反対ポジションがあれば先に決済(ドテン)
        if (action == "buy" and pos < 0) or (action == "sell" and pos > 0):
            out.append(self.handle(sig("close")))
        if (action == "buy" and pos > 0) or (action == "sell" and pos < 0):
            return out  # 同方向を既に持っている
        out.append(self.handle(sig(action, self.order_notional / price)))
        return out

    async def loop(self, every_sec: int) -> None:
        while True:
            try:
                for r in await asyncio.to_thread(self.run_once):
                    log.info("runner: %s", r)
            except Exception:
                log.exception("runner cycle failed")
            await asyncio.sleep(every_sec)
