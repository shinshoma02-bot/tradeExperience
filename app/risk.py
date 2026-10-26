from datetime import date

from app.config import Settings
from app.models import Signal


class RiskError(Exception):
    pass


class RiskManager:
    """注文を出す前の最後の関門。ここで弾いた注文は絶対に発注されない。"""

    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self._day = date.today()
        self._orders = 0
        self._loss = 0.0

    def _roll(self) -> None:
        today = date.today()
        if today != self._day:
            self._day, self._orders, self._loss = today, 0, 0.0

    def check(self, signal: Signal) -> None:
        self._roll()
        if self.s.kill_switch:
            raise RiskError("kill switch が有効です")
        if signal.market not in self.s.enabled_markets:
            raise RiskError(f"マーケット {signal.market} は無効です")
        if self._loss >= self.s.max_daily_loss:
            raise RiskError("1日の損失上限に達したため本日は停止中")
        if self._orders >= self.s.max_orders_per_day:
            raise RiskError("1日の注文回数の上限に達しました")
        notional = signal.price * signal.qty
        if signal.action != "close" and notional > self.s.max_notional_per_order:
            raise RiskError(f"1注文の上限金額超過: {notional:.2f} > {self.s.max_notional_per_order}")

    def record(self, realized_pnl: float) -> None:
        self._orders += 1
        if realized_pnl < 0:
            self._loss += -realized_pnl
