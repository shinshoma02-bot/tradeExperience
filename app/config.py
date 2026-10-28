import os
from dataclasses import dataclass, field


def _bool(v: str) -> bool:
    return v.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    webhook_secret: str
    trading_mode: str = "paper"
    max_notional_per_order: float = 1000.0
    max_orders_per_day: int = 20
    max_daily_loss: float = 200.0
    kill_switch: bool = False
    enabled_markets: frozenset = field(default_factory=frozenset)
    journal_path: str = "data/journal.jsonl"
    runner_enabled: bool = False
    runner_instruments: str = (
        "crypto:kraken:XBTUSD,crypto:kraken:ETHUSD,"
        "fx:yahoo:USDJPY=X:JPY,fx:yahoo:EURUSD=X,"
        "us_stock:yahoo:AAPL,us_stock:yahoo:NVDA,"
        "jp_stock:yahoo:7203.T:JPY,futures:yahoo:ES=F"
    )
    runner_interval_min: int = 60
    runner_order_notional: float = 100.0
    jpy_per_usd: float = 150.0  # 円建て銘柄を米ドルに直す概算レート(模擬用)
    ntfy_topic: str = ""
    ntfy_server: str = "https://ntfy.sh"

    def to_usd(self, amount: float, quote: str) -> float:
        return amount / self.jpy_per_usd if quote == "JPY" else amount

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ.get
        return cls(
            webhook_secret=env("WEBHOOK_SECRET", ""),
            trading_mode=env("TRADING_MODE", "paper"),
            max_notional_per_order=float(env("MAX_NOTIONAL_PER_ORDER", "1000")),
            max_orders_per_day=int(env("MAX_ORDERS_PER_DAY", "20")),
            max_daily_loss=float(env("MAX_DAILY_LOSS", "200")),
            kill_switch=_bool(env("KILL_SWITCH", "false")),
            enabled_markets=frozenset(
                m.strip()
                for m in env("ENABLED_MARKETS", "crypto,fx,us_stock,jp_stock,futures").split(",")
                if m.strip()
            ),
            journal_path=env("JOURNAL_PATH", "data/journal.jsonl"),
            runner_enabled=_bool(env("RUNNER_ENABLED", "false")),
            runner_instruments=env("RUNNER_INSTRUMENTS", cls.runner_instruments),
            runner_interval_min=int(env("RUNNER_INTERVAL_MIN", "60")),
            runner_order_notional=float(env("RUNNER_ORDER_NOTIONAL", "100")),
            jpy_per_usd=float(env("JPY_PER_USD", "150")),
            ntfy_topic=env("NTFY_TOPIC", ""),
            ntfy_server=env("NTFY_SERVER", "https://ntfy.sh"),
        )
