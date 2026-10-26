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
        )
