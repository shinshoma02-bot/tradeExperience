import hmac
from collections import defaultdict

from fastapi import FastAPI, HTTPException

from app.brokers.base import Broker
from app.brokers.paper import PaperBroker
from app.config import Settings
from app.journal import Journal
from app.models import Signal
from app.risk import RiskError, RiskManager


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    if settings.trading_mode != "paper":
        # 実ブローカー実装までは安全のため paper 以外では起動させない
        raise RuntimeError("現在は TRADING_MODE=paper のみ対応です")

    app = FastAPI(title="tradeExperience")
    risk = RiskManager(settings)
    journal = Journal(settings.journal_path)
    # マーケットごとに1つブローカーを持つ。実ブローカーはここを差し替えるだけ。
    brokers: dict[str, Broker] = defaultdict(PaperBroker)

    @app.get("/health")
    def health():
        return {
            "ok": True,
            "mode": settings.trading_mode,
            "secret_configured": bool(settings.webhook_secret),
        }

    @app.post("/webhook")
    def webhook(signal: Signal):
        if not settings.webhook_secret:
            # 合言葉が未設定のままでは全注文を受け付けない(落ちずに待機する)
            raise HTTPException(status_code=503, detail="WEBHOOK_SECRET is not configured")
        if not hmac.compare_digest(signal.secret, settings.webhook_secret):
            raise HTTPException(status_code=401, detail="bad secret")

        payload = signal.model_dump(exclude={"secret"})
        try:
            risk.check(signal)
        except RiskError as e:
            journal.write("rejected", signal=payload, reason=str(e))
            return {"status": "rejected", "reason": str(e)}

        fill = brokers[signal.market].execute(signal)
        if fill is None:
            journal.write("ignored", signal=payload)
            return {"status": "ignored"}

        risk.record(fill.realized_pnl)
        journal.write("fill", signal=payload, fill=fill.model_dump())
        return {"status": "filled", "fill": fill.model_dump()}

    @app.get("/positions")
    def positions():
        return {m: b.positions() for m, b in brokers.items()}

    @app.get("/report")
    def report():
        """マーケット×戦略ごとの成績(確定損益・約定回数)。どれが勝てるか比べる用。"""
        stats: dict[str, dict] = {}
        for row in journal.read():
            if row["kind"] != "fill":
                continue
            f = row["fill"]
            s = stats.setdefault(f"{f['market']}/{f['strategy']}", {"fills": 0, "realized_pnl": 0.0})
            s["fills"] += 1
            s["realized_pnl"] = round(s["realized_pnl"] + f["realized_pnl"], 8)
        return stats

    return app
