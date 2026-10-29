import asyncio
import hmac
import threading
from datetime import date
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.brokers.base import Broker
from app.brokers.paper import PaperBroker
from app.config import Settings
from app.journal import Journal
from app.models import Signal
from app.notify import Notifier
from app.risk import RiskError, RiskManager
from app.runner.data import parse_instruments
from app.runner.runner import StrategyRunner


def create_app(settings: Settings | None = None, runner_fetch=None, notifier: Notifier | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    if settings.trading_mode != "paper":
        # 実ブローカー実装までは安全のため paper 以外では起動させない
        raise RuntimeError("現在は TRADING_MODE=paper のみ対応です")

    notifier = notifier or Notifier(settings.ntfy_topic, settings.ntfy_server)
    risk = RiskManager(settings)
    journal = Journal(settings.journal_path)
    # マーケットごとに1つブローカーを持つ。実ブローカーはここを差し替えるだけ。
    brokers: dict[str, Broker] = defaultdict(PaperBroker)
    lock = threading.Lock()  # webhookと戦略ランナーが同時に状態を触らないように

    def process(signal: Signal) -> dict:
        """リスク確認→発注→記録。webhookも戦略ランナーも必ずここを通る。"""
        payload = signal.model_dump(exclude={"secret"})
        with lock:
            try:
                risk.check(signal)
            except RiskError as e:
                journal.write("rejected", signal=payload, reason=str(e))
                notifier.send("注文を拒否", f"{signal.market}/{signal.strategy} {signal.symbol} {signal.action}\n理由: {e}",
                              "high", dedupe_key=f"rej:{e}")
                return {"status": "rejected", "reason": str(e)}

            fill = brokers[signal.market].execute(signal)
            if fill is None:
                journal.write("ignored", signal=payload)
                return {"status": "ignored"}

            risk.record(fill.realized_pnl, fill.quote)
            journal.write("fill", signal=payload, fill=fill.model_dump())
            pnl = f" 損益 {fill.realized_pnl:+.2f}" if fill.realized_pnl else ""
            notifier.send(
                f"約定 {fill.side} {fill.symbol}",
                f"{fill.market}/{fill.strategy}\n{fill.qty:g} @ {fill.price:g}{pnl}(模擬)",
            )
            return {"status": "filled", "fill": fill.model_dump()}

    def restore_from_journal() -> int:
        """再起動で消えた模擬ポジションと本日のリスク集計を、記録(約定の履歴)を再生して復元する。"""
        today = date.today().isoformat()
        n = 0
        for row in journal.read():
            if row.get("kind") != "fill":
                continue
            try:
                sig = Signal(secret="", **row["signal"])
                fill = brokers[sig.market].execute(sig)
            except Exception:
                continue  # 壊れた行は飛ばす
            if fill and row["ts"].startswith(today):
                risk.record(fill.realized_pnl, fill.quote)
            n += 1
        return n

    restored = restore_from_journal()

    def get_position(market: str, strategy: str, symbol: str) -> float:
        with lock:
            pos = brokers[market].positions().get(f"{strategy}:{symbol}")
        return pos["qty"] if pos else 0.0

    runner = None
    if settings.runner_enabled:
        kw = {"fetch": runner_fetch} if runner_fetch else {}
        runner = StrategyRunner(
            handle=process,
            get_position=get_position,
            instruments=parse_instruments(settings.runner_instruments),
            interval_min=settings.runner_interval_min,
            order_notional_usd=settings.runner_order_notional,
            to_quote=lambda usd, quote: usd * settings.jpy_per_usd if quote == "JPY" else usd,
            alert=lambda m: notifier.send("ランナー異常", m, "urgent", dedupe_key="runner-fail"),
            **kw,
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        task = None
        notifier.send("サーバー起動", f"tradeExperience 起動(mode={settings.trading_mode}, runner={bool(runner)})", "low")
        if runner and runner_fetch is None:  # テスト時は自動ループを動かさない
            task = asyncio.create_task(runner.loop(every_sec=60))
        yield
        if task:
            task.cancel()

    app = FastAPI(title="tradeExperience", lifespan=lifespan)
    app.state.runner = runner

    @app.get("/health")
    def health():
        return {
            "ok": True,
            "mode": settings.trading_mode,
            "secret_configured": bool(settings.webhook_secret),
            "runner": bool(runner),
            "notify": notifier.enabled,
            "restored_fills": restored,
        }

    @app.post("/webhook")
    def webhook(signal: Signal):
        if not settings.webhook_secret:
            # 合言葉が未設定のままでは全注文を受け付けない(落ちずに待機する)
            raise HTTPException(status_code=503, detail="WEBHOOK_SECRET is not configured")
        if not hmac.compare_digest(signal.secret, settings.webhook_secret):
            raise HTTPException(status_code=401, detail="bad secret")
        return process(signal)

    quotes = {(i.market, i.symbol): i.quote for i in parse_instruments(settings.runner_instruments)}

    def with_unrealized() -> dict[str, dict]:
        """ポジションに含み損益を付ける。評価価格は直近の確定足の終値(最大で足1本分=約1時間遅れ)。"""
        with lock:
            snap = {m: b.positions() for m, b in brokers.items()}
        marks = runner.last_prices if runner else {}
        out: dict[str, dict] = {}
        for market, poss in snap.items():
            out[market] = {}
            for key, p in poss.items():
                mark = marks.get((market, p["symbol"]))
                quote = quotes.get((market, p["symbol"]), "USD")
                row = {**p, "side": "long" if p["qty"] > 0 else "short", "quote": quote, "mark_price": mark}
                if mark is not None:
                    pnl = (mark - p["avg_price"]) * p["qty"]
                    row["unrealized_pnl"] = round(pnl, 8)
                    row["unrealized_pnl_usd"] = round(settings.to_usd(pnl, quote), 8)
                out[market][key] = row
        return out

    @app.get("/positions")
    def positions():
        return with_unrealized()

    @app.get("/report")
    def report():
        """マーケット×戦略ごとの成績(確定損益はUSD換算・約定回数)。どれが勝てるか比べる用。"""
        stats: dict[str, dict] = {}
        for row in journal.read():
            if row["kind"] != "fill":
                continue
            f = row["fill"]
            s = stats.setdefault(f"{f['market']}/{f['strategy']}", {"fills": 0, "realized_pnl_usd": 0.0})
            s["fills"] += 1
            usd = settings.to_usd(f["realized_pnl"], f.get("quote", "USD"))
            s["realized_pnl_usd"] = round(s["realized_pnl_usd"] + usd, 8)
        for market, poss in with_unrealized().items():
            for p in poss.values():
                s = stats.setdefault(f"{market}/{p['strategy']}", {"fills": 0, "realized_pnl_usd": 0.0})
                s["unrealized_pnl_usd"] = round(s.get("unrealized_pnl_usd", 0.0) + p.get("unrealized_pnl_usd", 0.0), 8)
        for s in stats.values():
            s["total_pnl_usd"] = round(s["realized_pnl_usd"] + s.get("unrealized_pnl_usd", 0.0), 8)
        return stats

    return app
