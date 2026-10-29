import asyncio
import hmac
import os
import threading
import time
from datetime import date, datetime, timezone
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.brokers.base import Broker
from app.brokers.paper import PaperBroker
from app.config import Settings
from app.journal import Journal
from app.models import Signal
from app.forward.ai_judge import TARGETS as AI_TARGETS
from app.forward.ai_judge import AIForwardLog, recent_returns
from app.forward.ai_judge import evaluate as ai_evaluate
from app.maintenance import (crash_messages, drawdown_from_high, iso_week_key, max_drawdown_pct, milestone_messages,
                             month_key, monthly_message, weekly_message)
from app.notify import Notifier
from app.risk import RiskError, RiskManager
from app.runner.backtest import DEFAULT_FEE_BPS, run_all
from app.runner.data import Instrument, fetch_candles, parse_instruments
from app.runner.runner import StrategyRunner
from app.runner.strategies import STRATEGIES


def create_app(settings: Settings | None = None, runner_fetch=None, notifier: Notifier | None = None,
               backtest_fetch=None, benchmark_fetch=None, ai_llm=None, ai_headlines=None, ai_candles=None) -> FastAPI:
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

    instruments = parse_instruments(settings.runner_instruments)
    n_bots = len(instruments) * len(STRATEGIES)
    runner = None
    if settings.runner_enabled:
        kw = {"fetch": runner_fetch} if runner_fetch else {}
        runner = StrategyRunner(
            handle=process,
            get_position=get_position,
            instruments=instruments,
            interval_min=settings.runner_interval_min,
            order_notional_usd=settings.order_notional_usd(n_bots),
            to_quote=lambda usd, quote: usd * settings.jpy_per_usd if quote == "JPY" else usd,
            alert=lambda m: notifier.send("ランナー異常", m, "urgent", dedupe_key="runner-fail"),
            **kw,
        )

    ai_path = settings.ai_log_path or os.path.join(os.path.dirname(settings.journal_path) or ".", "ai_forecasts.jsonl")
    ai_log = AIForwardLog(ai_path)
    ai_on = settings.ai_log_enabled and bool(settings.anthropic_api_key)

    def ai_candles_for_targets():
        if ai_candles:
            return ai_candles()
        return {t: fetch_candles(Instrument("us_stock" if t == "SPY" else "jp_stock" if t.endswith(".T") else "fx", "yahoo", t), 1440, days=400)
                for t in AI_TARGETS}

    def ai_returns():
        return recent_returns(ai_candles_for_targets())

    async def ai_loop():
        """毎日 ai_log_hour_utc 以降に1回だけ、見出しからの予測を追記する(売買には使わない)。"""
        while True:
            try:
                now = datetime.now(timezone.utc)
                if now.hour >= settings.ai_log_hour_utc:
                    kw = {}
                    if ai_llm:
                        kw["llm"] = ai_llm
                    if ai_headlines:
                        kw["fetch"] = ai_headlines
                    kw["returns"] = ai_returns
                    rows = await asyncio.to_thread(ai_log.run_once, settings.anthropic_api_key, settings.ai_model, now, **kw)
                    for row in rows:
                        journal.write("ai_forecast", variant=row["variant"])
            except Exception:
                pass
            await asyncio.sleep(1800)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        task = None
        ai_task = asyncio.create_task(ai_loop()) if ai_on else None
        maint_task = asyncio.create_task(maintenance_loop())
        notifier.send("サーバー起動", f"tradeExperience 起動(mode={settings.trading_mode}, runner={bool(runner)})", "low")
        if runner and runner_fetch is None:  # テスト時は自動ループを動かさない
            task = asyncio.create_task(runner.loop(every_sec=60 if settings.runner_interval_min < 240 else 600))
        yield
        if task:
            task.cancel()
        if ai_task:
            ai_task.cancel()
        maint_task.cancel()

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
            "ai_log": ai_on,
        }

    @app.post("/webhook")
    def webhook(signal: Signal):
        if not settings.webhook_secret:
            # 合言葉が未設定のままでは全注文を受け付けない(落ちずに待機する)
            raise HTTPException(status_code=503, detail="WEBHOOK_SECRET is not configured")
        if not hmac.compare_digest(signal.secret, settings.webhook_secret):
            raise HTTPException(status_code=401, detail="bad secret")
        return process(signal)

    quotes = {(i.market, i.symbol): i.quote for i in instruments}

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

    def fee_usd(f: dict) -> float:
        """約定1回あたりの手数料+スプレッドの想定(バックテストと同じ率)。記録には持たず、集計時に計算する。"""
        bps = DEFAULT_FEE_BPS.get(f["market"], 10.0)
        return settings.to_usd(f["qty"] * f["price"], f.get("quote", "USD")) * bps / 1e4

    @app.get("/report")
    def report():
        """マーケット×戦略ごとの成績(USD換算)。total_pnl_usd は確定+含み損益から想定手数料を引いた値。"""
        stats: dict[str, dict] = {}
        for row in journal.read():
            if row["kind"] != "fill":
                continue
            f = row["fill"]
            s = stats.setdefault(f"{f['market']}/{f['strategy']}", {"fills": 0, "realized_pnl_usd": 0.0, "fees_usd": 0.0})
            s["fills"] += 1
            quote = f.get("quote", "USD")
            s["realized_pnl_usd"] = round(s["realized_pnl_usd"] + settings.to_usd(f["realized_pnl"], quote), 8)
            s["fees_usd"] = round(s["fees_usd"] + fee_usd(f), 8)
        for market, poss in with_unrealized().items():
            for p in poss.values():
                s = stats.setdefault(f"{market}/{p['strategy']}", {"fills": 0, "realized_pnl_usd": 0.0, "fees_usd": 0.0})
                s["unrealized_pnl_usd"] = round(s.get("unrealized_pnl_usd", 0.0) + p.get("unrealized_pnl_usd", 0.0), 8)
        for s in stats.values():
            s["total_pnl_usd"] = round(s["realized_pnl_usd"] + s.get("unrealized_pnl_usd", 0.0) - s["fees_usd"], 8)
        return stats

    bench_cache: dict[str, tuple[float, float | None]] = {}

    def benchmark_return_pct(since: str) -> float | None:
        """同じ期間にSPYを持ちっぱなしにしていた場合の値動き(%)。取れなければNone。"""
        hit = bench_cache.get(since)
        if hit and time.time() - hit[0] < 3600:
            return hit[1]
        val = None
        try:
            fetch = benchmark_fetch or (lambda: fetch_candles(Instrument("us_stock", "yahoo", "SPY"), 1440, days=400))
            candles = fetch()
            t0 = datetime.fromisoformat(since).timestamp()
            before = [c for c in candles if c.ts <= t0]
            base = before[-1] if before else (candles[0] if candles else None)  # 開始時点を含む足の終値を基準にする
            if base and candles:
                val = round((candles[-1].close / base.close - 1) * 100, 2)
        except Exception:
            pass
        bench_cache[since] = (time.time(), val)
        return val

    def status(with_benchmark: bool = False) -> dict:
        """元手に対する今の損益(確定+含み損益-想定手数料)と、合格ラインの判定。ネットワークは使わない(ベンチマーク以外)。"""
        rep = report()
        pnl_usd = sum(v["total_pnl_usd"] for v in rep.values())
        fees_usd = sum(v["fees_usd"] for v in rep.values())
        pnl_jpy = pnl_usd * settings.jpy_per_usd
        by_market: dict[str, float] = {}
        for k, v in rep.items():
            m = k.split("/")[0]
            by_market[m] = round(by_market.get(m, 0.0) + v["total_pnl_usd"] * settings.jpy_per_usd, 1)
        rows = journal.read()
        started = rows[0]["ts"] if rows else None
        days = (datetime.now(timezone.utc) - datetime.fromisoformat(started)).days if started else 0
        closed = sum(1 for r in rows if r["kind"] == "fill" and r["fill"]["realized_pnl"] != 0)
        ret = pnl_jpy / settings.capital_jpy * 100 if settings.capital_jpy else 0.0
        bench = benchmark_return_pct(started) if (started and with_benchmark) else None
        gates = {
            "90日以上の運用": days >= 90,
            "決済した取引が30回以上": closed >= 30,
            "手数料込みでプラス": pnl_jpy > 0,
        }
        equities = [r["equity_jpy"] for r in rows if r["kind"] == "snapshot"]
        return {
            "capital_jpy": settings.capital_jpy,
            "equity_jpy": round(settings.capital_jpy + pnl_jpy, 1),
            "pnl_jpy": round(pnl_jpy, 1),
            "return_pct": round(ret, 2),
            "fees_jpy": round(fees_usd * settings.jpy_per_usd, 1),
            "benchmark_spy_return_pct": bench,
            "beats_spy": (ret > bench) if bench is not None else None,
            "started_at": started,
            "days_running": days,
            "closed_trades": closed,
            "pnl_jpy_by_market": by_market,
            "bots": n_bots,
            "per_bot_jpy": round(settings.capital_jpy / max(n_bots, 1), 1),
            "snapshots": len(equities),
            "max_drawdown_pct": max_drawdown_pct(equities) if equities else None,
            "real_money_gates": gates,
            "ready_for_real_money": all(gates.values()),
            "note": "模擬。手数料は想定率(暗号資産0.2%/FX0.02%/株0.1%)、スリッページなし。円換算は固定レート。"
                    "合格ライン(real_money_gates)を全部満たしても実資金は保証ではなく、最初は失ってもよい額で",
        }

    @app.get("/summary")
    def summary():
        """元手に対して今いくらか(円)。SPY持ちっぱなしとの比較、最大の落ち込み、実資金に進む合格ラインつき。"""
        return status(with_benchmark=True)

    def maintenance(now: datetime | None = None) -> list[str]:
        """放置運用の定期処理(30分ごとに呼ぶ)。日次の成績記録、週次レポート、月次リマインダー、節目の通知。
        通知は記録(kind=notice)に残して、再起動しても二重に送らない。戻り値は今回やったことの一覧。"""
        now = now or datetime.now(timezone.utc)
        rows = journal.read()
        done: list[str] = []
        sent = {r["key"] for r in rows if r["kind"] == "notice"}
        today = now.date().isoformat()

        st = status()
        if not any(r["kind"] == "snapshot" and r.get("day") == today for r in rows):
            journal.write("snapshot", day=today, equity_jpy=st["equity_jpy"], pnl_jpy=st["pnl_jpy"], fees_jpy=st["fees_jpy"])
            done.append("snapshot")
            st = status()

        def notify_once(key: str, title: str, body: str, priority: str = "low") -> None:
            if key in sent:
                return
            journal.write("notice", key=key)
            sent.add(key)
            notifier.send(title, body, priority)
            done.append(key)

        if now.weekday() == 0:  # 月曜(UTC)=日本の月曜の朝9時以降
            wk = iso_week_key(now)
            if wk not in sent:
                st_b = status(with_benchmark=True)
                ai_days = len({r["ts"][:10] for r in ai_log.rows()}) if ai_on else None
                title, body = weekly_message(st_b, st_b["max_drawdown_pct"] or 0.0, ai_days)
                notify_once(wk, title, body)
        if now.day == 1:
            title, body = monthly_message()
            notify_once(month_key(now), title, body)
        for key, title, body in milestone_messages(st):
            notify_once(key, title, body, "default")

        # 相場(SPY)が高値から-15%/-25%のとき、積立を続ける理由を1通だけ(下落の局面ごと)
        try:
            fetch = benchmark_fetch or (lambda: fetch_candles(Instrument("us_stock", "yahoo", "SPY"), 1440, days=1500))
            dd = drawdown_from_high([(c.ts, c.close) for c in fetch()])
            if dd:
                for key, title, body in crash_messages(*dd):
                    notify_once(key, title, body, "high")
        except Exception:
            pass
        return done

    app.state.maintenance = maintenance

    async def maintenance_loop():
        while True:
            try:
                await asyncio.to_thread(maintenance)
            except Exception:
                pass
            await asyncio.sleep(1800)

    @app.get("/ai/forecasts")
    def ai_forecasts(limit: int = 10):
        """記録されたAIの予測(新しい順)。"""
        rows = ai_log.rows()[-max(1, min(limit, 100)):][::-1]
        return [{"ts": r["ts"], "variant": r.get("variant", "news"), "model": r["model"], "prompt_version": r["prompt_version"],
                 "n_headlines": len(r["headlines"]), "forecast": r["forecast"]} for r in rows]

    @app.post("/ai/run")
    def ai_run(body: dict):
        """動作確認用: 時刻に関係なく今すぐ1回だけ予測を記録する(今日分が既にあれば何もしない)。合言葉が必要。"""
        if not settings.webhook_secret or not hmac.compare_digest(str(body.get("secret", "")), settings.webhook_secret):
            raise HTTPException(status_code=401, detail="bad secret")
        if not ai_on:
            raise HTTPException(status_code=503, detail="AI_LOG_ENABLED と ANTHROPIC_API_KEY が設定されていない")
        kw = {"returns": ai_returns}
        if ai_llm:
            kw["llm"] = ai_llm
        if ai_headlines:
            kw["fetch"] = ai_headlines
        try:
            rows = ai_log.run_once(settings.anthropic_api_key, settings.ai_model, **kw)
        except Exception as e:  # APIキーなどの秘密を返さないよう、種類と状態だけ返す
            code = getattr(e, "code", None)
            raise HTTPException(status_code=502, detail=f"AI呼び出しに失敗: {type(e).__name__} {code or ''}".strip())
        return {"recorded": [r["variant"] for r in rows], "note": "空なら今日分は記録済み、または見出しが少ない"}

    @app.get("/ai/eval")
    def ai_eval():
        """AIの予測が当たったか(1日後・5日後)。ランダム・『いつも上』と比べて見る。売買は一切していない。"""
        rows = ai_log.rows()
        if not rows:
            return {"forecasts": 0, "note": "まだ記録がない"}
        try:
            candles = ai_candles_for_targets()
        except Exception as e:
            return {"forecasts": len(rows), "error": f"価格取得に失敗: {e}"}
        return {"forecasts": len(rows), "since": rows[0]["ts"], "results": ai_evaluate(rows, candles)}

    bt_cache: dict[tuple, tuple[float, dict]] = {}

    @app.get("/backtest")
    def backtest(days: int = 3650, fee_bps: float | None = None):
        """過去の足で戦略を試した結果。公開データを取りに行くので重い(結果は1時間キャッシュ)。"""
        days = max(7, min(days, 3650))
        fee_bps = None if fee_bps is None else max(0.0, min(fee_bps, 100.0))
        key = (days, fee_bps)
        hit = bt_cache.get(key)
        if hit and time.time() - hit[0] < 3600:
            return hit[1]
        fetch = backtest_fetch or (lambda inst: fetch_candles(inst, settings.runner_interval_min, days=days))
        out = run_all(
            instruments, fetch,
            order_notional_usd=settings.order_notional_usd(n_bots), fee_bps=fee_bps, to_usd=settings.to_usd,
            to_quote=lambda usd, q: usd * settings.jpy_per_usd if q == "JPY" else usd,
        )
        out["note"] = (f"元手{settings.capital_jpy:.0f}円を銘柄×戦略に均等配分した想定。手数料は片道(既定: 暗号資産0.2%/FX0.02%/株0.1%)、"
                       "スリッページなし。期間は銘柄ごとに異なる(暗号資産は直近約2年)。過去の結果は将来を保証しない")
        out["days"], out["fee_bps"] = days, fee_bps
        bt_cache[key] = (time.time(), out)
        return out

    return app
