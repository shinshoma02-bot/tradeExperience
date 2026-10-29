from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.runner.data import Candle
from app.runner.strategies import donchian_breakout, rsi_reversion, sma_cross


def candles(closes, spread=1.0):
    return [Candle(i, c + spread, c - spread, c) for i, c in enumerate(closes)]


def test_sma_cross_buy_and_sell():
    up = [100.0] * 60 + [101, 105, 110]  # 横ばいから急上昇 → 短期が長期を上抜け
    assert "buy" in [sma_cross(candles(up[:k]), 0) for k in range(55, len(up) + 1)]
    down = [100.0] * 60 + [99, 95, 90]
    assert "sell" in [sma_cross(candles(down[:k]), 0) for k in range(55, len(down) + 1)]


def test_sma_cross_flat_is_none():
    assert sma_cross(candles([100.0] * 80), 0) is None


def test_donchian():
    base = [100.0] * 25
    assert donchian_breakout(candles(base + [110.0]), 0) == "buy"
    assert donchian_breakout(candles(base + [90.0]), 0) == "sell"
    assert donchian_breakout(candles(base + [100.0]), 0) is None


def test_rsi_buy_after_oversold_and_close_at_50():
    falling = [100 - i for i in range(30)]          # ずっと下げ → RSI低位
    sigs = [rsi_reversion(candles(falling + [70 + i * 3 for i in range(k)]), 0) for k in range(1, 12)]
    assert "buy" in sigs
    sigs = [rsi_reversion(candles(falling + [70 + i * 3 for i in range(k)]), 1.0) for k in range(1, 12)]
    assert "close" in sigs


def make_client(tmp_path, feed):
    s = Settings(webhook_secret="x", journal_path=str(tmp_path / "j.jsonl"),
                 runner_enabled=True, runner_instruments="crypto:kraken:XBTUSD", runner_order_notional=100,
                 max_notional_per_order=1000, enabled_markets=frozenset({"crypto"}))
    app = create_app(s, runner_fetch=lambda inst, iv: feed["c"])
    return TestClient(app), app.state.runner


def test_runner_warmup_then_trades_on_new_bar(tmp_path):
    feed = {"c": candles([100.0] * 59)}
    client, runner = make_client(tmp_path, feed)
    assert runner.run_once() == []                 # 初回は何もしない(ウォームアップ)
    feed["c"] = candles([100.0] * 59 + [110.0])    # 新しい足で急上昇(SMAクロス+ブレイクアウト)
    out = runner.run_once()
    assert any(r["status"] == "filled" and r["fill"]["side"] == "buy" for r in out)
    assert runner.run_once() == []                 # 同じ足では二重発注しない
    rep = client.get("/report").json()
    assert "crypto/donchian_breakout" in rep
    assert client.get("/health").json()["runner"] is True


def test_runner_reverses_position(tmp_path):
    feed = {"c": candles([100.0] * 59)}
    client, runner = make_client(tmp_path, feed)
    runner.run_once()
    feed["c"] = candles([100.0] * 59 + [110.0])
    runner.run_once()                              # donchian long
    feed["c"] = candles([100.0] * 59 + [110.0] + [80.0])
    out = runner.run_once()                        # 下抜け → 決済してから売り
    sides = [r["fill"]["side"] for r in out if r["status"] == "filled"]
    assert "sell" in sides
    pos = client.get("/positions").json()["crypto"]
    assert pos["donchian_breakout:XBTUSD"]["qty"] < 0


def test_notifications_fill_reject_and_runner_failure(tmp_path):
    from app.notify import Notifier

    sent = []
    n = Notifier("t", sender=lambda title, msg, pr: sent.append((title, msg, pr)))
    # 同期的に確認できるよう、スレッドを使わず直接送る
    n.send = lambda title, message, priority="default", dedupe_key=None, dedupe_sec=3600: sent.append((title, priority))

    feed = {"c": candles([100.0] * 59)}
    s = Settings(webhook_secret="x", journal_path=str(tmp_path / "j.jsonl"), runner_enabled=True,
                 runner_instruments="crypto:kraken:XBTUSD", runner_order_notional=100, enabled_markets=frozenset({"crypto"}),
                 max_notional_per_order=1000)
    app = create_app(s, runner_fetch=lambda inst, iv: feed["c"], notifier=n)
    runner = app.state.runner
    runner.run_once()
    feed["c"] = candles([100.0] * 59 + [110.0])
    runner.run_once()
    assert any(t.startswith("約定") for t, _ in sent)

    client = TestClient(app)
    client.post("/webhook", json=dict(secret="x", market="fx", strategy="t", symbol="USDJPY",
                                      action="buy", price=1.0, qty=1))
    assert ("注文を拒否", "high") in sent

    def boom(inst, iv):
        raise OSError("down")
    runner.fetch = boom
    for _ in range(5):
        runner.run_once()
    assert ("ランナー異常", "urgent") in sent


def test_notifier_dedupe_and_disabled():
    from app.notify import Notifier

    calls = []
    n = Notifier("t", sender=lambda *a: calls.append(a))
    n.send("a", "b", dedupe_key="k")
    n.send("a", "b", dedupe_key="k")
    import time; time.sleep(0.2)
    assert len(calls) == 1
    off = Notifier("", sender=lambda *a: calls.append(a))
    off.send("a", "b")
    time.sleep(0.1)
    assert len(calls) == 1


def test_stock_never_shorts_and_jpy_is_converted(tmp_path):
    feed = {"c": candles([100.0] * 59)}
    s = Settings(webhook_secret="x", journal_path=str(tmp_path / "j.jsonl"), runner_enabled=True,
                 runner_instruments="us_stock:yahoo:AAPL,jp_stock:yahoo:7203.T:JPY",
                 runner_order_notional=100, max_notional_per_order=1000, jpy_per_usd=150,
                 enabled_markets=frozenset({"us_stock", "jp_stock"}))
    app = create_app(s, runner_fetch=lambda inst, iv: feed["c"])
    runner = app.state.runner
    runner.run_once()
    feed["c"] = candles([100.0] * 59 + [80.0])          # 下抜け: 株は空売りしない
    assert runner.run_once() == []
    assert all(not v for v in TestClient(app).get("/positions").json().values())
    feed["c"] = candles([100.0] * 59 + [80.0] + [200.0])  # 上抜け: 買い
    out = runner.run_once()
    jp = [r["fill"] for r in out if r["status"] == "filled" and r["fill"]["market"] == "jp_stock"]
    assert jp and abs(jp[0]["qty"] * jp[0]["price"] - 100 * 150) < 1e-6  # 100USD相当=15000円


def test_parse_instruments_and_incomplete_candle_dropped():
    from app.runner.data import _drop_incomplete, parse_instruments

    ins = parse_instruments("fx:yahoo:USDJPY=X:JPY, crypto:kraken:XBTUSD")
    assert (ins[0].quote, ins[1].quote, ins[1].provider) == ("JPY", "USD", "kraken")
    cs = [Candle(0, 1, 1, 1), Candle(3600, 1, 1, 1), Candle(7200, 1, 1, 1)]
    assert [c.ts for c in _drop_incomplete(cs, 60, now=7300)] == [0, 3600]


def test_unrealized_pnl_in_positions_and_report(tmp_path):
    feed = {"c": candles([100.0] * 59)}
    client, runner = make_client(tmp_path, feed)
    runner.run_once()
    feed["c"] = candles([100.0] * 59 + [110.0])      # donchian が 110 で買い(100USD分)
    runner.run_once()
    feed["c"] = candles([100.0] * 59 + [110.0] + [121.0])  # 同方向のままなので保有継続
    runner.run_once()
    p = client.get("/positions").json()["crypto"]["donchian_breakout:XBTUSD"]
    assert p["side"] == "long" and p["mark_price"] == 121.0
    assert abs(p["unrealized_pnl"] - (121 - 110) * p["qty"]) < 1e-9
    rep = client.get("/report").json()["crypto/donchian_breakout"]
    assert abs(rep["total_pnl_usd"] - rep["unrealized_pnl_usd"]) < 1e-9 and rep["realized_pnl_usd"] == 0


def test_positions_restored_from_journal_after_restart(tmp_path):
    s = Settings(webhook_secret="x", journal_path=str(tmp_path / "j.jsonl"),
                 enabled_markets=frozenset({"futures", "crypto"}), max_notional_per_order=1000)
    c1 = TestClient(create_app(s))
    sig = dict(secret="x", market="futures", strategy="donchian_breakout", symbol="ES=F",
               action="sell", price=7744.75, qty=0.0129, quote="USD")
    assert c1.post("/webhook", json=sig).json()["status"] == "filled"
    # 別のアプリ(=再起動後のサーバー)が同じ記録から復元する
    c2 = TestClient(create_app(s))
    p = c2.get("/positions").json()["futures"]["donchian_breakout:ES=F"]
    assert p["qty"] == -0.0129 and p["avg_price"] == 7744.75 and p["side"] == "short"
    assert c2.get("/health").json()["restored_fills"] == 1
    # 復元後の決済で正しく損益が出る(価格が下がった=売りの利益)
    r = c2.post("/webhook", json={**sig, "action": "close", "price": 7700.0, "qty": 0}).json()
    assert r["fill"]["realized_pnl"] > 0
