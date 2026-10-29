from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.runner.data import Candle
from app.runner.strategies import donchian_breakout, rsi_reversion, sma_cross


def candles(closes, spread=1.0):
    return [Candle(i, c + spread, c - spread, c) for i, c in enumerate(closes)]


def test_sma_cross_follows_trend_with_position():
    up = candles([100.0] * 100 + [100.0 + i for i in range(1, 31)])
    assert sma_cross(up, 0) == "buy" and sma_cross(up, 1) is None and sma_cross(up, -1) == "buy"
    down = candles([100.0] * 100 + [100.0 - i for i in range(1, 31)])
    assert sma_cross(down, 0) == "sell" and sma_cross(down, 1) == "sell" and sma_cross(down, -1) is None
    assert sma_cross(candles([100.0] * 120), 0) is None           # 横ばいは何もしない
    assert sma_cross(candles([100.0] * 50), 0) is None            # 本数不足


def test_donchian_turtle_entry_and_exit():
    base = [100.0] * 60
    assert donchian_breakout(candles(base + [110.0]), 0) == "buy"
    assert donchian_breakout(candles(base + [90.0]), 0) == "sell"
    assert donchian_breakout(candles(base + [100.0]), 0) is None
    assert donchian_breakout(candles(base + [90.0]), 1) == "close"    # 買い保有中に20日安値割れ → 降りる
    assert donchian_breakout(candles(base + [110.0]), -1) == "close"  # 売り保有中に20日高値超え → 買い戻し
    assert donchian_breakout(candles(base + [110.0]), 1) is None


def test_rsi_buys_oversold_and_exits_at_50():
    falling = [200 - i * 3 for i in range(40)]
    assert rsi_reversion(candles(falling), 0) == "buy"
    assert rsi_reversion(candles(falling), 1) is None                 # 既に持っているなら追加しない
    rebound = falling + [falling[-1] + i * 8 for i in range(1, 15)]
    assert rsi_reversion(candles(rebound), 1) == "close"
    assert rsi_reversion(candles(rebound), 0) is None


def make_client(tmp_path, feed):
    s = Settings(webhook_secret="x", journal_path=str(tmp_path / "j.jsonl"),
                 runner_enabled=True, runner_instruments="crypto:kraken:XBTUSD", runner_order_notional=100,
                 max_notional_per_order=1000, enabled_markets=frozenset({"crypto"}))
    app = create_app(s, runner_fetch=lambda inst, iv: feed["c"])
    return TestClient(app), app.state.runner


def test_runner_warmup_then_trades_on_new_bar(tmp_path):
    feed = {"c": candles([100.0] * 120)}
    client, runner = make_client(tmp_path, feed)
    assert runner.run_once() == []                 # 初回は何もしない(ウォームアップ)
    feed["c"] = candles([100.0] * 120 + [110.0])    # 新しい足で急上昇(SMAクロス+ブレイクアウト)
    out = runner.run_once()
    assert any(r["status"] == "filled" and r["fill"]["side"] == "buy" for r in out)
    assert runner.run_once() == []                 # 同じ足では二重発注しない
    rep = client.get("/report").json()
    assert "crypto/donchian_breakout" in rep
    assert client.get("/health").json()["runner"] is True


def test_spot_market_exits_but_never_shorts(tmp_path):
    feed = {"c": candles([100.0] * 120)}
    client, runner = make_client(tmp_path, feed)
    runner.run_once()
    feed["c"] = candles([100.0] * 120 + [110.0])
    runner.run_once()                                        # long
    feed["c"] = candles([100.0] * 120 + [110.0] + [80.0])
    out = runner.run_once()                                  # 急落 → 降りる(暗号資産は売りから入らない)
    assert any(r["status"] == "filled" and r["fill"]["side"] == "sell" for r in out)
    assert all(p["qty"] > 0 for p in client.get("/positions").json()["crypto"].values())  # 空売りは持たない


def test_fx_can_short(tmp_path):
    feed = {"c": candles([100.0] * 120)}
    s = Settings(webhook_secret="x", journal_path=str(tmp_path / "j.jsonl"), runner_enabled=True,
                 runner_instruments="fx:yahoo:USDJPY=X:JPY", runner_order_notional=10, jpy_per_usd=150,
                 max_notional_per_order=1000, enabled_markets=frozenset({"fx"}))
    app = create_app(s, runner_fetch=lambda inst, iv: feed["c"])
    runner = app.state.runner
    runner.run_once()
    feed["c"] = candles([100.0] * 120 + [80.0])
    runner.run_once()
    pos = TestClient(app).get("/positions").json()["fx"]
    assert pos["donchian_breakout:USDJPY=X"]["side"] == "short"


def test_stock_never_shorts_and_jpy_is_converted(tmp_path):
    feed = {"c": candles([100.0] * 120)}
    s = Settings(webhook_secret="x", journal_path=str(tmp_path / "j.jsonl"), runner_enabled=True,
                 runner_instruments="us_stock:yahoo:AAPL,jp_stock:yahoo:7203.T:JPY",
                 runner_order_notional=100, max_notional_per_order=1000, jpy_per_usd=150,
                 enabled_markets=frozenset({"us_stock", "jp_stock"}))
    app = create_app(s, runner_fetch=lambda inst, iv: feed["c"])
    runner = app.state.runner
    runner.run_once()
    feed["c"] = candles([100.0] * 120 + [80.0])          # 急落
    out = runner.run_once()
    assert all(r["fill"]["side"] == "buy" for r in out if r["status"] == "filled")  # 空売りは出ない
    pos = TestClient(app).get("/positions").json()
    assert all(p["qty"] > 0 for m in pos.values() for p in m.values())
    jp = [r["fill"] for r in out if r["status"] == "filled" and r["fill"]["market"] == "jp_stock"]
    assert jp and jp[0]["qty"] == int(jp[0]["qty"])                  # 日本株は1株単位
    assert 15000 - 80 < jp[0]["qty"] * jp[0]["price"] <= 15000        # 100USD相当=15000円まで


def test_order_qty_and_allocation():
    from app.runner.runner import order_qty

    assert order_qty("jp_stock", 15000, 3051.0) == 4.0 and order_qty("jp_stock", 2000, 3051.0) == 0.0
    assert abs(order_qty("crypto", 100, 80000.0) - 0.00125) < 1e-12
    s = Settings(webhook_secret="x", capital_jpy=10000, jpy_per_usd=150)
    assert abs(s.order_notional_usd(12) - 10000 / 150 / 12) < 1e-9
    assert Settings(webhook_secret="x", runner_order_notional=5).order_notional_usd(12) == 5


def test_runner_skips_when_allocation_too_small(tmp_path):
    feed = {"c": candles([3000.0] * 120)}
    s = Settings(webhook_secret="x", journal_path=str(tmp_path / "j.jsonl"), runner_enabled=True,
                 runner_instruments="jp_stock:yahoo:7203.T:JPY", runner_order_notional=5, jpy_per_usd=150,
                 enabled_markets=frozenset({"jp_stock"}))
    app = create_app(s, runner_fetch=lambda inst, iv: feed["c"])
    runner = app.state.runner
    runner.run_once()
    feed["c"] = candles([3000.0] * 120 + [3300.0])        # 5USD=750円では3300円の1株が買えない
    out = runner.run_once()
    assert out and all(r["status"] == "skipped" for r in out)


def test_parse_instruments_and_incomplete_candle_dropped():
    from app.runner.data import _drop_incomplete, parse_instruments

    ins = parse_instruments("fx:yahoo:USDJPY=X:JPY, crypto:kraken:XBTUSD")
    assert (ins[0].quote, ins[1].quote, ins[1].provider) == ("JPY", "USD", "kraken")
    cs = [Candle(0, 1, 1, 1), Candle(3600, 1, 1, 1), Candle(7200, 1, 1, 1)]
    assert [c.ts for c in _drop_incomplete(cs, 60, now=7300)] == [0, 3600]


def test_unrealized_pnl_in_positions_and_report(tmp_path):
    feed = {"c": candles([100.0] * 120)}
    client, runner = make_client(tmp_path, feed)
    runner.run_once()
    feed["c"] = candles([100.0] * 120 + [110.0])      # donchian が 110 で買い(100USD分)
    runner.run_once()
    feed["c"] = candles([100.0] * 120 + [110.0] + [121.0])  # 同方向のままなので保有継続
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


def test_backtest_endpoint_and_plan_orders(tmp_path):
    from app.runner.runner import plan_orders

    assert plan_orders("fx", "buy", -1) == ["close", "buy"]
    assert plan_orders("fx", "buy", 1) == []
    assert plan_orders("us_stock", "sell", 0) == []
    assert plan_orders("us_stock", "sell", 1) == ["close"]
    assert plan_orders("crypto", "close", 0) == ["close"]

    up = [100.0] * 120 + [100 + i * 2 for i in range(1, 60)]  # 横ばい→上昇トレンド
    s = Settings(webhook_secret="x", journal_path=str(tmp_path / "j.jsonl"),
                 runner_instruments="crypto:kraken:XBTUSD")
    c = TestClient(create_app(s, backtest_fetch=lambda inst: candles(up)))
    r = c.get("/backtest").json()
    assert r["errors"] == [] and len(r["results"]) == 3
    donch = next(x for x in r["results"] if x["strategy"] == "donchian_breakout")
    assert donch["pnl_usd"] > 0 and donch["buy_hold_pct"] > 0   # 上昇相場でブレイクアウトは儲かる


def test_summary_in_yen(tmp_path):
    s = Settings(webhook_secret="x", journal_path=str(tmp_path / "j.jsonl"), capital_jpy=10000, jpy_per_usd=150,
                 runner_instruments="crypto:kraken:XBTUSD", enabled_markets=frozenset({"crypto"}))
    c = TestClient(create_app(s))
    sig = dict(secret="x", market="crypto", strategy="t", symbol="XBTUSD", price=100.0, qty=1.0)
    c.post("/webhook", json={**sig, "action": "buy"})
    c.post("/webhook", json={**sig, "action": "close", "price": 101.5})   # +1.5USD = +225円
    r = c.get("/summary").json()
    assert r["pnl_jpy"] == 225.0 and r["equity_jpy"] == 10225.0 and r["return_pct"] == 2.25
