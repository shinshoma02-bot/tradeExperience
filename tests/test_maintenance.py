from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.maintenance import max_drawdown_pct
from app.notify import Notifier


def make(tmp_path, sent):
    n = Notifier("t")
    n.send = lambda title, message, priority="default", dedupe_key=None, dedupe_sec=3600: sent.append((title, message))
    s = Settings(webhook_secret="x", journal_path=str(tmp_path / "j.jsonl"), runner_instruments="crypto:kraken:XBTUSD",
                 enabled_markets=frozenset({"crypto"}), capital_jpy=10000, jpy_per_usd=150)
    app = create_app(s, notifier=n, benchmark_fetch=lambda: [])
    return app, TestClient(app)


def utc(y, m, d, h=0):
    return datetime(y, m, d, h, 5, tzinfo=timezone.utc)


def test_max_drawdown():
    assert max_drawdown_pct([100, 110, 99, 120, 108]) == -10.0
    assert max_drawdown_pct([100, 101, 102]) == 0.0 and max_drawdown_pct([]) == 0.0


def test_daily_snapshot_once_and_drawdown_in_summary(tmp_path):
    sent = []
    app, c = make(tmp_path, sent)
    m = app.state.maintenance
    assert "snapshot" in m(utc(2026, 11, 4))            # 水曜: スナップショットだけ
    assert m(utc(2026, 11, 4, 12)) == []                 # 同じ日は2回やらない
    assert sent == []                                    # 通知は出ない(週次・月次・節目の日ではない)
    assert "snapshot" in m(utc(2026, 11, 5))
    r = c.get("/summary").json()
    assert r["snapshots"] == 2 and r["max_drawdown_pct"] == 0.0


def test_weekly_monthly_and_milestones_are_sent_once_even_after_restart(tmp_path):
    sent = []
    app, c = make(tmp_path, sent)
    m = app.state.maintenance
    assert any(k.startswith("weekly:") for k in m(utc(2026, 11, 2)))    # 月曜
    assert any(t.startswith("週次レポート") for t, _ in sent)
    body = next(b for t, b in sent if t.startswith("週次レポート"))
    assert "積立は止めない" in body and "合格ライン" in body
    assert [k for k in m(utc(2026, 11, 2, 12)) if k.startswith("weekly")] == []   # 同じ週は二度送らない
    assert any(k.startswith("monthly:") for k in m(utc(2026, 12, 1)))             # 毎月1日
    n_before = len(sent)
    app2, _ = make(tmp_path, sent)                       # 再起動(同じ記録)
    assert [k for k in app2.state.maintenance(utc(2026, 11, 2, 13)) if k.startswith("weekly")] == []
    assert len(sent) == n_before


def test_milestone_90_days_and_30_trades(tmp_path):
    sent = []
    app, c = make(tmp_path, sent)
    import json
    # 90日以上前の記録と、決済30回分の約定を作る
    rows = [{"ts": "2026-06-01T00:00:00+00:00", "kind": "rejected", "signal": {}, "reason": "x"}]
    for i in range(30):
        rows.append({"ts": "2026-06-02T00:00:00+00:00", "kind": "fill",
                     "fill": {"market": "crypto", "strategy": "t", "symbol": "XBTUSD", "side": "sell", "qty": 1, "price": 100,
                              "realized_pnl": 1.0, "quote": "USD"}, "signal": {}})
    with open(tmp_path / "j.jsonl", "w") as f:
        f.write("\n".join(json.dumps(r) for r in rows) + "\n")
    app2, _ = make(tmp_path, sent)
    done = app2.state.maintenance(utc(2026, 11, 4))
    assert {"milestone:days90", "milestone:trades30", "milestone:ready"} <= set(done)
    assert any("合格ライン" in t for t, _ in sent)
    assert [k for k in app2.state.maintenance(utc(2026, 11, 4, 12)) if k.startswith("milestone")] == []


def test_crash_notice_once_per_episode(tmp_path):
    from app.maintenance import crash_messages, drawdown_from_high
    from app.runner.data import Candle

    assert drawdown_from_high([]) is None
    dd, ts = drawdown_from_high([(1, 100.0), (2, 120.0), (3, 90.0)])
    assert dd == -25.0 and ts == 2
    assert [k for k, _, _ in crash_messages(dd, ts)] == ["crash15:2", "crash25:2"]
    assert crash_messages(-10.0, 2) == []

    sent = []
    n = Notifier("t")
    n.send = lambda title, message, priority="default", dedupe_key=None, dedupe_sec=3600: sent.append((title, message))
    candles = [Candle(i, 1, 1, c) for i, c in [(1, 100.0), (2, 120.0), (3, 96.0)]]   # 高値から-20%
    s = Settings(webhook_secret="x", journal_path=str(tmp_path / "j.jsonl"), runner_instruments="crypto:kraken:XBTUSD",
                 enabled_markets=frozenset({"crypto"}))
    app = create_app(s, notifier=n, benchmark_fetch=lambda: candles)
    done = app.state.maintenance(utc(2026, 11, 4))
    assert "crash15:2" in done and "crash25:2" not in done
    assert any("積立は続けましょう" in t for t, _ in sent)
    assert "crash15:2" not in app.state.maintenance(utc(2026, 11, 5))                 # 同じ局面で二度送らない
