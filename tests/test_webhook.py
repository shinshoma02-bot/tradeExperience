import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

SECRET = "s3cret"


@pytest.fixture
def client(tmp_path):
    settings = Settings(
        webhook_secret=SECRET,
        max_notional_per_order=1000,
        max_orders_per_day=5,
        max_daily_loss=50,
        enabled_markets=frozenset({"crypto", "fx"}),
        journal_path=str(tmp_path / "j.jsonl"),
    )
    return TestClient(create_app(settings))


def sig(**kw):
    base = dict(secret=SECRET, market="crypto", strategy="t", symbol="BTCUSDT",
                action="buy", price=100.0, qty=1.0)
    base.update(kw)
    return base


def test_bad_secret(client):
    assert client.post("/webhook", json=sig(secret="x")).status_code == 401


def test_buy_then_close_pnl(client):
    assert client.post("/webhook", json=sig()).json()["status"] == "filled"
    r = client.post("/webhook", json=sig(action="close", price=110.0, qty=0)).json()
    assert r["fill"]["realized_pnl"] == 10.0
    assert client.get("/positions").json()["crypto"] == {}


def test_short_pnl(client):
    client.post("/webhook", json=sig(action="sell"))
    r = client.post("/webhook", json=sig(action="close", price=90.0, qty=0)).json()
    assert r["fill"]["realized_pnl"] == 10.0


def test_notional_limit(client):
    r = client.post("/webhook", json=sig(qty=100)).json()
    assert r["status"] == "rejected"


def test_disabled_market(client):
    r = client.post("/webhook", json=sig(market="futures")).json()
    assert r["status"] == "rejected"


def test_daily_loss_stops_trading(client):
    client.post("/webhook", json=sig(price=100, qty=10))
    client.post("/webhook", json=sig(action="close", price=90, qty=0))  # -100
    r = client.post("/webhook", json=sig()).json()
    assert r["status"] == "rejected"


def test_close_without_position_ignored(client):
    assert client.post("/webhook", json=sig(action="close", qty=0)).json()["status"] == "ignored"


def test_report(client):
    client.post("/webhook", json=sig())
    client.post("/webhook", json=sig(action="close", price=105.0, qty=0))
    assert client.get("/report").json()["crypto/t"] == {"fills": 2, "realized_pnl_usd": 5.0, "total_pnl_usd": 5.0}


def test_no_secret_configured_rejects_everything(tmp_path):
    c = TestClient(create_app(Settings(webhook_secret="", journal_path=str(tmp_path / "j.jsonl"))))
    assert c.get("/health").json()["secret_configured"] is False
    assert c.post("/webhook", json=sig(secret="")).status_code == 503
