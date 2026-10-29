import json
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.config import Settings
from app.forward.ai_judge import AIForwardLog, build_prompt, evaluate, parse_forecast
from app.main import create_app
from app.runner.data import Candle

GOOD = json.dumps({"targets": {"1348.T": {"1d": {"dir": "up", "conf": 0.6}, "5d": {"dir": "flat", "conf": 0.2}},
                               "SPY": {"1d": {"dir": "down", "conf": 0.5}, "5d": {"dir": "up", "conf": 0.5}},
                               "USDJPY=X": {"1d": {"dir": "up", "conf": 0.5}, "5d": {"dir": "up", "conf": 0.5}}},
                   "rationale": "テスト"})
HEADS = [f"見出し{i}" for i in range(10)]


def test_parse_forecast_handles_noise_and_bad_json():
    assert parse_forecast("前置き " + GOOD + " 後書き")["rationale"] == "テスト"
    assert parse_forecast("not json") is None
    assert parse_forecast('{"targets": {"SPY": {"1d": {"dir": "sideways"}}}}') is None   # 不正な値は捨てる
    assert "見出し0" in build_prompt(HEADS) and "SPY" in build_prompt(HEADS)


def test_log_is_append_only_once_per_day_and_skips_thin_news(tmp_path):
    log = AIForwardLog(str(tmp_path / "f.jsonl"))
    now = datetime(2026, 11, 2, 22, 5, tzinfo=timezone.utc)
    calls = []
    llm = lambda prompt, key, model: calls.append(1) or GOOD
    rets = lambda: {"1348.T": {"1d": 0.1, "5d": 0.5, "20d": 1.0}}
    thin = log.run_once("k", "m", now, fetch=lambda: HEADS[:3], llm=llm, returns=rets)
    assert [r["variant"] for r in thin] == ["control"]                                  # 見出しが少ない日はニュース系を記録しない
    full = log.run_once("k", "m", now, fetch=lambda: HEADS, llm=llm, returns=rets)
    assert sorted(r["variant"] for r in full) == ["news", "news_price"]                 # 残りのバリアントだけ追記
    assert log.run_once("k", "m", now, fetch=lambda: HEADS, llm=llm, returns=rets) == []  # 同じ日の2回目は何もしない
    assert len(calls) == 3 and len(log.rows()) == 3
    assert log.run_once("k", "m", now.replace(day=3), fetch=lambda: HEADS, llm=lambda *a: "壊れた応答", returns=rets) == []
    assert len(log.rows()) == 3
    # コントロールのプロンプトにはニュースも価格も含まれない
    from app.forward.ai_judge import build_prompt as bp
    assert "見出し0" not in bp(HEADS, "control", "2026-11-02") and "直近1日" in bp(HEADS, "news_price", "", {"SPY": {"1d": 1, "5d": 2, "20d": 3}})


def test_evaluate_hit_rate_and_pending():
    t0 = datetime(2026, 11, 2, 22, 0, tzinfo=timezone.utc).timestamp()
    day = 86400
    # 予測時点までに確定していた最後の日足の終値 100。翌日 101(上)、5日後 99(下)
    candles = [Candle(t0 - 2 * day, 0, 0, 98), Candle(t0 - day * 1.5 + 100, 0, 0, 100)] + \
              [Candle(t0 + day * k, 0, 0, c) for k, c in enumerate([101, 100, 100, 100, 100, 99], start=0)]
    rows = [{"ts": datetime.fromtimestamp(t0, timezone.utc).isoformat(), "variant": "news", "forecast": json.loads(GOOD)}]
    r = evaluate(rows, {"1348.T": candles})
    assert r["news/1348.T/1d"]["called"] == 1 and r["news/1348.T/1d"]["hit_rate"] == 1.0   # 上と予測→+1%
    assert r["news/1348.T/5d"]["called"] == 0                                              # flat は的中を数えない
    assert r["control/1348.T/1d"]["evaluated"] == 0                                        # 別バリアントの記録は混ぜない
    assert "偶然と区別できない" in r["_note"]


def test_endpoints_and_disabled_by_default(tmp_path):
    s = Settings(webhook_secret="x", journal_path=str(tmp_path / "j.jsonl"))
    c = TestClient(create_app(s))
    assert c.get("/health").json()["ai_log"] is False                # APIキーが無ければ動かない(勝手に課金しない)
    assert c.get("/ai/eval").json()["forecasts"] == 0
    log = AIForwardLog(str(tmp_path / "ai_forecasts.jsonl"))
    log.run_once("k", "m", datetime(2026, 11, 2, 22, 5, tzinfo=timezone.utc), fetch=lambda: HEADS, llm=lambda *a: GOOD,
                 returns=lambda: {"SPY": {"1d": 0, "5d": 0, "20d": 0}})
    assert len(c.get("/ai/forecasts").json()) == 3


def test_manual_run_endpoint_requires_secret_and_key(tmp_path):
    base = dict(webhook_secret="s", journal_path=str(tmp_path / "j.jsonl"))
    off = TestClient(create_app(Settings(**base)))
    assert off.post("/ai/run", json={"secret": "wrong"}).status_code == 401
    assert off.post("/ai/run", json={"secret": "s"}).status_code == 503               # キーが無ければ動かない
    on = TestClient(create_app(Settings(**base, ai_log_enabled=True, anthropic_api_key="k"),
                               ai_llm=lambda *a: GOOD, ai_headlines=lambda: HEADS,
                               ai_candles=lambda: {"SPY": [Candle(i * 86400, 1, 1, 100 + i) for i in range(30)]}))
    assert on.get("/health").json()["ai_log"] is True
    r = on.post("/ai/run", json={"secret": "s"}).json()
    assert sorted(r["recorded"]) == ["control", "news", "news_price"]
    assert on.post("/ai/run", json={"secret": "s"}).json()["recorded"] == []         # 同じ日は二重に記録しない
