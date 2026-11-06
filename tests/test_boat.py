from datetime import date
from pathlib import Path

import numpy as np

from boat import model, parse, upcoming

FX = Path(__file__).parent / "fixtures"
D = date(2026, 10, 2)


def _load():
    b = (FX / "boat_B_20261002.txt").read_text(encoding="utf-8")
    k = (FX / "boat_K_20261002.txt").read_text(encoding="utf-8")
    return parse.parse_b(b, D), parse.parse_k(k, D)


def test_parse_program_and_results():
    (rb, eb), (rk, ek) = _load()
    assert len(rb) == 12 and len(eb) == 72
    e = eb[0]
    assert (e["race_no"], e["lane"], e["reg"], e["name"], e["cls"]) == (1, 1, 3775, "泉啓文", "B1")
    assert e["nat_win"] == 3.85 and e["motor"] == 24 and e["boat_2r"] == 26.67
    assert len(rk) == 12 and len(ek) == 72
    r1 = rk[0]
    assert r1["win_lane"] == 6 and r1["win_pay"] == 570 and r1["3連単_combo"] == "6-1-3" and r1["3連単_pay"] == 11800
    assert r1["kimarite"] == "まくり差し" and r1["wind"] == 1
    # 1Rの1着は6号艇(進入5コース)、S0/S2は完走せず(finish=0)
    w = [x for x in ek if x["race_no"] == 1 and x["finish"] == 1][0]
    assert w["lane"] == 6 and w["course"] == 5 and w["st"] == 0.11
    assert sum(1 for x in ek if x["race_no"] == 1 and x["finish"] == 0) == 2


def test_conditional_logit_recovers_signal():
    rng = np.random.default_rng(0)
    R, F = 3000, 2
    X = rng.normal(size=(R, 6, F))
    true = np.array([1.2, -0.5])
    s = X @ true
    p = np.exp(s) / np.exp(s).sum(1, keepdims=True)
    y = np.array([rng.choice(6, p=pp) for pp in p])
    w = model.fit(X, np.ones((R, 6), bool), y, lam=1.0)
    assert np.allclose(w, true, atol=0.15)


def test_trifecta_sums_to_one_and_respects_mask():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(4, 6, 3))
    mask = np.ones((4, 6), bool)
    mask[3, 5] = False  # 5艇立てのレース
    ws = [rng.normal(size=3) * 0.3 for _ in range(3)]
    T = model.trifecta(X, mask, ws)
    assert np.allclose(T[:3].sum(1), 1.0)
    assert np.isclose(T[3].sum(), 1.0)
    for c, (i, j, k) in enumerate(model.COMBOS):
        if 5 in (i, j, k):
            assert T[3, c] == 0


def test_parse_odds_pages():
    html = """<table><tbody><tr>
    <td class="is-fs14 is-boatColor2" rowspan="4">2</td><td class="is-boatColor2">3</td><td class="oddsPoint">16.6</td>
    <td class="is-fs14 is-boatColor1" rowspan="4">1</td><td class="is-boatColor1">3</td><td class="oddsPoint">32.0</td>
    <td class="is-fs14" rowspan="4">1</td><td>2</td><td class="oddsPoint">57.9</td>
    <td class="is-fs14" rowspan="4">1</td><td>2</td><td class="oddsPoint">60.8</td>
    <td class="is-fs14" rowspan="4">1</td><td>2</td><td class="oddsPoint">776.2</td>
    <td class="is-fs14" rowspan="4">1</td><td>2</td><td class="oddsPoint">66.9</td></tr>
    <tr><td>4</td><td class="oddsPoint">19.6</td><td>4</td><td class="oddsPoint">27.9</td><td>4</td><td class="oddsPoint">52.7</td>
    <td>3</td><td class="oddsPoint">70.0</td><td>3</td><td class="oddsPoint">695.4</td><td>3</td><td class="oddsPoint">1,027</td></tr>
    </tbody></table>"""
    o = upcoming.parse_3t(html)
    assert o["1-2-3"] == 16.6 and o["2-1-3"] == 32.0 and o["1-2-4"] == 19.6 and o["6-1-3"] == 1027.0


def test_forward_record_and_evaluate(tmp_path, monkeypatch):
    import pandas as pd
    from boat import forward
    monkeypatch.setattr(forward, "LOG", tmp_path / "log.jsonl")
    cards = [{"date": "2026-10-02", "race_no": 1, "boats": [{"lane": l, "p": 1 / 6} for l in range(1, 7)],
              "win_odds": {1: 3.0, 6: 8.0}, "ev_all": [{"c": "6-1-3", "p": 0.05, "odds": 40.0}]},
             {"date": "2026-10-02", "race_no": 2, "boats": [], "win_odds": {}}]  # オッズ無しは記録しない
    assert forward.record(cards) == 1
    races = pd.DataFrame([{"date": pd.Timestamp("2026-10-02"), "race_no": 1, "win_pay": 800, "win_lane": 6,
                           "3連単_combo": "6-1-3", "3連単_pay": 4000}])
    ev = forward.evaluate(races)
    assert ev["settled"] == 1
    s = {x["label"]: x for x in ev["strategies"]}
    # 単勝: 6号艇は 1/6×8.0=1.33 ≥1.0 → 的中で8.0倍 / 3連単 6-1-3 は 0.05×40=2.0 → 40倍
    assert s["単勝: 期待値1.0以上の艇すべて"]["roi"] == 8.0
    assert s["3連単: 期待値1.5以上の組合せすべて(確率1%以上)"]["roi"] == 40.0


def test_forward_blend_strategy(tmp_path, monkeypatch):
    import pandas as pd
    from boat import forward
    monkeypatch.setattr(forward, "LOG", tmp_path / "log.jsonl")
    # モデルが6号艇を強く推し、市場は6号艇を軽視 → 混合確率×オッズが基準を超えて買われる
    cards = [{"date": "2026-10-02", "race_no": 1,
              "boats": [{"lane": l, "p": (0.5 if l == 6 else 0.1)} for l in range(1, 7)],
              "win_odds": {1: 2.0, 2: 6.0, 3: 8.0, 4: 10.0, 5: 12.0, 6: 14.0}, "ev_all": []}]
    forward.record(cards)
    races = pd.DataFrame([{"date": pd.Timestamp("2026-10-02"), "race_no": 1, "win_pay": 1400, "win_lane": 6,
                           "3連単_combo": "6-1-2", "3連単_pay": 9000}])
    ev = forward.evaluate(races)
    s = [x for x in ev["strategies"] if "混合" in x["label"]]
    assert s and any(x["bets"] >= 1 and x["hits"] >= 1 for x in s)


def test_boat_data_served_from_github_with_fallback(tmp_path, monkeypatch):
    import httpx
    from fastapi.testclient import TestClient

    from app.config import Settings
    from app.main import create_app

    def make():
        return TestClient(create_app(Settings(webhook_secret="x", journal_path=str(tmp_path / "j.jsonl"))))

    class R:
        status_code = 200
        content = b'{"built": "GITHUB"}'

    monkeypatch.setattr(httpx, "get", lambda *a, **k: R())
    assert make().get("/boat/data.json").json() == {"built": "GITHUB"}  # GitHub の最新が出る

    def boom(*a, **k):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx, "get", boom)
    r = make().get("/boat/data.json")  # 取れないときは同梱のデータ
    assert r.status_code == 200 and "validation" in r.json()
    assert make().get("/boat/").status_code == 200  # ページ本体も出る
