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
