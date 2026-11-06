import random
from datetime import date, timedelta

from pachinko import ingest, model


def _rows(days=40, halls=("A店",), seed=1):
    rnd = random.Random(seed)
    out = []
    d0 = date(2026, 10, 1)
    for i in range(days):
        d = (d0 + timedelta(days=i)).isoformat()
        for h in halls:
            for no in range(1, 31):
                out.append({"date": d, "hall": h, "no": str(no), "name": "X", "kind": "slot",
                            "games": str(rnd.randint(2000, 8000)), "bonus": "10",
                            "diff": str(rnd.randint(-1500, 1500))})
    return out


def test_clean_rejects_bad():
    assert ingest.clean({"date": "2026-10-01", "hall": "A", "no": "1", "name": "n", "kind": "slot",
                         "games": "1,000", "bonus": "3", "diff": "+120"})["diff"] == "120"
    assert ingest.clean({"date": "bad", "hall": "A", "no": "1", "kind": "slot", "games": "1", "bonus": "1", "diff": "1"}) is None
    assert ingest.clean({"date": "2026-10-01", "hall": "A", "no": "1", "kind": "x", "games": "1", "bonus": "1", "diff": "1"}) is None


def test_merge_dedup(tmp_path):
    p = tmp_path / "r.csv"
    rows = _rows(2)
    assert ingest.merge(rows, p) == (60, 0)
    assert ingest.merge(rows, p) == (0, 0)
    assert len(ingest.load(p)) == 60


def test_random_data_has_no_edge():
    # 乱数データでは、候補も基準も勝率はほぼ同じになるはず(予想が効くと誤認しない)
    rows = model._typed([ingest.clean(r) for r in _rows(60, ("A店", "B店"))])
    ev = model.evaluate(rows)["slot"]
    for s in model.STRATS:
        assert abs(ev[s]["win"] - ev["base"]["win"]) < 4 * ev[s]["win_se"] + 0.02


def test_picks_use_only_given_day():
    rows = model._typed([ingest.clean(r) for r in _rows(2)])
    pk = model.picks([r for r in rows if r["date"] == rows[0]["date"]])
    assert len(pk["cont"]) <= model.TOP_N and all(p["diff"] > 0 for p in pk["cont"])
    assert all(p["diff"] < 0 for p in pk["rev"])
