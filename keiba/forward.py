"""前向き記録: 予想した時点のモデル勝率とオッズを保存し、結果が出たら答え合わせする。

過去のレースの最終オッズ(全頭ぶん)は無料で取れないので、『期待値(勝率×オッズ)が高い馬を買うと儲かるか』
は過去データでは検証できない。そこで毎週、発走前の予想とオッズを記録しておき、結果が出てから
実際の払戻で評価する(予想時点の情報だけなので、後知恵が入らない)。

- record(races): predict_upcoming から呼ばれる。発走後の予想は記録しない。同じレースは
  「オッズ付きの新しい予想」で置き換える
- python -m keiba.forward : 答え合わせして keiba/site/forward.json を作る
"""
from __future__ import annotations

import csv
import gzip
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
STORE = HERE / "data" / "forward.json"
OUT = HERE / "site" / "forward.json"
ALL = HERE / "data" / "all"
JST = timezone(timedelta(hours=9))


def _start(date: str, start: str) -> datetime | None:
    m = re.match(r"(\d+)時(\d+)分", start or "")
    if not m:
        return None
    return datetime.fromisoformat(date).replace(hour=int(m[1]), minute=int(m[2]), tzinfo=JST)


def load_store() -> dict:
    return json.loads(STORE.read_text(encoding="utf-8")) if STORE.exists() else {}


def record(races: list[dict], now: datetime | None = None) -> int:
    """発走前のレースだけ記録する。戻り値=記録(更新)したレース数。"""
    now = now or datetime.now(JST)
    store = load_store()
    n = 0
    for r in races:
        st = _start(r["date"], r.get("start", ""))
        if st is None or now >= st:
            continue
        old = store.get(r["id"])
        if old and old["with_odds"] and not r["with_odds"]:
            continue  # オッズ付きの記録がすでにあるなら、オッズ無しで上書きしない
        store[r["id"]] = dict(
            date=r["date"], venue=r["venue"], no=r["no"], name=r["name"], klass=r["klass"], grade=r["grade"],
            start=r["start"], snap=now.strftime("%Y-%m-%d %H:%M"), with_odds=r["with_odds"],
            horses=[[h["num"], h["name"], h["pw"], h["odds"]] for h in r["horses"]],
        )
        n += 1
    STORE.parent.mkdir(exist_ok=True)
    STORE.write_text(json.dumps(store, ensure_ascii=False, separators=(",", ":"), sort_keys=True), encoding="utf-8")
    return n


def _results(years: set[str]) -> dict:
    """(日付, 場, R, 馬番) -> (着順, 単勝払戻, 複勝払戻)"""
    out = {}
    for y in years:
        f = ALL / f"{y}.csv.gz"
        if not f.exists():
            continue
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                out[(row["date"], row["venue"], int(row["race_no"]), int(row["num"]))] = (
                    int(float(row["place"] or 0)), int(float(row["win_pay"] or 0)), int(float(row["place_pay"] or 0)))
    return out


def _roi(rows: list[tuple]) -> dict:
    """rows: (単勝払戻, 複勝払戻, 勝ったか) を1点100円で買い続けた成績"""
    n = len(rows)
    if n == 0:
        return dict(bets=0)
    w = np.array([r[0] / 100.0 for r in rows])
    p = np.array([r[1] / 100.0 for r in rows])
    return dict(bets=n, hit=float(np.mean([r[2] for r in rows])), win_roi=float(w.mean()),
                win_se=float(w.std(ddof=1) / np.sqrt(n)) if n > 1 else None, place_roi=float(p.mean()))


def evaluate() -> dict:
    store = load_store()
    res = _results({v["date"][:4] for v in store.values()})
    picks, favs = [], []
    ev = {1.0: [], 1.2: [], 1.5: []}
    graded = graded_odds = 0
    for rid, v in store.items():
        rows = []
        for num, name, pw, odds in v["horses"]:
            r = res.get((v["date"], v["venue"], v["no"], num))
            if r and r[0] > 0:
                rows.append((num, pw, odds, r))
        if len(rows) < max(2, len(v["horses"]) - 3):  # 結果がまだ/不完全(中止・除外を考慮)
            continue
        graded += 1
        top = max(rows, key=lambda x: x[1])
        picks.append((top[3][1], top[3][2], top[3][0] == 1))
        if v["with_odds"]:
            graded_odds += 1
            fav = min((x for x in rows if x[2]), key=lambda x: x[2], default=None)
            if fav:
                favs.append((fav[3][1], fav[3][2], fav[3][0] == 1))
            for k in ev:
                for num, pw, odds, r in rows:
                    if odds and pw * odds >= k:
                        ev[k].append((r[1], r[2], r[0] == 1))
    first = min((v["date"] for v in store.values()), default=None)
    out = dict(
        generated=datetime.now(JST).strftime("%Y-%m-%d %H:%M"), recorded=len(store), graded=graded,
        graded_with_odds=graded_odds, since=first,
        strategies=[dict(name="モデルの本命(各レース1頭)", **_roi(picks)),
                    dict(name="1番人気(各レース1頭)", **_roi(favs))] +
                   [dict(name=f"期待値{k}以上の馬すべて", **_roi(v)) for k, v in ev.items()],
    )
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return out


if __name__ == "__main__":
    o = evaluate()
    print(f"記録 {o['recorded']}レース / 答え合わせ済み {o['graded']}(オッズ付き {o['graded_with_odds']})")
    for s in o["strategies"]:
        if s["bets"]:
            se = f"±{s['win_se']:.0%}" if s.get("win_se") is not None else ""
            print(f"  {s['name']:<22}{s['bets']:>5}点 的中{s['hit']:.1%} 単勝{s['win_roi']:.1%}{se} 複勝{s['place_roi']:.1%}")
