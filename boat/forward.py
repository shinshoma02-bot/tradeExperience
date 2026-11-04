"""前向き記録: 開催前に「予想の確率」と「その時点のオッズ」を記録し、結果が出たら答え合わせする。

過去のオッズ(全組合せ)は無料で取れないので、「確率×オッズ(期待値)が高い買い目は本当に得か」は
過去データでは検証できない。そこで、予想を出した時点のオッズごと毎回ここに残していく。
- 記録: boat/data/forward_log.jsonl(git管理する)。同じレースは何度記録してもよい(評価では最後の1件を使う)
- 評価: 結果(競走成績K)が出たレースだけ。払戻は公式の確定払戻を使う
- 注意: 最後の記録がレース開始後だと「締切時オッズ」に近くなる。記録はなるべく締切前に
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

LOG = Path(__file__).parent / "data" / "forward_log.jsonl"
JST = timezone(timedelta(hours=9))


def record(cards: list[dict]) -> int:
    """オッズが取れているレースだけ記録。戻り値は記録した件数。"""
    rows = []
    now = datetime.now(JST).strftime("%Y-%m-%d %H:%M")
    for c in cards:
        if not (c.get("win_odds") or c.get("ev_all")):
            continue
        rows.append({
            "logged": now, "date": c["date"], "race_no": c["race_no"],
            "with_ex": c.get("top_x") is not None,
            "p": {b["lane"]: b["px"] if (c.get("top_x") is not None and b.get("px") is not None) else b["p"]
                  for b in c["boats"]},
            "win_odds": c.get("win_odds") or {},
            "combos": c.get("ev_all") or [],  # {c, p, odds}: 確率1%以上の3連単
        })
    if rows:
        LOG.parent.mkdir(exist_ok=True)
        with LOG.open("a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def _last_per_race() -> list[dict]:
    if not LOG.exists():
        return []
    last: dict = {}
    for ln in LOG.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln)
            last[(r["date"], r["race_no"])] = r
    return list(last.values())


def evaluate(races: pd.DataFrame) -> dict:
    logs = _last_per_race()
    res = races.dropna(subset=["win_pay"]).copy()
    res["date"] = res["date"].dt.strftime("%Y-%m-%d")
    rmap = res.set_index(["date", "race_no"])
    done = [(r, rmap.loc[(r["date"], r["race_no"])]) for r in logs if (r["date"], r["race_no"]) in rmap.index]
    out = {"logged": len(logs), "settled": len(done), "pending": len(logs) - len(done), "strategies": []}

    def add(label, rets, nbets):
        if not nbets:
            return
        rets = np.array(rets, float)
        out["strategies"].append({"label": label, "bets": int(nbets), "hits": int((rets > 0).sum()),
                                  "roi": float(rets.sum() / nbets)})

    # 単勝: 確率×オッズが基準以上の艇に100円ずつ / 基準: 一番人気(最低オッズ)の艇
    for th in (1.0, 1.2, 1.5):
        rets, n = [], 0
        for r, x in done:
            for lane, o in r["win_odds"].items():
                if r["p"].get(str(lane), r["p"].get(lane, 0)) * o >= th:
                    n += 1
                    rets.append(x["win_pay"] / 100.0 if int(x["win_lane"]) == int(lane) else 0.0)
        add(f"単勝: 期待値{th}以上の艇すべて", rets, n)
    rets, n = [], 0
    for r, x in done:
        if r["win_odds"]:
            fav = min(r["win_odds"], key=lambda k: r["win_odds"][k])
            n += 1
            rets.append(x["win_pay"] / 100.0 if int(x["win_lane"]) == int(fav) else 0.0)
    add("単勝: 一番人気(基準)", rets, n)
    # 3連単: 確率1%以上で期待値が基準以上の組合せすべて
    for th in (1.0, 1.2, 1.5):
        rets, n = [], 0
        for r, x in done:
            for cb in r["combos"]:
                if cb["p"] * cb["odds"] >= th:
                    n += 1
                    rets.append(x["3連単_pay"] / 100.0 if x["3連単_combo"] == cb["c"] else 0.0)
        add(f"3連単: 期待値{th}以上の組合せすべて(確率1%以上)", rets, n)
    return out
