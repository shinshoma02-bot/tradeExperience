"""台データ → 狙い目候補 + 答え合わせ。  python -m pachinko.model

ルール(結果を見る前に固定。後から調整しない):
  cont 前日好調継続: 前日の差が+で、稼働が十分な台のうち差が大きい上位N台
  rev  前日凹み反発: 前日の差が-で、稼働が十分な台のうち差が小さい(凹んだ)上位N台
  base 基準: 稼働が十分な全台(これと比べて初めて意味がある)
結果: 翌営業日に差がプラスだった割合(勝率)と、1台あたり平均差。
予想は forward.json に「その日の前に」固定して残す(後出しを防ぐ)。
"""
import json
import math
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

from .ingest import load

HERE = Path(__file__).parent
SITE = HERE / "site" / "data.json"
FORWARD = HERE / "data" / "forward.json"
TOP_N = 5
MIN_GAMES = {"slot": 3000, "pachi": 2000}  # 稼働が少ない台は偶然が大きいので除く
STRATS = ("cont", "rev")


def _next_day(d: str) -> str:
    return (date.fromisoformat(d) + timedelta(days=1)).isoformat()


def picks(day_rows: list[dict]) -> dict[str, list[dict]]:
    """1日分の台(同じ店・同じ種類)から、翌日用の候補を返す。"""
    out = {s: [] for s in STRATS}
    groups = defaultdict(list)
    for r in day_rows:
        if r["games"] >= MIN_GAMES[r["kind"]]:
            groups[(r["hall"], r["kind"])].append(r)
    for rows in groups.values():
        up = sorted((r for r in rows if r["diff"] > 0), key=lambda r: -r["diff"])[:TOP_N]
        dn = sorted((r for r in rows if r["diff"] < 0), key=lambda r: r["diff"])[:TOP_N]
        out["cont"] += up
        out["rev"] += dn
    return out


def _typed(rows):
    return [{**r, "games": int(r["games"]), "bonus": int(r["bonus"]), "diff": int(r["diff"])} for r in rows]


def _stat(vals: list[int]) -> dict:
    n = len(vals)
    if not n:
        return {"n": 0}
    p = sum(v > 0 for v in vals) / n
    mean = sum(vals) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    return {"n": n, "win": round(p, 4), "win_se": round(math.sqrt(p * (1 - p) / n), 4),
            "mean": round(mean, 1), "mean_se": round(sd / math.sqrt(n), 1)}


def evaluate(rows: list[dict]) -> dict:
    """前日までのデータだけで選び、翌日の実績で採点(ウォークフォワード)。"""
    by_day = defaultdict(list)
    for r in rows:
        by_day[r["date"]].append(r)
    res = {k: {s: [] for s in (*STRATS, "base")} for k in ("slot", "pachi")}
    days = sorted(by_day)
    for d in days:
        prev = (date.fromisoformat(d) - timedelta(days=1)).isoformat()
        if prev not in by_day:  # 連続した営業日だけ比べる
            continue
        today = {(r["hall"], r["no"], r["name"]): r for r in by_day[d] if r["games"] >= MIN_GAMES[r["kind"]]}
        for r in today.values():
            res[r["kind"]]["base"].append(r["diff"])
        for s, lst in picks(by_day[prev]).items():
            for p in lst:
                t = today.get((p["hall"], p["no"], p["name"]))  # 機種入替は除く
                if t:
                    res[t["kind"]][s].append(t["diff"])
    return {k: {s: _stat(v) for s, v in d.items()} for k, d in res.items()}


def freeze_forward(rows: list[dict]) -> dict:
    """最新日のデータから翌日の候補を作り、forward.json に一度だけ固定する。"""
    if not rows:
        return {}
    last = max(r["date"] for r in rows)
    target = _next_day(last)
    fw = json.loads(FORWARD.read_text("utf-8")) if FORWARD.exists() else {}
    if target not in fw:
        pk = picks([r for r in rows if r["date"] == last])
        fw[target] = {s: [{k: p[k] for k in ("hall", "no", "name", "kind", "diff", "games", "bonus")} for p in lst]
                      for s, lst in pk.items()}
        FORWARD.parent.mkdir(parents=True, exist_ok=True)
        FORWARD.write_text(json.dumps(fw, ensure_ascii=False, indent=1), "utf-8")
    return fw


def score_forward(rows: list[dict], fw: dict) -> dict:
    """固定済みの予想を、結果が出た日だけ採点する(後出しなし)。"""
    actual = {(r["date"], r["hall"], r["no"], r["name"]): r["diff"] for r in rows}
    res = {k: {s: [] for s in STRATS} for k in ("slot", "pachi")}
    for d, strats in fw.items():
        for s, lst in strats.items():
            for p in lst:
                v = actual.get((d, p["hall"], p["no"], p["name"]))
                if v is not None:
                    res[p["kind"]][s].append(v)
    return {k: {s: _stat(v) for s, v in d.items()} for k, d in res.items()}


def hall_tendency(rows: list[dict]) -> list[dict]:
    """店×曜日の平均差(参考)。日数が少ないうちは偶然。"""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["games"] >= MIN_GAMES[r["kind"]]:
            wd = "月火水木金土日"[date.fromisoformat(r["date"]).weekday()]
            acc[(r["hall"], r["kind"])][wd].append(r["diff"])
    out = []
    for (hall, kind), w in sorted(acc.items()):
        out.append({"hall": hall, "kind": kind,
                    "weekday": {d: {"n": len(v), "mean": round(sum(v) / len(v), 1)} for d, v in w.items()}})
    return out


def build() -> dict:
    rows = _typed(load())
    fw = freeze_forward(rows)
    last = max((r["date"] for r in rows), default=None)
    target = _next_day(last) if last else None
    data = {
        "updated": last, "target": target,
        "days": len({r["date"] for r in rows}), "rows": len(rows),
        "halls": sorted({r["hall"] for r in rows}),
        "walk_forward": evaluate(rows),
        "forward": score_forward(rows, fw),
        "picks": fw.get(target, {}) if target else {},
        "tendency": hall_tendency(rows),
        "rules": {"top_n": TOP_N, "min_games": MIN_GAMES},
    }
    SITE.parent.mkdir(parents=True, exist_ok=True)
    SITE.write_text(json.dumps(data, ensure_ascii=False), "utf-8")
    return data


if __name__ == "__main__":
    d = build()
    print(f"{d['days']}日 {d['rows']}行 最新 {d['updated']} → 候補 {d['target']}")
    for k, v in d["walk_forward"].items():
        for s, x in v.items():
            if x["n"]:
                print(f"  {k:5s} {s:5s} n={x['n']:5d} 勝率 {x['win']:.1%}±{x['win_se']:.1%} 平均差 {x['mean']:+.0f}±{x['mean_se']:.0f}")
