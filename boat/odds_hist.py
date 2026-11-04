"""過去レースの締切時オッズ(単勝・3連単の全組合せ)を公式サイトから集める。

これで「確率×オッズ(期待値)が高い買い目は本当に得か」を過去データで検証できる。
- 取得元: boatrace.jp の oddstf / odds3t (robots.txt は全許可。1秒に1リクエスト以内)
- 取得済みのレースは取り直さない(boat/_cache/odds/YYYYMMDD_RR.json)。新しい日から順に取る
- 持ち運び用に boat/data/odds/YYYY-MM.json.gz にまとめる(python -m boat.odds_hist pack / unpack)
使い方: python -m boat.odds_hist fetch 2025-01-01
"""
from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

from boat import scrape, upcoming

HERE = Path(__file__).parent
CACHE = HERE / "_cache" / "odds"
PACK = HERE / "data" / "odds"


def racing_days(start: date) -> list[date]:
    """結果ファイル(K)が空でない日 = 丸亀の開催日。新しい順。"""
    days = []
    for f in scrape.CACHE.glob("K*.txt"):
        d = date(int(f.stem[1:5]), int(f.stem[5:7]), int(f.stem[7:9]))
        if d >= start and f.stat().st_size > 0:
            days.append(d)
    return sorted(days, reverse=True)


def fetch_one(job: tuple[date, int]) -> None:
    d, rno = job
    f = CACHE / f"{d:%Y%m%d}_{rno:02d}.json"
    if f.exists():
        return
    win = upcoming.parse_win_odds(scrape.get_html(upcoming._url("oddstf", d, rno)))
    o3 = upcoming.parse_3t(scrape.get_html(upcoming._url("odds3t", d, rno)))
    f.write_text(json.dumps({"win": win, "o3": o3}, ensure_ascii=False), encoding="utf-8")


def fetch(start: date) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    jobs = [(d, r) for d in racing_days(start) for r in range(1, 13)]
    done = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for _ in ex.map(fetch_one, jobs):
            done += 1
            if done % 200 == 0:
                print(done, "/", len(jobs), flush=True)


def load() -> dict[tuple[str, int], dict]:
    """{(日付 'YYYY-MM-DD', レース番号): {'win': {枠: オッズ}, 'o3': {'1-2-3': オッズ}}}"""
    out = {}
    for f in CACHE.glob("*.json"):
        d, r = f.stem.split("_")
        v = json.loads(f.read_text(encoding="utf-8"))
        if v["o3"] or v["win"]:
            out[(f"{d[:4]}-{d[4:6]}-{d[6:]}", int(r))] = {"win": {int(k): x for k, x in v["win"].items()}, "o3": v["o3"]}
    return out


def pack() -> int:
    groups: dict[str, dict] = defaultdict(dict)
    for f in sorted(CACHE.glob("*.json")):
        groups[f"{f.stem[:4]}-{f.stem[4:6]}"][f.name] = json.loads(f.read_text(encoding="utf-8"))
    PACK.mkdir(parents=True, exist_ok=True)
    n = 0
    for ym, files in groups.items():
        data = gzip.compress(json.dumps(files, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(), 9, mtime=0)
        out = PACK / f"{ym}.json.gz"
        if not out.exists() or out.read_bytes() != data:
            out.write_bytes(data)
            n += 1
    return n


def unpack() -> int:
    CACHE.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(PACK.glob("*.json.gz")):
        for name, v in json.loads(gzip.decompress(f.read_bytes())).items():
            p = CACHE / name
            if not p.exists():
                p.write_text(json.dumps(v, ensure_ascii=False), encoding="utf-8")
                n += 1
    return n


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "fetch":
        fetch(date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else date(2025, 1, 1))
    else:
        print({"pack": pack, "unpack": unpack}[cmd]())
