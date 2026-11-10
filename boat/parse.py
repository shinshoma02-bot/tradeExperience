"""番組表B・競走成績Kの丸亀ブロックを表(DataFrame)にする。

entries: 1行=1艇の出走(番組表の情報 + あれば結果)
races  : 1行=1レース(天候・風・払戻など)
"""
from __future__ import annotations

import re
import unicodedata
from datetime import date
from pathlib import Path

import pandas as pd

from boat import venues

CACHE = Path(__file__).parent / "_cache"  # 丸亀(既定)の置き場所。他の場は venues.cache_dir(場コード)

CLASS = {"A1": 3, "A2": 2, "B1": 1, "B2": 0}
# 丸亀の水面は海水(潮の影響あり)。風向は16方位のまま持つ
_RACE_B = re.compile(r"^\s*(\d{1,2})R\s+(.+?)\s+H(\d+)m\s+電話投票締切予定(\d{1,2}):(\d{2})")
_ENT_B = re.compile(
    r"^([1-6]) (\d{4})(.{4})(\d{2})(\S{2})(\d{2})([AB][12])\s*"
    r"(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)\s+(\d+\.\d+)\s+(\d+)\s+(\d+\.\d+)"
)
_RACE_K = re.compile(r"^\s*(\d{1,2})R\s+(.+?)\s+H(\d+)m\s+(\S+)\s+風\s+(\S*)\s*(\d+)m\s+波\s+(\d+)cm")
_ENT_K = re.compile(
    r"^\s+(\S+)\s+([1-6])\s+(\d{4})\s+(.+?)\s+(\d+)\s+(\d+)\s+(?:(\d\.\d\d)|\S*)\s+(?:([1-6])|\S*)\s+(\S+)\s+(\d\.\d{2}|\.)"
)
_PAY = {
    "単勝": re.compile(r"^\s+単勝\s+([1-6])\s+(\d+)"),
    "2連単": re.compile(r"^\s+2連単\s+([1-6])-([1-6])\s+(\d+)\s+人気\s+(\d+)"),
    "2連複": re.compile(r"^\s+2連複\s+([1-6])-([1-6])\s+(\d+)\s+人気\s+(\d+)"),
    "3連単": re.compile(r"^\s+3連単\s+([1-6])-([1-6])-([1-6])\s+(\d+)\s+人気\s+(\d+)"),
    "3連複": re.compile(r"^\s+3連複\s+([1-6])-([1-6])-([1-6])\s+(\d+)\s+人気\s+(\d+)"),
}


def _norm(t: str) -> str:
    return unicodedata.normalize("NFKC", t)


def parse_b(text: str, d: date, name: str = "丸亀") -> tuple[list[dict], list[dict]]:
    """番組表 → (races, entries)。name は場名(開催名の行を探すのに使う)。"""
    text = _norm(text)
    m = re.search(r"第\s*(\d+)日", text)
    day_no = int(m.group(1)) if m else None
    m2 = re.search(r"\s*".join(re.escape(c) for c in name) + r"\s+\d+月\s*\d+日\s+(\S+)", text)
    title = m2.group(1) if m2 else ""
    races, ents = [], []
    cur = None
    for ln in text.splitlines():
        r = _RACE_B.match(ln)
        if r:
            no = int(r.group(1))
            cur = {"date": d, "race_no": no, "kind": r.group(2).split()[0], "fixed": "進入固定" in r.group(2), "dist": int(r.group(3)),
                   "close": f"{int(r.group(4)):02d}:{r.group(5)}", "day_no": day_no, "title": title}
            races.append(cur)
            continue
        e = _ENT_B.match(ln)
        if e and cur:
            g = e.groups()
            ents.append({
                "date": d, "race_no": cur["race_no"], "lane": int(g[0]), "reg": int(g[1]),
                "name": g[2].replace(" ", "").replace("　", ""), "age": int(g[3]), "branch": g[4],
                "weight": int(g[5]), "cls": g[6], "nat_win": float(g[7]), "nat_2r": float(g[8]),
                "loc_win": float(g[9]), "loc_2r": float(g[10]), "motor": int(g[11]),
                "motor_2r": float(g[12]), "boat": int(g[13]), "boat_2r": float(g[14]),
            })
    return races, ents


def parse_k(text: str, d: date) -> tuple[list[dict], list[dict]]:
    """競走成績 → (races, results)。races は天候・風・払戻、results は着順・展示・ST。"""
    text = _norm(text)
    races, res = [], []
    cur = None
    for ln in text.splitlines():
        r = _RACE_K.match(ln)
        if r:
            cur = {"date": d, "race_no": int(r.group(1)), "weather": r.group(4), "wind_dir": r.group(5),
                   "wind": int(r.group(6)), "wave": int(r.group(7)),
                   "kimarite": ""}
            races.append(cur)
            continue
        if cur is None:
            continue
        if ln.lstrip().startswith("着 艇"):
            toks = ln.split()
            cur["kimarite"] = toks[-1] if toks[-1] not in ("ﾚｰｽﾀｲﾑ", "レースタイム") else ""
            continue
        for k, rx in _PAY.items():
            p = rx.match(ln)
            if p:
                g = p.groups()
                if k == "単勝":
                    cur["win_lane"], cur["win_pay"] = int(g[0]), int(g[1])
                elif k in ("2連単", "2連複"):
                    cur[f"{k}_combo"] = "-".join(g[:2]); cur[f"{k}_pay"] = int(g[2]); cur[f"{k}_pop"] = int(g[3])
                else:
                    cur[f"{k}_combo"] = "-".join(g[:3]); cur[f"{k}_pay"] = int(g[3]); cur[f"{k}_pop"] = int(g[4])
                break
        else:
            e = _ENT_K.match(ln)
            if e:
                g = e.groups()
                finish = int(g[0]) if g[0].isdigit() else 0  # 0=完走せず(F/L/K/S/欠)
                st = re.sub(r"^[FL]", "", g[8])  # F/Lは「フライング/出遅れ」。数値だけ取る
                res.append({
                    "date": d, "race_no": cur["race_no"], "lane": int(g[1]), "reg": int(g[2]),
                    "finish": finish, "flag": "" if g[0].isdigit() else g[0],
                    "ex_time": float(g[6]) if g[6] else None,
                    "course": int(g[7]) if g[7] else None,
                    "st": float(st) if re.fullmatch(r"\d\.\d{2}", st) else None,
                })
    return races, res


def load(start: date | None = None, venue: str = venues.DEFAULT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """その場のキャッシュ全部を読んで (entries, races) を返す。結果が無いレースも entries には残る(今後の予想用)。"""
    CACHE = venues.cache_dir(venue)
    name = venues.VENUES[venue]
    ent_rows, race_b, res_rows, race_k = [], [], [], []
    for f in sorted(CACHE.glob("B*.txt")):
        d = date(int(f.stem[1:5]), int(f.stem[5:7]), int(f.stem[7:9]))
        if start and d < start:
            continue
        t = f.read_text(encoding="utf-8")
        if not t:
            continue
        rb, eb = parse_b(t, d, name)
        race_b += rb
        ent_rows += eb
    for f in sorted(CACHE.glob("K*.txt")):
        d = date(int(f.stem[1:5]), int(f.stem[5:7]), int(f.stem[7:9]))
        if start and d < start:
            continue
        t = f.read_text(encoding="utf-8")
        if not t:
            continue
        rk, ek = parse_k(t, d)
        race_k += rk
        res_rows += ek
    ent = pd.DataFrame(ent_rows)
    res = pd.DataFrame(res_rows)
    races = pd.DataFrame(race_b)
    rk = pd.DataFrame(race_k)
    if len(rk):
        races = races.merge(rk, on=["date", "race_no"], how="outer")
    if len(ent) and len(res):
        ent = ent.merge(res, on=["date", "race_no", "lane"], how="left", suffixes=("", "_k"))
    for df in (ent, races):
        if len(df):
            df["date"] = pd.to_datetime(df["date"])
    return ent, races
