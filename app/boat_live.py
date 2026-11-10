"""丸亀ボート: 当日のレース結果を、公式の「レース結果」ページから取ってキャッシュする。

- 終わったレースだけ(締切の数分後から)取りに行く。1レースにつき1分に1回まで、取れたら取り直さない
- 公式サイトへは1秒に1リクエスト以内(ロックで直列化)。閲覧者が何人いてもアクセス数は増えない
- 依存を増やさないため、HTMLはタグを剥がして正規表現で読む(bs4は使わない)
"""
from __future__ import annotations

import html as _html
import math
import re
import threading
import time
import unicodedata
from datetime import datetime, timedelta, timezone
from itertools import permutations

import httpx

JST = timezone(timedelta(hours=9))
URL = "https://www.boatrace.jp/owpc/pc/race/raceresult?rno={rno}&jcd={jcd}&hd={ymd}"
BEFORE_URL = "https://www.boatrace.jp/owpc/pc/race/beforeinfo?rno={rno}&jcd={jcd}&hd={ymd}"
PRE_BEFORE_MIN = 20  # 締切の何分前から直前の展示を取りに行くか
PRE_CACHE_SEC = 120  # 展示は2分ごとに取り直す(取りすぎない)
VENUES = ("15", "14", "16")  # 丸亀・鳴門・児島(boat/venues.py と同じ。Flyのイメージには boat/ が入らないので別に持つ)
UA = "tradeExperience-boat-research/0.1 (personal study; 1 req/sec)"
WAIT_AFTER_CLOSE_MIN = 3  # 締切からこの分数たったら結果ページを見に行く
RETRY_SEC = 60
MAX_FETCH_PER_CALL = 3

_lock = threading.Lock()
_results: dict[str, dict[int, dict]] = {}  # 場:日付 -> レース番号 -> 結果
_tried: dict[tuple[str, str, int], float] = {}
_last_fetch = 0.0


def page_text(html: str) -> str:
    html = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
    text = _html.unescape(re.sub(r"<[^>]+>", " ", html))
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text))


def parse_result_text(text: str) -> dict:
    """結果ページ(タグを剥がした文字列)から着順・払戻・決まり手。未終了なら空。"""
    m3 = re.search(r"3連単 ([1-6]) - ([1-6]) - ([1-6]) ¥([\d,]+) (\d+)", text)
    if not m3:
        return {}
    out: dict = {"tri": "-".join(m3.group(i) for i in (1, 2, 3)), "tri_pay": int(m3.group(4).replace(",", ""))}
    order = re.findall(r"(?<![\d.])([1-6]) ([1-6]) (\d{4}) \S+(?: \S+)? \d'\d\d\"\d", text)
    out["order"] = [int(lane) for _, lane, _ in order]  # 着順(1着から)の枠番
    w = re.search(r"単勝 ([1-6]) ¥([\d,]+)", text)
    if w:
        out["win_lane"], out["win_pay"] = int(w.group(1)), int(w.group(2).replace(",", ""))
    k = re.search(r"決まり手 (\S+)", text)
    if k:
        out["kimarite"] = k.group(1)
    seg = re.search(r"スタート情報(.*?)勝式", text)
    if seg:
        out["st"] = {int(a): float(b.lstrip("FL")) for a, b in re.findall(r"([1-6]) (F?L?\.\d+)", seg.group(1))}
    return out


def _fetch(day: str, rno: int, venue: str = "15") -> dict:
    global _last_fetch
    wait = 1.0 - (time.time() - _last_fetch)
    if wait > 0:
        time.sleep(wait)
    _last_fetch = time.time()
    try:
        r = httpx.get(URL.format(rno=rno, jcd=venue, ymd=day.replace("-", "")), headers={"User-Agent": UA}, timeout=15)
        if r.status_code == 200:
            return parse_result_text(page_text(r.text))
    except httpx.HTTPError:
        pass
    return {}


def results_for_today(races: list[dict], now: datetime | None = None, venue: str = "15") -> dict:
    """races: [{date, race_no, close('HH:MM')}]。今日ぶんで、締切後のレースの結果を取って返す。venue は場コード。"""
    now = now or datetime.now(JST)
    today = now.strftime("%Y-%m-%d")
    with _lock:
        have = _results.setdefault(f"{venue}:{today}", {})
        fetched = 0
        for r in sorted((x for x in races if x.get("date") == today), key=lambda x: x["race_no"]):
            rno = r["race_no"]
            if rno in have or fetched >= MAX_FETCH_PER_CALL or not r.get("close"):
                continue
            hh, mm = (int(x) for x in r["close"].split(":"))
            due = now.replace(hour=hh, minute=mm, second=0, microsecond=0) + timedelta(minutes=WAIT_AFTER_CLOSE_MIN)
            if now < due or time.time() - _tried.get((venue, today, rno), 0) < RETRY_SEC:
                continue
            _tried[(venue, today, rno)] = time.time()
            fetched += 1
            res = _fetch(today, rno, venue)
            if res:
                have[rno] = res
        return {"date": today, "asof": now.strftime("%H:%M"),
                "results": {str(k): v for k, v in have.items()}}


# ───────── 直前予想(展示タイム・水面の情報で確率を出し直す) ─────────
# boat/model.py が書いた site/pre*.json(重み・出走表の素材)を使い、標準ライブラリだけで計算する
# (サーバーには numpy が無いため)。計算式は boat/model.py の build / win_probs / trifecta と同じ。
_BEFORE_RE = re.compile(r"(?<![\d.])([1-6]) (\S+(?: \S+)?) (\d{2}\.\d)kg (\d\.\d{2}) (-?\d\.\d)")
_net_lock = threading.Lock()
_last_net = 0.0
_before_cache: dict[tuple[str, str, int], tuple[float, dict]] = {}
_live_pred: dict[tuple[str, str, int], dict] = {}


def parse_ex(html: str) -> dict[int, float]:
    """直前情報のページ → {枠: 展示タイム}。6艇そろっていなければ空。"""
    t = page_text(html)
    # 並び: 枠・選手名・体重(kg)・展示タイム・チルト。展示タイムは4番目の数字
    out = {int(a): float(d) for a, _b, _c, d, _e in _BEFORE_RE.findall(t)}
    if len(out) < 6 or all(v == 0 for v in out.values()):
        return {}
    return out


def _softmax(scores: list[float], mask: list[bool]) -> list[float]:
    m = [sc for sc, ok in zip(scores, mask) if ok]
    if not m:
        return [0.0] * len(scores)
    mx = max(m)
    e = [math.exp(sc - mx) if ok else 0.0 for sc, ok in zip(scores, mask)]
    t = sum(e)
    return [v / t for v in e]


def live_probs(pre_race: dict, weights: list, ex_cols: list[int], ex: dict[int, float]):
    """展示タイムを入れて、1着の確率(6艇)と3連単120通りの確率を返す。"""
    mask = pre_race["mask"]
    x = [list(row) for row in pre_race["x"]]
    vals = [ex[l] for l in range(1, 7) if mask[l - 1] and ex.get(l) is not None]
    mean = sum(vals) / len(vals) if vals else 0.0
    for slot in range(6):
        L = slot + 1
        v = ex.get(L)
        d = (v - mean) if (mask[slot] and v is not None and vals) else 0.0
        x[slot][ex_cols[0]] = d
        x[slot][ex_cols[1]] = d if L == 1 else 0.0
        x[slot][ex_cols[2]] = d if L >= 4 else 0.0

    def score(w):
        return [sum(a * b for a, b in zip(row, w)) for row in x]

    s1, s2, s3 = (score(w) for w in weights)
    p1 = _softmax(s1, mask)
    combos = []
    for i, j, k in permutations(range(6), 3):
        m = list(mask)
        m[i] = False
        q2 = _softmax(s2, m)
        m[j] = False
        q3 = _softmax(s3, m)
        combos.append((i, j, k, p1[i] * q2[j] * q3[k] if sum(mask) >= 3 else 0.0))
    return p1, combos


def _top(combos, k=10):
    best = sorted(combos, key=lambda c: -c[3])[:k]
    return [{"c": f"{i + 1}-{j + 1}-{kk + 1}", "p": round(p, 5)} for i, j, kk, p in best]


def _top2(combos, k=6):
    ex: dict[tuple[int, int], float] = {}
    for i, j, _k, p in combos:
        ex[(i, j)] = ex.get((i, j), 0.0) + p
    best = sorted(ex.items(), key=lambda kv: -kv[1])[:k]
    return [{"c": f"{i + 1}-{j + 1}", "p": round(p, 5)} for (i, j), p in best]


def _throttle() -> None:
    global _last_net
    with _net_lock:
        wait = 1.0 - (time.time() - _last_net)
        if wait > 0:
            time.sleep(wait)
        _last_net = time.time()


def _fetch_before(day: str, rno: int, venue: str) -> dict[int, float]:
    key = (venue, day, rno)
    hit = _before_cache.get(key)
    if hit and time.time() - hit[0] < PRE_CACHE_SEC:
        return hit[1]
    _throttle()
    ex: dict[int, float] = {}
    try:
        r = httpx.get(BEFORE_URL.format(rno=rno, jcd=venue, ymd=day.replace("-", "")),
                      headers={"User-Agent": UA}, timeout=15)
        if r.status_code == 200:
            ex = parse_ex(r.text)
    except httpx.HTTPError:
        pass
    _before_cache[key] = (time.time(), ex)
    return ex


def live_predictions(venue: str, pre: dict, data: dict, now: datetime | None = None) -> dict:
    """締切前の時間帯のレースだけ、展示タイムを取って予想を出し直す(取れたものは2分キャッシュ)。"""
    now = now or datetime.now(JST)
    close = {(c["date"], c["race_no"]): c.get("close") for c in data.get("upcoming", [])}
    weights, ex_cols = pre["weights"], pre["ex_cols"]
    for r in pre.get("races", []):
        key = (r["date"], r["race_no"])
        cl = close.get(key)
        if not cl or r["date"] != now.strftime("%Y-%m-%d"):
            continue
        hh, mm = (int(x) for x in cl.split(":"))
        close_t = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if not (close_t - timedelta(minutes=PRE_BEFORE_MIN) <= now <= close_t + timedelta(minutes=3)):
            continue
        ex = _fetch_before(r["date"], r["race_no"], venue)
        if len(ex) < 6:
            continue
        p1, combos = live_probs(r, weights, ex_cols, ex)
        _live_pred[(venue, r["date"], r["race_no"])] = {
            "p": [round(v, 4) for v in p1], "top": _top(combos), "top2": _top2(combos),
            "ex": {str(k): v for k, v in ex.items()}, "asof": now.strftime("%H:%M")}
    out = [{"date": d, "race_no": n, **v} for (vn, d, n), v in _live_pred.items() if vn == venue]
    return {"venue": venue, "asof": now.strftime("%H:%M"), "races": out}
