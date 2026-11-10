"""丸亀ボート: 当日のレース結果を、公式の「レース結果」ページから取ってキャッシュする。

- 終わったレースだけ(締切の数分後から)取りに行く。1レースにつき1分に1回まで、取れたら取り直さない
- 公式サイトへは1秒に1リクエスト以内(ロックで直列化)。閲覧者が何人いてもアクセス数は増えない
- 依存を増やさないため、HTMLはタグを剥がして正規表現で読む(bs4は使わない)
"""
from __future__ import annotations

import html as _html
import re
import threading
import time
import unicodedata
from datetime import datetime, timedelta, timezone

import httpx

JST = timezone(timedelta(hours=9))
URL = "https://www.boatrace.jp/owpc/pc/race/raceresult?rno={rno}&jcd={jcd}&hd={ymd}"
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
