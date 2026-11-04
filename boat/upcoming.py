"""開催前の丸亀について、公式サイトの直前情報(展示タイム・天候・オッズ)を取る。

- 番組表(B)は前日の午後に出る。展示タイムは各レースの約10〜15分前、オッズは発売開始後に出る
- 取れたものだけを boat/site/live.json に保存(取れなかった項目は欠ける。モデル側は欠けても動く)
- 使い方: python -m boat.upcoming [YYYY-MM-DD ...]   (日付を省略すると 今日と明日)
"""
from __future__ import annotations

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from bs4 import BeautifulSoup

from boat import scrape

JCD = "15"  # 丸亀
BASE = "https://www.boatrace.jp/owpc/pc/race"
OUT = Path(__file__).parent / "site" / "live.json"
JST = timezone(timedelta(hours=9))


def refresh_programs(days: list[date]) -> None:
    """今日以降の番組表は(出走変更があり得るので)キャッシュを捨てて取り直す。"""
    for d in days:
        f = scrape.CACHE / f"B{d:%Y%m%d}.txt"
        if f.exists():
            f.unlink()
        scrape.get_day("B", d, date.today())


def _url(page: str, d: date, rno: int) -> str:
    return f"{BASE}/{page}?rno={rno}&jcd={JCD}&hd={d:%Y%m%d}"


def parse_before(html: str) -> dict:
    """直前情報: 艇ごとの展示タイム・チルト・体重、スタート展示の進入順、水面気象。"""
    if not html:
        return {}
    soup = BeautifulSoup(html, "html.parser")
    text = re.sub(r"\s+", " ", soup.get_text(" "))
    out: dict = {"boats": {}}
    for m in re.finditer(r"(?<![\d.])([1-6]) (\S+(?: \S+)?) (\d{2}\.\d)kg (\d\.\d{2}) (-?\d\.\d)", text):
        out["boats"][int(m.group(1))] = {"weight": float(m.group(3)), "ex_time": float(m.group(4)),
                                         "tilt": float(m.group(5))}
    m = re.search(r"スタート展示 コース 並び ST((?: [1-6] [F.L]*\.?\d+)+)", text)
    if m:
        out["course_order"] = [int(x) for x in re.findall(r" ([1-6]) [F.L]*\.?\d+", m.group(1))][:6]
    w = re.search(r"気温 ([\d.]+)℃ (\S+) 風速 (\d+)m 水温 ([\d.]+)℃ 波高 (\d+)cm", text)
    if w:
        out["weather"] = {"temp": float(w.group(1)), "sky": w.group(2), "wind": int(w.group(3)),
                          "water": float(w.group(4)), "wave": int(w.group(5))}
    # 展示タイムが全艇そろっていなければ「まだ」とみなす
    if len(out["boats"]) < 6 or all(b["ex_time"] == 0 for b in out["boats"].values()):
        out["boats"] = {}
    return out


def parse_win_odds(html: str) -> dict[int, float]:
    if not html:
        return {}
    text = re.sub(r"\s+", " ", BeautifulSoup(html, "html.parser").get_text(" "))
    i = text.find("単勝オッズ ボートレーサー")
    seg = text[i:] if i >= 0 else ""
    seg = seg.split("複勝オッズ")[0]
    return {int(a): float(b.replace(",", "")) for a, b in re.findall(r" ([1-6]) \S+(?: \S+)? ([\d,]+\.?\d*)(?= |$)", seg)}


def parse_3t(html: str) -> dict[str, float]:
    """3連単オッズ表。1着の艇ごとの列に 2着(rowspan) 3着 オッズ が並ぶ。"""
    if not html:
        return {}
    soup = BeautifulSoup(html, "html.parser")
    out: dict[str, float] = {}
    for tbl in soup.find_all("table"):
        if not tbl.find("td", class_="oddsPoint"):
            continue
        second = [0] * 6
        for tr in tbl.find("tbody").find_all("tr"):
            tds = tr.find_all("td")
            p = 0
            for c in range(6):
                if p >= len(tds):
                    break
                if tds[p].has_attr("rowspan"):
                    second[c] = int(tds[p].get_text(strip=True))
                    p += 1
                third = int(tds[p].get_text(strip=True))
                v = tds[p + 1].get_text(strip=True).replace(",", "")
                p += 2
                try:
                    out[f"{c + 1}-{second[c]}-{third}"] = float(v)
                except ValueError:
                    pass  # 欠場などで「欠」「−」のとき
        break
    return out


def fetch_day(d: date) -> dict:
    def one(rno: int) -> dict:
        r = {"race_no": rno}
        before = parse_before(scrape.get_html(_url("beforeinfo", d, rno)))
        r.update(before)
        r["win_odds"] = parse_win_odds(scrape.get_html(_url("oddstf", d, rno)))
        r["odds3t"] = parse_3t(scrape.get_html(_url("odds3t", d, rno)))
        return r

    with ThreadPoolExecutor(max_workers=6) as ex:
        return {"date": str(d), "races": list(ex.map(one, range(1, 13)))}


def main(days: list[date]) -> dict:
    refresh_programs(days)
    live = {"asof": datetime.now(JST).strftime("%Y-%m-%d %H:%M"), "days": [fetch_day(d) for d in days]}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(live, ensure_ascii=False), encoding="utf-8")
    return live


if __name__ == "__main__":
    if len(sys.argv) > 1:
        ds = [date.fromisoformat(a) for a in sys.argv[1:]]
    else:
        t = datetime.now(JST).date()
        ds = [t, t + timedelta(days=1)]
    live = main(ds)
    for dd in live["days"]:
        n = sum(1 for r in dd["races"] if r.get("boats"))
        o = sum(1 for r in dd["races"] if r.get("odds3t"))
        print(dd["date"], f"展示あり{n}R / 3連単オッズあり{o}R")
