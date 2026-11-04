"""JRA公式「レース成績データ」から重賞レースの結果を集めてCSVにする。

- 取得元: https://www.jra.go.jp/datafile/seiseki/replay/<年>/jyusyo.html と各レースページ
  (robots.txt は全許可。それでも1秒あけて、同じページは二度取らない)
- 生HTMLは keiba/_cache/ (git管理外)、整形後のCSVは keiba/data/ (コミットする)
使い方: python -m keiba.scrape 2019 2026
"""
from __future__ import annotations

import csv
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE = "https://www.jra.go.jp"
HERE = Path(__file__).parent
CACHE = HERE / "_cache"
DATA = HERE / "data"
UA = "tradeExperience-keiba-research/0.1 (personal study; 1 req/sec)"
DELAY = 1.0

_last = 0.0


def get(path: str) -> str:
    """キャッシュ付きGET。JRAのページはShift_JIS。"""
    global _last
    f = CACHE / path.strip("/").replace("/", "_")
    if f.exists():
        return f.read_bytes().decode("shift_jis", "ignore")
    wait = DELAY - (time.time() - _last)
    if wait > 0:
        time.sleep(wait)
    for i in range(3):
        try:
            r = requests.get(BASE + path, headers={"User-Agent": UA}, timeout=30)
            _last = time.time()
            if r.status_code == 404:
                return ""
            r.raise_for_status()
            CACHE.mkdir(exist_ok=True)
            f.write_bytes(r.content)
            return r.content.decode("shift_jis", "ignore")
        except requests.RequestException:
            time.sleep(3 * (i + 1))
    return ""


def num(s: str, default=None):
    s = (s or "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return default


def parse_race(html: str, year: int, rid: str, grade: str, name: str) -> tuple[list[dict], dict]:
    return parse_scope(BeautifulSoup(html, "lxml"), year, rid, grade, name)


def parse_scope(soup, year: int, rid: str, grade: str, name: str) -> tuple[list[dict], dict]:
    """soup はページ全体でも、1レース分のブロック(div.race_result_unit)でもよい。"""
    cap = soup.select_one("div.date_line")
    date_txt = cap.select_one(".date").get_text(" ", strip=True) if cap else ""
    m = re.search(r"(\d+)年(\d+)月(\d+)日", date_txt)
    date = f"{int(m[1]):04d}-{int(m[2]):02d}-{int(m[3]):02d}" if m else ""
    turf = soup.select_one("li.turf .txt")
    dirt = soup.select_one("li.durt .txt") or soup.select_one("li.dirt .txt")
    ground = "芝" if turf else "ダ"
    going = (turf or dirt).get_text(strip=True) if (turf or dirt) else ""
    course = soup.select_one("div.type .course")
    ctxt = course.get_text("", strip=True) if course else ""
    dist = re.search(r"([\d,]+)メートル", ctxt)
    rtitle = soup.select_one("span.race_name")
    cond = soup.select_one("div.type .category")
    wt = soup.select_one("div.type .weight")
    klass = soup.select_one("div.type .class")
    kai = re.search(r"(\d+)回(\D+?)(\d+)日", date_txt)
    meta = dict(
        venue=kai[2] if kai else "", race_id=f"{year}-{rid}", date=date, grade=grade,
        race=(rtitle.get_text("", strip=True) if rtitle else name),
        ground=ground, going=going, dist=int(num(dist[1], 0)) if dist else 0,
        cond=cond.get_text(strip=True) if cond else "", wt_rule=wt.get_text(strip=True) if wt else "",
        klass=klass.get_text(strip=True) if klass else "",
    )
    rows = []
    for tr in soup.select("table.basic.narrow-xy tbody tr"):
        td = {c.get("class", [""])[0]: c for c in tr.find_all("td")}
        if "horse" not in td:
            continue
        sexage = td["age"].get_text(strip=True)
        hw = td["h_weight"]
        hw_diff = hw.find("span")
        corners = [num(li.get_text(strip=True)) for li in td["corner"].select("li")]
        corners = [c for c in corners if c is not None]
        place = td["place"].get_text(strip=True)
        rows.append(dict(
            **meta,
            place=int(place) if place.isdigit() else 0,  # 0=除外/中止/失格など
            num=int(num(td["num"].get_text(strip=True), 0)),
            horse=td["horse"].get_text(strip=True),
            sex=sexage[:1], age=int(num(sexage[1:], 0)),
            weight=num(td["weight"].get_text(strip=True)),
            jockey=re.sub(r"\s+", "", td["jockey"].get_text()),
            time=td["time"].get_text(strip=True),
            last3f=num(td["f_time"].get_text(strip=True)),
            first_corner=corners[0] if corners else None,
            bw=num(hw.get_text("", strip=True).split("(")[0]),
            bw_diff=num((hw_diff.get_text(strip=True) if hw_diff else "").strip("()+")),
            trainer=re.sub(r"\s+", "", td["trainer"].get_text()),
            pop=int(num(td["pop"].get_text(strip=True), 0)),
        ))
    # 払戻(単勝・複勝)。馬番 -> 100円あたり払戻
    pay = {"win": {}, "place": {}}
    for key in ("win", "place"):
        li = soup.select_one(f"div.refund_unit li.{key}")
        if li:
            for line in li.select("div.line"):
                n = line.select_one(".num")
                y = line.select_one(".yen")
                if n and y:
                    pay[key][int(num(n.get_text(strip=True), 0))] = int(num(y.get_text("", strip=True).replace("円", ""), 0))
    for r in rows:
        r["win_pay"] = pay["win"].get(r["num"], 0)
        r["place_pay"] = pay["place"].get(r["num"], 0)
    return rows, meta


def scrape_year(year: int) -> list[dict]:
    idx = get(f"/datafile/seiseki/replay/{year}/jyusyo.html")
    if not idx:
        print(year, "一覧なし")
        return []
    soup = BeautifulSoup(idx, "lxml")
    out = []
    for tr in soup.select("table tbody tr"):
        a = tr.select_one("td.result a")
        if not a:
            continue
        m = re.search(r"/(\w+)\.html", a["href"])
        rid = m[1] if m else a["href"].strip("/").replace("/", "_")
        if "J・" in tr.select_one("td.race").get_text():
            continue  # 障害レースは平地とは別物なので対象外
        g = tr.select_one(".grade_icon")
        grade = {"G&#8544;": "G1", "GⅠ": "G1", "GⅡ": "G2", "GⅢ": "G3"}.get(g.get_text(strip=True) if g else "", "")
        if not grade and g:
            t = g.get_text(strip=True)
            grade = "G1" if t.endswith("Ⅰ") else "G2" if t.endswith("Ⅱ") else "G3" if t.endswith("Ⅲ") else "他"
        name = tr.select_one("td.race").get_text(strip=True)
        html = get(a["href"])
        if not html:
            continue
        rows, _ = parse_race(html, year, rid, grade or "他", name)
        out += rows
        print(year, rid, name, len(rows), flush=True)
    return out


def main(y0: int, y1: int) -> None:
    DATA.mkdir(exist_ok=True)
    allrows = []
    for y in range(y0, y1 + 1):
        allrows += scrape_year(y)
    if not allrows:
        return
    out = DATA / "jra_graded_results.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(allrows[0].keys()))
        w.writeheader()
        w.writerows(allrows)
    print("書き出し", out, len(allrows), "行")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if a else 2019, int(a[1]) if len(a) > 1 else 2026)
