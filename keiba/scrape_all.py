"""JRADBをたどって、重賞以外も含む全レースの結果を集める(新しい月から古い月へ)。

流れ: 月ページ → 開催日(場ごと) → 「全てのレースを表示」(1場1日=全12R) → 各レースの表を解析
- cnameの末尾にチェックサムがあるため、URLは組み立てずページ内のリンクだけを辿る
- 1秒間隔。1場1日ぶんごとに keiba/_cache/rows/ にJSONで保存するので、途中で止めても再開できる
- 仕上げに keiba/data/all/<年>.csv.gz を作る(コミット対象)
使い方: python -m keiba.scrape_all [遡る下限の年(既定2016) [開始する年]]  / 最後に build でCSVを作る
"""
from __future__ import annotations

import csv
import gzip
import json
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from keiba.scrape import BASE, HERE, UA, parse_scope

ROWS = HERE / "_cache" / "rows"
OUTDIR = HERE / "data" / "all"
START = "pw01skl00999999/B3"  # 「過去レース結果検索」の最新月(JRAトップ→レース結果から辿れるリンク)
_last = 0.0


def post(path: str, cname: str) -> str:
    global _last
    for i in range(4):
        wait = 1.5 - (time.time() - _last)
        if wait > 0:
            time.sleep(wait)
        try:
            r = requests.post(BASE + path, data={"cname": cname}, headers={"User-Agent": UA}, timeout=60)
            _last = time.time()
            if r.status_code == 200:
                return r.content.decode("shift_jis", "ignore")
        except requests.RequestException:
            pass
        time.sleep(5 * (i + 1))
    return ""


def links(html: str, prefix: str) -> list[str]:
    return re.findall(r"doAction\('/JRADB/accessS\.html',\s*'(" + prefix + r"[^']+)'\)", html)


def day_rows(cname: str) -> list[dict]:
    """開催日ページ(srl)→全レース表示(ses)→全レース分の行。"""
    html = post("/JRADB/accessS.html", cname)
    ses = links(html, "pw01ses")
    if not ses:
        return []
    page = post("/JRADB/accessS.html", ses[0])
    soup = BeautifulSoup(page, "lxml")
    out = []
    for unit in soup.select("div.race_result_unit"):
        rno = re.search(r"race_result_(\d+)R", unit.get("id", ""))
        h2 = unit.select_one("div.race_title h2")
        name = h2.get_text("", strip=True) if h2 else ""
        icon = unit.select_one("div.race_title img[alt^='G']")
        alt = icon.get("alt", "") if icon else ""
        grade = "G1" if alt.endswith("Ⅰ") or alt == "GI" else "G2" if alt.endswith("Ⅱ") else "G3" if alt.endswith("Ⅲ") else ""
        m = re.search(r"(\d{4})(\d{2})(\d{2})/", cname + "/")
        rows, meta = parse_scope(unit, int(m[1]) if m else 0, f"{m[2]}{m[3]}-{cname[9:11]}-{int(rno[1]):02d}" if (m and rno) else "x", grade, name)
        for r in rows:
            r["race_no"] = int(rno[1]) if rno else 0
        out += rows
    return out


def crawl(stop_year: int, start_year: int = 9999) -> None:
    ROWS.mkdir(parents=True, exist_ok=True)
    month = post("/JRADB/accessS.html", START)
    n_new = 0
    while month:
        cur = re.search(r'class="current"><span>(\d+)年(\d+)月', month)
        if not cur:
            break
        y, mth = int(cur[1]), int(cur[2])
        if y < stop_year:
            break
        days = links(month, "pw01srl") if y <= start_year else []  # start_yearより新しい月は読み飛ばす
        todo = []
        for c in dict.fromkeys(days):
            # cname の形: pw01srl10<場2桁><年4桁><回2桁><日2桁><YYYYMMDD>/<チェックサム>
            date = re.search(r"(\d{8})/[0-9A-F]{2}$", c)
            if date and date[1][:4] == str(y) and int(date[1][4:6]) == mth:
                todo.append((date[1], c))
        for date, c in todo:
            f = ROWS / (c.replace("/", "_") + ".json")
            if f.exists():
                continue
            rows = day_rows(c)
            f.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            n_new += 1
            print(f"{y}-{mth:02d} {c} {len(rows)}行", flush=True)
        prev = re.search(r"doAction\('/JRADB/accessS\.html',\s*'(pw01skl1[^']+)'\);\"><i[^>]*></i>\d+年\d+月", month)
        month = post("/JRADB/accessS.html", prev[1]) if prev else ""
    print("新規", n_new, "開催日")


def build() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    by_year: dict[str, list[dict]] = {}
    for f in sorted(ROWS.glob("*.json")):
        for r in json.loads(f.read_text(encoding="utf-8")):
            by_year.setdefault(r["date"][:4], []).append(r)
    for y, rows in sorted(by_year.items()):
        rows.sort(key=lambda r: (r["date"], r["venue"], r["race_no"], r["place"] or 99, r["num"]))
        out = OUTDIR / f"{y}.csv.gz"
        with gzip.open(out, "wt", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(out.name, len(rows), "行", out.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "build":
        build()
    else:
        crawl(int(a[0]) if a else 2016, int(a[1]) if len(a) > 1 else 9999)
        build()
