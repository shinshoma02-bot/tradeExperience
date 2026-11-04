"""今週の出馬表(JRA公式)を取って、出走馬の勝率を予想する。

流れ: 「今週の出馬表」→ 開催(場ごと) → 「全てのレースを表示」→ 各レースの表(枠・馬番・馬体重・斤量・騎手・単勝オッズ)
- 出馬表は木曜16時ごろ(馬番なし)、金・土曜に馬番つきで出る。オッズは発売開始後
- オッズ(=人気)がまだ無いときは「人気なし」モデルで予想する
使い方: python -m keiba.upcoming  → keiba/site/upcoming.json
"""
from __future__ import annotations

import re
import time

import requests
from bs4 import BeautifulSoup

from keiba.scrape import BASE, UA, num

TOP = "pw01dli00/F3"  # 「今週の出馬表」(JRA > 出馬表)
_last = 0.0


def post(cname: str) -> str:
    global _last
    for i in range(4):
        wait = 1.0 - (time.time() - _last)
        if wait > 0:
            time.sleep(wait)
        try:
            r = requests.post(BASE + "/JRADB/accessD.html", data={"cname": cname},
                              headers={"User-Agent": UA}, timeout=60)
            _last = time.time()
            if r.status_code == 200:
                return r.content.decode("shift_jis", "ignore")
        except requests.RequestException:
            pass
        time.sleep(5 * (i + 1))
    return ""


def links(html: str, prefix: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"doAction\('/JRADB/accessD\.html',\s*'(" + prefix + r"[^']+)'\)", html)))


def parse_entries(html: str) -> list[dict]:
    """1開催日ぶん(全レース表示ページ)の出走馬を行にして返す。"""
    soup = BeautifulSoup(html, "lxml")
    rows = []
    for tb in soup.select("table.basic.narrow-xy"):
        cap = tb.select_one("caption")
        if not cap or not tb.select_one("td.horse"):
            continue
        date_txt = cap.select_one(".date_line .date").get_text(" ", strip=True)
        m = re.search(r"(\d+)年(\d+)月(\d+)日", date_txt)
        kai = re.search(r"(\d+)回(\D+?)(\d+)日", date_txt)
        date = f"{int(m[1]):04d}-{int(m[2]):02d}-{int(m[3]):02d}"
        rn = cap.select_one(".race_number img")
        rno = int(re.search(r"(\d+)", rn.get("alt", "0"))[1]) if rn else 0
        h2 = cap.select_one(".race_title h2")
        icon = cap.select_one(".race_title img[alt^='G']")
        alt = icon.get("alt", "") if icon else ""
        grade = "G1" if alt.endswith("Ⅰ") else "G2" if alt.endswith("Ⅱ") else "G3" if alt.endswith("Ⅲ") else ""
        course = cap.select_one(".race_title .course")
        ctxt = course.get_text("", strip=True) if course else ""
        dist = re.search(r"([\d,]+)メートル", ctxt)
        ground = "芝" if "芝" in ctxt else "ダ"
        cat, klass = cap.select_one(".race_title .category"), cap.select_one(".race_title .class")
        time_ = cap.select_one(".time strong")
        meta = dict(
            date=date, venue=kai[2] if kai else "", race_no=rno,
            race=h2.get_text("", strip=True) if h2 else "", grade=grade, ground=ground,
            dist=int(num(dist[1], 0)) if dist else 0,
            cond=cat.get_text(strip=True) if cat else "", klass=klass.get_text(strip=True) if klass else "",
            start=time_.get_text(strip=True) if time_ else "",
        )
        meta["race_id"] = f"{date}-{meta['venue']}-{rno:02d}"
        for tr in tb.select("tbody tr"):
            td = {c.get("class", [""])[0]: c for c in tr.find_all("td")}
            if "horse" not in td:
                continue
            hw = td.get("h_weight")
            diff = re.search(r"\(([+\-]?\d+)\)", hw.get_text("", strip=True)) if hw else None
            sexage = td["age"].get_text(strip=True)
            rows.append(dict(
                **meta,
                num=int(num(td["num"].get_text(strip=True), 0)) if "num" in td else 0,
                horse=td["horse"].get_text(strip=True),
                sex=sexage[:1], age=int(num(sexage[1:], 0)),
                weight=num(re.sub(r"[^\d.]", "", td["weight"].get_text(strip=True))) if "weight" in td else None,
                jockey=re.sub(r"\s+", "", re.sub(r"^[▲△☆★◇]", "", td["jockey"].get_text(" ", strip=True))),
                trainer=re.sub(r"\s+", "", td["trainer"].get_text()),
                bw=num(re.sub(r"\(.*", "", hw.get_text("", strip=True)).replace("kg", "")) if hw else None,
                bw_diff=num(diff[1]) if diff else None,
                odds=num(td["odds"].get_text(strip=True)) if "odds" in td else None,
            ))
    # オッズの低い順に人気(オッズがあるレースだけ)
    by_race: dict[str, list[dict]] = {}
    for r in rows:
        by_race.setdefault(r["race_id"], []).append(r)
    for rs in by_race.values():
        if all(r["odds"] for r in rs):
            for i, r in enumerate(sorted(rs, key=lambda r: r["odds"]), 1):
                r["pop"] = i
        else:
            for r in rs:
                r["pop"] = 0
    return rows


def fetch() -> list[dict]:
    top = post(TOP)
    out = []
    for c in links(top, "pw01drl"):
        day = post(c)
        for d in links(day, "pw01des")[:1]:
            out += parse_entries(post(d))
    return out


if __name__ == "__main__":
    rs = fetch()
    print(len(rs), "頭", len({r["race_id"] for r in rs}), "レース")
