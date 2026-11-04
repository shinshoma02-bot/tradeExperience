"""ボートレース公式の過去データ(番組表B・競走成績K)から丸亀(場コード15)だけ抜き出して保存する。

- 取得元: https://www1.mbrace.or.jp/od2/{B,K}/YYYYMM/{b,k}YYMMDD.lzh (robots.txt は全許可)
  それでも1秒あけて、同じ日は二度取らない(取得済みはキャッシュ)
- 1日分のファイルは全24場ぶんが入っているので、丸亀のブロックだけ boat/_cache/ に保存(git管理外)
- 開催のない日は空ファイルを置く(次回から取りに行かない)
使い方: python -m boat.scrape 2023-01-01 2026-11-04
"""
from __future__ import annotations

import io
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import lhafile
import requests

BASE = "https://www1.mbrace.or.jp/od2"
HERE = Path(__file__).parent
CACHE = HERE / "_cache"
UA = "tradeExperience-boat-research/0.1 (personal study; 1 req/sec)"
DELAY = 1.0
VENUE = "15"  # 丸亀

_last = 0.0
_lock = threading.Lock()


def fetch_lzh(kind: str, d: date) -> bytes | None:
    """kind='B'|'K'。404なら None。"""
    global _last
    ym = d.strftime("%Y%m")
    url = f"{BASE}/{kind}/{ym}/{kind.lower()}{d.strftime('%y%m%d')}.lzh"
    for i in range(3):
        # 全スレッド合計で1秒に1リクエストまで(応答が遅いので並列で待ち時間を埋める)
        with _lock:
            wait = DELAY - (time.time() - _last)
            if wait > 0:
                time.sleep(wait)
            _last = time.time()
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            time.sleep(3 * (i + 1))
    return None


def get_html(url: str) -> str:
    """公式サイトのページを取る(全スレッド合計で1秒に1リクエスト)。失敗したら空文字。"""
    global _last
    for i in range(3):
        with _lock:
            wait = DELAY - (time.time() - _last)
            if wait > 0:
                time.sleep(wait)
            _last = time.time()
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=60)
            if r.status_code == 200:
                return r.text
            if r.status_code == 404:
                return ""
        except requests.RequestException:
            pass
        time.sleep(3 * (i + 1))
    return ""


def venue_block(text: str, kind: str) -> str:
    """全場ぶんのテキストから丸亀のブロック(24BBGN〜24BEND など)を取り出す。"""
    beg, end = f"{VENUE}{kind}BGN", f"{VENUE}{kind}END"
    i = text.find(beg)
    if i < 0:
        return ""
    j = text.find(end, i)
    return text[i : (j + len(end) if j >= 0 else len(text))]


def extract(raw: bytes, kind: str) -> str:
    z = lhafile.Lhafile(io.BytesIO(raw))
    for info in z.infolist():
        text = z.read(info.filename).decode("cp932", "ignore")
        return venue_block(text, kind)
    return ""


def get_day(kind: str, d: date, today: date) -> str:
    f = CACHE / f"{kind}{d:%Y%m%d}.txt"
    if f.exists():
        return f.read_text(encoding="utf-8")
    raw = fetch_lzh(kind, d)
    if raw is None:
        # 直近は「まだ公開されていない」だけかもしれないので、キャッシュしない
        if (today - d).days > 3:
            CACHE.mkdir(exist_ok=True)
            f.write_text("", encoding="utf-8")
        return ""
    block = extract(raw, kind)
    CACHE.mkdir(exist_ok=True)
    f.write_text(block, encoding="utf-8")
    return block


def main(a: date, b: date) -> None:
    today = date.today()
    days = [b - timedelta(days=i) for i in range((b - a).days + 1)]  # 新しい日から
    jobs = [(k, d) for d in days for k in ("B", "K")]
    done = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for _ in ex.map(lambda j: get_day(j[0], j[1], today), jobs):
            done += 1
            if done % 100 == 0:
                print(done, "/", len(jobs), flush=True)


if __name__ == "__main__":
    s = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date(2023, 1, 1)
    e = date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else date.today()
    main(s, e)
