"""取得済みの生データ(boat/_cache)を、月ごとの圧縮ファイル(boat/data/raw/YYYY-MM.json.gz)で持ち運ぶ。

定期実行は毎回まっさらな環境で始まるので、2,000ファイルを取り直さずに済むようにする。
中身が変わらない月はファイルも変わらない(gzipの時刻を固定)ので、gitはほとんど増えない。
使い方: python -m boat.cache unpack | pack
"""
from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
CACHE = HERE / "_cache"
RAW = HERE / "data" / "raw"


def pack() -> int:
    groups: dict[str, dict[str, str]] = defaultdict(dict)
    for f in sorted(CACHE.glob("[BK]*.txt")):
        d = f.stem[1:]
        groups[f"{d[:4]}-{d[4:6]}"][f.name] = f.read_text(encoding="utf-8")
    RAW.mkdir(parents=True, exist_ok=True)
    changed = 0
    for ym, files in groups.items():
        data = gzip.compress(json.dumps(files, ensure_ascii=False, sort_keys=True).encode("utf-8"), 9, mtime=0)
        out = RAW / f"{ym}.json.gz"
        if not out.exists() or out.read_bytes() != data:
            out.write_bytes(data)
            changed += 1
    return changed


def unpack() -> int:
    CACHE.mkdir(exist_ok=True)
    n = 0
    for f in sorted(RAW.glob("*.json.gz")):
        for name, text in json.loads(gzip.decompress(f.read_bytes()).decode("utf-8")).items():
            p = CACHE / name
            if not p.exists():
                p.write_text(text, encoding="utf-8")
                n += 1
    return n


if __name__ == "__main__":
    print({"pack": pack, "unpack": unpack}[sys.argv[1]]())
