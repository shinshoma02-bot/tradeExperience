"""対象のボートレース場。場コードは公式のもの(桐生01 … 鳴門14 丸亀15 児島16 … 大村24)。

丸亀(DEFAULT)だけは昔からの置き場所をそのまま使う(移行で壊さないため)。
他の場は同じ名前のサブフォルダ/サフィックス付き: _cache/14/, data/raw/14/, site/data_14.json など。
場を増やすときは VENUES に1行足して `python -m boat.scrape 2024-01-01 <今日>` で取り直す。
"""
from __future__ import annotations

from pathlib import Path

VENUES = {"15": "丸亀", "14": "鳴門", "16": "児島"}
DEFAULT = "15"
HERE = Path(__file__).parent


def _sub(base: Path, v: str) -> Path:
    return base if v == DEFAULT else base / v


def cache_dir(v: str) -> Path:
    return _sub(HERE / "_cache", v)


def raw_dir(v: str) -> Path:
    return _sub(HERE / "data" / "raw", v)


def odds_cache(v: str) -> Path:
    return _sub(HERE / "_cache" / "odds", v)


def odds_pack(v: str) -> Path:
    return _sub(HERE / "data" / "odds", v)


def site_file(prefix: str, v: str) -> Path:
    """boat/site の出力。丸亀は data.json / live.json / next.json のまま、他は data_14.json など。"""
    return HERE / "site" / (f"{prefix}.json" if v == DEFAULT else f"{prefix}_{v}.json")
