"""予想JSONを GitHub の raw から取り、1分キャッシュする(ボート・競馬で共通)。

- 定期更新で data.json などが push されると、Fly を再デプロイしなくても、1分後には画面に出る
- raw が取れない時は、Docker イメージに同梱されたファイルを返す(前回のデプロイ時点のもの)
"""
from __future__ import annotations

import time
from pathlib import Path

import httpx
from fastapi.responses import FileResponse, Response

RAW_BASE = ("https://raw.githubusercontent.com/shinshoma02-bot/tradeExperience/"
            "claude/trading-skeleton/")


class SiteJSON:
    def __init__(self, site_dir: Path, sub: str, ttl: float = 60.0, fetch=None):
        self.site_dir = site_dir
        self.url = RAW_BASE + sub  # 例: "keiba/site/"
        self.ttl = ttl
        self.cache: dict[str, dict] = {}  # ファイル名 -> {"t": 取得時刻, "body": 中身}
        self.fetch = fetch or (lambda url: httpx.get(url, timeout=10))

    def response(self, fname: str):
        c = self.cache.setdefault(fname, {"t": 0.0, "body": None})
        if c["body"] is None or time.time() - c["t"] > self.ttl:
            try:
                r = self.fetch(self.url + fname)
                if r.status_code == 200 and r.content.startswith(b"{"):
                    c["body"], c["t"] = r.content, time.time()
            except httpx.HTTPError:
                pass
        if c["body"] is not None:
            return Response(c["body"], media_type="application/json", headers={"Cache-Control": "no-cache"})
        return FileResponse(self.site_dir / fname, media_type="application/json")
