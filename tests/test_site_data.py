"""予想JSONの配信: GitHub raw から取れたら中身を返し、取れなければ同梱ファイルに戻る。"""
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from app import main as app_main
from app import site_data
from app.config import Settings
from app.site_data import SiteJSON


class Resp:
    def __init__(self, status: int, body: bytes):
        self.status_code, self.content = status, body


def make(tmp_path: Path, responder):
    (tmp_path / "data.json").write_text('{"src":"bundled"}', encoding="utf-8")
    calls = []

    def fetch(url):
        calls.append(url)
        return responder(url)
    return SiteJSON(tmp_path, "keiba/site/", ttl=60, fetch=fetch), calls


def test_uses_raw_when_available(tmp_path):
    s, calls = make(tmp_path, lambda url: Resp(200, b'{"src":"raw"}'))
    r = s.response("data.json")
    assert r.body == b'{"src":"raw"}'
    assert calls[0].endswith("/keiba/site/data.json")


def test_falls_back_to_bundled_on_error(tmp_path):
    s, _ = make(tmp_path, lambda url: Resp(404, b"not found"))
    assert s.response("data.json").path.name == "data.json"  # FileResponse(同梱)


def test_falls_back_on_network_error(tmp_path):
    def boom(url):
        raise httpx.ConnectError("down")
    s, _ = make(tmp_path, boom)
    assert s.response("data.json").path.name == "data.json"


def test_cache_avoids_refetch_within_ttl(tmp_path):
    s, calls = make(tmp_path, lambda url: Resp(200, b'{"n":1}'))
    s.response("data.json")
    s.response("data.json")
    assert len(calls) == 1


def test_keiba_routes_are_served_before_static_mount(tmp_path, monkeypatch):
    # 取得を偽物に差し替えて、/keiba/data.json が raw の中身を返すことを確認
    monkeypatch.setattr(site_data, "SiteJSON", lambda d, sub: SiteJSON(d, sub, fetch=lambda u: Resp(200, b'{"from":"raw"}')))
    client = TestClient(app_main.create_app(Settings(webhook_secret="", journal_path=str(tmp_path / "j.jsonl"))))
    r = client.get("/keiba/data.json")
    assert r.status_code == 200 and r.json() == {"from": "raw"}
    # 予想以外(画面)は静的ファイルのまま
    assert client.get("/keiba/").status_code == 200
