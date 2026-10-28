import json
import threading
import time
import urllib.request

PRIORITY = {"min": 1, "low": 2, "default": 3, "high": 4, "urgent": 5}


class Notifier:
    """スマホ通知(ntfy.sh)。topic が空なら何もしない。送信失敗でも売買は止めない。"""

    def __init__(self, topic: str, server: str = "https://ntfy.sh", sender=None) -> None:
        self.topic, self.server = topic, server.rstrip("/")
        self._sender = sender or self._post
        self._last: dict[str, float] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.topic)

    def send(self, title: str, message: str, priority: str = "default", dedupe_key: str | None = None,
             dedupe_sec: int = 3600) -> None:
        if not self.enabled:
            return
        if dedupe_key:  # 同じ内容を連投しない(上限到達の拒否が延々続く場合など)
            now = time.time()
            if now - self._last.get(dedupe_key, 0) < dedupe_sec:
                return
            self._last[dedupe_key] = now
        threading.Thread(target=self._safe, args=(title, message, priority), daemon=True).start()

    def _safe(self, title: str, message: str, priority: str) -> None:
        try:
            self._sender(title, message, priority)
        except Exception:
            pass

    def _post(self, title: str, message: str, priority: str) -> None:
        # ntfyのJSON投稿なら日本語のタイトルもそのまま送れる
        body = json.dumps({"topic": self.topic, "title": title, "message": message,
                           "priority": PRIORITY.get(priority, 3)}).encode("utf-8")
        req = urllib.request.Request(self.server, data=body, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10).read()
