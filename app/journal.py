import json
import os
from datetime import datetime, timezone


class Journal:
    """全シグナルと結果を1行1JSONで記録する(後で成績分析に使う)。"""

    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def write(self, kind: str, **data) -> None:
        row = {"ts": datetime.now(timezone.utc).isoformat(), "kind": kind, **data}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def read(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
