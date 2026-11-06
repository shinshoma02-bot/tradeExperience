"""台データCSVの取り込み。  python -m pachinko.ingest ファイル.csv [...]

列: date,hall,no,name,kind,games,bonus,diff
  date  YYYY-MM-DD(営業日)  hall 店名  no 台番  name 機種名
  kind  slot / pachi
  games スロット=総ゲーム数 / パチンコ=回転数(スタート)
  bonus スロット=BB+RB回数 / パチンコ=大当たり回数
  diff  スロット=差枚 / パチンコ=差玉
同じ (date,hall,no) は新しい行で上書きする。
"""
import csv
import sys
from pathlib import Path

DATA = Path(__file__).parent / "data" / "records.csv"
COLS = ["date", "hall", "no", "name", "kind", "games", "bonus", "diff"]


def clean(row: dict) -> dict | None:
    """1行を検証して整える。不正なら None。"""
    try:
        r = {k: (row.get(k) or "").strip() for k in COLS}
        if not (len(r["date"]) == 10 and r["date"][4] == "-" and r["date"][7] == "-"):
            return None
        if r["kind"] not in ("slot", "pachi") or not r["hall"] or not r["no"]:
            return None
        for k in ("games", "bonus"):
            r[k] = str(int(float(r[k].replace(",", ""))))
            if int(r[k]) < 0:
                return None
        r["diff"] = str(int(float(r["diff"].replace(",", "").replace("+", ""))))
        return r
    except (ValueError, AttributeError):
        return None


def load(path: Path = DATA) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return [r for r in (clean(x) for x in csv.DictReader(f)) if r]


def merge(new_rows: list[dict], path: Path = DATA) -> tuple[int, int]:
    """(追加した行数, 不正で捨てた行数)。"""
    old = {(r["date"], r["hall"], r["no"]): r for r in load(path)}
    bad = 0
    add = 0
    for x in new_rows:
        r = clean(x)
        if not r:
            bad += 1
            continue
        key = (r["date"], r["hall"], r["no"])
        add += key not in old
        old[key] = r
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for k in sorted(old):
            w.writerow(old[k])
    return add, bad


if __name__ == "__main__":
    for p in sys.argv[1:]:
        with open(p, encoding="utf-8-sig", newline="") as f:
            add, bad = merge(list(csv.DictReader(f)))
        print(f"{p}: 追加 {add} 行 / 不正 {bad} 行")
