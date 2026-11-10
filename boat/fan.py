"""公式の「レーサー期別成績」(全選手の半年ごとの成績。コース別の着順・平均STつき)。

- 取得元: https://www.boatrace.jp/static_extra/pc_static/download/data/kibetsu/fanYYMM.lzh (robots全許可)
  fan2510 = 集計期間 2025-05-01〜2025-10-31(2026年前期の級別のもと)。年2回しか出ない
- 先読みを避けるため、集計期間の終わりから2か月たった日(級別が切り替わる 1/1・7/1)以降のレースにだけ使う
- 生ファイルは boat/data/fan/ に置いてコミットする(1つ180KB、半年に1つ増えるだけ)
使い方: python -m boat.fan   (足りないファイルを取りに行く)
"""
from __future__ import annotations

import io
import time
from datetime import date
from pathlib import Path

import lhafile
import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).parent
DIR = HERE / "data" / "fan"
URL = "https://www.boatrace.jp/static_extra/pc_static/download/data/kibetsu/fan{}.lzh"
UA = "tradeExperience-boat-research/0.1 (personal study; 1 req/sec)"
K_SHRINK = 10  # コース別の率は、そのコースの全選手平均に引き寄せる(出走の少ないコースの過大評価を防ぐ)


def _codes(today: date) -> list[str]:
    """2023年10月の期から今日までの fanYYMM(4月・10月)。"""
    out, y, m = [], 2023, 10
    while date(y, m, 1) <= today:
        out.append(f"{y % 100:02d}{m:02d}")
        y, m = (y + 1, 4) if m == 10 else (y, 10)
    return out


def fetch(today: date | None = None) -> int:
    DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for c in _codes(today or date.today()):
        f = DIR / f"fan{c}.lzh"
        if f.exists():
            continue
        time.sleep(1.0)
        try:
            r = requests.get(URL.format(c), headers={"User-Agent": UA}, timeout=60)
        except requests.RequestException:
            continue
        if r.status_code == 200 and len(r.content) > 10000:
            f.write_bytes(r.content)
            n += 1
    return n


def parse_file(raw: bytes) -> pd.DataFrame:
    """1行=1選手×1コース。固定長(cp932のバイト位置)。"""
    z = lhafile.Lhafile(io.BytesIO(raw))
    body = z.read(z.infolist()[0].filename)
    rows = []
    for ln in body.split(b"\r\n"):
        if len(ln) < 400:
            continue
        o = 0

        def t(k: int) -> bytes:
            nonlocal o
            s = ln[o:o + k]
            o += k
            return s

        try:
            reg = int(t(4))
            t(16 + 15 + 4 + 2 + 1 + 6 + 1 + 2 + 3 + 2 + 2)
            win = int(t(4)) / 100
            t(4 + 3 + 3)
            n_all = int(t(3))
            t(2 + 2)
            st_all = int(t(3)) / 100
            crs = [(int(t(3)), int(t(4)) / 10, int(t(3)) / 100, int(t(3)) / 100) for _ in range(6)]
            t(6 + 4)
            idx = int(t(4)) / 100
            t(4 + 1 + 8)
            to = t(8).decode()
            t(3)
            fins = []
            for _ in range(6):
                fins.append([int(t(3)) for _ in range(6)])
                t(16)
        except ValueError:
            continue
        for k in range(6):
            n = crs[k][0]
            rows.append({"reg": reg, "lane": k + 1, "to": to, "f_n": n, "f_w": fins[k][0],
                         "f_t3": sum(fins[k][:3]), "f_cst_raw": crs[k][2] if n else np.nan,
                         "f_idx": idx if n_all else np.nan, "f_st": st_all if n_all else np.nan,
                         "win": win, "n_all": n_all})
    return pd.DataFrame(rows)


def load() -> pd.DataFrame:
    """全ファイルを読んで、コース別の率(引き寄せ済み)と「使ってよい日」を付ける。"""
    out = []
    for f in sorted(DIR.glob("fan*.lzh")):
        d = parse_file(f.read_bytes())
        if not len(d):
            continue
        tot = d.groupby("lane")[["f_n", "f_w", "f_t3"]].sum()
        d = d.merge((tot["f_w"] / tot["f_n"]).rename("pw"), left_on="lane", right_index=True)
        d = d.merge((tot["f_t3"] / tot["f_n"]).rename("p3"), left_on="lane", right_index=True)
        d["f_cw"] = (d["f_w"] + d["pw"] * K_SHRINK) / (d["f_n"] + K_SHRINK)
        d["f_c3"] = (d["f_t3"] + d["p3"] * K_SHRINK) / (d["f_n"] + K_SHRINK)
        d["f_cst"] = d["f_cst_raw"].fillna(d["f_st"])
        end = pd.Timestamp(d["to"].iloc[0])
        d["valid_from"] = end + pd.DateOffset(days=1) + pd.DateOffset(months=2)
        out.append(d[["reg", "lane", "valid_from", "f_n", "f_cw", "f_c3", "f_cst", "f_idx", "f_st"]])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


FEATS = ["f_cw", "f_c3", "f_cst", "f_idx", "f_st"]


def attach(ent: pd.DataFrame) -> pd.DataFrame:
    """出走表の各艇に、そのレース日に使える最新の期別成績(その枠=コースの数字)を付ける。"""
    fan = load()
    ent = ent.copy()
    if not len(fan):
        for c in FEATS + ["f_n"]:
            ent[c] = np.nan
        return ent
    ent["_i"] = np.arange(len(ent))
    left = ent[["_i", "date", "reg", "lane"]].copy()
    left["date"] = pd.to_datetime(left["date"]).astype("datetime64[ns]")
    left = left.sort_values("date")
    fan["valid_from"] = fan["valid_from"].astype("datetime64[ns]")
    m = pd.merge_asof(left, fan.sort_values("valid_from"), left_on="date", right_on="valid_from",
                      by=["reg", "lane"], direction="backward")
    # 1つ前の期にしか載っていない選手(長期欠場など)に古い数字が付くのは許す(先読みではない)
    ent = ent.merge(m[["_i", "f_n"] + FEATS], on="_i", how="left").drop(columns="_i")
    return ent


def latest_table() -> dict[int, dict]:
    """いちばん新しい期別成績から、選手ごとの 勝率・平均ST・1コースの1着率(その期の出走が少なければ None)。"""
    fs = sorted(DIR.glob("fan*.lzh"))
    if not fs:
        return {}
    d = parse_file(fs[-1].read_bytes())
    out: dict[int, dict] = {}
    for reg, g in d.groupby("reg"):
        r1 = g[g["lane"] == 1].iloc[0]
        if not r1["n_all"]:
            continue
        out[int(reg)] = {"win": float(r1["win"]), "st": float(r1["f_st"]), "n": int(r1["n_all"]),
                         "c1n": int(r1["f_n"]), "c1w": round(float(r1["f_w"] / r1["f_n"]), 3) if r1["f_n"] >= 5 else None}
    return out


if __name__ == "__main__":
    print("取得", fetch(), "件 /", len(list(DIR.glob('fan*.lzh'))), "ファイル")
