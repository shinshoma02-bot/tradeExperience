"""AIにニュース見出しを読ませた予測を、前向きに記録して後で検証する(売買には使わない)。

過去のニュースでAIをバックテストすると、モデルの学習データに「その後の株価の結果」が含まれていて
未来の情報が漏れる。そこで今日以降の予測だけを追記専用で記録し、数か月後に当たったかを評価する。
結果を見てから書き換えないよう、記録は追記のみ(1日1回)。
"""
import json
import os
import re
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Callable

PROMPT_VERSION = "v2"
VARIANTS = ("news", "control", "news_price")   # 事前に決めた比較。news=見出しのみ、control=情報なし(AIの癖の基準)、news_price=見出し+直近の値動き
TARGETS = {"1348.T": "日本株(TOPIX連動ETF)", "SPY": "米国株(S&P500連動ETF)", "USDJPY=X": "ドル円"}
FEEDS = [
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://www3.nhk.or.jp/rss/news/cat5.xml",
    "https://news.yahoo.co.jp/rss/topics/business.xml",
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
]
HORIZONS = {"1d": 1, "5d": 5}


def fetch_headlines(feeds: list[str] = FEEDS, max_age_hours: int = 36, limit: int = 40) -> list[str]:
    """RSSから、直近の見出し(タイトルのみ)を集める。古い記事は捨てる。"""
    now = datetime.now(timezone.utc)
    out: list[str] = []
    for url in feeds:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            root = ET.fromstring(urllib.request.urlopen(req, timeout=15).read())
        except Exception:
            continue
        for item in root.iter("item"):
            title = (item.findtext("title") or "").strip()
            pub = item.findtext("pubDate")
            try:
                if pub and now - parsedate_to_datetime(pub) > timedelta(hours=max_age_hours):
                    continue
            except Exception:
                pass
            if title:
                out.append(title)
    seen, uniq = set(), []
    for t in out:  # 重複除去(順序は保つ)
        if t not in seen:
            seen.add(t); uniq.append(t)
    return uniq[:limit]


def build_prompt(headlines: list[str], variant: str = "news", date: str = "", returns: dict | None = None) -> str:
    targets = "\n".join(f"- {k}: {v}" for k, v in TARGETS.items())
    schema = ('次のJSONだけを出力してください(説明文やコードブロックは不要):\n'
              '{"targets": {"<対象コード>": {"1d": {"dir": "up|down|flat", "conf": 0.0-1.0}, "5d": {"dir": "up|down|flat", "conf": 0.0-1.0}}}, "rationale": "50字以内"}\n'
              "1d は次の営業日の終値、5d は5営業日後の終値が、今日の終値より上か下かの予測です。")
    if variant == "control":   # ニュースなし。AIが何も知らない時にどちらへ偏るか(癖)を測る
        return (f"あなたは金融市場のアナリストです。今日は{date}です。ニュースも価格も与えられていません。"
                f"次の対象の方向を予測してください。確信がなければ flat としてください。\n\n対象:\n{targets}\n\n{schema}")
    news = "\n".join(f"{i + 1}. {h}" for i, h in enumerate(headlines))
    price = ""
    if variant == "news_price" and returns:
        lines = "\n".join(f"- {k}: 直近1日 {v['1d']:+.2f}%、5日 {v['5d']:+.2f}%、20日 {v['20d']:+.2f}%" for k, v in returns.items())
        price = f"\n直近の値動き(終値ベース):\n{lines}\n"
        basis = "次のニュース見出し(直近36時間)と直近の値動きを根拠に"
    else:
        basis = "次のニュース見出し(直近36時間)だけを根拠に"
    return (f"あなたは金融市場のアナリストです。{basis}、各対象の方向を予測してください。確信がなければ flat としてください。\n\n"
            f"対象:\n{targets}\n{price}\nニュース見出し:\n{news}\n\n{schema}")


def call_anthropic(prompt: str, api_key: str, model: str) -> str:
    """Anthropic Messages API を直接呼ぶ(依存を増やさないため urllib)。"""
    body = json.dumps({"model": model, "max_tokens": 600, "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=body,
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
    )
    res = json.load(urllib.request.urlopen(req, timeout=60))
    return "".join(b.get("text", "") for b in res.get("content", []))


def parse_forecast(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        for t, v in d["targets"].items():
            for h in HORIZONS:
                assert v[h]["dir"] in ("up", "down", "flat")
        return d
    except Exception:
        return None


class AIForwardLog:
    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def rows(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, encoding="utf-8") as f:
            return [json.loads(x) for x in f if x.strip()]

    def done_today(self, now: datetime, variant: str = "news") -> bool:
        return any(r["ts"][:10] == now.date().isoformat() and r.get("variant", "news") == variant for r in self.rows())

    def run_once(self, api_key: str, model: str, now: datetime | None = None,
                 fetch: Callable[[], list[str]] = fetch_headlines,
                 llm: Callable[[str, str, str], str] = call_anthropic,
                 returns: Callable[[], dict] | None = None) -> list[dict]:
        """1日1回だけ、各バリアントの予測を追記する。既に今日分があるバリアントは飛ばす。"""
        now = now or datetime.now(timezone.utc)
        todo = [v for v in VARIANTS if not self.done_today(now, v)]
        if not todo:
            return []
        heads = fetch()
        out = []
        for v in todo:
            if v != "control" and len(heads) < 5:  # 見出しが少なすぎる日は、ニュースを使うバリアントは記録しない
                continue
            rets = None
            if v == "news_price":
                try:
                    rets = returns() if returns else None
                except Exception:
                    rets = None
                if not rets:
                    continue
            fc = parse_forecast(llm(build_prompt(heads, v, now.date().isoformat(), rets), api_key, model))
            if not fc:
                continue
            row = {"ts": now.isoformat(), "variant": v, "model": model, "prompt_version": PROMPT_VERSION,
                   "headlines": heads if v != "control" else [], "forecast": fc}
            with open(self.path, "a", encoding="utf-8") as f:  # 追記のみ。書き換えない
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.append(row)
        return out


def recent_returns(candles_by_target: dict) -> dict:
    """直近1/5/20営業日の騰落率(%)。予測のプロンプト用(news_price)。"""
    out = {}
    for t, c in candles_by_target.items():
        if len(c) > 21:
            out[t] = {k: (c[-1].close / c[-1 - n].close - 1) * 100 for k, n in (("1d", 1), ("5d", 5), ("20d", 20))}
    return out


def evaluate(rows: list[dict], candles_by_target: dict, costs_bps: float = 5.0) -> dict:
    """記録した予測が、その後の実際の値動きと合っていたかを集計する。

    candles_by_target: {対象: [Candle(ts, high, low, close), ...]}(確定した日足、古い順)
    基準価格は『予測した時点までで最後の終値』、結果は horizon 営業日後の終値。
    比較対象: 『いつも上と予測』(その期間の上昇比率)、『ランダム』(的中率50%)。
    """
    out: dict = {}
    for variant in VARIANTS:
      vrows = [r for r in rows if r.get("variant", "news") == variant]
      for t, candles in candles_by_target.items():
        for h, n in HORIZONS.items():
            hits = total = up_base = 0
            pnl: list[float] = []
            for r in vrows:
                d = r["forecast"]["targets"].get(t, {}).get(h)
                if not d:
                    continue
                t0 = datetime.fromisoformat(r["ts"]).timestamp()
                idx = [i for i, c in enumerate(candles) if c.ts + 86400 <= t0]  # 予測時点で確定していた日足
                if not idx or idx[-1] + n >= len(candles):
                    continue  # まだ結果が出ていない
                base, res = candles[idx[-1]].close, candles[idx[-1] + n].close
                ret = res / base - 1
                total += 1
                up_base += ret > 0
                if d["dir"] == "flat":
                    continue
                side = 1 if d["dir"] == "up" else -1
                hits += (ret > 0) == (side > 0)
                pnl.append(side * ret - 2 * costs_bps / 1e4)
            called = len(pnl)
            out[f"{variant}/{t}/{h}"] = {
                "evaluated": total, "called": called,
                "hit_rate": round(hits / called, 3) if called else None,
                "always_up_hit_rate": round(up_base / total, 3) if total else None,
                "mean_net_return_pct": round(sum(pnl) / called * 100, 3) if called else None,
            }
    n_min = min((v["evaluated"] for v in out.values()), default=0)
    out["_note"] = ("評価できた予測が60件を超えるまでは、的中率の差は偶然と区別できない。"
                    "news が control(情報なし)や『いつも上』を上回るか、news_price が news を上回るかを見る"
                    if n_min < 60 else "標本は一応たまった。news と control の差、ランダム・『いつも上』との差で判断する(それでも多重比較に注意)")
    return out
