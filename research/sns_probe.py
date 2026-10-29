"""SNSデータの入手性の確認(各ソース1回だけ試す)と、取れた公開データでの検証、前向き記録の1回分。
  python research/sns_probe.py            # 入手性の表 + 検証 + 前向き記録の取得テスト
  python research/sns_probe.py --no-test  # 入手性の表だけ
規約の方針: 公式API・公開データセットだけ。403/401/429が返ったら『取得不可』と記録して終わり(回避しない)。
個人のユーザー名・本文は保存しない(集計値だけ)。検証用データはHugging Faceの公開CSV(apache-2.0)。
検証: mjw/stock_market_tweets (2015-01-01〜2019-12-31, AAPL/AMZN/GOOG/MSFT/TSLA の5銘柄, 170万件, 日付のみ)。
  注目度 = その日の投稿数の、過去20日平均に対するzスコア。論調 = 単語リストの強気/弱気比率(MLなし)。
  ルックアヘッド対策: ツイートは日付しかない(時刻・時間帯不明)ので、日付tの集計は『t日の引け後に分かった』ものとし、
  t+1日の終値で入って、そこからh営業日後の終値で出る(超過 = 銘柄 - SPY)。調整後終値(分割・配当込み)を使用。
"""
import json, os, re, sys, time, urllib.request, urllib.error
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get("SNS_CACHE", os.path.join(HERE, "_cache"))
UA = "tradeExperience-research/0.1 (personal, low-volume; one request per source)"
HF_CSV = "https://huggingface.co/datasets/mjw/stock_market_tweets/resolve/main/stock_market_tweets.csv"
TESTS = []


def get(url, ua=UA, timeout=25):
    """1回だけ取得。失敗しても再試行・回避しない。(status, body_bytes)"""
    try:
        r = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": ua}), timeout=timeout)
        return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b""
    except Exception as e:
        return 0, str(e).encode()[:200]


# ---------- Step 1: 入手性 ----------
def probe():
    rows = []
    def add(name, url, note_ok, note_ng, ok_codes=(200,)):
        s, b = get(url)
        rows.append((name, s, note_ok if s in ok_codes else note_ng))
        return s, b
    add("X 公式API v2 検索(無認証)", "https://api.x.com/2/tweets/search/recent?query=SPY", "取れた",
        "取得不可: 認証必須。現行の公式APIは有料/従量課金が中心(料金は要確認)。個人の1万円運用には見合わない")
    add("Reddit r/wallstreetbets new.json", "https://www.reddit.com/r/wallstreetbets/new.json?limit=5", "取れた",
        "取得不可: 403(User-Agent名乗りでも拒否)。OAuth+アプリ登録が必要で、回避はしない。規約上も利用は要申請の方向")
    s, b = add("StockTwits 公開API(SPY)", "https://api.stocktwits.com/api/2/streams/symbol/SPY.json",
               "取れた(直近30件のみ。過去分なし)。強気/弱気タグ付き。再配布・商用は規約要確認", "取得不可")
    add("Bluesky 公開検索API", "https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts?q=%24SPY&limit=5", "取れた",
        "取得不可: 403(無認証の検索は拒否)。アカウント作成はしない")
    add("Mastodon タグTL(mastodon.social)", "https://mastodon.social/api/v1/timelines/tag/stocks?limit=5",
        "取れた(直近のみ、金融の投稿はごく少数。全文検索は空)", "取得不可")
    add("Hugging Face dataset検索", "https://huggingface.co/api/datasets?search=stock%20tweets&limit=5", "取れた", "取得不可")
    rows.append(("Yahoo!ファイナンス掲示板 / 5ch / note", None,
                 "試していない(自動取得は規約で禁止/制限されている可能性が高いので使わない。要確認)"))
    rows.append(("日本語の公開データ(HF)", None,
                 "検索では Finance-Instruct-500k-Japanese(英語の翻訳QA)程度。投稿日時+銘柄つきのSNSデータは見つからず"))
    print("== Step1 入手性(各1回) ==")
    for n, s, t in rows:
        print(f"- {n} [{s if s is not None else '-'}]: {t}")


# ---------- Step 2: 検証 ----------
BULL = r"\b(?:buy|buying|bull|bullish|long|calls?|moon|rally|breakout|upgrade|beat|strong|green|undervalued|up)\b"
BEAR = r"\b(?:sell|selling|bear|bearish|short|puts?|crash|dump|downgrade|miss|weak|red|overvalued|down|drop)\b"


def daily_agg():
    f = os.path.join(CACHE, "sns_daily_agg.csv")
    if os.path.exists(f):
        return pd.read_csv(f, parse_dates=["date"])
    os.makedirs(CACHE, exist_ok=True)
    raw = os.path.join(CACHE, "mjw_tweets.csv")
    if not os.path.exists(raw):
        print("  公開CSVを取得中(約300MB)...")
        urllib.request.urlretrieve(HF_CSV, raw)
    out = []
    for ch in pd.read_csv(raw, usecols=["post_date", "body", "ticker_symbol"], chunksize=200000):
        t = ch.body.fillna("").str.lower()
        ch["bull"] = t.str.contains(BULL, regex=True).astype(int)
        ch["bear"] = t.str.contains(BEAR, regex=True).astype(int)
        out.append(ch.groupby(["post_date", "ticker_symbol"]).agg(n=("bull", "size"), bull=("bull", "sum"), bear=("bear", "sum")))
    a = pd.concat(out).groupby(level=[0, 1]).sum().reset_index()
    a.columns = ["date", "sym", "n", "bull", "bear"]
    a["date"] = pd.to_datetime(a["date"])
    a.to_csv(f, index=False)
    os.remove(raw)  # 本文を含む生データは残さない
    return a


def px(sym):
    f = os.path.join(CACHE, f"sns_px_{sym}.json")
    if os.path.exists(f):
        res = json.load(open(f))
    else:
        u = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval=1d&period1=1400000000&period2=1893000000"
        res = json.loads(get(u, ua="Mozilla/5.0", timeout=40)[1])["chart"]["result"][0]
        json.dump(res, open(f, "w"))
    s = pd.Series(res["indicators"]["adjclose"][0]["adjclose"],
                  index=pd.to_datetime(np.array(res["timestamp"]) + res["meta"].get("gmtoffset", 0), unit="s").normalize())
    s = s.dropna()
    s = s[~s.index.duplicated()]
    r = s.pct_change().abs()
    if (r > 0.15).any():
        print(f"  注意: {sym} に日次±15%超 {int((r > .15).sum())}件")
    return s


def tstat(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return (x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else np.nan), len(x)


def study():
    print("\n== Step2 検証 (mjw/stock_market_tweets 2015-2019, 5銘柄) ==")
    a = daily_agg()
    spy = px("SPY")
    rows = []
    for sym in sorted(a.sym.unique()):
        p = px(sym)
        d = a[a.sym == sym].set_index("date").sort_index()
        d = d.reindex(spy.index.intersection(p.index)).fillna({"n": 0, "bull": 0, "bear": 0})  # 営業日に揃える
        d = d[(d.index >= "2015-01-01") & (d.index <= "2019-12-31")]
        d["attn"] = (d.n - d.n.rolling(20).mean().shift(1)) / d.n.rolling(20).std().shift(1)  # t日までの情報だけ
        s = (d.bull - d.bear) / (d.bull + d.bear).clip(lower=1)
        d["sent"] = (s - s.rolling(60).mean().shift(1)) / s.rolling(60).std().shift(1)
        for h in (1, 5, 20):
            # t日の集計 -> t+1の終値で入り t+1+h の終値で出る
            ex = (p.shift(-(1 + h)) / p.shift(-1) - 1) - (spy.shift(-(1 + h)) / spy.shift(-1) - 1)
            d[f"ex{h}"] = ex.reindex(d.index)
        d["sym"] = sym
        rows.append(d)
    D = pd.concat(rows)
    print(f"  標本: {len(D)} 銘柄日, 営業日 {D.index.nunique()}, 2015-2019のみ(上場廃止なし=生存者・勝ち組5銘柄)")
    print("  単位: 超過リターン(対SPY, %)。t値は『h日以上あけた重ならないイベント』で計算(銘柄をまたいで1つずつ)")
    res = []
    for h in (1, 5, 20):
        col = f"ex{h}"
        for name, mask_hi, mask_lo in (
            ("注目度急増(z>2) vs それ以外", D.attn > 2, D.attn <= 2),
            ("論調 強気(z>1) vs 弱気(z<-1)", D.sent > 1, D.sent < -1),
        ):
            TESTS.append((name, h))
            # 重なり対策: 各銘柄でh日あけてサンプリング
            def pick(mask):
                out = []
                for sym, g in D[mask].groupby("sym"):
                    last = None
                    for dt in g.index:
                        if last is None or (dt - last).days >= h * 1.5:
                            out.append(g.loc[dt, col]); last = dt
                return np.array(out, float)
            hi, lo = pick(mask_hi), pick(mask_lo)
            hi, lo = hi[~np.isnan(hi)], lo[~np.isnan(lo)]
            diff = hi.mean() - lo.mean()
            se = np.sqrt(hi.var(ddof=1) / len(hi) + lo.var(ddof=1) / len(lo))
            res.append((name, h, len(hi), hi.mean() * 100, len(lo), lo.mean() * 100, diff * 100, diff / se))
    print(f"  {'シグナル':<28}{'h':>3}{'n高':>6}{'平均高':>8}{'n低':>6}{'平均低':>8}{'差':>8}{'t':>7}")
    for r in res:
        print(f"  {r[0]:<28}{r[1]:>3}{r[2]:>6}{r[3]:>8.2f}{r[4]:>6}{r[5]:>8.2f}{r[6]:>8.2f}{r[7]:>7.2f}")
    k = len(res)
    print(f"  試したパターン: {k}(2シグナル x 3期間)。ボンフェローニ補正で |t|>{2.64:.2f} が必要(5%, 6本)。")
    print(f"  |t|>2.64 は {sum(abs(r[7]) > 2.64 for r in res)} 本。")
    print("  限界: 5つの超大型株のみ・5年・日付だけ(時刻なし)・生存者バイアス(全員が勝ち組)・"
          "単語リストの論調は粗い・ボット/重複投稿の影響。結論は大きく割り引く。")


# ---------- Step 3: 前向き記録(追記専用・集計値のみ) ----------
def forward_record(path=None):
    path = path or os.path.join(CACHE, "sns_forward.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stocktwits": {}}
    for sym in ("SPY", "QQQ", "AAPL", "TSLA", "NVDA"):
        s, b = get(f"https://api.stocktwits.com/api/2/streams/symbol/{sym}.json")
        if s != 200:
            rec["stocktwits"][sym] = {"status": s}
            continue
        j = json.loads(b)
        sent = [((m.get("entities") or {}).get("sentiment") or {}).get("basic") for m in j.get("messages", [])]
        rec["stocktwits"][sym] = {"n": len(sent), "bull": sent.count("Bullish"), "bear": sent.count("Bearish"),
                                  "watchers": j.get("symbol", {}).get("watchlist_count")}
        time.sleep(1)
    s, b = get("https://mastodon.social/api/v1/timelines/tag/stocks?limit=40")
    rec["mastodon_stocks_tag"] = {"status": s, "n": len(json.loads(b)) if s == 200 else None}
    with open(path, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("\n== Step3 前向き記録(1回分, 集計値のみ) ==")
    print(json.dumps(rec, ensure_ascii=False))
    print(f"  追記先: {path}  ※StockTwitsは直近30件のスナップショットで、1日1回では取りこぼす。")
    print("  日次の件数は『取得時点の直近30件の時間幅』も記録すると注目度に使える(設計メモ)。")


if __name__ == "__main__":
    probe()
    if "--no-test" not in sys.argv:
        study()
        forward_record()
