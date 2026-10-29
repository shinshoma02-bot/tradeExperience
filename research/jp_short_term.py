"""日本の大型高配当株: 権利取り・先回り・寄り付きギャップ・権利月の検証。

実行: python research/jp_short_term.py   (Yahoo日足を取得。キャッシュは環境変数 JP_CACHE か research/data/jp_cache)
- 価格は Yahoo の raw close/open(分割調整済み・配当未調整)。配当は events.dividends(権利落ち日・分割調整済み金額)。
- 生存者バイアス: 現在も大型で流動性のある銘柄だけを選んでいる(過去に落ちた銘柄は含まない)。
  このため絶対水準は上振れ方向。権利取りの「落ちる幅 vs 配当」は銘柄の勝ち負けに依らないので影響は小さいが、
  先回り・ギャップの水準は過大の可能性。
- 検証した仮説の総数は main() の最後に表示する。t値は権利落ち日(同日の銘柄は相関)でクラスタした頑健t。
"""
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

TICKERS = ["8306.T", "8316.T", "8411.T", "9433.T", "9432.T", "2914.T", "8058.T", "8031.T",
           "8001.T", "8053.T", "8002.T", "8766.T", "8591.T", "5108.T", "7203.T", "4502.T",
           "9434.T", "8604.T", "3382.T", "5401.T", "6501.T", "4063.T", "8801.T", "9020.T",
           "2502.T", "4503.T", "8725.T", "8630.T", "1605.T", "5020.T"]
MARKET = "^N225"
CACHE = Path(os.environ.get("JP_CACHE", Path(__file__).parent / "data" / "jp_cache"))
TAX = 0.20315
SPLIT = pd.Timestamp("2019-01-01")
N_TESTS = 0


def fetch(sym):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / (sym.replace("^", "_") + ".json")
    if f.exists():
        return json.loads(f.read_text())
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
           "?interval=1d&period1=0&period2=1790000000&events=div")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    d = json.loads(urllib.request.urlopen(req, timeout=30).read())
    f.write_text(json.dumps(d))
    return d


def parse(d):
    r = d["chart"]["result"][0]
    idx = pd.to_datetime(r["timestamp"], unit="s") + pd.Timedelta(hours=9)
    q = r["indicators"]["quote"][0]
    df = pd.DataFrame({"o": q["open"], "c": q["close"]}, index=idx.normalize()).dropna()
    df = df[(df.o > 0) & (df.c > 0)]
    df = df[~df.index.duplicated()]
    divs = {}
    for v in (r.get("events", {}) or {}).get("dividends", {}).values():
        dt = (pd.Timestamp(v["date"], unit="s") + pd.Timedelta(hours=9)).normalize()
        divs[dt] = divs.get(dt, 0) + v["amount"]
    return df, pd.Series(divs, dtype=float)


def cluster_t(x, dates):
    """平均と、日付クラスタ頑健なt値。"""
    x = np.asarray(x, float)
    n = len(x)
    if n < 5:
        return np.nan, np.nan
    m = x.mean()
    s = pd.Series(x - m).groupby(np.asarray(dates)).sum()
    se = np.sqrt((s ** 2).sum()) / n
    return m, (m / se if se > 0 else np.nan)


def line(label, x, dates, cost_bp=0.0):
    """x: 1取引の収益率(小数)。bp表示。"""
    global N_TESTS
    N_TESTS += 1
    x = np.asarray(x, float)
    if len(x) < 5:
        print(f"{label:46s} n={len(x)} (少なすぎ)")
        return
    m, t = cluster_t(x, dates)
    print(f"{label:46s} n={len(x):5d} 平均{m*1e4:7.1f}bp t={t:5.2f} 勝率{(x>0).mean()*100:4.1f}%"
          f" | 費用後(往復{cost_bp:.0f}bp){(m*1e4-cost_bp):7.1f}bp")


def main():
    data, divs, skipped = {}, {}, []
    for s in TICKERS:
        try:
            df, dv = parse(fetch(s))
            if len(df) < 1000:
                skipped.append(s); continue
            data[s], divs[s] = df, dv
        except Exception as e:  # noqa
            skipped.append(s)
    mk, _ = parse(fetch(MARKET))
    print(f"取得成功 {len(data)}銘柄 / 取得失敗・短すぎ: {skipped}")
    first = min(df.index[0] for df in data.values()); last = max(df.index[-1] for df in data.values())
    print(f"期間 {first.date()} 〜 {last.date()}  (前半=〜2018, 後半=2019〜)")
    print("価格(最新close)中央値: %.0f円 -> 100株 %.0f円" % (
        np.median([df.c.iloc[-1] for df in data.values()]),
        100 * np.median([df.c.iloc[-1] for df in data.values()])))
    mkt_ret = mk.c.pct_change()

    # ---- イベント表(権利落ち日) ----
    ev = []
    for s, df in data.items():
        dv = divs[s]
        pos = {d: i for i, d in enumerate(df.index)}
        uncond1 = (df.c.pct_change()).mean()
        for d, amt in dv.items():
            if d not in pos or pos[d] < 21 or amt <= 0:
                continue
            i = pos[d]
            pm1, p0 = df.c.iloc[i - 1], df.c.iloc[i]
            row = dict(sym=s, date=d, div=amt, divy=amt / pm1, pm1=pm1,
                       dropr=(pm1 - p0) / amt,
                       r_ex=(p0 - pm1) / pm1,
                       mkt=mkt_ret.get(d, np.nan),
                       uncond1=uncond1,
                       month=d.month)
            # 翌日終値まで
            if i + 1 < len(df):
                row["r_ex2"] = (df.c.iloc[i + 1] - pm1) / pm1
            for k in (5, 10, 20):
                row[f"run{k}"] = pm1 / df.c.iloc[i - 1 - k] - 1
                row[f"unc{k}"] = np.nan
                dm = df.index[i - 1]; d0 = df.index[i - 1 - k]
                try:
                    row[f"mk{k}"] = mk.c.asof(dm) / mk.c.asof(d0) - 1
                except Exception:
                    row[f"mk{k}"] = np.nan
            ev.append(row)
    ev = pd.DataFrame(ev)
    # 無条件k日リターン(銘柄別平均)
    for k in (5, 10, 20):
        for s, df in data.items():
            u = (df.c / df.c.shift(k) - 1).mean()
            ev.loc[ev.sym == s, f"unc{k}"] = u
    ev["late"] = ev.date >= SPLIT
    print(f"\n権利落ちイベント数 {len(ev)} (前半 {(~ev.late).sum()} / 後半 {ev.late.sum()})")
    print(f"平均配当利回り(権利付き終値比) {ev.divy.mean()*1e4:.0f}bp、 落ち幅/配当 の中央値 {ev.dropr.median():.2f} 平均 {ev.dropr.mean():.2f}")

    # ---- (a) 権利取り ----
    print("\n=== (a) 権利付き最終日の終値で買い→権利落ち日の終値で売り(配当込み) ===")
    ev["gross"] = ev.r_ex + ev.divy   # 配当は税引前
    # 税: 合計損益がプラスなら20.315%、配当は源泉で先に取られる(損益通算前は配当分にかかる)ものとして
    # 保守的に「配当に課税、値下がりは損失として通算できない(最悪ケース)」
    ev["net_taxdiv"] = ev.r_ex + ev.divy * (1 - TAX)
    for nm, sub in [("全期間", ev), ("前半〜2018", ev[~ev.late]), ("後半2019〜", ev[ev.late])]:
        line(f"(a) 配当込み gross [{nm}]", sub.gross, sub.date, 4)
    line("(a) 値下がりのみ(配当除く) 全期間", ev.r_ex, ev.date)
    line("(a) 配当税引後(配当に20.315%) 全期間", ev.net_taxdiv, ev.date, 4)
    ev["excess"] = ev.gross - ev.mkt.fillna(0)
    line("(a) 日経平均超過 gross 全期間", ev.excess, ev.date, 4)
    line("(a) 無条件の1日平均を引いた超過 全期間", ev.gross - ev.uncond1, ev.date, 4)
    # 配当落ち幅/配当
    print(f"  参考: 配当(平均)={ev.divy.mean()*1e4:.0f}bp、権利落ち日の値下がり(平均)={-ev.r_ex.mean()*1e4:.0f}bp"
          f" -> 値下がり/配当 = {(-ev.r_ex.mean()/ev.divy.mean()):.2f} (1未満なら配当取り得)")
    for nm, sub in [("前半", ev[~ev.late]), ("後半", ev[ev.late])]:
        print(f"  {nm}: 値下がり/配当 = {(-sub.r_ex.mean()/sub.divy.mean()):.2f}")
    line("(a2) 権利落ち翌日終値で売る(配当込み) 全期間", (ev.r_ex2 + ev.divy).dropna(), ev.date[ev.r_ex2.notna()], 4)

    # ---- (d) 権利月 ----
    print("\n=== (d) 3月・9月 vs その他の権利月(a の超過=gross-無条件1日) ===")
    ev["ex_ex"] = ev.gross - ev.uncond1
    for nm, m in [("3月・9月", ev.month.isin([3, 9])), ("その他", ~ev.month.isin([3, 9]))]:
        sub = ev[m]
        line(f"(d) {nm} gross", sub.gross, sub.date, 4)
    for nm, m in [("3月", ev.month == 3), ("9月", ev.month == 9)]:
        sub = ev[m]
        print(f"   参考 {nm}: 値下がり/配当 = {(-sub.r_ex.mean()/sub.divy.mean()):.2f}  n={len(sub)}")
    a, b = ev[ev.month.isin([3, 9])].gross, ev[~ev.month.isin([3, 9])].gross
    diff = a.mean() - b.mean()
    print(f"   差(3,9月-その他) = {diff*1e4:.1f}bp (単純比較。検定は上の2行のtから概算)")
    global N_TESTS
    N_TESTS += 1

    # ---- (b) 先回り ----
    print("\n=== (b) 権利落ち日のk営業日前の終値で買い→権利付き最終日の終値で売り(配当は受け取らない) ===")
    for k in (5, 10, 20):
        x = ev[f"run{k}"]
        for nm, sub in [("全期間", ev), ("前半", ev[~ev.late]), ("後半", ev[ev.late])]:
            line(f"(b) k={k:2d} 生リターン [{nm}]", sub[f"run{k}"], sub.date, 8)
        line(f"(b) k={k:2d} 日経平均の同期間を引いた超過 全期間", ev[f"run{k}"] - ev[f"mk{k}"], ev.date, 8)
        line(f"(b) k={k:2d} 日経平均超過 後半2019〜", (ev[f"run{k}"] - ev[f"mk{k}"])[ev.late], ev.date[ev.late], 8)
        sub = ev
        line(f"(b) k={k:2d} 無条件k日平均を引いた超過 全期間", sub[f"run{k}"] - sub[f"unc{k}"], sub.date, 8)
        line(f"(b) k={k:2d} 超過 後半2019〜", (sub[f"run{k}"] - sub[f"unc{k}"])[sub.late], sub.date[sub.late], 8)

    # ---- (c) ギャップ ----
    print("\n=== (c) 寄り付きギャップ(前日終値→始値)後の日中(始値→終値)。権利落ち日は除外 ===")
    rows = []
    for s, df in data.items():
        exd = set(divs[s].index)
        g = df.o / df.c.shift(1) - 1
        intr = df.c / df.o - 1
        t = pd.DataFrame({"gap": g, "intr": intr, "sym": s})
        t = t[~t.index.isin(exd)].dropna()
        t = t[(t.gap.abs() < 0.2)]  # 分割等の異常を除く
        # 2日連続歩み寄り: 翌日が権利落ちでも当日の日中は影響なし
        rows.append(t)
    gp = pd.concat(rows)
    gp["date"] = gp.index
    print(f"  全ギャップ日 {len(gp)}。無条件の日中平均 {gp.intr.mean()*1e4:.1f}bp")
    for th in (0.01, 0.02):
        up = gp[gp.gap >= th]; dn = gp[gp.gap <= -th]
        for nm, sub, sgn, lab in [("ギャップアップ→買い(続伸)", up, 1, "up"),
                                   ("ギャップアップ→売り(反転)", up, -1, "up"),
                                   ("ギャップダウン→買い(反転)", dn, 1, "dn"),
                                   ("ギャップダウン→売り(続落)", dn, -1, "dn")]:
            # 4通りは2通りの符号違いで同一データ(重複検定を避けるため買い方向のみカウント)
            for per, ss in [("全期間", sub), ("前半", sub[sub.index < SPLIT]), ("後半", sub[sub.index >= SPLIT])]:
                x = sgn * ss.intr
                if per == "全期間":
                    line(f"(c) |gap|>={th*100:.0f}% {nm} [{per}]", x, ss.date, 10)
                else:
                    line(f"(c)     同上 [{per}]", x, ss.date, 10)
            if sgn == -1:
                N_TESTS -= 3  # 符号反転は同じ検定なので数え直さない
    # 指数全体のギャップ日の影響か(銘柄固有か)確認: 日経平均ギャップを引く
    mg = (mk.o / mk.c.shift(1) - 1)
    mi = (mk.c / mk.o - 1)
    gp["mgap"] = mg.reindex(gp.index).values
    gp["mintr"] = mi.reindex(gp.index).values
    idio = gp[(gp.gap - gp.mgap).abs() >= 0.01]
    print("  銘柄固有(個別ギャップ-指数ギャップ>=1%)の日中(指数の日中を引く):")
    for nm, m, sg in [("up", (idio.gap - idio.mgap) > 0, 1), ("dn", (idio.gap - idio.mgap) < 0, 1)]:
        ss = idio[m]
        line(f"(c') 固有ギャップ{nm} 日中超過(買い方向)", (ss.intr - ss.mintr), ss.date, 10)

    # ---- 回数・まとめ ----
    print("\n=== 売買回数の目安 ===")
    print(f"  (a) 権利取り: {len(ev)}件/{len(data)}銘柄/約{(last-first).days/365:.0f}年 = 銘柄あたり年約{len(ev)/len(data)/((last-first).days/365):.1f}回")
    print(f"  (c) |gap|>=1%: 銘柄あたり年約{(gp.gap.abs()>=0.01).sum()/len(data)/((last-first).days/365):.1f}回")
    print(f"\n検定した仮説の行数(重複除く概算): {N_TESTS}。Bonferroni 的には |t|>~3 が目安。")

    # ---- Part 3: 低位株 ----
    print("\n=== Part3: 1万円で買える銘柄(100株=1万円以下 => 株価100円以下)の呼値コスト ===")
    print("呼値(TOPIX500以外/Prime・Standard、JPX表。要確認): 〜3,000円=1円, 3,000超〜5,000円=5円")
    print("株価  100株金額  1ティック(%)  往復で最低1ティック分(円/100株) 必要な値幅(%)")
    for p in (30, 50, 80, 100, 150, 300):
        tick = 1
        cost_yen = tick * 100
        print(f"{p:4d}円  {100*p:6d}円   {tick/p*100:5.2f}%   {cost_yen:5d}円(往復1ティック)   {tick/p*100:5.2f}%")
    print("-> 買い気配と売り気配が1ティック差なら、成行で往復すると約1〜3%を失う。"
          "(a)(c)の平均優位(数bp〜数十bp)より1〜2桁大きい。")
    prices = sorted([(df.c.iloc[-1], s) for s, df in data.items()])
    print("  参考: 本検証の銘柄の最新株価 最安 %s=%.0f円 -> 100株 %.0f円(1万円では買えない)" %
          (prices[0][1], prices[0][0], prices[0][0] * 100))
    print("  1万円を100円株(100株)に全額入れ、勝率55%・平均+/-2%動く想定でも期待値: "
          "(0.55*2%-0.45*2%)*10000 - 100円(往復1ティック) = 20 - 100 = -80円/回")


if __name__ == "__main__":
    sys.exit(main())
