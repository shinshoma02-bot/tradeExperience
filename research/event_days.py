"""重要イベント日の値動き検証: FOMC / 米雇用統計 / 米CPI / 日銀会合 / 日本SQ / 米オプション満期 / 月末・四半期末。
  python research/event_days.py
対象: SPY, QQQ, USDJPY=X, 1348.T(MAXIS TOPIX), ^N225。2011-01〜2026-10。
日付の出所(推測なし):
  FOMC   = federalreserve.gov のカレンダー(2011-2020 は fomchistorical{年}.htm、2021- は fomccalendars.htm)。決定日=会合最終日。
           臨時(unscheduled)・書面決議は除外。
  日銀   = boj.or.jp の過去会合一覧/現行ページに載る決定公表文のファイル名(kYYMMDD)。最終日=決定日。2020-05-22の臨時会合は除外。
  雇用統計・CPI = BLS の年別スケジュール(bls.gov/schedule/{年}/home.htm)。2013年(閉鎖後の再設定)・2025年(閉鎖)は
           実際の発表日が載った版。2026年分は予定。発表されなかった月(2025-10の雇用統計・11月CPI)は無し。
  日本SQ = 第2金曜(休場なら直前の営業日)。3・6・9・12月=メジャー、他=月次(オプションSQ)。ルールで計算(取引日カレンダーで補正)。
  米opex = 第3金曜(休場なら直前の営業日)。3・6・9・12月=四半期(クアッドウィッチング)。ルールで計算。
  月末/四半期末 = 各資産の月/四半期の最終取引日。
指標(すべて event日の値と「通常日(そのイベントの日を除く全取引日)」の差、Welch t):
  cc   前日終値→当日終値(配当込み=調整後終値)
  oc   当日 寄り→引け
  gap  前日終値→当日寄り
  post3 当日終値→3営業日後終値(イベント後の値動き)
米イベントを日本の資産(1348, ^N225)で見るときは、日本時間で反応する「米国日付の翌日本営業日」をイベント日とする。
ドル円は日足バーの日付=イベント日(FOMC/雇用統計/CPI/日銀会合は全て同じUTC日に入る)。
前半=〜2018、後半=2019〜。前半で符号を決め、後半で確かめる。コストは片道、往復=2倍。
キャッシュ: research/data/ev_*.json(.gitignore済み)。
"""
import datetime as dt
import json, os, re, sys, time, urllib.request
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data")
UA = {"User-Agent": "Mozilla/5.0"}
START, END = pd.Timestamp("2011-01-01"), pd.Timestamp("2026-10-28")
SYMS = ["SPY", "QQQ", "USDJPY=X", "1348.T", "^N225"]
JP = {"1348.T", "^N225"}
COSTS_BP = [2, 5, 10]  # 片道 0.02/0.05/0.10%

# ---------------------------------------------------------------- 日付
BLS_NFP = """2011-01-07 2011-02-04 2011-03-04 2011-04-01 2011-05-06 2011-06-03 2011-07-08 2011-08-05 2011-09-02 2011-10-07 2011-11-04 2011-12-02
2012-01-06 2012-02-03 2012-03-09 2012-04-06 2012-05-04 2012-06-01 2012-07-06 2012-08-03 2012-09-07 2012-10-05 2012-11-02 2012-12-07
2013-01-04 2013-02-01 2013-03-08 2013-04-05 2013-05-03 2013-06-07 2013-07-05 2013-08-02 2013-09-06 2013-10-22 2013-11-08 2013-12-06
2014-01-10 2014-02-07 2014-03-07 2014-04-04 2014-05-02 2014-06-06 2014-07-03 2014-08-01 2014-09-05 2014-10-03 2014-11-07 2014-12-05
2015-01-09 2015-02-06 2015-03-06 2015-04-03 2015-05-08 2015-06-05 2015-07-02 2015-08-07 2015-09-04 2015-10-02 2015-11-06 2015-12-04
2016-01-08 2016-02-05 2016-03-04 2016-04-01 2016-05-06 2016-06-03 2016-07-08 2016-08-05 2016-09-02 2016-10-07 2016-11-04 2016-12-02
2017-01-06 2017-02-03 2017-03-10 2017-04-07 2017-05-05 2017-06-02 2017-07-07 2017-08-04 2017-09-01 2017-10-06 2017-11-03 2017-12-08
2018-01-05 2018-02-02 2018-03-09 2018-04-06 2018-05-04 2018-06-01 2018-07-06 2018-08-03 2018-09-07 2018-10-05 2018-11-02 2018-12-07
2019-01-04 2019-02-01 2019-03-08 2019-04-05 2019-05-03 2019-06-07 2019-07-05 2019-08-02 2019-09-06 2019-10-04 2019-11-01 2019-12-06
2020-01-10 2020-02-07 2020-03-06 2020-04-03 2020-05-08 2020-06-05 2020-07-02 2020-08-07 2020-09-04 2020-10-02 2020-11-06 2020-12-04
2021-01-08 2021-02-05 2021-03-05 2021-04-02 2021-05-07 2021-06-04 2021-07-02 2021-08-06 2021-09-03 2021-10-08 2021-11-05 2021-12-03
2022-01-07 2022-02-04 2022-03-04 2022-04-01 2022-05-06 2022-06-03 2022-07-08 2022-08-05 2022-09-02 2022-10-07 2022-11-04 2022-12-02
2023-01-06 2023-02-03 2023-03-10 2023-04-07 2023-05-05 2023-06-02 2023-07-07 2023-08-04 2023-09-01 2023-10-06 2023-11-03 2023-12-08
2024-01-05 2024-02-02 2024-03-08 2024-04-05 2024-05-03 2024-06-07 2024-07-05 2024-08-02 2024-09-06 2024-10-04 2024-11-01 2024-12-06
2025-01-10 2025-02-07 2025-03-07 2025-04-04 2025-05-02 2025-06-06 2025-07-03 2025-08-01 2025-09-05 2025-11-20 2025-12-16
2026-01-09 2026-02-11 2026-03-06 2026-04-03 2026-05-08 2026-06-05 2026-07-02 2026-08-07 2026-09-04 2026-10-02""".split()
BLS_CPI = """2011-01-14 2011-02-17 2011-03-17 2011-04-15 2011-05-13 2011-06-15 2011-07-15 2011-08-18 2011-09-15 2011-10-19 2011-11-16 2011-12-16
2012-01-19 2012-02-17 2012-03-16 2012-04-13 2012-05-15 2012-06-14 2012-07-17 2012-08-15 2012-09-14 2012-10-16 2012-11-15 2012-12-14
2013-01-16 2013-02-21 2013-03-15 2013-04-16 2013-05-16 2013-06-18 2013-07-16 2013-08-15 2013-09-17 2013-10-30 2013-11-20 2013-12-17
2014-01-16 2014-02-20 2014-03-18 2014-04-15 2014-05-15 2014-06-17 2014-07-22 2014-08-19 2014-09-17 2014-10-22 2014-11-20 2014-12-17
2015-01-16 2015-02-26 2015-03-24 2015-04-17 2015-05-22 2015-06-18 2015-07-17 2015-08-19 2015-09-16 2015-10-15 2015-11-17 2015-12-15
2016-01-20 2016-02-19 2016-03-16 2016-04-14 2016-05-17 2016-06-16 2016-07-15 2016-08-16 2016-09-16 2016-10-18 2016-11-17 2016-12-15
2017-01-18 2017-02-15 2017-03-15 2017-04-14 2017-05-12 2017-06-14 2017-07-14 2017-08-11 2017-09-14 2017-10-13 2017-11-15 2017-12-13
2018-01-12 2018-02-14 2018-03-13 2018-04-11 2018-05-10 2018-06-12 2018-07-12 2018-08-10 2018-09-13 2018-10-11 2018-11-14 2018-12-12
2019-01-11 2019-02-13 2019-03-12 2019-04-10 2019-05-10 2019-06-12 2019-07-11 2019-08-13 2019-09-12 2019-10-10 2019-11-13 2019-12-11
2020-01-14 2020-02-13 2020-03-11 2020-04-10 2020-05-12 2020-06-10 2020-07-14 2020-08-12 2020-09-11 2020-10-13 2020-11-12 2020-12-10
2021-01-13 2021-02-10 2021-03-10 2021-04-13 2021-05-12 2021-06-10 2021-07-13 2021-08-11 2021-09-14 2021-10-13 2021-11-10 2021-12-10
2022-01-12 2022-02-10 2022-03-10 2022-04-12 2022-05-11 2022-06-10 2022-07-13 2022-08-10 2022-09-13 2022-10-13 2022-11-10 2022-12-13
2023-01-12 2023-02-14 2023-03-14 2023-04-12 2023-05-10 2023-06-13 2023-07-12 2023-08-10 2023-09-13 2023-10-12 2023-11-14 2023-12-12
2024-01-11 2024-02-13 2024-03-12 2024-04-10 2024-05-15 2024-06-12 2024-07-11 2024-08-14 2024-09-11 2024-10-10 2024-11-13 2024-12-11
2025-01-15 2025-02-12 2025-03-12 2025-04-10 2025-05-13 2025-06-11 2025-07-15 2025-08-12 2025-09-11 2025-10-24 2025-12-18
2026-01-13 2026-02-13 2026-03-11 2026-04-10 2026-05-12 2026-06-10 2026-07-14 2026-08-12 2026-09-11 2026-10-14""".split()


def fetch_text(url, name):
    os.makedirs(CACHE, exist_ok=True)
    f = os.path.join(CACHE, f"ev_{name}.html")
    if not os.path.exists(f):
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40) as r:
            open(f, "wb").write(r.read())
    return open(f, encoding="utf-8", errors="ignore").read()


def flat(html):
    s = re.sub(r"<[^>]+>", "|", html)
    s = re.sub(r"\s+", " ", s)
    return re.sub(r"(\| ?)+", "|", s)


MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


def month_no(tok):
    """'Jan/Feb' や 'Feb'、'March' → 最後の月の番号(会合は終了月の日付で決定)"""
    last = tok.split("/")[-1].strip()[:3]
    return [m[:3] for m in MONTHS].index(last) + 1


def fomc_dates():
    out = set()
    for y in range(2011, 2021):
        s = flat(fetch_text(f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{y}.htm", f"fomc{y}"))
        # 例: "January 25-26 Meeting - 2011" / 月またぎは "Feb 31-1 Meeting"(=1/31-2/1、月は終了月)。
        # unscheduled/cancelled は ' Meeting' の直前に付くので一致しない
        for m in re.finditer(r"([A-Z][a-z]+(?:/[A-Z][a-z]+)?) (\d{1,2})(?:-(\d{1,2}))? Meeting - (\d{4})", s):
            mo, d1, d2, yy = m.groups()
            if int(yy) != y:
                continue
            out.add(dt.date(y, month_no(mo), int(d2 or d1)))
    t = fetch_text("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm", "fomccal")
    for y in range(2021, 2027):
        i = t.find(f"{y} FOMC Meetings")
        j = t.find("FOMC Meetings", i + 20)
        s = flat(t[i:j])
        for mo, days in re.findall(r"\|((?:%s)[a-z]*(?:/[A-Z][a-z]+)?)\|([^|]*)\|" % "|".join(m[:3] for m in MONTHS), s):
            if "notation" in days or "unscheduled" in days.lower() or "cancel" in days.lower():
                continue
            m = re.match(r"(\d{1,2})(?:-(\d{1,2}))?", days.strip())
            if not m:
                continue
            out.add(dt.date(y, month_no(mo), int(m.group(2) or m.group(1))))
    return sorted(d for d in out if START.date() <= d <= END.date())


def boj_dates():
    out = set()
    for url, nm in [("https://www.boj.or.jp/en/mopo/mpmsche_minu/past.htm", "bojpast"),
                    ("https://www.boj.or.jp/en/mopo/mpmsche_minu/index.htm", "bojnow")]:
        t = fetch_text(url, nm)
        for s in re.findall(r"/k(\d{6})[a-z]\.pdf", t):
            out.add(dt.datetime.strptime(s, "%y%m%d").date())
    out.discard(dt.date(2020, 5, 22))  # 臨時会合(金融機関への資金供給)
    return sorted(d for d in out if START.date() <= d <= END.date())


def nth_friday(y, m, n):
    d = dt.date(y, m, 1)
    d += dt.timedelta(days=(4 - d.weekday()) % 7)
    return d + dt.timedelta(weeks=n - 1)


# ---------------------------------------------------------------- データ
def load_prices(sym):
    os.makedirs(CACHE, exist_ok=True)
    f = os.path.join(CACHE, "ev_%s.json" % sym.replace("^", "_").replace("=", "_"))
    if not os.path.exists(f):
        u = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval=1d"
             f"&period1=1285000000&period2={int(time.time()) + 86400}")
        res = json.load(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=40))["chart"]["result"][0]
        json.dump(res, open(f, "w"))
    res = json.load(open(f))
    q = res["indicators"]["quote"][0]
    off = res["meta"].get("gmtoffset", 0)
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose", q["close"])
    d = pd.DataFrame({"o": q["open"], "c": q["close"], "a": adj},
                     index=pd.to_datetime(np.array(res["timestamp"]) + off, unit="s").normalize())
    d = d.dropna()
    d = d[(d.o > 0) & (d.c > 0) & (d.a > 0)]
    d = d[~d.index.duplicated()]
    return d


def build_metrics(d):
    m = pd.DataFrame(index=d.index)
    m["cc"] = d.a.pct_change()
    m["oc"] = d.c / d.o - 1
    m["gap"] = (1 + m.cc) / (1 + m.oc) - 1
    m["post3"] = d.a.shift(-3) / d.a - 1
    bad = (m.cc.abs() > .15) | (m.oc.abs() > .15)
    nbad = int(bad.sum())
    m[bad] = np.nan
    m = m[(m.index >= START) & (m.index <= END)]
    return m, nbad


def snap(idx, day, mode):
    """カレンダー日 day を資産の取引日に対応させる。same=当日(無ければ None)、next=翌取引日、prev=以前の直近取引日"""
    day = pd.Timestamp(day)
    if mode == "same":
        return day if day in idx else None
    if mode == "next":
        k = idx.searchsorted(day, side="right")
        return idx[k] if k < len(idx) else None
    k = idx.searchsorted(day, side="right") - 1
    return idx[k] if k >= 0 else None


def welch(a, b):
    a, b = a.dropna(), b.dropna()
    if len(a) < 5 or len(b) < 5:
        return np.nan
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return (a.mean() - b.mean()) / se if se > 0 else np.nan


def stat(x, base):
    x = x.dropna()
    if len(x) < 5:
        return dict(n=len(x), mean=np.nan, exc=np.nan, t=np.nan)
    return dict(n=len(x), mean=x.mean() * 1e4, exc=(x.mean() - base.mean()) * 1e4, t=welch(x, base))


def main():
    fomc, boj = fomc_dates(), boj_dates()
    nfp, cpi = [dt.date.fromisoformat(s) for s in BLS_NFP], [dt.date.fromisoformat(s) for s in BLS_CPI]
    nfp = [d for d in nfp if d <= END.date()]
    cpi = [d for d in cpi if d <= END.date()]
    print(f"日付数: FOMC {len(fomc)}, 日銀 {len(boj)}, 雇用統計 {len(nfp)}, CPI {len(cpi)}")
    print("FOMC 年別:", {y: sum(d.year == y for d in fomc) for y in range(2011, 2027)})
    print("日銀 年別:", {y: sum(d.year == y for d in boj) for y in range(2011, 2027)})
    # 検算: 雇用統計・CPIの日付が平日か、月に1回か、(雇用統計)金曜でない例外を表示
    for nm, L in [("雇用統計", nfp), ("CPI", cpi)]:
        assert all(d.weekday() < 5 for d in L) and len({(d.year, d.month) for d in L}) == len(L) - sum(
            1 for i in range(1, len(L)) if (L[i].year, L[i].month) == (L[i - 1].year, L[i - 1].month))
        print(nm, "金曜以外:", [str(d) for d in L if d.weekday() != 4][:12] if nm == "雇用統計" else "")
    px, mets, nbads = {}, {}, {}
    for s in SYMS:
        px[s] = load_prices(s)
        mets[s], nbads[s] = build_metrics(px[s])
        print(f"{s}: {px[s].index[0].date()}〜{px[s].index[-1].date()} 異常値除外 {nbads[s]}日, 直近終値 {px[s].c.iloc[-1]:.2f}")
    usd = px["USDJPY=X"]
    print("USDJPY 寄り==前日終値 の割合: %.0f%% (高いなら oc/gap は信頼できない)" % (100 * (usd.o.round(3) == usd.c.shift().round(3)).mean()))

    months = pd.period_range("2011-01", "2026-10", freq="M")
    results = []  # 全パターン

    def run(event, asset, dates, mode):
        m = mets[asset]
        idx = m.index
        days = sorted({snap(idx, d, mode) for d in dates} - {None})
        days = [d for d in days if START <= d <= END]
        if len(days) < 8:
            return
        evset = set(days)
        normal = m.loc[[i for i in idx if i not in evset]]
        ev = m.loc[days]
        yrs = (days[-1] - days[0]).days / 365.25
        for met in ["cc", "oc", "gap", "post3"]:
            b = normal[met]
            allS = stat(ev[met], b)
            h1 = stat(ev[met][ev.index <= "2018-12-31"], normal[met][normal.index <= "2018-12-31"])
            h2 = stat(ev[met][ev.index >= "2019-01-01"], normal[met][normal.index >= "2019-01-01"])
            s15 = stat(ev[met][ev.index >= "2015-01-01"], normal[met][normal.index >= "2015-01-01"])
            results.append(dict(event=event, asset=asset, met=met, per_yr=len(days) / yrs, all=allS, h1=h1, h2=h2, s15=s15,
                                ev=ev[met].dropna()))

    # --- 米イベント(日本資産は翌営業日)と日銀
    for name, dates in [("FOMC", fomc), ("雇用統計", nfp), ("米CPI", cpi)]:
        for a in SYMS:
            run(name, a, dates, "next" if a in JP else "same")
    for a in SYMS:
        run("日銀会合", a, boj, "same")
    # --- 日本SQ(第2金曜、休場なら前営業日)
    for a in ["1348.T", "^N225"]:
        major = [nth_friday(p.year, p.month, 2) for p in months if p.month in (3, 6, 9, 12)]
        minor = [nth_friday(p.year, p.month, 2) for p in months if p.month not in (3, 6, 9, 12)]
        run("日本SQ(メジャー)", a, major, "prev")
        run("日本SQ(月次)", a, minor, "prev")
    # --- 米opex(第3金曜)
    for a in ["SPY", "QQQ"]:
        quad = [nth_friday(p.year, p.month, 3) for p in months if p.month in (3, 6, 9, 12)]
        mon = [nth_friday(p.year, p.month, 3) for p in months if p.month not in (3, 6, 9, 12)]
        run("米opex(四半期)", a, quad, "prev")
        run("米opex(月次)", a, mon, "prev")
    # --- 月末・四半期末(各資産の最終取引日)
    for a in SYMS:
        idx = mets[a].index
        s = pd.Series(idx, index=idx)
        me = list(s.groupby(idx.to_period("M")).last())
        qe = list(s.groupby(idx.to_period("Q")).last())
        me = [d for d in me if d.to_period("M") != idx[-1].to_period("M")]  # 最終月は未完了
        qe = [d for d in qe if d.to_period("Q") != idx[-1].to_period("Q")]
        run("月末", a, me, "same")
        run("四半期末", a, qe, "same")

    # --- 1時間足: FOMC前24時間ドリフト(直近730日のみ)
    hourly = []
    for a in ["SPY", "QQQ"]:
        f = os.path.join(CACHE, f"ev_h_{a}.json")
        if not os.path.exists(f):
            u = f"https://query1.finance.yahoo.com/v8/finance/chart/{a}?interval=1h&range=730d"
            json.dump(json.load(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=40))["chart"]["result"][0], open(f, "w"))
        res = json.load(open(f))
        off = res["meta"].get("gmtoffset", -18000)
        q = res["indicators"]["quote"][0]
        # 夏時間があるので固定オフセットではなく America/New_York に変換する
        h = pd.DataFrame({"o": q["open"], "c": q["close"]},
                         index=pd.to_datetime(np.array(res["timestamp"]), unit="s", utc=True).tz_convert("America/New_York").tz_localize(None)).dropna()
        h["date"] = h.index.normalize()
        h["hm"] = h.index.hour * 60 + h.index.minute
        # 13:30(現地)開始のバーの寄り値 ≒ 13:30の価格(発表14:00の30分前)
        p1330 = h[h.hm == 13 * 60 + 30].set_index("date").o
        p1530 = h[h.hm == 15 * 60 + 30].set_index("date").o  # 15:30バーの寄り=15:30の価格
        pre = p1330 / p1330.shift() - 1  # 前営業日13:30→当日13:30
        ann = p1530 / p1330 - 1  # 13:30→15:30(発表を含む2時間)
        evd = [pd.Timestamp(d) for d in fomc if pd.Timestamp(d) in pre.index]
        for nm, ser in [("FOMC前24h(前日13:30→当日13:30)", pre), ("FOMC発表2h(13:30→15:30)", ann)]:
            ser = ser.dropna()
            x = ser.loc[[d for d in evd if d in ser.index]]
            b = ser.drop(x.index)
            hourly.append((a, nm, len(x), x.mean() * 1e4, (x.mean() - b.mean()) * 1e4, welch(x, b)))

    # ------------------------------------------------------------ 集計
    N = len(results) + len(hourly)
    from math import erf, sqrt
    from statistics import NormalDist
    z_bonf = NormalDist().inv_cdf(1 - 0.05 / 2 / N)
    print(f"\n試した全パターン数 N = {N} (イベント×資産×指標 {len(results)} + 1時間足 {len(hourly)})")
    print(f"Bonferroni(両側5%) 閾値 |t| > {z_bonf:.2f}。2.0超は偶然でも {N * 0.0455:.1f} 個出る見込み\n")

    rows = []
    for r in results:
        a, h1, h2, s15 = r["all"], r["h1"], r["h2"], r["s15"]
        sgn = np.sign(h1["exc"]) if h1["n"] >= 5 else np.nan
        # 前半の符号で取引した場合の1回あたり(後半・全期間)。粗リターン(超過ではなく生の平均)
        raw_h2 = h2["mean"] * sgn if h2["n"] >= 5 else np.nan
        rows.append(dict(event=r["event"], asset=r["asset"], met=r["met"], n=a["n"], per_yr=round(r["per_yr"], 1),
                         exc=a["exc"], t=a["t"], h1_exc=h1["exc"], h1_t=h1["t"], h2_exc=h2["exc"], h2_t=h2["t"],
                         s15_exc=s15["exc"], s15_t=s15["t"], sgn=sgn, raw_h2=raw_h2,
                         net2=raw_h2 - 4, net5=raw_h2 - 10, net10=raw_h2 - 20,
                         year_net5=(raw_h2 - 10) * r["per_yr"]))
    df = pd.DataFrame(rows)
    df["absT"] = df.t.abs()
    pd.set_option("display.width", 250, "display.max_rows", 400, "display.max_columns", 30)
    fmt = df.drop(columns=["absT"]).round(2)
    out = os.path.join(os.environ.get("EV_OUT", "/tmp"), "event_days_all.csv")
    try:
        fmt.to_csv(out, index=False)
    except Exception:
        pass
    print("=== |t|上位25 (全期間、通常日との差 exc[bp]) ===")
    print(df.sort_values("absT", ascending=False).head(25).drop(columns=["absT", "sgn", "net2", "net10"]).round(2).to_string(index=False))

    d = df.dropna(subset=["t"])
    print(f"\n全期間 |t|>2: {int((d.t.abs() > 2).sum())}/{len(d)}, |t|>Bonferroni({z_bonf:.2f}): {int((d.t.abs() > z_bonf).sum())}")
    cand = d[(d.h1_t.abs() > 2)]
    same = cand[(np.sign(cand.h2_exc) == cand.sgn)]
    conf = cand[(np.sign(cand.h2_exc) == cand.sgn) & (cand.h2_t.abs() > 2)]
    print(f"前半(〜2018) |t|>2: {len(cand)} 件 → 後半も同符号 {len(same)} 件 → 後半も|t|>2 {len(conf)} 件 (偶然なら後半|t|>2は約{len(cand) * 0.0455 / 2:.1f}件)")
    if len(conf):
        print(conf[["event", "asset", "met", "n", "h1_exc", "h1_t", "h2_exc", "h2_t", "s15_exc", "s15_t", "raw_h2", "net5", "per_yr"]].round(2).to_string(index=False))
    print("\n=== 前半で |t|>2 だった全件の後半の様子 ===")
    print(cand[["event", "asset", "met", "n", "h1_exc", "h1_t", "h2_exc", "h2_t", "s15_exc", "s15_t", "raw_h2", "net5", "year_net5"]].round(2).to_string(index=False))

    print("\n=== 1時間足(直近730日、SPY/QQQ、FOMC) mean[bp], 通常比exc[bp], t ===")
    for a, nm, n, mn, ex, t in hourly:
        print(f"{a:4s} {nm:34s} n={n:2d} mean={mn:6.1f} exc={ex:6.1f} t={t:5.2f}")

    # 主要イベントの要約: cc / oc の全体表
    print("\n=== イベント別・資産別の cc(前日終→当日終) と oc(寄り→引け) [exc bp (t)], 前半/後半 ===")
    for ev in df.event.unique():
        for a in SYMS:
            sub = df[(df.event == ev) & (df.asset == a)]
            if sub.empty:
                continue
            cc = sub[sub.met == "cc"].iloc[0]
            oc = sub[sub.met == "oc"].iloc[0]
            print(f"{ev:10s} {a:9s} n={int(cc.n):3d} ({cc.per_yr:4.1f}/年) cc {cc.exc:6.1f}({cc.t:5.2f}) 前{cc.h1_exc:6.1f}/後{cc.h2_exc:6.1f} 15+ {cc.s15_exc:6.1f} | oc {oc.exc:6.1f}({oc.t:5.2f}) 前{oc.h1_exc:6.1f}/後{oc.h2_exc:6.1f}")
    # FOMC当日の寄りまでの上昇(前日終値→当日寄り, 配当込み): 年別と勝率、コスト後
    print("\n=== FOMC前日終値→当日寄り(発表前の夜間) 詳細 ===")
    for a in ["SPY", "QQQ"]:
        m = mets[a]
        days = sorted({snap(m.index, d, "same") for d in fomc} - {None})
        g = m.loc[days, "gap"].dropna() * 1e4
        base = m.gap.drop(days).dropna() * 1e4
        print(f"{a}: n={len(g)} 平均{g.mean():.1f}bp 中央値{g.median():.1f} 勝率{(g > 0).mean() * 100:.0f}% 標準偏差{g.std():.0f} / 通常日の夜間平均{base.mean():.1f}bp, 勝率{(base > 0).mean() * 100:.0f}%")
        print("  年別平均bp:", {y: round(v, 1) for y, v in g.groupby(g.index.year).mean().items()})
        for c in COSTS_BP:
            print(f"  片道{c}bp: 1回あたり {g.mean() - 2 * c:.1f}bp, 年{len(g) / 15.8:.1f}回で 年{(g.mean() - 2 * c) * len(g) / 15.8:.0f}bp(資本比)")
        top = g.sort_values()
        print(f"  下位3 {top.head(3).round(0).tolist()} 上位3 {top.tail(3).round(0).tolist()}; 上位3回を除いた平均 {g.drop(top.tail(3).index).mean():.1f}bp")
    # 平均的な通常日
    print("\n通常日の平均(bp/日): " + ", ".join(f"{s} cc={mets[s].cc.mean() * 1e4:.1f}" for s in SYMS))


if __name__ == "__main__":
    main()
