"""カレンダー効果の検証: 夜間 vs 日中、月末月初、月別、曜日別(SPY, QQQ, 1348.T)。
  python research/anomaly_calendar.py
データ: Yahoo Finance 日足(period1/period2 指定。range=max だと間引かれるので使わない)。
  始値・終値(分割調整のみ)と調整後終値(配当込み)。日次±15%超は異常として除外する。
  日次total return = adj終値の前日比。日中 = 終値/始値-1。夜間 = (1+total)/(1+日中)-1(配当落ちを含む)。
前半/後半 = 各銘柄の標本を日付で半分に割る。「2010年以降」も別に出す。
試したパターンの数は最後に集計して表示する(多重比較の目安)。
"""
import json, os, urllib.request
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data")
H = {"User-Agent": "Mozilla/5.0"}
SYMS = {"SPY": 728000000, "QQQ": 920000000, "1348.T": 1230000000}
COSTS = [0.0002, 0.0005, 0.001]  # 片道
TESTS = []  # (区分, 名前) 試した数の記録


def load(sym):
    os.makedirs(CACHE, exist_ok=True)
    f = os.path.join(CACHE, f"cal_{sym}.json")
    if os.path.exists(f):
        res = json.load(open(f))
    else:
        u = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval=1d"
             f"&period1={SYMS[sym]}&period2=1893000000")
        res = json.load(urllib.request.urlopen(urllib.request.Request(u, headers=H), timeout=40))["chart"]["result"][0]
        json.dump(res, open(f, "w"))
    q = res["indicators"]["quote"][0]
    off = res["meta"].get("gmtoffset", 0)
    d = pd.DataFrame({"o": q["open"], "c": q["close"], "a": res["indicators"]["adjclose"][0]["adjclose"]},
                     index=pd.to_datetime(np.array(res["timestamp"]) + off, unit="s").normalize())
    d = d.dropna()
    d = d[(d.o > 0) & (d.c > 0)]
    d = d[~d.index.duplicated()]
    d["tot"] = d.a.pct_change()
    d["intra"] = d.c / d.o - 1
    d["ovn"] = (1 + d.tot) / (1 + d.intra) - 1
    bad = (d.tot.abs() > .15) | (d.intra.abs() > .15) | (d.ovn.abs() > .15)
    nbad = int(bad.sum())
    d = d[~bad].dropna()
    return d, nbad


def tstat(x):
    x = pd.Series(x).dropna()
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else np.nan


def welch(a, b):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return (a.mean() - b.mean()) / se if se > 0 else np.nan


def ann(r):  # 日次リターン列の幾何年率(取引日252)
    r = pd.Series(r).dropna()
    return (1 + r).prod() ** (252 / len(r)) - 1 if len(r) else np.nan


def periods(d):
    mid = d.index[len(d) // 2]
    out = {"全期間": d, "前半": d[d.index < mid], "後半": d[d.index >= mid]}
    out["2010〜"] = d[d.index >= "2010-01-01"]
    out["〜2009"] = d[d.index < "2010-01-01"]
    return out, mid


def pct(x):
    return f"{x*100:6.1f}%"


# ---------------- 1. 夜間 vs 日中 ----------------
def overnight(name, d):
    print(f"\n[{name}] 夜間 vs 日中(毎日売買、片道コストを2回/日で引く)")
    ps, mid = periods(d)
    print(f"  分割日(前半/後半): {mid.date()}  期間 {d.index[0].date()}〜{d.index[-1].date()}  日数 {len(d)}")
    print("  期間      | 持ちっぱなし | 夜間のみ gross / 0.02% / 0.05% / 0.1% | 日中のみ gross / 0.02% / 0.05% / 0.1% | 夜間 t / 日中 t")
    for k, p in ps.items():
        if len(p) < 250:
            continue
        row = f"  {k:8s}| {pct(ann(p.tot))} |"
        for col in ("ovn", "intra"):
            for c in [0] + COSTS:
                row += f" {pct(ann(p[col] - 2 * c))}"
            row += " |"
        row += f" {tstat(p.ovn):5.2f} / {tstat(p.intra):5.2f}"
        print(row)
    TESTS.append(("夜間/日中", name + " 夜間・日中×コスト4"))


# ---------------- 2. ターン・オブ・ザ・マンス ----------------
def tom_flags(d, pre, post):
    """月末から数えて pre 営業日目〜翌月 post 営業日目を保有する日(その日のtotal returnを得る)のフラグ。
    保有日 = 月末pre日目の「翌日」〜翌月post日目 のリターン。つまり pre日目の終値で買い、post日目の終値で売る。"""
    ym = d.index.to_period("M")
    pos_from_start = d.groupby(ym).cumcount().values + 1
    n = d.groupby(ym)["tot"].transform("size").values
    pos_from_end = n - pos_from_start + 1  # 1 = 最終営業日
    # 月末側: 最終営業日から数えて pre 日目より後(= pos_from_end < pre)の日が保有日(pre日目の終値で買う)
    in_end = pos_from_end < pre
    in_start = pos_from_start <= post
    return in_end | in_start


def tom(name, d):
    grid = [(pre, post) for pre in (1, 2, 3, 5) for post in (1, 3, 5)]
    print(f"\n[{name}] ターン・オブ・ザ・マンス(買う: 月末から数えてpre日目の終値、売る: 翌月post日目の終値)")
    print("  標準(事前決定) = pre=2(最終営業日の1つ前の終値で買う), post=3(月初3日目の終値で売る)。他の組合せは参考・多重比較の対象")
    ps, mid = periods(d)
    h1 = d[d.index < mid]
    rows = []
    for pre, post in grid:
        fl = pd.Series(tom_flags(d, pre, post), index=d.index)
        rows.append((pre, post, fl))
        TESTS.append(("TOM", f"{name} pre{pre} post{post}"))
    # 前半の t(窓内日 vs 窓外日)で最良を選ぶ
    best = None
    for pre, post, fl in rows:
        t1 = welch(h1.tot[fl[h1.index]], h1.tot[~fl[h1.index]])
        if best is None or t1 > best[0]:
            best = (t1, pre, post)
    print(f"  前半だけで選ぶと pre={best[1]} post={best[2]} (前半 t={best[0]:.2f})")
    print("  pre post | 期間 | 窓内日平均(bp/日) 窓外日平均 | t(窓内-窓外) | 年率: 窓のみ保有 gross / 0.02%往復 / 0.1%片道 | 窓内日数/年 | 持ちっぱなし年率")
    for pre, post, fl in rows:
        if (pre, post) not in ((2, 3), (best[1], best[2])):
            continue
        for k in ("全期間", "前半", "後半", "2010〜"):
            p = ps[k]
            f = fl[p.index]
            inn, out = p.tot[f], p.tot[~f]
            # 1か月1回の往復 = 2片道。窓内の日数に按分せず、月数分を窓の最終日に引く
            months = p.index.to_period("M").nunique()
            res = []
            for c in (0, 0.0002, 0.001):
                r = p.tot.where(f, 0.0).copy()
                # コストは窓の開始日に一括(月1回、2片道)
                starts = f & ~f.shift(1, fill_value=False)
                r = r - starts.astype(float) * 2 * c
                res.append(ann(r))
            tag = "(標準)" if (pre, post) == (2, 3) else "(前半選択)"
            print(f"  {pre:3d} {post:3d} {tag} {k:5s}| {inn.mean()*1e4:5.1f} {out.mean()*1e4:5.1f} | {welch(inn,out):5.2f} | "
                  f"{pct(res[0])} {pct(res[1])} {pct(res[2])} | {f.sum()/len(p)*252:4.0f} | {pct(ann(p.tot))}")
    # 全グリッドの全期間・前半・後半の t 一覧
    print("  全12通りの t 値(窓内日-窓外日):   pre,post: 前半 / 後半")
    cnt = 0
    for pre, post, fl in rows:
        a = d[d.index < mid]; b = d[d.index >= mid]
        t1 = welch(a.tot[fl[a.index]], a.tot[~fl[a.index]]); t2 = welch(b.tot[fl[b.index]], b.tot[~fl[b.index]])
        cnt += (t1 > 0 and t2 > 0)
        print(f"    {pre},{post}: {t1:5.2f} / {t2:5.2f}", end=";")
    print(f"\n  前半・後半ともにプラスの組合せ {cnt}/12")


# ---------------- 3. 月別・曜日別 ----------------
def monthly(name, d):
    m = (1 + d.tot).groupby(d.index.to_period("M")).prod() - 1
    mm = pd.DataFrame({"r": m.values, "mo": m.index.month, "y": m.index.year}, index=m.index)
    mid = mm.index[len(mm) // 2]
    print(f"\n[{name}] 月別リターン(その月 vs 他の月の差と t / 前半t / 後半t / 2010〜t)。月数 {len(mm)}")
    hits = []
    for mo in range(1, 13):
        TESTS.append(("月別", f"{name} {mo}月"))
        a = mm[mm.mo == mo]
        o = mm[mm.mo != mo]
        t_all = welch(a.r, o.r)
        t1 = welch(a.r[a.index < mid], o.r[o.index < mid])
        t2 = welch(a.r[a.index >= mid], o.r[o.index >= mid])
        t3 = welch(a.r[a.y >= 2010], o.r[o.y >= 2010])
        d_all = a.r.mean() - o.r.mean()
        hits.append((mo, d_all, t_all, t1, t2, t3, a.r.mean()))
        print(f"  {mo:2d}月 平均 {a.r.mean()*100:5.2f}% 差 {d_all*100:5.2f}%  t={t_all:5.2f} 前半 {t1:5.2f} 後半 {t2:5.2f} 2010〜 {t3:5.2f}")
    return mm, mid


def sell_in_may(name, mm, mid):
    TESTS.append(("5月売り", name))
    win = mm.mo.isin([11, 12, 1, 2, 3, 4])
    print(f"[{name}] 5月売り(11〜4月だけ保有): 年率(月次、手数料なし/往復0.2%を年1回)")
    for k, sel in (("全期間", mm.index == mm.index), ("前半", mm.index < mid), ("後半", mm.index >= mid), ("2010〜", mm.y >= 2010)):
        p = mm[sel]
        w = win[sel]
        n = len(p)
        g = lambda r: (1 + r).prod() ** (12 / n) - 1
        hold = g(p.r)
        strat = g(p.r.where(w, 0.0))
        yrs = n / 12
        cost = (1 + p.r.where(w, 0.0)).prod() * (1 - 0.002) ** (2 * yrs) if False else None
        strat_c = g(p.r.where(w, 0.0) - (p.mo.isin([11, 5])).astype(float) * 0.001)
        print(f"  {k:5s} 持ちっぱなし {pct(hold)}  11-4月のみ {pct(strat)} (片道0.1%コスト後 {pct(strat_c)})  "
              f"t(11-4月 vs 5-10月)={welch(p.r[w], p.r[~w]):5.2f}")


def weekday(name, d):
    names = "月火水木金"
    ps, mid = periods(d)
    print(f"\n[{name}] 曜日別(その曜日 vs 他の曜日、日次total return)  平均bp/日・t(全/前半/後半/2010〜)")
    for wd in range(5):
        TESTS.append(("曜日", f"{name} {names[wd]}"))
        row = f"  {names[wd]}曜 "
        first = True
        for k in ("全期間", "前半", "後半", "2010〜"):
            p = ps[k]
            a, o = p.tot[p.index.weekday == wd], p.tot[p.index.weekday != wd]
            if first:
                row += f"平均 {a.mean()*1e4:5.1f}bp | "
                first = False
            row += f"{welch(a, o):5.2f} "
        print(row)


def japan_months(name, mm, mid):
    print(f"[{name}] 日本の権利確定月(3月・9月)と、その他の月の比較は上の月別に含まれる。3・9月合算 vs 他:")
    TESTS.append(("権利月", name + " 3,9月合算"))
    s = mm.mo.isin([3, 9])
    for k, sel in (("全期間", mm.index == mm.index), ("前半", mm.index < mid), ("後半", mm.index >= mid)):
        p = mm[sel]; q = s[sel]
        print(f"  {k}: 3・9月 平均 {p.r[q].mean()*100:5.2f}% / 他 {p.r[~q].mean()*100:5.2f}%  t={welch(p.r[q], p.r[~q]):5.2f}")


def main():
    for name in SYMS:
        d, nbad = load(name)
        print("=" * 100)
        print(f"{name}: {d.index[0].date()}〜{d.index[-1].date()} {len(d)}日  異常(±15%超)除外 {nbad}日  "
              f"持ちっぱなし年率 {pct(ann(d.tot))}  ※終値=始値の日 {((d.o==d.c).mean())*100:.1f}%")
        overnight(name, d)
        tom(name, d)
        mm, mid = monthly(name, d)
        sell_in_may(name, mm, mid)
        weekday(name, d)
        if name.endswith(".T"):
            japan_months(name, mm, mid)
    from collections import Counter
    c = Counter(t[0] for t in TESTS)
    print("\n" + "=" * 100)
    print("試したパターン数:", dict(c), "合計", len(TESTS))
    from statistics import NormalDist
    thr = NormalDist().inv_cdf(1 - 0.05 / len(TESTS) / 2)
    print("目安: 独立な検定をN回すると、偶然で|t|>2になるのは約5%%(N=%d なら約%.1f個)。Bonferroni(両側5%%)の閾値は |t|≈%.2f"
          % (len(TESTS), len(TESTS) * .05, thr))


if __name__ == "__main__":
    main()
