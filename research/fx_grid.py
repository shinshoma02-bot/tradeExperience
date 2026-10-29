"""FX リピート系(トラリピ型・グリッド・ループイフダン型)の実態検証。 python research/fx_grid.py [quick]

- Yahoo 日足(period1=0)を research/_cache/fxgrid_*.json に取得して使う(無ければ自動取得)。BIS政策金利でスワップを概算(業者上乗せ年0.5%を差引)。
- 約定の近似(保守側): 日足の安値/高値で指値の約定を判定。ロスカット判定を先に行い(不利な順序)、当日新規に建てた玉は当日は決済しない。
  含み損益は日中の不利な極値(ロング=安値、ショート=高値)で評価してロスカット判定。両建ての必要証拠金は『買い・売りの大きい方』と仮定。
- グリッド: 現在値(前日終値)の下(買い)/上(売り)のN本の価格(=値幅の整数倍)に指値、約定したら +値幅 で利確(買い。売りは逆)。
- 事前固定したパラメータ: 値幅=開始価格の0.5%/1%/2%、本数=5/10/20、フル約定時のレバ=5倍(感度:3倍/10倍)、
  維持率ロスカット=100%/50%、元手=1万/10万/30万円、スプレッド=片道0.015%(往復0.03%)、レバ上限25倍(証拠金4%)。
- 税(申告分離20.315%)は最後に概算の注記だけ(損失繰越は未考慮)。投資助言ではなく情報整理。
"""
import json, os, sys, time, urllib.request
from multiprocessing import Pool
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "_cache"); os.makedirs(CACHE, exist_ok=True)
H = {"User-Agent": "Mozilla/5.0"}
SYMS = ["USDJPY=X", "AUDJPY=X", "EURJPY=X", "NZDJPY=X", "GBPJPY=X", "CADJPY=X", "CHFJPY=X", "AUDUSD=X", "AUDNZD=X", "EURGBP=X", "USDCAD=X", "USDCHF=X"]
PAIRS = ["USDJPY=X", "AUDJPY=X", "EURJPY=X", "NZDJPY=X", "AUDUSD=X", "AUDNZD=X", "EURGBP=X", "USDCAD=X", "USDCHF=X"]
Q2J = {"USD": "USDJPY=X", "NZD": "NZDJPY=X", "GBP": "GBPJPY=X", "CAD": "CADJPY=X", "CHF": "CHFJPY=X"}
AREA = {"USD": "US", "EUR": "XM", "GBP": "GB", "AUD": "AU", "NZD": "NZ", "CAD": "CA", "CHF": "CH", "JPY": "JP"}
HAIRCUT = 0.005; SPREAD_SIDE = 0.00015; MARGIN = 0.04


def load():
    p = os.path.join(CACHE, "fxgrid_daily.json")
    if not os.path.exists(p):
        out = {}
        for s in SYMS:
            u = f"https://query1.finance.yahoo.com/v8/finance/chart/{s}?interval=1d&period1=0&period2={int(time.time())}"
            r = json.load(urllib.request.urlopen(urllib.request.Request(u, headers=H), timeout=40))["chart"]["result"][0]
            q = r["indicators"]["quote"][0]
            out[s] = [(t, h, l, c) for t, h, l, c in zip(r["timestamp"], q["high"], q["low"], q["close"]) if None not in (h, l, c)]
        json.dump(out, open(p, "w"))
    raw = json.load(open(p)); dfs = {}
    for s, rows in raw.items():
        d = pd.DataFrame(rows, columns=["t", "h", "l", "c"]); d.index = pd.to_datetime(d.t, unit="s").dt.normalize()
        dfs[s] = d[~d.index.duplicated()].drop(columns="t")
    pb = os.path.join(CACHE, "fxgrid_bis.csv")
    if not os.path.exists(pb):
        u = "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_CBPOL/1.0/M.JP+US+XM+GB+AU+NZ+CA+CH?format=csv&startPeriod=2003-01"
        open(pb, "wb").write(urllib.request.urlopen(urllib.request.Request(u, headers=H), timeout=60).read())
    b = pd.read_csv(pb); b["d"] = pd.to_datetime(b.TIME_PERIOD + "-01") + pd.offsets.MonthEnd(0)   # 月末値→月末以降に使う(先読み回避)
    rt = b.pivot(index="d", columns="REF_AREA", values="OBS_VALUE").sort_index() / 100.0
    return dfs, rt


def build(sym, dfs, rt):
    """ペアごとの配列(高値, 安値, 終値, 対円換算, 基軸/決済通貨の金利)"""
    base, quote = sym[:3], sym[3:6]
    d = dfs[sym].copy()
    bad = (d.c > d.h * 1.002) | (d.c < d.l * 0.998)          # Yahooの終値グリッチ(例: EURGBP 2022-10-09 +11%)は高安の中値に置換
    d.loc[bad, "c"] = (d.h + d.l)[bad] / 2
    d["h"] = np.maximum(d.h, d.c); d["l"] = np.minimum(d.l, d.c)
    q = pd.Series(1.0, index=d.index) if quote == "JPY" else dfs[Q2J[quote]].c.reindex(d.index).ffill()
    ridx = rt.reindex(rt.index.union(d.index)).ffill().reindex(d.index)
    rb, rq = ridx[AREA[base]].bfill(), ridx[AREA[quote]].bfill()
    return pd.DataFrame({"h": d.h, "l": d.l, "c": d.c, "q": q, "rb": rb, "rq": rq}).dropna()


def anomalies(dfs):
    for s in PAIRS:
        r = dfs[s].c.pct_change(); b = r[r.abs() > 0.05]
        print(f"  {s}: 日次±5%超 {len(b)}件 " + ", ".join(f"{i.date()}({v*100:+.1f}%)" for i, v in b.iloc[:6].items()))


def sim(x, i0, i1, side, step_pct, nlines, cap, lev=5.0, lc=1.0):
    """x: build()のDataFrame、[i0,i1)日。side: 'L','S','B'。"""
    h = x.h.values; l = x.l.values; c = x.c.values; q = x.q.values; rb = x.rb.values; rq = x.rq.values; dt = x.index.values
    p0 = c[i0]; step = float(f"{step_pct * p0:.3g}")
    lot = max(1, int(cap * lev / (nlines * p0 * q[i0])))
    R = float(cap); LU = {}; SU = {}
    lc_n = 0; wins = 0; loss = 0; win_amt = 0.0; loss_amt = 0.0; swap_tot = 0.0; ruin = False; peak = float(cap); mdd = 0.0
    minEq = float(cap); eq_close = float(cap); blocked = 0
    useL = side in "LB"; useS = side in "SB"
    last = i1 - 1
    for i in range(i0 + 1, i1):
        last = i
        pc = c[i - 1]; H_, L_, C_, Q = h[i], l[i], c[i], q[i]
        newL = []; newS = []
        if useL:
            kmax = int(np.floor(pc / step - 1e-9)); lo = max(kmax - nlines + 1, int(np.ceil(L_ / step - 1e-9)))
            newL = [k for k in range(lo, kmax + 1) if k not in LU]
        if useS:
            kmin = int(np.ceil(pc / step + 1e-9)); hi = min(kmin + nlines - 1, int(np.floor(H_ / step + 1e-9)))
            newS = [k for k in range(kmin, hi + 1) if k not in SU]

        def req(nl, ns, px): return MARGIN * lot * max(nl, ns) * px * Q
        fl = set(); fs = set()
        for k in newL:
            if eq_close >= req(len(LU) + 1, len(SU), k * step): LU[k] = 1; fl.add(k); R -= lot * k * step * SPREAD_SIDE * Q
            else: blocked += 1
        for k in newS:
            if eq_close >= req(len(LU), len(SU) + 1, k * step): SU[k] = 1; fs.add(k); R -= lot * k * step * SPREAD_SIDE * Q
            else: blocked += 1

        def upnl(px):
            u = 0.0
            for k in LU: u += lot * (px - k * step)
            for k in SU: u += lot * (k * step - px)
            return u * Q
        nl, ns = len(LU), len(SU); did_lc = False
        if nl or ns:
            e_lo = R + upnl(L_); e_hi = R + upnl(H_)
            m_lo = req(nl, ns, L_); m_hi = req(nl, ns, H_)
            r_lo = e_lo / m_lo; r_hi = e_hi / m_hi
            worst = min(e_lo, e_hi); minEq = min(minEq, worst)
            if peak > 0: mdd = max(mdd, (peak - worst) / peak)
            if min(r_lo, r_hi) < lc:
                px, e = (L_, e_lo) if r_lo <= r_hi else (H_, e_hi)
                for k in LU:
                    pl = lot * (px - k * step) * Q
                    if pl >= 0: wins += 1; win_amt += pl
                    else: loss += 1; loss_amt += pl
                for k in SU:
                    pl = lot * (k * step - px) * Q
                    if pl >= 0: wins += 1; win_amt += pl
                    else: loss += 1; loss_amt += pl
                R = e - (nl + ns) * lot * px * SPREAD_SIDE * Q
                LU = {}; SU = {}; lc_n += 1; did_lc = True
                if R <= 0: ruin = True; R = 0.0; eq_close = 0.0; break
        if not did_lc:
            for k in [k for k in LU if k not in fl and (k + 1) * step <= H_ + 1e-12]:
                pl = lot * step * Q; R += pl - lot * (k + 1) * step * SPREAD_SIDE * Q; wins += 1; win_amt += pl; del LU[k]
            for k in [k for k in SU if k not in fs and (k - 1) * step >= L_ - 1e-12]:
                pl = lot * step * Q; R += pl - lot * (k - 1) * step * SPREAD_SIDE * Q; wins += 1; win_amt += pl; del SU[k]
        gap = max(1, int((dt[i] - dt[i - 1]) / np.timedelta64(1, "D")))
        sw = (lot * len(LU) * C_ * Q * (rb[i] - rq[i] - HAIRCUT) + lot * len(SU) * C_ * Q * (rq[i] - rb[i] - HAIRCUT)) * gap / 365.0
        R += sw; swap_tot += sw
        eq_close = R + upnl(C_); peak = max(peak, eq_close); minEq = min(minEq, eq_close)
        if eq_close > 0 and peak > 0: mdd = max(mdd, (peak - eq_close) / peak)
        if eq_close <= 0: ruin = True; eq_close = 0.0; break
    days = (dt[last] - dt[i0]) / np.timedelta64(1, "D")
    ann = (eq_close / cap) ** (365.0 / max(days, 1)) - 1 if eq_close > 0 else -1.0
    return dict(ret=eq_close / cap - 1, ann=ann, mdd=min(mdd, 1.0), lc=lc_n, ruin=int(ruin), half=int(minEq <= 0.5 * cap), lost80=int(minEq <= 0.2 * cap), wins=wins, losses=loss,
                win_amt=win_amt, loss_amt=loss_amt, swap=swap_tot / cap, lot=lot, open=len(LU) + len(SU), blocked=blocked)


def hold(x, i0, i1, side, cap):
    """同じ元手での1倍の持ちっぱなし(買い or 売り)。スワップ込み"""
    c = x.c.values; q = x.q.values; rb = x.rb.values; rq = x.rq.values; dt = x.index.values
    sg = 1 if side == "L" else -1
    u = cap / (c[i0] * q[i0]); R = float(cap); pk = cap; mdd = 0; e = cap
    for i in range(i0 + 1, i1):
        gap = max(1, int((dt[i] - dt[i - 1]) / np.timedelta64(1, "D")))
        R += u * c[i] * q[i] * sg * (rb[i] - rq[i] - HAIRCUT) * gap / 365.0
        e = R + sg * u * (c[i] - c[i0]) * q[i]; pk = max(pk, e); mdd = max(mdd, (pk - e) / pk)
    days = (dt[i1 - 1] - dt[i0]) / np.timedelta64(1, "D")
    return dict(ret=e / cap - 1, ann=(max(e, 1e-9) / cap) ** (365 / days) - 1, mdd=mdd)


_X = {}
def _init():
    global _X
    dfs, rt = load(); _X = {s: build(s, dfs, rt) for s in PAIRS}


def _job(a):
    key, s, i0, i1, side, st, n, cap, lev, lc = a
    r = sim(_X[s], i0, i1, side, st, n, cap, lev, lc); r.update(dict(key=key, pair=s, i0=i0, side=side, step=st, n=n, cap=cap, lev=lev, lcr=lc)); return r


def rolling_starts(x, years=3, every=126, first="2004-09-01"):
    idx = x.index; st = []
    i = int(idx.searchsorted(pd.Timestamp(first)))
    while i < len(idx):
        t1 = idx[i] + pd.DateOffset(years=years)
        if t1 > idx[-1]: break
        st.append((i, int(idx.searchsorted(t1)))); i += every
    return st


REGIMES = [  # 事前に宣言した局面(結果を見て選んでいない)
    ("USDJPY 円安トレンド2012-10〜2015-06", "USDJPY=X", "2012-10-01", "2015-06-30"),
    ("USDJPY 円安トレンド2021-01〜2024-07", "USDJPY=X", "2021-01-01", "2024-07-31"),
    ("USDJPY レンジ気味2016-01〜2019-12", "USDJPY=X", "2016-01-01", "2019-12-31"),
    ("AUDJPY リーマン2007-07〜2008-12", "AUDJPY=X", "2007-07-01", "2008-12-31"),
    ("AUDJPY 2015-01〜2016-12(ブレグジット含む)", "AUDJPY=X", "2015-01-01", "2016-12-31"),
    ("AUDJPY 2024-07〜2024-09(8月急落)", "AUDJPY=X", "2024-07-01", "2024-09-30"),
    ("EURJPY 2016-01〜2016-12(ブレグジット)", "EURJPY=X", "2016-01-01", "2016-12-31"),
    ("AUDNZD レンジ2016-01〜2019-12", "AUDNZD=X", "2016-01-01", "2019-12-31"),
    ("EURGBP 2016-06〜2019-12(ブレグジット後)", "EURGBP=X", "2016-06-01", "2019-12-31"),
    ("USDCHF スイスショック2014-07〜2015-12", "USDCHF=X", "2014-07-01", "2015-12-31"),
]
STEPS = [0.005, 0.01, 0.02]; NS = [5, 10, 20]; SIDES = ["L", "S", "B"]
CORE = ["USDJPY=X", "AUDJPY=X", "AUDNZD=X"]


def main():
    quick = len(sys.argv) > 1 and sys.argv[1] == "quick"
    dfs, rt = load(); X = {s: build(s, dfs, rt) for s in PAIRS}
    print("データ範囲/異常値チェック"); anomalies(dfs)
    for s in PAIRS: print(f"  {s} {X[s].index[0].date()}〜{X[s].index[-1].date()} {len(X[s])}日")
    jobs = []
    wins = {s: rolling_starts(X[s]) for s in PAIRS}
    if quick: wins = {s: v[::4] for s, v in wins.items()}
    for s in PAIRS:   # A: 全ペア×売買×値幅3×本数3, 元手10万, 維持率100%, レバ5倍
        for side in SIDES:
            for st in STEPS:
                for n in NS:
                    for (a, b) in wins[s]: jobs.append(("A", s, a, b, side, st, n, 100000, 5.0, 1.0))
    for s in CORE:    # B: 感度
        for side in SIDES:
            for st in STEPS:
                for n in NS:
                    for (a, b) in wins[s]:
                        for cap, lev, lc in [(10000, 5.0, 1.0), (300000, 5.0, 1.0), (100000, 5.0, 0.5), (100000, 3.0, 1.0), (100000, 10.0, 1.0), (10000, 2.0, 1.0), (100000, 2.0, 1.0), (100000, 1.0, 1.0)]:
                            jobs.append(("B", s, a, b, side, st, n, cap, lev, lc))
    for name, s, d0, d1 in REGIMES:   # C: 局面別
        a = int(X[s].index.searchsorted(pd.Timestamp(d0))); b = int(X[s].index.searchsorted(pd.Timestamp(d1)))
        for side in SIDES:
            for st in STEPS:
                for n in NS:
                    for lc in (1.0, 0.5): jobs.append(("C:" + name, s, a, b, side, st, n, 100000, 5.0, lc))
    print(f"試したシミュレーション総数: {len(jobs)} (A={sum(j[0]=='A' for j in jobs)}, B={sum(j[0]=='B' for j in jobs)}, C={sum(j[0].startswith('C') for j in jobs)})", flush=True)
    t = time.time()
    with Pool(4, initializer=_init) as pool: res = pool.map(_job, jobs, chunksize=100)
    df = pd.DataFrame(res); print(f"計算 {time.time()-t:.0f}秒")
    df.to_csv(os.environ.get("FXGRID_OUT", os.path.join(CACHE, "fxgrid_results.csv")), index=False)
    report(df, X)


def agg(g):
    tw = g.wins.sum(); tl = g.losses.sum()
    return pd.Series(dict(窓数=len(g), 年率中央=g.ann.median() * 100, 年率平均=g.ann.mean() * 100, 年率p10=g.ann.quantile(.1) * 100, 最大下落中央=g.mdd.median() * 100, 最大下落p90=g.mdd.quantile(.9) * 100,
                          ロスカット回数平均=g.lc.mean(), ロスカット発生窓=(g.lc > 0).mean() * 100, 破産確率=g.ruin.mean() * 100, 半減以下=g.half.mean() * 100, 八割喪失=g.lost80.mean() * 100,
                          勝率=tw / max(tw + tl, 1) * 100, 平均利益=g.win_amt.sum() / max(tw, 1), 平均損失=g.loss_amt.sum() / max(tl, 1), スワップ寄与=g.swap.mean() * 100))


def report(df, X):
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 300); pd.set_option("display.float_format", lambda v: f"{v:,.1f}")
    G = lambda d, k: d.groupby(k).apply(agg, include_groups=False)
    A = df[df.key == "A"]
    print("\n=== A. 3年ローリング(開始を半年ずつずらす)・元手10万・維持率100%・フル約定5倍。ペア×売買(値幅3×本数3の27設定込み) ===")
    print(G(A, ["pair", "side"])[["窓数", "年率中央", "年率p10", "最大下落中央", "最大下落p90", "ロスカット回数平均", "ロスカット発生窓", "破産確率", "勝率", "平均利益", "平均損失", "スワップ寄与"]])
    print("\n--- A: 売買別まとめ ---"); print(G(A, "side")[["年率中央", "年率平均", "年率p10", "最大下落中央", "ロスカット発生窓", "破産確率", "半減以下", "勝率"]])
    print("\n--- A: 値幅×本数(全ペア・全売買) ---"); print(G(A, ["step", "n"])[["年率中央", "年率平均", "最大下落中央", "ロスカット発生窓", "破産確率", "勝率"]])
    A = A.assign(has=(A.lc > 0))
    print("\n--- A: ロスカットあり/なし の窓: 件数/年率中央%/年率平均% ---"); t = A.groupby("has").ann.agg(["count", "median", "mean"]); t[["median", "mean"]] *= 100; print(t)
    tw, tl = A.win_amt.sum(), A.loss_amt.sum()
    print(f"\n--- A: 決済益合計(円) {tw:,.0f} / 強制決済等の損失合計 {tl:,.0f} / 比 {tw/max(-tl,1):.2f} ---")
    B = df[df.key == "B"]
    print("\n=== B. 感度(コア3ペア: USDJPY, AUDJPY, AUDNZD ×売買×27設定×3年窓) ===")
    allc = pd.concat([A[A.pair.isin(CORE)].drop(columns="has"), B])
    print(G(allc, ["cap", "lev", "lcr"])[["年率中央", "年率平均", "年率p10", "最大下落中央", "ロスカット回数平均", "ロスカット発生窓", "破産確率", "半減以下", "八割喪失", "勝率"]])
    cc = allc[(allc.lev == 5.0) & (allc.lcr == 1.0)]
    print("\n--- B: 元手別×ペア×売買(維持率100%, 5倍): 年率中央 / ロスカット回数平均 / 破産確率 ---")
    print(G(cc, ["pair", "side", "cap"])[["年率中央", "最大下落中央", "ロスカット回数平均", "破産確率"]].unstack("cap"))
    print("\n--- B: 維持率50% vs 100%(10万, 5倍) ---"); print(G(allc[(allc.cap == 100000) & (allc.lev == 5.0)], ["pair", "lcr"])[["年率中央", "最大下落中央", "ロスカット回数平均", "破産確率"]].unstack("lcr"))
    print("\n=== C. 局面別(元手10万・5倍・27設定×売買3、維持率100%/50%) ===")
    C = df[df.key.str.startswith("C:")]; rows = []
    for (k, side, lcr), g in C.groupby(["key", "side", "lcr"]):
        tw, tl = g.wins.sum(), g.losses.sum()
        rows.append(dict(局面=k[2:], 売買=side, 維持率=lcr, 年率中央=g.ann.median() * 100, 年率最悪=g.ann.min() * 100, 最大下落中央=g.mdd.median() * 100, ロスカット回数平均=g.lc.mean(), ロスカット発生=(g.lc > 0).mean() * 100, 破産=g.ruin.mean() * 100, 勝率=tw / max(tw + tl, 1) * 100))
    print(pd.DataFrame(rows).set_index(["局面", "売買", "維持率"]))
    print("\n=== 参考: 同局面の1倍持ちっぱなし(スワップ込み) / 現金=0% ===")
    for name, s, d0, d1 in REGIMES:
        a = int(X[s].index.searchsorted(pd.Timestamp(d0))); b = int(X[s].index.searchsorted(pd.Timestamp(d1)))
        hl, hs = hold(X[s], a, b, "L", 100000), hold(X[s], a, b, "S", 100000)
        print(f"  {name}: 買い持ち 年率{hl['ann']*100:+.1f}% DD{hl['mdd']*100:.0f}% / 売り持ち 年率{hs['ann']*100:+.1f}% DD{hs['mdd']*100:.0f}%")
    print("\n=== 参考: 3年ローリングの1倍買い持ち(年率中央/最大下落中央) ===")
    for s in PAIRS:
        r = [hold(X[s], a, b, "L", 100000) for a, b in rolling_starts(X[s])]
        print(f"  {s}: 年率中央 {np.median([v['ann'] for v in r])*100:+.1f}% DD中央 {np.median([v['mdd'] for v in r])*100:.0f}%")
    print("\n税(申告分離20.315%)は利益のみ課税=プラス収益の約2割減(損失繰越・3年は未考慮。要確認)")


if __name__ == "__main__":
    main()
