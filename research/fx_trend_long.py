"""FXトレンドの長期史(2004〜)・劣化・ボラ目標・ストップ・少額(ロスカット)シミュレーション。
  python research/fx_trend_long.py        # 初回は Yahoo から20ペアを取得して research/data/fx20_full.json に保存
投資助言ではなく検証。スワップは含めない(トレンドは買い売り両方向で概ね相殺する前提。未検証)。
ルール: シグナルは当日終値、売買は翌営業日の終値(= 位置を2日シフト)。パラメータは定番値に固定。
試した設定の総数は最後に表示する。
"""
import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data") + os.sep
FILE = D + "fx20_full.json"
PAIRS = ["USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY",
         "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF",
         "EURGBP", "EURCHF", "EURAUD", "EURCAD", "GBPCHF", "AUDCAD", "AUDNZD"]
START = pd.Timestamp("2004-01-01")
PERIODS = [("2004-2007", "2004-01-01", "2007-12-31"), ("2008-2012", "2008-01-01", "2012-12-31"),
           ("2013-2017", "2013-01-01", "2017-12-31"), ("2018-2026", "2018-01-01", "2026-12-31")]
DPY = 261
N_TRIALS = 0


def fetch():
    if os.path.exists(FILE):
        return json.load(open(FILE))
    os.makedirs(D, exist_ok=True)
    out = {}
    for p in PAIRS:
        u = f"https://query1.finance.yahoo.com/v8/finance/chart/{p}=X?interval=1d&period1=0&period2={int(time.time())}"
        r = json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=40))["chart"]["result"][0]
        q = r["indicators"]["quote"][0]
        out[p] = [(t, h, l, c) for t, h, l, c in zip(r["timestamp"], q["high"], q["low"], q["close"]) if None not in (h, l, c)]
    json.dump(out, open(FILE, "w"))
    return out


def load():
    raw = fetch(); dfs = {}; log = []
    for p, rows in raw.items():
        d = pd.DataFrame(rows, columns=["t", "h", "l", "c"]); d.index = pd.to_datetime(d.t, unit="s").dt.normalize()
        d = d[~d.index.duplicated(keep="last")].drop(columns="t")
        d = d[d.index.dayofweek < 5]
        # 異常値: 日次±5%超で、翌日にほぼ戻る(スパイク)ものだけ除外。戻らないもの(2015 SNB等の実イベント)は残す
        for _ in range(3):
            r = d.c.pct_change(); bad = []
            for i in np.where(r.abs().values > 0.05)[0]:
                if i + 1 < len(d) and np.sign(r.iloc[i + 1]) != np.sign(r.iloc[i]) and abs(r.iloc[i] + r.iloc[i + 1]) < abs(r.iloc[i]) * 0.5:
                    bad.append(i)
            if not bad: break
            for i in bad: log.append((p, str(d.index[i].date()), round(r.iloc[i] * 100, 1), "除外"))
            d = d.drop(d.index[bad])
        r = d.c.pct_change()
        for i in np.where(r.abs().values > 0.05)[0]:
            log.append((p, str(d.index[i].date()), round(r.iloc[i] * 100, 1), "実イベントとして残す"))
        cp = d.c.shift(1).fillna(d.c)
        h = np.where(d.h / np.maximum(d.c, cp) - 1 > 0.05, np.maximum(d.c, cp), d.h)
        l = np.where(d.l / np.minimum(d.c, cp) - 1 < -0.05, np.minimum(d.c, cp), d.l)
        d["h"] = np.maximum(h, np.maximum(d.c, cp)); d["l"] = np.minimum(l, np.minimum(d.c, cp))
        dfs[p] = d
    return dfs, log


# ---------- シグナル(-1/0/+1。当日終値で決まる) ----------
def sig_sma(d, f=50, s=200):
    a, b = d.c.rolling(f).mean(), d.c.rolling(s).mean()
    return pd.Series(np.where(b.notna(), np.where(a > b, 1.0, -1.0), 0.0), index=d.index)


def sig_donch(d, en=55, ex=20):
    c = d.c.values; hi = d.c.rolling(en).max().shift(1).values; lo = d.c.rolling(en).min().shift(1).values
    xh = d.c.rolling(ex).max().shift(1).values; xl = d.c.rolling(ex).min().shift(1).values
    pos = np.zeros(len(c)); p = 0
    for i in range(len(c)):
        if p == 0:
            if c[i] > hi[i]: p = 1
            elif c[i] < lo[i]: p = -1
        elif p == 1 and c[i] < xl[i]: p = 0
        elif p == -1 and c[i] > xh[i]: p = 0
        pos[i] = p
    return pd.Series(pos, index=d.index)


def sig_mom(d, n=252):
    m = np.sign(d.c / d.c.shift(n) - 1)
    ym = d.index.to_series().dt.to_period("M")
    last = d.index.to_series().groupby(ym).transform("max")
    return m.where(d.index == last).ffill().fillna(0.0)


def sig_stop(d, base, mult=3.0, trailing=True, n=14):
    """baseシグナルにATR(14)×3のストップを重ねる。ストップ後は、baseが変わるまで再エントリーしない。"""
    pc = d.c.shift(1)
    atr = pd.concat([d.h - d.l, (d.h - pc).abs(), (d.l - pc).abs()], axis=1).max(axis=1).rolling(n).mean().values
    c = d.c.values; b = base.values; out = np.zeros(len(c)); pos = 0; ext = 0.0; ref = 0.0; stopped = 0
    for i in range(len(c)):
        if stopped != 0 and b[i] != stopped: stopped = 0
        if pos == 0:
            if stopped == 0 and b[i] != 0 and not np.isnan(atr[i]):
                pos = b[i]; ext = c[i]; ref = c[i]
        else:
            if b[i] != pos:
                pos = b[i] if (b[i] != 0 and not np.isnan(atr[i])) else 0; ext = c[i]; ref = c[i]
            else:
                if trailing:
                    ext = max(ext, c[i]) if pos > 0 else min(ext, c[i]); lvl = ext - pos * mult * atr[i]
                else:
                    lvl = ref - pos * mult * atr[i]
                if (pos > 0 and c[i] < lvl) or (pos < 0 and c[i] > lvl): stopped = pos; pos = 0
        out[i] = pos
    return pd.Series(out, index=d.index)


def cost_rate(p, mult=1.0):
    return (0.000075 if "JPY" in p else 0.0002) * mult   # 片道: 円クロス0.0075% / その他0.02%


def build(dfs):
    S = {k: {} for k in ["sma", "donch", "mom", "long"]}
    for p, d in dfs.items():
        S["sma"][p] = sig_sma(d); S["donch"][p] = sig_donch(d); S["mom"][p] = sig_mom(d); S["long"][p] = pd.Series(1.0, index=d.index)
    return S


def eff(sigs):
    return pd.DataFrame(sigs).shift(2)   # シグナルは当日終値、売買は翌営業日終値 → 実効ポジションは2日遅れ


def returns_from(pos, rets, costmult=1.0):
    pos = pos.reindex(rets.index)
    avail = rets.notna() & pos.notna()
    P = pos.where(avail, 0.0); R = rets.fillna(0.0)
    cr = pd.Series({p: cost_rate(p, costmult) for p in rets.columns})
    return P * R - P.diff().abs().fillna(0) * cr, avail


def stats(r, start=None, end=None):
    r = r.dropna()
    if start: r = r[r.index >= start]
    if end: r = r[r.index <= end]
    if len(r) < 50: return dict(cagr=np.nan, dd=np.nan, sh=np.nan, vol=np.nan)
    eq = (1 + r).cumprod(); yrs = len(r) / DPY
    return dict(cagr=(eq.iloc[-1] ** (1 / yrs) - 1) * 100, dd=(eq / eq.cummax() - 1).min() * 100,
                sh=r.mean() / r.std() * np.sqrt(DPY) if r.std() > 0 else 0, vol=r.std() * np.sqrt(DPY) * 100)


def ew(net, avail):
    return net.sum(axis=1) / avail.sum(axis=1).replace(0, np.nan)


def vol_target(S, rets, target=0.10, cap=8.0):
    """S: 実効シグナル(-1..1)。ペアは逆ボラ配分 → ポートフォリオ全体を目標ボラに調整(過去60日の実績ボラ、1日遅れ)。"""
    S = S.reindex(rets.index)
    avail = S.notna() & rets.notna(); S0 = S.where(avail, 0.0)
    sig = (rets.ewm(span=60, min_periods=30).std().shift(1) * np.sqrt(DPY))
    n = avail.sum(axis=1).replace(0, np.nan)
    w0 = (S0 / sig).replace([np.inf, -np.inf], 0).fillna(0).div(n, axis=0)
    cr = pd.Series({p: cost_rate(p) for p in rets.columns}); R = rets.fillna(0)
    raw = (w0 * R).sum(axis=1)
    pv = raw.ewm(span=60, min_periods=30).std().shift(1) * np.sqrt(DPY)
    scale = (target / pv).clip(upper=cap).fillna(0)
    w = w0.mul(scale, axis=0)
    net = (w * R).sum(axis=1) - (w.diff().abs() * cr).sum(axis=1)
    return net, w


def line(label, st, extra=""):
    return f"{label:32}{st['cagr']:7.2f}%{st['dd']:8.1f}%{st['sh']:7.2f}{extra}"


def main():
    global N_TRIALS
    dfs, log = load()
    print("## データ: 20ペア日足(Yahoo, period1=0)。1999年以降に始まるペアの開始日:")
    print("  ", {p: str(d.index[0].date()) for p, d in dfs.items() if d.index[0] > pd.Timestamp("1999-01-01")})
    print("## 日次±5%超の記録(スパイクは除外、戻らない実イベントは残す):")
    for x in log: print("  ", x)
    rets = pd.DataFrame({p: d.c.pct_change() for p, d in dfs.items()})
    rets = rets[rets.index >= START]
    S = build(dfs)
    E = {k: eff(v).reindex(rets.index) for k, v in S.items()}
    E["ens"] = (E["sma"] + E["donch"] + E["mom"]) / 3
    names = {"sma": "SMA 50/200(売買両方向)", "donch": "ドンチャン 55/20", "mom": "12か月モメンタム(月次)",
             "ens": "3つの平均(アンサンブル)", "long": "参考: 常にロング(基準通貨買い)"}
    N_TRIALS += 4

    print("\n## 1. 期間別(20ペア等分、1倍、手数料後)   年率 / 最大下落 / シャープ / プラスのペア数 / 手数料前シャープ")
    for k in ["sma", "donch", "mom", "ens", "long"]:
        net, av = returns_from(E[k], rets); gro, _ = returns_from(E[k], rets, 0.0)
        pn, pg = ew(net, av), ew(gro, av)
        print(f"\n[{names[k]}]")
        for nm, a, b in PERIODS + [("全期間", "2004-01-01", "2026-12-31"), ("前半2004-2012", "2004-01-01", "2012-12-31"), ("後半2013-2026", "2013-01-01", "2026-12-31")]:
            st = stats(pn, a, b); sg = stats(pg, a, b)
            sub = net[(net.index >= a) & (net.index <= b)]
            npos = int((sub.sum() > 0).sum())
            print(line(f"  {nm}", st, f"   {npos:2d}/20   gross SR {sg['sh']:5.2f}"))

    print("\n## 2. ボラ目標10%(逆ボラ配分→ポートフォリオ全体を10%に調整、レバ上限8倍、20ペア、手数料後)")
    for k in ["sma", "donch", "mom", "ens"]:
        net, w = vol_target(E[k], rets)
        g = w.abs().sum(axis=1)
        print(f"[{names[k]}] 平均総建玉(対元手) {g[g > 0].mean():.2f}倍")
        for nm, a, b in PERIODS + [("全期間", "2004-01-01", "2026-12-31")]:
            print(line(f"  {nm}", stats(net, a, b)))
    N_TRIALS += 4
    print("\n  ペア数の効果(アンサンブル・ランダム30組の平均、ボラ目標10%。年率 / 最大下落 / シャープ)")
    rng = np.random.default_rng(0)
    for n in [1, 3, 5, 10, 20]:
        res = []
        for _ in range(30 if n < 20 else 1):
            cols = list(rng.choice(PAIRS, n, replace=False)) if n < 20 else PAIRS
            net, _ = vol_target(E["ens"][cols], rets[cols])
            res.append((stats(net, "2004-01-01"), stats(net, "2013-01-01")))
        print(f"   {n:2d}ペア  {np.mean([a['cagr'] for a, _ in res]):6.2f}% {np.mean([a['dd'] for a, _ in res]):7.1f}%  SR {np.mean([a['sh'] for a, _ in res]):5.2f}  後半2013-26 SR {np.mean([b['sh'] for _, b in res]):5.2f}")
    N_TRIALS += 5
    net_all, av = returns_from(E["ens"], rets)
    first = net_all[net_all.index <= "2012-12-31"]
    C = first.corr().abs(); chosen = [C.mean().idxmin()]
    while len(chosen) < 6:
        rest = [p for p in PAIRS if p not in chosen]; chosen.append(min(rest, key=lambda p: C.loc[p, chosen].mean()))
    nl, _ = vol_target(E["ens"][chosen], rets[chosen])
    rs = [stats(vol_target(E["ens"][list(c)], rets[list(c)])[0], "2013-01-01") for c in (rng.choice(PAIRS, 6, replace=False) for _ in range(30))]
    print(f"\n  前半(〜2012)の相関で選んだ低相関6ペア {chosen}\n   後半2013-26: {line('低相関6ペア', stats(nl, '2013-01-01'))}\n   同ランダム6ペア平均: 年率 {np.mean([x['cagr'] for x in rs]):.2f}% SR {np.mean([x['sh'] for x in rs]):.2f}")
    tu = net_all[net_all.index >= "2013-01-01"].corr().values[np.triu_indices(20, 1)]
    print(f"   ペア別戦略リターンの平均相関(後半): {np.nanmean(tu):.3f}")
    N_TRIALS += 2

    print("\n## 3. ストップ(ATR14×3の定番値のみ。20ペア等分・1倍・手数料後)   年率 / 最大下落 / シャープ  [前半2004-12 / 後半2013-26のSR]")
    for bk in ["sma", "donch"]:
        base = S[bk]
        for lab, fn in [("ストップ無し", None), ("固定ATRストップ(建値-3ATR)", False), ("トレーリング3ATR", True)]:
            sigs = base if fn is None else {p: sig_stop(dfs[p], base[p], 3.0, fn) for p in dfs}
            net, av = returns_from(eff(sigs).reindex(rets.index), rets); pn = ew(net, av)
            print(line(f"  {bk} {lab}", stats(pn, "2004-01-01"), f"   前半SR {stats(pn,'2004-01-01','2012-12-31')['sh']:5.2f} 後半SR {stats(pn,'2013-01-01')['sh']:5.2f}"))
            N_TRIALS += 0 if fn is None else 1

    small_account(dfs, rets, E)
    print(f"\n## 試した設定の総数(概算): {N_TRIALS + 2*5*2 + 1}  (全て定番値で固定、チューニング無し。ペア・戦略の組合せ比較なので多重比較に注意)")


# ---------- 4,5. 少額・ロスカット ----------
def sim(W, rets, hi, lo, L, thr, capital=1.0):
    """W: 目標ウェイト(1倍時の対元手比、符号付き)。毎月初と符号変化時に『元手×L×W』へ建て直す(間は数量固定)。
    日中の最悪値(買いは安値、売りは高値、全ペア同時に起きると仮定=保守的)で
    維持率 = (純資産+最悪損益)/(必要証拠金=建玉合計/25) が thr を割ったらロスカット(全決済して残りで再開)。"""
    idx = rets.index; T, Np = rets.shape
    R = rets.fillna(0).values; Wv = W.reindex(idx).fillna(0).values
    adv_long = lo.reindex(idx).fillna(0).values; adv_short = hi.reindex(idx).fillna(0).values
    cr = np.array([cost_rate(p) for p in rets.columns])
    E = capital; N = np.zeros(Np); eqs = np.zeros(T); lc = 0; near = 0; minratio = np.inf; hold = False
    months = idx.to_period("M"); prevW = np.zeros(Np); bust = False
    for t in range(T):
        if bust: eqs[t] = E; continue
        newmonth = t == 0 or months[t] != months[t - 1]
        changed = bool(np.any(np.sign(Wv[t]) != np.sign(prevW)))
        if changed: hold = False
        if (newmonth or changed) and not hold:
            tgt = E * L * Wv[t]; E -= np.sum(np.abs(tgt - N) * cr); N = tgt
        prevW = Wv[t]
        worst = np.sum(np.where(N > 0, N * adv_long[t], N * adv_short[t]))
        req = np.sum(np.abs(N)) / 25
        if req > 0:
            ratio = (E + worst) / req
            minratio = min(minratio, ratio)
            if ratio < 1.5 * thr: near += 1
            if ratio < thr:
                E = max(E + worst, 0.0); N = np.zeros(Np); lc += 1; hold = True
                if E <= 0: bust = True
                eqs[t] = E; continue
        E += np.sum(N * R[t]); N = N * (1 + R[t])
        if E <= 0: E = 0.0; bust = True
        eqs[t] = E
    return pd.Series(eqs, index=idx), lc, near, minratio, bust


def small_account(dfs, rets, E):
    print("\n## 4. 少額の現実(レバ上限25倍・1通貨単位・ロスカット)")
    d = dfs["USDJPY"]; px = d.c.iloc[-1]; r = d.c.pct_change()
    vol_long = r[r.index >= START].std() * np.sqrt(DPY); vol_now = r.iloc[-60:].std() * np.sqrt(DPY)
    print(f"USDJPY 直近 {px:.2f}円({d.index[-1].date()}) 年率ボラ: 長期{vol_long*100:.1f}% / 直近60日{vol_now*100:.1f}%  → ボラ目標10%の建玉 = 元手×0.10/ボラ")
    for cap in [10000, 30000, 100000]:
        notional = cap * 0.10 / vol_now; units = int(notional / px); notional = units * px; margin = notional / 25
        ratio = cap / margin * 100 if margin else np.nan
        mv = lambda thr: (cap - thr / 100 * margin) / units if units else np.nan
        print(f" 元手{cap:>7,}円: {units:>5,}通貨(約{notional:,.0f}円=元手の{notional/cap:.2f}倍) 必要証拠金{margin:,.0f}円 維持率{ratio:.0f}%  1円動くと{units:,}円損益  "
              f"維持率100%割れまで{mv(100):.1f}円({mv(100)/px*100:.1f}%逆行)  50%割れまで{mv(50):.1f}円({mv(50)/px*100:.1f}%)")
    print(f"  (1通貨単位の業者を想定。多くの業者は1,000通貨単位で、1万円・ボラ目標10%の建玉は約{int(10000*0.10/vol_now/px)}通貨=最小単位に届かない)")
    r20 = d.c.pct_change(20)
    print(f"  USDJPY 2004以降の最大日次変動 {r[r.index>=START].min()*100:.1f}% / +{r[r.index>=START].max()*100:.1f}%、20営業日の最悪 {r20[r20.index>=START].min()*100:.1f}% / 最良 +{r20[r20.index>=START].max()*100:.1f}%")

    hi = pd.DataFrame({p: dfs[p].h / dfs[p].c.shift(1) - 1 for p in PAIRS}).reindex(rets.index)
    lo = pd.DataFrame({p: dfs[p].l / dfs[p].c.shift(1) - 1 for p in PAIRS}).reindex(rets.index)
    W20 = E["ens"] / 20; W1 = pd.DataFrame(0.0, index=rets.index, columns=PAIRS); W1["USDJPY"] = E["ens"]["USDJPY"].fillna(0)
    res = {}
    for label, W in [("20ペア分散(3戦略平均)", W20), ("USDJPY単独(3戦略平均)", W1)]:
        gross = W.abs().sum(axis=1).mean()
        print(f"\n[{label}]  1倍=全シグナルON時の建玉が元手と同額。平均総建玉 {gross:.2f}倍(1倍設定時)。2004〜2026")
        print(f"{'設定':>6}{'実効平均レバ':>12}{'年率':>8}{'最大下落':>9}{'年ボラ':>8}{'LC(100%)':>9}{'近接日':>7}{'LC(50%)':>8}{'最小維持率':>10}{'全損':>5}")
        for L in [1, 3, 5, 10, 25]:
            e1, lc1, nr1, mr1, b1 = sim(W, rets, hi, lo, L, 1.0)
            e5, lc5, nr5, mr5, b5 = sim(W, rets, hi, lo, L, 0.5)
            s = stats(e1.pct_change().fillna(0), "2004-01-01")
            s5 = stats(e5.pct_change().fillna(0), "2004-01-01")
            print(f"{L:>5}倍{L*gross:12.2f}{s['cagr']:8.2f}%{s['dd']:8.1f}%{s['vol']:7.1f}%{lc1:9d}{nr1:7d}{lc5:8d}{mr1*100:9.0f}%{'  有' if b1 else '  無':>5}   (50%版: 年率{s5['cagr']:.2f}% 最大下落{s5['dd']:.1f}%{' 全損' if b5 else ''})")
            res[(label, L)] = e1
    print("\n## 5. 期待値(円)  カレンダー年ごとの損益。ロスカット(維持率100%)込み。税は利益のみ20.315%と仮定(損失繰越は無視・要確認)")
    for label in ["20ペア分散(3戦略平均)", "USDJPY単独(3戦略平均)"]:
        for L in [1, 3, 5]:
            e = res[(label, L)]; ye = e.groupby(e.index.year).last(); prev = ye.shift(1); prev.iloc[0] = 1.0
            yr = (ye / prev - 1)
            print(f"\n[{label} / 設定{L}倍]  年次リターン 平均 {yr.mean()*100:.1f}%  中央値 {yr.median()*100:.1f}%  最悪 {yr.min()*100:.1f}%({yr.idxmin()})  マイナスの年 {int((yr<0).sum())}/{len(yr)}  下位10% {yr.quantile(.1)*100:.1f}%")
            for cap in [10000, 100000]:
                pnl = yr * cap; after = pnl.where(pnl < 0, pnl * (1 - 0.20315))
                print(f"   元手{cap:>7,}円: 平均 {pnl.mean():+8,.0f}円/年(税引後 {after.mean():+,.0f})  中央値 {pnl.median():+,.0f}  最悪年 {pnl.min():+,.0f}円  下位10% {pnl.quantile(.1):+,.0f}円  最高年 {pnl.max():+,.0f}円")


if __name__ == "__main__":
    main()
