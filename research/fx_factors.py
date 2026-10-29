"""FXファクター合成(キャリー+バリュー代用+モメンタム)、リスクレジーム(VIX)フィルタ、レンジ型クロスの検証。
  python research/fx_factors.py
データはYahoo日足とBIS政策金利(research/data/fxf_* にキャッシュ。gitignore済み)。投資助言ではなく情報整理。

ルックアヘッド対策:
 - Yahoo FX日足の日付tの終値は実質0:00UTC。FXだけのシグナル(日付tまでの情報)は t+1 の収益から(shift 1)。
 - VIX(t日の米国終値)は t+2 の収益から(shift 2。FXのt+1終値より後にしか確定しないので1日余分にラグ)。
 - BIS政策金利は月次。その月の値は翌月から使う(1か月ラグ)。
事前に固定した定番値: キャリー=政策金利差、モメンタム=12か月(252営業日)と3か月(63)、バリュー=PPPの代用
 (為替の対数価格の直近最大5年平均(最低2年)からの乖離の符号反転。実質実効為替レートではない)。
 VIXフィルタ=「過去250日の80%点超」or「30超」、為替実現ボラフィルタ=主要通貨バスケットの20日実現ボラが250日80%点超。
 ボリンジャー=20日・2σ(中心で決済)、RSI=14日 30/70(50で決済)。月次リバランス。
試したバリアント数は最後に出力する(多重比較の注意)。
"""
import io
import json
import os
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")
os.makedirs(D, exist_ok=True)
H = {"User-Agent": "Mozilla/5.0"}
SPLIT = pd.Timestamp("2013-01-01")
START = pd.Timestamp("2006-01-01")     # 全戦略共通の開始(バリューの履歴が溜まってから)
HAIRCUT = 0.005                         # スワップの業者上乗せ(年率、建玉の総額に対して)
COST_JPY, COST_OTH = 0.0001, 0.0002     # 片道コスト。円クロスは中間値0.01%、他は0.02%(感度は別途)
CCY = ["USD", "EUR", "GBP", "AUD", "NZD", "CAD", "CHF", "JPY"]
AREA = {"USD": "US", "EUR": "XM", "GBP": "GB", "AUD": "AU", "NZD": "NZ", "CAD": "CA", "CHF": "CH", "JPY": "JP"}
NVAR = 0
VARIANTS = []


def yahoo(sym):
    f = os.path.join(D, "fxf_" + sym.replace("^", "_").replace("=", "_") + ".json")
    if not os.path.exists(f):
        u = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym.replace('^', '%5E')}?interval=1d&period1=0&period2=1893456000"
        res = json.load(urllib.request.urlopen(urllib.request.Request(u, headers=H), timeout=30))["chart"]["result"][0]
        json.dump([res["timestamp"], res["indicators"]["quote"][0]["close"]], open(f, "w"))
    ts, c = json.load(open(f))
    s = pd.Series(c, index=pd.to_datetime(ts, unit="s").normalize(), dtype=float).dropna()
    return s[~s.index.duplicated()]


def bis():
    f = os.path.join(D, "fxf_bis.csv")
    if not os.path.exists(f):
        u = ("https://stats.bis.org/api/v2/data/dataflow/BIS/WS_CBPOL/1.0/M.JP+US+XM+GB+AU+NZ+CA+CH?format=csv&startPeriod=2002-01")
        open(f, "wb").write(urllib.request.urlopen(urllib.request.Request(u, headers=H), timeout=60).read())
    r = pd.read_csv(f)
    r["d"] = pd.to_datetime(r.TIME_PERIOD + "-01")
    return r.pivot(index="d", columns="REF_AREA", values="OBS_VALUE").sort_index() / 100.0


PAIRS = ["USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY", "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF",
         "AUDNZD", "EURGBP", "EURCHF", "AUDCAD", "NZDCAD", "GBPCHF", "EURAUD"]
FX = {}
for p in PAIRS:
    try:
        FX[p] = yahoo(p + "=X")
    except Exception as e:
        print("取得失敗", p, e)
VIX = yahoo("^VIX")
idx = FX["USDJPY"].index
idx = idx[idx >= "2003-06-01"]
PX = pd.DataFrame({k: v.reindex(idx).ffill() for k, v in FX.items()})
RM = bis()
RATE = (RM.shift(1).reindex(RM.index.union(idx)).ffill().reindex(idx))      # 1か月ラグ、日次の年率
GAP = pd.Series(idx, index=idx).diff().dt.days.fillna(1).values


def usd_prices():   # 各通貨1単位のUSD価格
    return pd.DataFrame({"USD": 1.0, "EUR": PX.EURUSD, "GBP": PX.GBPUSD, "AUD": PX.AUDUSD, "NZD": PX.NZDUSD,
                         "CAD": 1 / PX.USDCAD, "CHF": 1 / PX.USDCHF, "JPY": 1 / PX.USDJPY}, index=idx)


def jpy_prices():   # 各通貨1単位の円価格
    return pd.DataFrame({"USD": PX.USDJPY, "EUR": PX.EURJPY, "GBP": PX.GBPJPY, "AUD": PX.AUDJPY, "NZD": PX.NZDJPY,
                         "CAD": PX.CADJPY, "CHF": PX.CHFJPY}, index=idx)


def stats(r):
    r = r.dropna()
    if len(r) < 50: return dict(cagr=np.nan, dd=np.nan, sh=np.nan, cal=np.nan)
    eq = (1 + r).cumprod(); yrs = len(r) / 252
    cagr = (eq.iloc[-1] ** (1 / yrs) - 1) * 100; dd = (eq / eq.cummax() - 1).min() * 100
    sh = r.mean() * 252 / (r.std() * np.sqrt(252)) if r.std() > 0 else 0
    return dict(cagr=cagr, dd=dd, sh=sh, cal=cagr / abs(dd) if dd < 0 else np.nan)


# ---------- ファクターの戦略(月次リバランス、通貨間の相対順位) ----------
def month_end_mask(index):
    s = pd.Series(index, index=index)
    return index.isin(s.groupby([index.year, index.month]).max().values)


def zs(df): return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)


def signals(P, ccys, base):
    lp = np.log(P[ccys].astype(float))
    carry = pd.DataFrame({c: (RATE[AREA[c]] - RATE[AREA[base]]) if c != base else 0 for c in ccys}, index=idx)
    mom12 = lp - lp.shift(252); mom3 = lp - lp.shift(63)
    value = -(lp - lp.rolling(1260, min_periods=504).mean())
    return dict(carry=carry, mom12=mom12, mom3=mom3, value=value)


def weights(sig, kind, nl, ns):
    if kind == "comp3": s = (zs(sig["carry"]) + zs(sig["value"]) + zs(sig["mom12"])) / 3
    else: s = sig[kind]
    me = month_end_mask(s.index)
    W = pd.DataFrame(np.nan, index=s.index, columns=s.columns)
    for d in s.index[me]:
        row = s.loc[d].dropna()
        if len(row) < len(s.columns) or d < START - pd.Timedelta(days=5): continue
        rk = row.sort_values(ascending=False); w = pd.Series(0.0, index=s.columns)
        w[rk.index[:nl]] += 1.0 / nl
        if ns: w[rk.index[len(rk) - ns:]] -= 1.0 / ns
        W.loc[d] = w
    return W.ffill().fillna(0.0)          # 日付tまでの情報で決めた目標ウェイト


def leg_returns(P, ccys, base):
    ret = P[ccys].pct_change().fillna(0)
    car = pd.DataFrame({c: ((RATE[AREA[c]] - RATE[AREA[base]]) * GAP / 365.0) if c != base else 0.0 for c in ccys}, index=idx)
    return ret + car


def realized_vol_flag():
    u = usd_prices().pct_change().drop(columns="USD").mean(axis=1)
    rv = u.rolling(20).std() * np.sqrt(252)
    return (rv > rv.rolling(250).quantile(0.8)).fillna(False)


def vix_flag(mode):
    v = VIX.reindex(idx).ffill()
    if mode == "p80": return (v > v.rolling(250).quantile(0.8)).fillna(False)
    return (v > 30).fillna(False)


def backtest(W, R, cost, lag_filter=None):
    """W: 日付tまでの情報の目標ウェイト。lag_filter=(flag, lag)なら flag が立つ間は現金。"""
    pos = W.shift(1).fillna(0)
    if lag_filter is not None:
        flag, lag = lag_filter
        pos = pos.mul((~flag.shift(lag).fillna(False).astype(bool)).astype(float), axis=0)   # t日の情報を lag 日後から
    gross = pos.abs().sum(axis=1)
    turn = pos.diff().abs().sum(axis=1).fillna(0)
    r = (pos * R).sum(axis=1) - turn * cost - gross * HAIRCUT * GAP / 365.0
    return r[r.index >= START]


def reg(name, r):
    global NVAR
    NVAR += 1; VARIANTS.append(name); return r


# ---------- レンジ型クロス ----------
def pair_ccys(p): return p[:3], p[3:]


def range_pos(c, kind):
    n = len(c); pos = np.zeros(n); v = c.values
    if kind == "boll":
        ma = c.rolling(20).mean().values; sd = c.rolling(20).std().values; p = 0
        for i in range(n):
            if np.isnan(sd[i]): continue
            if p == 0:
                if v[i] < ma[i] - 2 * sd[i]: p = 1
                elif v[i] > ma[i] + 2 * sd[i]: p = -1
            elif p == 1 and v[i] >= ma[i]: p = 0
            elif p == -1 and v[i] <= ma[i]: p = 0
            pos[i] = p
    else:
        ch = c.diff(); g = ch.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean(); l = (-ch.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
        r = (100 - 100 / (1 + g / l)).values; p = 0
        for i in range(n):
            if np.isnan(r[i]): continue
            if p == 0:
                p = 1 if r[i] < 30 else (-1 if r[i] > 70 else 0)
            elif p == 1 and r[i] > 50: p = 0
            elif p == -1 and r[i] < 50: p = 0
            pos[i] = p
    return pd.Series(pos, index=c.index)


def range_ret(p, kind, spread_pips, swap=True):
    c = PX[p]; b, q = pair_ccys(p)
    pos = range_pos(c, kind).shift(1).fillna(0)
    sw = ((RATE[AREA[b]] - RATE[AREA[q]]) * GAP / 365.0) if swap else 0
    cost1 = spread_pips * 1e-4 / 2 / c.shift(1)               # 片道=スプレッドの半分(価格比)
    r = pos * (c.pct_change().fillna(0) + sw) - pos.abs() * HAIRCUT * GAP / 365.0 - pos.diff().abs().fillna(0) * cost1
    return r[r.index >= START]


# ---------- 出力 ----------
def split_stats(r):
    return stats(r), stats(r[r.index < SPLIT]), stats(r[r.index >= SPLIT])


def line(name, r):
    a, b, c = split_stats(r)
    f = lambda x: f"{x['cagr']:5.1f} {x['dd']:5.0f} {x['sh']:5.2f}"
    print(f"{name:34}{f(a):>20}{f(b):>20}{f(c):>20}  Cal {a['cal']:.2f}")


def regimes(r):
    yen = (PX.USDJPY / PX.USDJPY.shift(250) - 1).shift(1).reindex(r.index)   # 直近1年で円安=USDJPY上昇(前日までの情報)
    out = []
    for lab, m in (("円安期", yen > 0), ("円高期", yen <= 0)):
        x = r[m.values]
        out.append(f"{lab}{(x.mean() * 252 * 100):5.1f}%/年(n={len(x)})")
    return " ".join(out)


CRASH = [("2008/7-12", "2008-07-01", "2008-12-31"), ("2015/8", "2015-08-10", "2015-09-30"),
         ("2020/2-3", "2020-02-14", "2020-03-31"), ("2024/7-8", "2024-07-10", "2024-08-15")]


def crash_line(name, r):
    s = [f"{lab}:{((1 + r[(r.index >= a) & (r.index <= b)]).prod() - 1) * 100:6.1f}%" for lab, a, b in CRASH]
    print(f"  {name:32}" + "  ".join(s))


if __name__ == "__main__":
    up, jp = usd_prices(), jpy_prices()
    CU = [c for c in CCY if c != "USD"]            # USD基準の長短: USD以外7通貨
    CJ = [c for c in CCY if c != "JPY"]            # 円基準: 円以外7通貨
    sU, sJ = signals(up, CU, "USD"), signals(jp, CJ, "JPY")
    RU, RJ = leg_returns(up, CU, "USD"), leg_returns(jp, CJ, "JPY")
    rvf = realized_vol_flag()
    res = {}
    hdr = f"{'':34}{'全期間 年率/最大下落/シャープ':>20}{'前半(〜2012)':>20}{'後半(2013〜)':>20}"
    print("【A】USD基準・7通貨の長短(上位2買い/下位2売り、各側1倍) コスト後 片道0.02%+スワップ上乗せ0.5%")
    print(hdr)
    WA = {k: weights(sU, k, 2, 2) for k in ("carry", "mom12", "mom3", "value", "comp3")}
    for k, lab in (("carry", "キャリー"), ("mom12", "モメンタム12m"), ("mom3", "モメンタム3m"), ("value", "バリュー(代用)"), ("comp3", "合成3因子")):
        res["A_" + lab] = reg("A_" + lab, backtest(WA[k], RU, COST_OTH)); line("A " + lab + " (L/S)", res["A_" + lab])
    print("\n【B】円基準・7通貨の買いのみ(上位3、各1/3、円で調達) コスト後 片道0.01%")
    print(hdr)
    WB = {k: weights(sJ, k, 3, 0) for k in ("carry", "mom12", "value", "comp3")}
    for k, lab in (("carry", "キャリー"), ("mom12", "モメンタム12m"), ("value", "バリュー(代用)"), ("comp3", "合成3因子")):
        res["B_" + lab] = reg("B_" + lab, backtest(WB[k], RJ, COST_JPY)); line("B " + lab + " (買のみ)", res["B_" + lab])
    ew = pd.DataFrame(1.0 / len(CJ), index=idx, columns=CJ)
    res["B_全通貨等分"] = reg("B_全通貨等分", backtest(ew, RJ, COST_JPY)); line("B 7通貨等分(参考)", res["B_全通貨等分"])
    WC = {k: weights(sJ, k, 2, 2) for k in ("carry", "comp3")}
    for k, lab in (("carry", "キャリー"), ("comp3", "合成3因子")):
        res["C_" + lab] = reg("C_" + lab, backtest(WC[k], RJ, COST_JPY)); line("C 円基準 " + lab + " (L/S)", res["C_" + lab])
    print("   回転コスト感度(Aキャリー、片道):", " ".join(f"{c*100:.2f}%→{stats(backtest(WA['carry'], RU, c))['cagr']:.1f}%" for c in (0.0001, 0.0002, 0.0003)))

    print("\n【D】VIX/実現ボラ・フィルタ(リスクオフは現金)  年率/最大下落/シャープ")
    print(hdr)
    flags = {"VIX>250日p80": (vix_flag("p80"), 2), "VIX>30": (vix_flag("30"), 2), "為替実現ボラ>p80": (rvf, 1)}
    base_sets = {"B キャリー買のみ": (WB["carry"], RJ, COST_JPY), "A キャリーL/S": (WA["carry"], RU, COST_OTH), "B 合成3因子": (WB["comp3"], RJ, COST_JPY)}
    for bn, (W, R, c) in base_sets.items():
        res["D_" + bn + "_none"] = backtest(W, R, c)
        crash_line(bn + " フィルタなし", res["D_" + bn + "_none"])
        for fn, (fl, lag) in flags.items():
            r = reg(f"D {bn}+{fn}", backtest(W, R, c, (fl, lag))); res[f"D_{bn}_{fn}"] = r
            line(f"{bn}+{fn}", r); crash_line(f"  +{fn}", r)
    print("  (上記crash行: 2008/7-12, 2015/8, 2020/2-3, 2024/7-8 の累積収益)")

    print("\n【E】レンジ型クロス(日足)。スプレッド感度(pips、全幅) 年率%/最大下落%/シャープ 全期間, 2006〜")
    rp = ["AUDNZD", "EURGBP", "EURCHF", "AUDCAD", "NZDCAD", "GBPCHF", "EURAUD"]
    print(hdr)
    for kind, kl in (("boll", "ボリンジャー20/2σ"), ("rsi", "RSI14 30/70")):
        for sp in (0.0, 0.5, 1.0, 2.0, 3.0):
            r = pd.concat([range_ret(p, kind, sp) for p in rp], axis=1).fillna(0).mean(axis=1)
            if sp == 1.0: r = reg("E " + kl + "(7ペア等分)", r); res["E_" + kl] = r
            line(f"E {kl} 7ペア等分 sp={sp}", r)
    for kind, kl in (("boll", "ボリンジャー"), ("rsi", "RSI")):
        s = []
        for p in rp:
            a = stats(range_ret(p, kind, 1.0)); b = stats(range_ret(p, kind, 1.0)[lambda x: x.index >= SPLIT])
            s.append(f"{p}:{a['cagr']:.1f}/{b['cagr']:.1f}")
        print(f"  個別 {kl}(sp=1pip, 全期間/後半の年率%) " + " ".join(s))
    print("  レンジ(ボリンジャー)の最悪日:", " ".join(f"{p}:{range_ret(p, 'boll', 1.0).min()*100:.1f}%" for p in rp))

    print("\n【F】円安期/円高期(直近1年のUSDJPY上昇/下落、前日までの情報で判定)の年率")
    for k in ("A_キャリー", "A_合成3因子", "B_キャリー", "B_合成3因子", "B_全通貨等分", "E_ボリンジャー20/2σ"):
        if k in res: print(f"  {k:24}{regimes(res[k])}")

    print("\n【G】5b 合成: 前半(〜2012)で シャープ>0.3 かつ年率>0 の戦略を候補にし、後半で確認")
    cand = [k for k, r in res.items() if not k.startswith("D_") or "_none" not in k]
    cand = [k for k in cand if (lambda s: s["sh"] > 0.3 and s["cagr"] > 0)(stats(res[k][res[k].index < SPLIT]))]
    print("  候補:", cand)
    if len(cand) >= 2:
        M = pd.DataFrame({k: res[k] for k in cand}).fillna(0)
        print("  相関(日次):"); print(M.corr().round(2).to_string())
        ewp = M.mean(axis=1)
        vol = M.rolling(60).std().shift(1); iv = (1 / vol).div((1 / vol).sum(axis=1), axis=0).fillna(1 / len(cand))
        ivp = (M * iv).sum(axis=1)
        print(hdr); line("等ウェイト合成", ewp); line("逆ボラ合成", ivp)
    elif cand:
        print("  候補が1つだけで合成不可")
    print(f"\n試したバリアント数(スプレッド/コスト感度除く): {NVAR}")
