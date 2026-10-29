"""逆張り・群衆の逆を行く手法の検証(急落買い / VIX / IBS / RSI(2) / 連続下落 / 極端な日の翌日)。
  python research/anomaly_contrarian.py
ルール: シグナルは当日終値で決め、売買は翌営業日の始値(ルックアヘッドなし)。
  日足は「寄り→引け」と「引け→翌寄り」に分け、保有区間だけ収益を計上する。
居ない時間は現金利回り(米国は ^IRX、日本は0)を加えて持ちっぱなしと公平に比べる。
データ: Yahoo。OHLCは配当込み調整後終値の比率でスケール。日次±15%超は異常として除外。
"""
import json, os, time, urllib.request
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get("CONTRARIAN_CACHE", "/tmp/claude-0/-home-user-tradeExperience/ce681a55-ab87-515d-97f7-3cd9e02c3837/scratchpad")
H = {"User-Agent": "Mozilla/5.0"}
COSTS = [0.0, 0.0002, 0.001]  # 片道
HALF_SPLIT = None


def fetch(sym):
    os.makedirs(CACHE, exist_ok=True)
    fn = os.path.join(CACHE, sym.replace("^", "_") + "_daily.json")
    if os.path.exists(fn):
        return json.load(open(fn))
    u = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval=1d&period1=0&period2={int(time.time())}"  # range=maxは月足に落ちるので使わない
    res = json.load(urllib.request.urlopen(urllib.request.Request(u, headers=H), timeout=40))["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose") or q["close"]
    d = {"t": res["timestamp"], "o": q["open"], "h": q["high"], "l": q["low"], "c": q["close"], "a": adj}
    json.dump(d, open(fn, "w"))
    return d


def load(sym, use_adj=True):
    d = fetch(sym)
    df = pd.DataFrame({k: d[k] for k in "ohlca"}, index=pd.to_datetime(d["t"], unit="s").normalize())
    df = df[~df.index.duplicated(keep="last")].dropna()
    df = df[(df.o > 0) & (df.h > df.l)]
    f = (df.a / df.c) if use_adj else 1.0
    out = pd.DataFrame({"o": df.o * f, "h": df.h * f, "l": df.l * f, "c": df.c * f}) if use_adj else df[list("ohlc")].copy()
    r = out.c.pct_change()
    bad = r.abs() > 0.15
    if bad.any():
        print(f"  [{sym}] 日次±15%超 {int(bad.sum())}件を除外: {[str(x.date()) for x in out.index[bad]][:6]}")
        out = out[~bad]
    return out


# ---------- シグナル ----------
def sma(c, n): return c.rolling(n).mean()


def rsi_wilder(c, n=2):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def state_machine(enter, exit_, maxhold=None):
    """enter/exit_: bool配列。保有中はexit_でまで保有(maxhold日で強制終了)。"""
    s = np.zeros(len(enter)); on = False; k = 0
    for i in range(len(enter)):
        if on:
            k += 1
            if exit_[i] or (maxhold and k >= maxhold):
                on = False
        elif enter[i]:
            on = True; k = 0
        s[i] = on
    return s


def drawdown_target(c, th, maxhold):
    runmax = c.cummax(); dd = (c / runmax - 1).values
    s = np.zeros(len(c)); on = False; k = 0; armed = True
    for i in range(len(c)):
        if on:
            k += 1
            if dd[i] >= -1e-12 or (maxhold and k >= maxhold):
                on = False; armed = dd[i] > -th * 0.5 if dd[i] < -1e-12 else True
        else:
            if not armed and dd[i] > -th * 0.5:
                armed = True
            if armed and dd[i] <= -th:
                on = True; k = 0
        s[i] = on
    return pd.Series(s, index=c.index)


def build_strategies(df, vix):
    c, h, l = df.c, df.h, df.l
    ret = c.pct_change()
    trend = (c > sma(c, 200)).values
    S = {}
    for th in (0.05, 0.10, 0.15):
        S[f"急落買い-{int(th*100)}%→新高値で売"] = drawdown_target(c, th, None)
        S[f"急落買い-{int(th*100)}%→40日で売"] = drawdown_target(c, th, 40)
    if vix is not None:
        v = vix.reindex(df.index, method="ffill")
        for th in (20, 30, 40):
            S[f"VIX>{th}の間"] = (v > th).astype(float)
    ibs = ((c - l) / (h - l)).values
    for th in (0.1, 0.2, 0.3):
        S[f"IBS<{th}"] = pd.Series((ibs < th).astype(float), index=c.index)
        S[f"IBS<{th}&200日線上"] = pd.Series(((ibs < th) & trend).astype(float), index=c.index)
    rsi = rsi_wilder(c, 2).values
    for th in (5, 10, 20):
        S[f"RSI2<{th}→>70で売"] = pd.Series(state_machine(rsi < th, rsi > 70), index=c.index)
        S[f"RSI2<{th}&200日線上"] = pd.Series(state_machine((rsi < th) & trend, rsi > 70), index=c.index)
    down = (ret < 0).astype(int)
    for n in (2, 3, 4):
        run = down.groupby((down == 0).cumsum()).cumsum().values
        S[f"{n}連続下落"] = pd.Series((run >= n).astype(float), index=c.index)
        S[f"{n}連続下落&200日線上"] = pd.Series(((run >= n) & trend).astype(float), index=c.index)
    for x in (0.02, 0.03, 0.05):
        S[f"前日{int(x*100)}%超下落の翌日"] = (ret <= -x).astype(float)
    for x in (0.02, 0.03):
        S[f"前日{int(x*100)}%超上昇の翌日"] = (ret >= x).astype(float)
    return S


# ---------- バックテスト ----------
def backtest(df, target, cash, cost):
    o, c = df.o, df.c
    r_on = (o / c.shift(1) - 1).fillna(0); r_id = (c / o - 1)
    h_id = target.shift(1).fillna(0); h_on = h_id.shift(1).fillna(0)
    gross = (1 + r_on * h_on) * (1 + r_id * h_id) - 1
    buy = ((h_id == 1) & (h_on == 0)).astype(float); sell = ((h_on == 1) & (h_id == 0)).astype(float)
    net = gross - cost * (buy + sell)
    inm = ((h_on == 1) | (h_id == 1))
    tot = net.where(inm, cash)
    return net, tot, inm, int(buy.sum())


def stats(df, target, cash, cost, bh_r):
    net, tot, inm, ntr = backtest(df, target, cash, cost)
    yrs = len(df) / 252
    eq = (1 + tot).cumprod()
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    mdd = (eq / eq.cummax() - 1).min()
    return net, tot, inm, ntr, cagr, mdd


def tstat(x):
    x = x.dropna()
    return (x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 20 and x.std() > 0 else np.nan


def evaluate(name, df, cash, S, bh_ret, mid, split2010=True):
    rows = []
    for k, tg in S.items():
        net0, tot0, inm, ntr, cagr0, mdd = stats(df, tg, cash, 0.0, bh_ret)
        _, _, _, _, cagr2, _ = stats(df, tg, cash, 0.0002, bh_ret)
        _, _, _, _, cagr10, _ = stats(df, tg, cash, 0.001, bh_ret)
        ex = (net0 - cash)[inm]
        tim = lambda m: tstat(net0[inm & m] - bh_ret[m].mean()) if (inm & m).sum() > 20 else np.nan
        i1 = df.index < mid; i2 = df.index >= mid; ip = df.index >= "2010-01-01"; ia = df.index < "2010-01-01"
        n_in = int(inm.sum())
        rows.append(dict(戦略=k, 居る比率=inm.mean(), 売買=ntr, 年率0=cagr0, 年率0_02=cagr2, 年率0_1=cagr10, 最大下落=mdd,
                         居る日の年率超過=ex.mean() * 252, t_超過=tstat(ex), t_対平均=tim(np.ones(len(df), bool)),
                         t前=tim(i1), t後=tim(i2), t_2010前=tim(ia), t_2010後=tim(ip),
                         一回平均=(net0[inm].sum() / max(ntr, 1))))
    return pd.DataFrame(rows)


def main():
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    vix = load("^VIX", use_adj=False).c
    irx = load("^IRX", use_adj=False).c / 100 / 252
    assets = [("SPY", "SPY", True, True), ("QQQ", "QQQ", True, True), ("1348.T", "MAXIS TOPIX 1348.T", True, False),
              ("^N225", "日経平均(指数・配当なし)", False, False)]
    all_res = {}
    for sym, label, adj, us in assets:
        print(f"\n===== {label} =====")
        df = load(sym, use_adj=adj)
        print(f"  期間 {df.index[0].date()}〜{df.index[-1].date()} ({len(df)}日)")
        cash = irx.reindex(df.index, method="ffill").fillna(0) if us else pd.Series(0.0, index=df.index)
        S = build_strategies(df, vix)
        bh_tg = pd.Series(1.0, index=df.index)
        bh_net, bh_tot, _, _ = backtest(df, bh_tg, cash, 0.0)
        eq = (1 + bh_tot).cumprod(); yrs = len(df) / 252
        mid = df.index[len(df) // 2]
        print(f"  持ちっぱなし: 年率 {eq.iloc[-1]**(1/yrs)-1:.2%}  最大下落 {(eq/eq.cummax()-1).min():.1%}  1日平均超過年率 {(bh_net-cash).mean()*252:.2%}  "
              f"t={tstat(bh_net-cash):.2f}  前後分割日 {mid.date()}  現金年率平均 {cash.mean()*252:.2%}")
        res = evaluate(sym, df, cash, S, bh_net, mid)
        all_res[sym] = res
        f = res.copy()
        for c in ["居る比率", "年率0", "年率0_02", "年率0_1", "最大下落", "居る日の年率超過", "一回平均"]:
            f[c] = (f[c] * 100).round(1)
        print(f.round(2).to_string(index=False))
    # ---- 集計: 多重比較 ----
    n_per = len(all_res["SPY"])
    print(f"\n===== 集計(試したパターン数 = {n_per}戦略 x {len(all_res)}資産 = {n_per*len(all_res)}。コストは3水準を別計算) =====")
    for sym, r in all_res.items():
        ok = r.dropna(subset=["t_対平均"])
        print(f"{sym}: t_対平均>2 は {(ok.t_対平均>2).sum()}/{len(ok)}、t<-2 は {(ok.t_対平均<-2).sum()}。"
              f" 前半t>2 {(ok.t前>2).sum()} → そのうち後半t>1 {((ok.t前>2)&(ok.t後>1)).sum()}、後半t<0 {((ok.t前>2)&(ok.t後<0)).sum()}")
        # 前半(2010前 or 前半)で最良を選び、後半で確認
        cand = ok[(ok.売買 >= 15)]
        best = cand.sort_values("t前", ascending=False).head(5)
        print("  前半t上位5の後半成績:")
        print(best[["戦略", "売買", "t前", "t後", "年率0_1", "最大下落"]].round(2).to_string(index=False))
    print("\n注: t_超過=居る日の(日次リターン-現金)の平均/標準誤差。t_対平均=居る日の平均が全日平均より高いか(タイミング効果)。")
    print("   多重比較: 試行数が多いのでt>2は約5%が偶然。|t|>3程度でないと信用しない。")


if __name__ == "__main__":
    main()
