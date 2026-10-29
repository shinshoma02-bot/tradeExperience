"""行動ギャップの定量化: 毎月定額積立を『続ける』場合と『止める/売る/減らす/増やす』場合の差。

  python research/behavior_gap.py            # research/data/dca_raw.json(dca_optimization.pyのキャッシュ)を使う。SPYは behavior_spy.json に取得
  python research/behavior_gap.py --refresh

ルール(事前固定。結果を見て変えていない。試したのは下の5ルール×4系列×窓長10年/20年+4つの下落局面のみ):
  毎月の初営業日の終値で月額1を買う。シグナルは『前営業日までの情報』のみ(ルックアヘッドなし)。
  資産指数 = 配当込み(^GSPC は価格×年2%上乗せの近似、^N225 は価格のみ=不利に出る、SPY/VT は調整後終値)。
  ドローダウン dd = 指数 / それまでの最高値 - 1 (データ先頭からの履歴。窓の開始時点で観測できる状態)。
  (a) 継続      : 毎月1を買う
  (b) 停止      : dd<=-20% になったら積立を止め(その月の1は現金で保有、利子0)、指数が最高値を更新(dd==0)したら再開。止めた間の現金は再投資しない(最終資産には現金として計上=寛大な設定)
  (c) 全売却    : dd<=-30% で保有を全て売って現金化(NISA想定で非課税)。以後の積立も現金で保有。最高値更新後に、溜まった現金も全て投資して再開
  (d) 半減      : dd<=-20% の間(最高値更新まで)は買付を0.5に減らす(残りは現金)
  (e) 追随      : 直近252営業日のリターンが +10% 超なら買付を1.5倍、それ以外は1(投入総額が増えるので IRR と 資産/投入額 で比較)
  比較は (a) に対する最終資産の差(%)、IRR(年率)の差。窓は開始月を1か月ずつずらした全ローリング窓(重なるので独立ではない)。
  下落局面: [ピーク-24か月, 底+36か月] を事前に固定(2000-03, 2007-10, 2020-02, 2022-01 の高値)。
"""
import argparse, json, os, time, urllib.request
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "dca_raw.json")
SPY_CACHE = os.path.join(HERE, "data", "behavior_spy.json")
UA = {"User-Agent": "Mozilla/5.0"}


def fetch(sym):
    u = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?period1=0&period2={int(time.time())}&interval=1d"
    r = json.load(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=40))["chart"]["result"][0]
    a = r["indicators"].get("adjclose", [{}])[0].get("adjclose") or r["indicators"]["quote"][0]["close"]
    return list(zip(r["timestamp"], a))


def load(refresh=False):
    raw = json.load(open(RAW)) if os.path.exists(RAW) and not refresh else {s: fetch(s) for s in ["^GSPC", "^N225", "VT"]}
    if os.path.exists(SPY_CACHE) and not refresh:
        raw["SPY"] = json.load(open(SPY_CACHE))
    else:
        try:
            raw["SPY"] = fetch("SPY"); json.dump(raw["SPY"], open(SPY_CACHE, "w"))
        except Exception as e:
            print("SPY取得失敗:", e)
    out = {}
    for s in ["^GSPC", "^N225", "VT", "SPY"]:
        if s not in raw: continue
        x = pd.Series({pd.Timestamp(t, unit="s").normalize(): v for t, v in raw[s] if v is not None}).sort_index().astype(float)
        x = x[~x.index.duplicated(keep="last")]
        bad = x.pct_change().abs()
        bad = bad[bad > 0.15]
        if len(bad): print(f"  [{s}] 日次±15%超 {len(bad)}件: " + ", ".join(f"{d.date()}" for d in bad.index[:5]))
        if s == "^GSPC":   # 配当年2%の近似
            yrs = (x.index - x.index[0]).days / 365.25
            x = x * np.exp(0.02 * yrs)
        out[s] = x
    return out


def monthly(x):
    d = pd.DataFrame({"v": x})
    d["dd"] = d.v / d.v.cummax() - 1
    d["mom"] = d.v / d.v.shift(252) - 1
    d[["dd", "mom"]] = d[["dd", "mom"]].shift(1)       # 前営業日までの情報
    m = d.groupby([d.index.year, d.index.month]).head(1).iloc[1:]
    return m


def sim(v, dd, mom, strat):
    """戻り値: 最終資産, 投入総額, 月次キャッシュフロー(投入=負, 最後に最終資産=正)。"""
    units = cash = 0.0; flag = False; out = False; flows = []; paid = 0.0
    for p, d, mo in zip(v, dd, mom):
        pay = 1.0; paid += 1.0
        if strat == "a":
            units += 1 / p
        elif strat in ("b", "d"):
            if d <= -0.20: flag = True
            if d >= 0: flag = False
            buy = 0.0 if (strat == "b" and flag) else (0.5 if (strat == "d" and flag) else 1.0)
            units += buy / p; cash += 1 - buy
        elif strat == "c":
            if not out and d <= -0.30:
                cash += units * p; units = 0.0; out = True
            if out and d >= 0:
                out = False; units += cash / p; cash = 0.0
            if out: cash += 1
            else: units += 1 / p
        elif strat == "e":
            mult = 1.5 if (mo == mo and mo > 0.10) else 1.0
            pay = mult; paid += mult - 1; units += mult / p
        flows.append(-pay)
    final = units * v[-1] + cash
    flows[-1] += final
    return final, paid, flows


def irr(flows):
    lo, hi = -0.05, 0.05
    f = lambda r: sum(c / (1 + r) ** i for i, c in enumerate(flows))
    for _ in range(80):
        mid = (lo + hi) / 2
        if f(mid) > 0: lo = mid
        else: hi = mid
    return (1 + mid) ** 12 - 1


STRATS = {"a": "継続", "b": "-20%で停止", "c": "-30%で全売却", "d": "-20%で半減", "e": "上昇で1.5倍"}


def run_window(m, i0, i1):
    sl = m.iloc[i0:i1]
    v, dd, mo = sl.v.values, sl.dd.values, sl.mom.values
    res = {}
    for k in STRATS:
        fin, paid, fl = sim(v, dd, mo, k)
        res[k] = dict(fin=fin, paid=paid, irr=irr(fl))
    return res


def rolling(name, m, years):
    n = years * 12
    rows = []
    for i in range(0, len(m) - n + 1):
        r = run_window(m, i, i + n)
        rows.append((m.index[i], r))
    print(f"\n=== {name}  {years}年窓  窓数={len(rows)} ({rows[0][0].date()}〜{rows[-1][0].date()}開始) ===")
    print(f"{'ルール':<12}{'資産差 中央':>9}{'p10':>8}{'最悪':>8}{'最悪開始':>12}{'(a)より悪い窓':>12}{'IRR差 中央':>10}{'IRR差 最悪':>10}{'資産/投入 中央':>13}")
    for k in STRATS:
        if k == "a":
            mo = np.median([r["a"]["fin"] / r["a"]["paid"] for _, r in rows])
            print(f"{k+' '+STRATS[k]:<12}{'基準':>9}{'':>8}{'':>8}{'':>12}{'':>12}{'':>10}{'':>10}{mo:>13.2f}")
            continue
        diff = np.array([r[k]["fin"] / r["a"]["fin"] - 1 for _, r in rows])
        dirr = np.array([r[k]["irr"] - r["a"]["irr"] for _, r in rows])
        mo = np.median([r[k]["fin"] / r[k]["paid"] for _, r in rows])
        w = int(diff.argmin())
        print(f"{k+' '+STRATS[k]:<12}{np.median(diff):>+9.1%}{np.percentile(diff,10):>+8.1%}{diff.min():>+8.1%}{str(rows[w][0].date()):>12}{(diff<0).mean():>12.0%}{np.median(dirr):>+10.2%}{dirr.min():>+10.2%}{mo:>13.2f}")


EPISODES = {"2000-02ITバブル(ピーク2000-03/底2002-10)": ("2000-03", "2002-10"),
            "2008-09金融危機(ピーク2007-10/底2009-03)": ("2007-10", "2009-03"),
            "2020コロナ(ピーク2020-02/底2020-03)": ("2020-02", "2020-03"),
            "2022利上げ(ピーク2022-01/底2022-10)": ("2022-01", "2022-10")}


def episodes(name, m):
    last = m.index[-1]
    print(f"\n=== {name}: 下落局面 [ピーク-24か月, 底+36か月] 最終資産の(a)比 ===")
    for en, (pk, tr) in EPISODES.items():
        s = pd.Timestamp(pk) - pd.DateOffset(months=24)
        e = min(pd.Timestamp(tr) + pd.DateOffset(months=36), last)
        if s < m.index[0]: continue
        sub = m[(m.index >= s) & (m.index <= e)]
        i0 = m.index.get_loc(sub.index[0]); i1 = m.index.get_loc(sub.index[-1]) + 1
        r = run_window(m, i0, i1)
        mdd = (sub.v / sub.v.cummax() - 1).min()
        cells = "  ".join(f"{k}:{r[k]['fin']/r['a']['fin']-1:+.1%}" for k in "bcde")
        print(f"  {en}  {sub.index[0].date()}〜{sub.index[-1].date()} 最大dd={mdd:.0%}  (a)資産/投入={r['a']['fin']/r['a']['paid']:.2f}  {cells}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    px = load(a.refresh)
    print("試したルール: a継続 / b-20%停止 / c-30%全売却 / d-20%半減 / e上昇1.5倍(固定・チューニングなし)。合計5ルール")
    for name, x in px.items():
        m = monthly(x)
        print(f"\n##### {name}: {m.index[0].date()}〜{m.index[-1].date()} ({len(m)}か月)")
        for yrs in (10, 20):
            if len(m) >= yrs * 12 + 12: rolling(name, m, yrs)
        episodes(name, m)
    print("\n注: 窓は重なるため独立ではない。20年窓は事実上の独立標本が数個。b/dの現金は利子0・再投資なし。^N225は配当なし。")
