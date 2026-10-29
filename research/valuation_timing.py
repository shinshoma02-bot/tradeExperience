"""割安度(バリュエーション)による超低回転タイミングと国別ローテーションの検証。

  python research/valuation_timing.py            # キャッシュを使う(無ければ取得)
  python research/valuation_timing.py --refresh

データ: Shiller ie_data.xls(1871-2023.09。それ以降は更新が無く取得不可)、Yahoo の国別ETF(配当込み adjclose)+USDJPY。
ルール: しきい値は『その時点までの拡張窓の中央値/分位』だけ(ルックアヘッド無し)。CAPEは2か月ラグ。
試したパターン数は最後に表示する(多重比較)。債券は GS10 利回りからパーボンドの時価変動で作った近似(実際の債券指数ではない)。
"""
import io, json, os, sys, time, urllib.request
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
XLS = os.path.join(DATA, "valuation_ie_data.xls")
PXC = os.path.join(DATA, "valuation_px.json")
UA = {"User-Agent": "Mozilla/5.0"}
SYMS = ["SPY", "VT", "EWJ", "EWG", "EWU", "EWY", "EWT", "EWH", "EWA", "EWC", "EWZ", "INDA", "FXI", "EWW", "EWS", "USDJPY=X"]
COUNTRIES = [s for s in SYMS if s not in ("VT", "USDJPY=X")]  # SPY=米国
LAG = 2          # CAPEの公表・確定遅れ(月)
TAX = 0.20315
COST1 = 0.002    # ETF売買の片道コスト(スプレッド+為替。想定)
NTRIED = {"part1": 0, "part1_dca": 0, "part2": 0, "part3": 0}
REFRESH = "--refresh" in sys.argv


# ---------------------------------------------------------------- 共通
def stats(r, ppy=12):
    r = pd.Series(r).dropna()
    if len(r) < 12: return dict(cagr=np.nan, mdd=np.nan, sharpe=np.nan, n=len(r))
    eq = (1 + r).cumprod()
    cagr = eq.iloc[-1] ** (ppy / len(r)) - 1
    return dict(cagr=cagr, mdd=(eq / eq.cummax() - 1).min(), sharpe=r.mean() / r.std() * np.sqrt(ppy), n=len(r))


def fmt(name, s, extra=""):
    return f"  {name:<34} 年率{s['cagr']*100:6.2f}% 最大下落{s['mdd']*100:7.1f}% Sharpe{s['sharpe']:5.2f} {extra}"


# ---------------------------------------------------------------- Part1: Shiller
def load_shiller():
    if REFRESH or not os.path.exists(XLS):
        os.makedirs(DATA, exist_ok=True)
        b = urllib.request.urlopen(urllib.request.Request("http://www.econ.yale.edu/~shiller/data/ie_data.xls", headers=UA), timeout=90).read()
        open(XLS, "wb").write(b)
    d = pd.read_excel(XLS, "Data", header=None, skiprows=8)
    d = d[pd.to_numeric(d[1], errors="coerce").notna()].reset_index(drop=True)
    idx = pd.period_range("1871-01", periods=len(d), freq="M")
    out = pd.DataFrame({"P": d[1].astype(float).values, "CPI": d[4].astype(float).values, "GS10": d[6].astype(float).values,
                        "RTR": d[9].astype(float).values, "CAPE": pd.to_numeric(d[12], errors="coerce").values}, index=idx)
    return out


def par_bond_price(c, y, n):  # 半年複利、額面1、クーポンc(年率%)、利回りy(%)、残存n年
    c, y = c / 100, y / 100
    k = np.arange(1, int(round(n * 2)) + 1)
    # 端数期間は無視せず残存を半年刻みに丸める(近似)
    return (c / 2 * (1 + y / 2) ** (-k)).sum() + (1 + y / 2) ** (-k[-1])


def bond_returns(gs10):
    y = gs10.values
    r = np.full(len(y), np.nan)
    for i in range(1, len(y)):
        # 前月末に利回りy[i-1]で額面発行→今月末に残存9.5年(丸め)、利回りy[i]で再評価+1か月分クーポン
        p1 = par_bond_price(y[i - 1], y[i], 9.5)
        r[i] = p1 - 1 + y[i - 1] / 100 / 12
    return pd.Series(r, index=gs10.index)


def build_monthly(sh):
    cpi = sh["CPI"]
    infl = cpi / cpi.shift(1)
    stock = sh["RTR"] / sh["RTR"].shift(1) - 1
    bnom = bond_returns(sh["GS10"])
    bond = (1 + bnom) / infl - 1
    return stock, bond


def signals(sh):
    """ラグ済みシグナル(月tの配分に使う値。t-LAGまでの情報のみ)と拡張窓の中央値・順位。"""
    cape = sh["CAPE"]
    infl12 = sh["CPI"] / sh["CPI"].shift(12) - 1
    realy = sh["GS10"] / 100 - infl12
    erp = 1 / cape - realy                 # 益利回り - 実質長期金利(株式リスクプレミアム)
    res = {}
    for name, s in (("cape", cape), ("erp", erp)):
        s = s.shift(LAG)
        med = s.expanding(min_periods=240).median().shift(0)  # 240か月(20年)以上の履歴がある時点から。当月値を含む=過去のみ
        q = s.expanding(min_periods=240).apply(lambda a: (a[:-1] < a[-1]).mean(), raw=True)  # 現在値の過去内パーセンタイル
        res[name] = (s, med, q)
    return res


def run_alloc(stock, bond, w):
    w = w.dropna()
    r = w * stock.reindex(w.index) + (1 - w) * bond.reindex(w.index)
    return r


def turnover(w):
    w = w.dropna()
    ch = w.diff().abs().fillna(0)
    yrs = len(w) / 12
    return (ch > 1e-9).sum(), ch.sum() / yrs


def part1():
    print("=" * 70, "\nPart1: Shiller CAPE による株式/債券 配分(実質リターン、税・手数料なし)")
    sh = load_shiller()
    print(f"  Shiller データ: {sh.index[0]}〜{sh.index[-1]}(CAPE開始 {sh['CAPE'].first_valid_index()})")
    stock, bond = build_monthly(sh)
    sig = signals(sh)
    cape, cmed, cq = sig["cape"]; erp, emed, eq_ = sig["erp"]
    W = {}
    W["A 株100%(持ちっぱなし)"] = pd.Series(1.0, index=sh.index)
    W["B 60/40 月次リバランス"] = pd.Series(0.6, index=sh.index)
    W["C CAPE>中央値→株50 else株100"] = pd.Series(np.where(cape > cmed, 0.5, 1.0), index=sh.index).where(cmed.notna())
    W["D CAPE分位で株100→50線形"] = (1 - 0.5 * cq)
    W["E CAPE四分位 100/75/50/25"] = pd.Series(np.select([cq < .25, cq < .5, cq < .75], [1.0, .75, .5], .25), index=sh.index).where(cq.notna())
    W["F ERP<中央値→株50 else株100"] = pd.Series(np.where(erp < emed, 0.5, 1.0), index=sh.index).where(emed.notna())
    NTRIED["part1"] = 6
    # 比較用: 同じ平均株式比率の固定配分
    periods = [("1901-2023(全期間)", "1901-01", "2023-09"), ("前半 1901-1960", "1901-01", "1960-12"),
               ("後半 1961-2023", "1961-01", "2023-09"), ("1990-2023", "1990-01", "2023-09"), ("2000-2023", "2000-01", "2023-09")]
    rets = {k: run_alloc(stock, bond, v) for k, v in W.items()}
    start = max(r.first_valid_index() for r in rets.values())
    for pname, a, b in periods:
        print(f"\n [{pname}]")
        for k, v in W.items():
            r = rets[k].loc[a:b]
            w = v.loc[a:b]
            nsw, to = turnover(w)
            avgw = w.mean()
            # 平均株式比率が同じ固定配分(同じリスク水準の目安)
            fixed = (avgw * stock + (1 - avgw) * bond).loc[a:b]
            fs = stats(fixed)
            print(fmt(k, stats(r), f"平均株{avgw*100:3.0f}% 配分変更{nsw}回 年間回転{to*100:4.0f}%  | 同比率固定: {fs['cagr']*100:5.2f}%/{fs['mdd']*100:6.1f}%/{fs['sharpe']:4.2f}"))
    # 10年先予測力
    print("\n [10年先リターンの予測力: log(CAPE) vs 将来10年の実質年率(株式)]")
    fw = (stock.add(1).rolling(120).apply(np.prod, raw=True).shift(-120)) ** (1 / 10) - 1
    x = np.log(sh["CAPE"]); df = pd.DataFrame({"x": x, "y": fw}).dropna()
    c = np.corrcoef(df.x, df.y)[0, 1]
    print(f"  全期間 {df.index[0]}〜{df.index[-1]}: 相関{c:.2f} R²={c*c:.2f} (月次重複データ。独立な10年は約{len(df)//120}本。t値は過大)")
    for a, b in (("1881", "1930"), ("1931", "1980"), ("1981", "2013")):
        d2 = df.loc[a:b]
        c2 = np.corrcoef(d2.x, d2.y)[0, 1]
        print(f"   サブ期間 {a}-{b}: 相関{c2:.2f} R²={c2*c2:.2f}")
    # アウト・オブ・サンプル: 拡張窓の回帰で、予測時点で既知のデータ(目標期間が終わったもののみ)だけ使う
    preds, base, act = [], [], []
    xs, ys = x.dropna(), fw
    for t in range(len(df)):
        ti = df.index[t]
        if ti < pd.Period("1921-01", "M"): continue
        known = df[df.index <= ti - 120]                 # 10年先が実現済みのサンプルのみ
        if len(known) < 240: continue
        b1, b0 = np.polyfit(known.x, known.y, 1)
        preds.append(b0 + b1 * df.x.iloc[t]); base.append(known.y.mean()); act.append(df.y.iloc[t])
    preds, base, act = map(np.array, (preds, base, act))
    oos = 1 - ((act - preds) ** 2).sum() / ((act - base) ** 2).sum()
    print(f"  OOS R²(Campbell-Thompson、過去平均比; 1921〜): {oos:.2f}  (正なら過去平均より良い)")
    for a, b in (("1921", "1980"), ("1981", "2013")):
        m = [(p, bb, aa) for p, bb, aa, ti in zip(preds, base, act, df.index[-len(preds):]) if a <= str(ti)[:4] <= b]
        p, bb, aa = map(np.array, zip(*m))
        print(f"   OOS R² {a}-{b}: {1 - ((aa-p)**2).sum()/((aa-bb)**2).sum():.2f}")
    NTRIED["part1"] += 1
    return sh, stock, bond, cq


def dca_cape(sh, stock, bond, cape_sig):
    print("\n" + "=" * 70, "\nPart1b/3b: CAPE で積立の株/債比率だけ変える(売らない)- 20年ローリング")
    cape, med, q = cape_sig["cape"]
    base_stock = 0.75
    res = []
    idx = stock.dropna().index.intersection(bond.dropna().index)
    for start in range(0, len(idx) - 240, 12):
        s0 = idx[start]
        if s0 < pd.Period("1901-01", "M") or med.get(s0) is None or np.isnan(med.get(s0, np.nan)): continue
        sub = idx[start:start + 240]
        out = {}
        for nm, rule in (("tilt", lambda t: 1.0 if cape[t] <= med[t] else 0.5), ("fix75", lambda t: 0.75), ("fix_avg", None)):
            if rule is None: continue
            S = B = 0.0
            ws = []
            for t in sub:
                S *= 1 + stock[t]; B *= 1 + bond[t]
                w = rule(t); ws.append(w); S += w; B += 1 - w
            out[nm] = (S + B, np.mean(ws))
        S = B = 0.0; a = out["tilt"][1]
        for t in sub:
            S *= 1 + stock[t]; B *= 1 + bond[t]; S += a; B += 1 - a
        out["fix_avg"] = (S + B, a)
        res.append((s0, out["tilt"][0] / out["fix_avg"][0] - 1, out["tilt"][0] / out["fix75"][0] - 1, out["tilt"][1]))
    r = pd.DataFrame(res, columns=["start", "vs_same_avg", "vs_fix75", "avgstock"]).set_index("start")
    NTRIED["part1_dca"] = 1
    print(f"  開始年{r.index[0]}〜{r.index[-1]} の20年窓 {len(r)} 本(重複大。独立なのは約{len(r)//20}本)")
    print(f"  積立の株比率を同じ平均に固定した場合との最終資産差: 平均{r.vs_same_avg.mean()*100:+.1f}% 中央値{r.vs_same_avg.median()*100:+.1f}% 勝率{(r.vs_same_avg>0).mean()*100:.0f}%")
    print(f"  株75%固定との差: 平均{r.vs_fix75.mean()*100:+.1f}% 勝率{(r.vs_fix75>0).mean()*100:.0f}%  (傾斜側の平均株比率 {r.avgstock.mean()*100:.0f}%)")
    for a, b in (("1901", "1950"), ("1951", "2003")):
        rr = r.loc[a:b]
        print(f"   開始 {a}-{b}: 同平均固定比 {rr.vs_same_avg.mean()*100:+.1f}% 勝率{(rr.vs_same_avg>0).mean()*100:.0f}%")


# ---------------------------------------------------------------- Part2/3: 国別
def fetch_px():
    if os.path.exists(PXC) and not REFRESH:
        return json.load(open(PXC))
    out = {}; now = int(time.time())
    for s in SYMS:
        u = f"https://query1.finance.yahoo.com/v8/finance/chart/{s}?period1=0&period2={now}&interval=1d"
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=40))["chart"]["result"][0]
            a = r["indicators"].get("adjclose", [{}])[0].get("adjclose") or r["indicators"]["quote"][0]["close"]
            out[s] = list(zip(r["timestamp"], a))
        except Exception as e:
            print("  取得不可:", s, e)
    os.makedirs(DATA, exist_ok=True)
    json.dump(out, open(PXC, "w"))
    return out


def monthly_jpy(raw):
    ser = {}
    for s, v in raw.items():
        d = pd.Series({pd.Timestamp(t, unit="s").normalize(): p for t, p in v if p is not None})
        d = d[~d.index.duplicated()].sort_index()
        ser[s] = d.resample("ME").last()
    px = pd.DataFrame(ser); px.index = px.index.to_period("M")
    fx = px.pop("USDJPY=X")
    jpy = px.mul(fx, axis=0)
    # 異常値チェック
    for c in jpy:
        r = jpy[c].pct_change()
        if (r.abs() > 0.35).any(): print("  警告: 月次±35%超", c, list(r[r.abs() > .35].round(2).items())[:3])
    return jpy


def rotation(ret, mom, val, kind, every, topn, countries):
    """月次リターン表 ret(JPY)、シグナルはt月末時点→t+1月の保有。重み表を返す。"""
    W = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    cur = None
    for i, t in enumerate(ret.index[:-1]):
        if i % every == 0 or cur is None:
            m = mom.loc[t, countries]; v = val.loc[t, countries]
            ok = m.notna() & v.notna() if kind == "combo" else (m.notna() if kind == "mom" else v.notna())
            ok = ok[ok].index
            if len(ok) < max(topn + 2, 5):
                cur = None; continue
            if kind == "mom": score = m[ok].rank()
            elif kind == "val": score = (-v[ok]).rank()
            else: score = m[ok].rank() + (-v[ok]).rank()
            top = score.nlargest(topn).index
            cur = pd.Series(0.0, index=ret.columns); cur[top] = 1 / topn
        if cur is not None:
            W.loc[ret.index[i + 1]] = cur
    return W


def apply_costs_tax(ret, W):
    port = (W.shift(0) * ret).sum(axis=1)  # W[t] は t月の保有(t-1末に決定)
    drift = (W.shift(1) * (1 + ret.shift(1))).div((W.shift(1) * (1 + ret.shift(1))).sum(axis=1), axis=0).fillna(0)  # 概算
    to = (W - drift).abs().sum(axis=1).where(W.sum(axis=1) > 0, 0)
    net = port - to * COST1
    # 年次課税(全額実現とみなす=最悪側。NISA内なら0)
    ann = (1 + net).groupby(net.index.year).prod() - 1
    tax = ann.clip(lower=0) * TAX
    return port, net, to, ann, tax


def after_tax_cagr(net, tax_by_year):
    yrs = sorted(set(net.index.year))
    eq = 1.0
    for y in yrs:
        a = (1 + net[net.index.year == y]).prod() - 1
        eq *= 1 + a - max(a, 0) * TAX
    return eq ** (12 / len(net)) - 1


def part2_3():
    print("\n" + "=" * 70, "\nPart2: 国別ETFローテーション(円建て、配当込み)")
    raw = fetch_px()
    missing = [s for s in SYMS if s not in raw]
    print("  取得できたETF:", ", ".join(s for s in raw if s != "USDJPY=X"), "| 取得不可:", missing or "なし")
    jpy = monthly_jpy(raw)
    ret = jpy.pct_change()
    first = {c: jpy[c].first_valid_index() for c in jpy}
    print("  開始月:", {c: str(v) for c, v in first.items()})
    mom = jpy / jpy.shift(12) - 1          # 12か月リターン(月末時点で既知)
    val = jpy / jpy.shift(60) - 1          # 過去5年リターン(低い=割安)
    countries = [c for c in COUNTRIES if c in jpy.columns]
    ret = ret.loc["2001-01":]
    results = {}
    bench = {"SPY持ちっぱなし": ret["SPY"], "VT持ちっぱなし(2008〜)": ret["VT"],
             "60/40(SPY60+現金0%固定は無効→SPY/IEFなし: 省略)": None}
    ew = ret[countries].mean(axis=1)
    bench = {"SPY持ちっぱなし": ret["SPY"], "VT持ちっぱなし": ret["VT"], "全国等配分(月次再均衡)": ew}
    for k, v in bench.items():
        pass
    cfgs = []
    for kind in ("mom", "val", "combo"):
        for every in (1, 3):
            for topn in (3, 5):
                cfgs.append((kind, every, topn))
    NTRIED["part2"] = len(cfgs)
    rows = {}
    for kind, every, topn in cfgs:
        W = rotation(ret, mom, val, kind, every, topn, countries)
        W = W.loc[ret.index]
        valid = W.sum(axis=1) > 0
        port, net, to, ann, tax = apply_costs_tax(ret.fillna(0), W)
        net = net[valid]; port = port[valid]
        rows[(kind, every, topn)] = (port, net, to[valid], W[valid])
    periods = [("全期間", None, None), ("前半", "first", None), ("後半", "second", None)]
    first_valid = min(v[1].index[0] for v in rows.values())
    last = ret.index[-1]
    mid = first_valid + (last - first_valid).n // 2
    print(f"  評価期間 {first_valid}〜{last}(前半/後半の境 {mid})。コスト片道{COST1*100:.1f}%、課税は年次全額実現の最悪側(NISA内は不要)")
    for pname, a, b in (("全期間", first_valid, last), ("前半", first_valid, mid), ("後半", mid + 1, last)):
        print(f"\n [{pname}]")
        for k, r in bench.items():
            rr = r.loc[a:b].dropna()
            print(fmt(k, stats(rr)))
        for key, (port, net, to, W) in rows.items():
            nm = f"{key[0]} {'月次' if key[1]==1 else '四半期'} 上位{key[2]}"
            s = stats(net.loc[a:b])
            extra = f"年間回転{to.loc[a:b].sum()/(len(net.loc[a:b])/12)*100:4.0f}% 税後年率{after_tax_cagr(net.loc[a:b], None)*100:5.2f}%"
            print(fmt(nm, s, extra))
    # 前半で選び後半で確認
    print("\n [前半で最良の設定を選び後半で確認(ただし12通りから選ぶので過剰適合に注意)]")
    sc = {k: stats(v[1].loc[first_valid:mid])["sharpe"] for k, v in rows.items()}
    best = max(sc, key=sc.get)
    print(f"  前半最良: {best} (前半Sharpe {sc[best]:.2f}) → 後半: {stats(rows[best][1].loc[mid+1:last])['sharpe']:.2f} / SPY後半 {stats(ret['SPY'].loc[mid+1:last])['sharpe']:.2f} / 全国等配分後半 {stats(ew.loc[mid+1:last])['sharpe']:.2f}")
    # 順位の安定性: 12通りのうち何通りが全国等配分(再均衡)・SPYを上回ったか
    nb = sum(stats(v[1].loc[first_valid:last])["cagr"] > stats(ew.loc[first_valid:last])["cagr"] for v in rows.values())
    ns = sum(stats(v[1].loc[first_valid:last])["cagr"] > stats(ret['SPY'].loc[first_valid:last])["cagr"] for v in rows.values())
    print(f"  全12通りのうち 全国等配分を上回った(コスト後・税前) {nb}/12、SPYを上回った {ns}/12")

    # ---- Part3: 積立のウェイト傾斜(売却なし)
    print("\n" + "=" * 70, "\nPart3: 『割安な国に積立を厚く』(毎月同額を積立、売らない)円建て、終了時評価額の比とIRR")
    def dca(weightfn):
        units = pd.Series(0.0, index=ret.columns); cf = []
        wealth = []
        px = jpy.loc[first_valid:last]
        for i, t in enumerate(px.index[:-1]):
            w = weightfn(t)
            if w is None: continue
            tn = px.index[i + 1]  # 翌月末価格で約定(月末シグナル→翌月)
            units[w.index] += w.values / px.loc[tn, w.index].values if False else 0
            cf.append((tn, w))
        return cf
    def dca_value(weightfn, a, b):
        pxx = jpy
        idx = [t for t in jpy.index if a <= t <= b]
        units = pd.Series(0.0, index=jpy.columns)
        flows = []
        for t in idx[:-1]:
            w = weightfn(t)
            if w is None: continue
            tn = idx[idx.index(t) + 1]
            ok = w.index
            units[ok] += w.values / pxx.loc[tn, ok].values
            flows.append((tn, -1.0))
        end = idx[-1]
        tv = (units * pxx.loc[end]).sum()
        n = len(flows); tot = float(n)
        # 月次IRR
        cfs = np.array([-1.0] * n + [tv]); times = np.arange(1, n + 2)
        lo, hi = -0.05, 0.05
        for _ in range(100):
            mid_ = (lo + hi) / 2
            npv = (cfs / (1 + mid_) ** times).sum()
            lo, hi = (mid_, hi) if npv > 0 else (lo, mid_)
        return tv / tot, (1 + mid_) ** 12 - 1
    def wfun(kind):
        def f(t):
            av = [c for c in countries if pd.notna(jpy.loc[t, c]) and pd.notna(val.loc[t, c]) and pd.notna(mom.loc[t, c])]
            if len(av) < 5: return None
            if kind == "eq": return pd.Series(1 / len(av), index=av)
            if kind == "val": s = (-val.loc[t, av]).rank()
            elif kind == "mom": s = mom.loc[t, av].rank()
            else: s = (-val.loc[t, av]).rank() + mom.loc[t, av].rank()
            return s / s.sum()
        return f
    NTRIED["part3"] = 3
    for pname, a, b in (("全期間", first_valid, last), ("前半", first_valid, mid), ("後半", mid, last)):
        print(f" [{pname}] 積立1円あたり終了評価額(1.0=元本) / IRR年率")
        base = dca_value(wfun("eq"), a, b)
        print(f"   等配分積立(基準)       {base[0]:.3f}  {base[1]*100:5.2f}%")
        for k, n in (("val", "割安ほど厚く(5年リターン順位)"), ("mom", "モメンタム順位で厚く"), ("combo", "割安+モメンタム順位")):
            r = dca_value(wfun(k), a, b)
            print(f"   {n:<22} {r[0]:.3f}  {r[1]*100:5.2f}%  (基準比 {(r[0]/base[0]-1)*100:+.1f}%)")
    spy_only = dca_value(lambda t: pd.Series([1.0], index=["SPY"]) if pd.notna(jpy.loc[t, "SPY"]) else None, first_valid, last)
    print(f"   SPY一本積立(参考)      {spy_only[0]:.3f}  {spy_only[1]*100:5.2f}%")


if __name__ == "__main__":
    sh, stock, bond, cq = part1()
    dca_cape(sh, stock, bond, signals(sh))
    part2_3()
    print("\n試したパターン数:", NTRIED, "合計", sum(NTRIED.values()))
