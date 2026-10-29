"""機械学習による短期予測の網羅探索(厳格なウォークフォワード + 順列検定)。
  python research/ml_scan.py [--perms 60] [--quick]
データ: Yahoo日足(period1=0)。キャッシュ: research/data/ml_cache.json(gitignore済み)。
設計の要点(過剰適合を疑う前提):
 - 特徴量は『その日の終値までの情報』のみ。日本ETF(1348.T)は『翌営業日の寄り→引け』を予測(米国の前日終値が寄り前に分かる)。
 - ウォークフォワード: 最初の5年で学習→次の1年を予測、毎年更新(拡張窓)。ラベルのオーバーラップ分はパージ。
 - 閾値(しきい値 d)は学習期間の最後1年(パージ付き)でのみ選ぶ。成績はOOSのみ。
 - 帰無分布: ラベルを特徴量に対し循環シフト(自己相関は保つが関係は壊す)して全パイプラインを再実行。
   全設定の『最大シャープ』の分布と比べる(Westfall-Young型の多重比較補正)。
"""
import argparse, json, os, sys, time, warnings
import numpy as np, pandas as pd
import urllib.request
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "ml_cache.json")
H = {"User-Agent": "Mozilla/5.0"}

ETF = ["SPY", "QQQ", "IWM", "TLT", "GLD", "EFA", "EEM"]
FX = ["USDJPY=X", "EURJPY=X", "AUDJPY=X", "EURUSD=X"]
JP = ["1348.T"]  # 寄り→引け
TARGETS = ETF + JP + FX
CROSS = ["SPY", "QQQ", "IWM", "TLT", "GLD", "EFA", "EEM", "1348.T", "^N225", "USDJPY=X", "EURJPY=X", "AUDJPY=X", "EURUSD=X"]
MACRO = ["^VIX", "^IRX", "^TNX"]
ALL = sorted(set(TARGETS + CROSS + MACRO))
COST = {**{s: 0.0005 for s in ETF}, "1348.T": 0.0005, **{s: 0.00005 for s in FX}}  # 片道
MODELS = ["logit", "ridge", "hgb"]
HORIZONS = [1, 5]
DGRID = [0.0, 0.5, 1.0]


def fetch():
    if os.path.exists(CACHE):
        return json.load(open(CACHE))
    out = {}
    for s in ALL:
        u = f"https://query1.finance.yahoo.com/v8/finance/chart/{s}?interval=1d&period1=0&period2=1793000000"
        r = json.load(urllib.request.urlopen(urllib.request.Request(u, headers=H), timeout=30))["chart"]["result"][0]
        q = r["indicators"]["quote"][0]
        adj = (r["indicators"].get("adjclose") or [{}])[0].get("adjclose") or q["close"]
        out[s] = {"t": r["timestamp"], "o": q["open"], "c": q["close"], "a": adj}
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    json.dump(out, open(CACHE, "w"))
    return out


def clean(raw):
    """日付キーのDataFrame群。日次±15%超は異常として除外。US ETFは調整後終値、他はclose。"""
    d = {}
    for s, v in raw.items():
        t = pd.to_datetime(v["t"], unit="s")
        # FXの足は前日23:00UTC(夏時間)/0:00UTCで日付がずれる→12時間足して丸める。週末の足は捨てる
        idx = (t + pd.Timedelta(hours=12)).normalize() if s.endswith("=X") else t.normalize()
        df = pd.DataFrame({"o": v["o"], "c": v["a"] if s in ETF + ["QQQ"] else v["c"]}, index=idx)
        df = df[~df.index.duplicated(keep="last")].dropna(subset=["c"]).sort_index()
        df = df[(df["c"] > 0) & (df.index.dayofweek < 5)]
        if s not in ("^VIX", "^IRX", "^TNX"):
            r = df["c"].pct_change()
            bad = r.abs() > 0.15
            # 1日だけ飛んで翌日戻る『悪い価格』(スパイク)を除外: |r_t|>1.5% かつ翌日にほぼ全戻し
            rn = r.shift(-1)
            spike = (r.abs() > 0.015) & (r * rn < 0) & ((r + rn).abs() < 0.3 * r.abs())
            df = df[~(bad | spike)]
        d[s] = df
    return d


def rsi(c, n=14):
    dl = c.diff()
    up = dl.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-dl.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def build(d):
    """各ターゲットの (X, y1, y5, 評価用翌日リターン, 日付, 特徴量名)。"""
    cal = sorted(set().union(*[set(x.index) for x in d.values()]))
    P = pd.DataFrame({s: d[s]["c"] for s in ALL}).reindex(cal).ffill(limit=5)
    ret = {s: np.log(P[s]).diff() for s in CROSS}
    macro = pd.DataFrame(index=P.index)
    macro["vix"] = P["^VIX"]; macro["vix_d1"] = P["^VIX"].pct_change(); macro["vix_d5"] = P["^VIX"].pct_change(5)
    macro["irx"] = P["^IRX"]; macro["tnx"] = P["^TNX"]
    macro["irx_d5"] = P["^IRX"].diff(5); macro["tnx_d5"] = P["^TNX"].diff(5); macro["slope"] = P["^TNX"] - P["^IRX"]
    out = {}
    for s in TARGETS:
        own = d[s]
        c = own["c"]
        f = pd.DataFrame(index=c.index)
        lc = np.log(c)
        for n in (1, 2, 5, 10, 20, 60):
            f[f"own_r{n}"] = lc.diff(n)
        f["own_vol20"] = lc.diff().rolling(20).std()
        f["own_rsi14"] = rsi(c)
        f["own_ma20dev"] = c / c.rolling(20).mean() - 1
        f["own_ma50dev"] = c / c.rolling(50).mean() - 1
        dw = f.index.dayofweek
        for k in range(4):
            f[f"dow{k}"] = (dw == k).astype(float)
        f["month_sin"] = np.sin(2 * np.pi * f.index.month / 12); f["month_cos"] = np.cos(2 * np.pi * f.index.month / 12)
        Pi = P.reindex(f.index)  # ffill済み: 日付<=tの最新値(JP/FXターゲットでも未来は含まない)
        for o in CROSS:
            if o == s:
                continue
            ro = ret[o].reindex(f.index)
            f[f"x_{o}_r1"] = ro
            f[f"x_{o}_r5"] = np.log(Pi[o]).diff(5)
        for k in macro.columns:
            f[f"m_{k}"] = macro[k].reindex(f.index)
        # ターゲット
        if s in JP:  # t+1 の 寄り→引け
            nxt = (own["c"] / own["o"] - 1).shift(-1)
            nxt = nxt.where(own["o"] > 0)
        else:
            nxt = c.pct_change().shift(-1)
        y5 = (c.shift(-5) / c - 1)
        okr = nxt.abs() < 0.15
        nxt = nxt.where(okr)
        df = f.assign(_r=nxt, _y5=y5).replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        cols = [x for x in df.columns if not x.startswith("_")]
        out[s] = dict(X=df[cols].values.astype(float), r=df["_r"].values, y5=df["_y5"].values, dates=df.index, names=cols)
    return out


def fit_score(model, Xf, yf_ret, yf_sgn, Xs_list):
    """標準化→学習→各Xのスコア。返り値: [scores...], fit-scoreのmean/std"""
    mu = Xf.mean(0); sd = Xf.std(0) + 1e-9
    z = lambda X: np.clip((X - mu) / sd, -5, 5)
    Zf = z(Xf)
    if model == "logit":
        m = LogisticRegression(C=0.02, max_iter=200).fit(Zf, yf_sgn); sc = m.decision_function
    elif model == "ridge":
        m = Ridge(alpha=3000.0).fit(Zf, yf_ret * 100); sc = m.predict
    else:
        m = HistGradientBoostingClassifier(max_iter=40, max_depth=2, learning_rate=0.08, min_samples_leaf=100,
                                           l2_regularization=5.0, random_state=0).fit(Zf, yf_sgn)
        sc = lambda Z: m.decision_function(Z) if hasattr(m, "decision_function") else m.predict_proba(Z)[:, 1]
    sf = sc(Zf)
    outs = [(sc(z(X)) - sf.mean()) / (sf.std() + 1e-12) for X in Xs_list]
    imp = None
    if model == "logit":
        imp = m.coef_[0]
    elif model == "ridge":
        imp = m.coef_
    return outs, imp


def positions(zs, d, longshort):
    p = (zs > d).astype(float)
    if longshort:
        p -= (zs < -d)
    return p


def net_pnl(pos, r, cost, jp):
    if jp:
        return pos * r - cost * 2 * np.abs(pos)
    prev = np.concatenate([[0.0], pos[:-1]])
    return pos * r - cost * np.abs(pos - prev)


def sharpe(x):
    return 0.0 if len(x) < 30 or x.std() == 0 else x.mean() / x.std() * np.sqrt(252)


def run_target(s, D, model, h, shift_frac=None, want_imp=False):
    """1ターゲット×モデル×ホライズンのウォークフォワード。OOSの日次ネット損益と日付を返す。"""
    X, r, dates = D["X"], D["r"], D["dates"]
    n = len(r)
    lab_r = r if h == 1 else D["y5"]
    if shift_frac is not None:  # 帰無: ラベルだけ循環シフト(特徴量との対応を壊す)
        k = int(n * shift_frac)
        lab_r = np.roll(lab_r, k)
    lab_s = (lab_r > 0).astype(int)
    years = dates.year.values
    y0 = dates[0].year + 5
    y0 = max(y0, 2009)
    jp = s == "1348.T"; ls = s in FX; cost = COST[s]
    pnl, pd_, imps = [], [], []
    chosen = []
    for Y in range(y0, dates[-1].year + 1):
        te = np.where(years == Y)[0]
        if len(te) < 20:
            continue
        tr_end = te[0] - h  # パージ
        va_start = np.searchsorted(dates, pd.Timestamp(Y - 1, 1, 1))  # 学習の最後1年
        fit_end = va_start - h
        if fit_end < 750:
            continue
        # 閾値選択: fit(〜va_start) → val(1年)
        (zv,), _ = fit_score(model, X[:fit_end], lab_r[:fit_end], lab_s[:fit_end], [X[va_start:tr_end]])
        rv = r[va_start:tr_end]
        best = max(DGRID, key=lambda d: sharpe(net_pnl(positions(zv, d, ls), rv, cost, jp)))
        (zt,), imp = fit_score(model, X[:tr_end], lab_r[:tr_end], lab_s[:tr_end], [X[te]])
        pos = positions(zt, best, ls)
        pnl.append(net_pnl(pos, r[te], cost, jp)); pd_.append(dates[te]); chosen.append(best)
        # 次の日のposition連続性のため、フォールド境界のコストは無視(年1回)
        if want_imp and imp is not None:
            imps.append(imp)
    if not pnl:
        return None
    return dict(pnl=np.concatenate(pnl), dates=pd.DatetimeIndex(np.concatenate([x.values for x in pd_])),
                imps=imps, ds=chosen)


def run_all(data, shift_frac=None, models=MODELS, want_imp=False):
    res = {}
    for s in TARGETS:
        for m in models:
            for h in HORIZONS:
                o = run_target(s, data[s], m, h, shift_frac, want_imp)
                if o is not None:
                    res[(s, m, h)] = o
    return res


def null_job(data, frac):
    res = run_all(data, shift_frac=frac)
    sh = {k: sharpe(v["pnl"]) for k, v in res.items()}
    return sh


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--perms", type=int, default=60); ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    data = build(clean(fetch()))
    nfeat = {s: len(v["names"]) for s, v in data.items()}
    print("特徴量数/ターゲット:", set(nfeat.values()), " 期間:", {s: (str(v['dates'][0].date()), str(v['dates'][-1].date())) for s, v in data.items() if s in ("SPY", "USDJPY=X", "1348.T")})
    jobs = [(s, m, h) for s in TARGETS for m in MODELS for h in HORIZONS]
    real = Parallel(n_jobs=4)(delayed(run_target)(s, data[s], m, h, None, True) for s, m, h in jobs)
    real = {k: v for k, v in zip(jobs, real) if v is not None}
    print(f"試した設定数: {len(jobs)}(ターゲット{len(TARGETS)}×モデル{len(MODELS)}×ホライズン{len(HORIZONS)})、閾値候補{len(DGRID)}は学習内で選択。実時間 {time.time()-t0:.0f}s")
    # ベンチ: 同期間の買い持ち(ETFは長期ロング、FXは単純ロング)
    rows = []
    for k, v in real.items():
        s = k[0]; D = data[s]
        pnl = v["pnl"]; ds = v["dates"]
        idx = pd.Series(D["r"], index=D["dates"]).reindex(ds).values
        bh = sharpe(idx)
        rows.append(dict(asset=s, model=k[1], h=k[2], sharpe=sharpe(pnl), ann_ret=pnl.mean() * 252 * 100,
                         hit=float((np.sign(pnl[pnl != 0]) > 0).mean()) * 100 if (pnl != 0).any() else np.nan,
                         exposure=float((pnl != 0).mean()) * 100, gross_sh=np.nan, bh_sharpe=bh, bh_ret=idx.mean() * 252 * 100,
                         years_pos=float(np.mean([pnl[ds.year == y].sum() > 0 for y in sorted(set(ds.year))])),
                         n=len(pnl)))
    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
    print("\n=== OOS(2009〜)ネット成績 上位15 ===")
    print(df.head(15).round(2).to_string(index=False))
    print("\n=== 全72設定のネットSharpe分布 === 平均 %.3f 中央値 %.3f 最大 %.3f 最小 %.3f 正の割合 %.0f%%" % (
        df.sharpe.mean(), df.sharpe.median(), df.sharpe.max(), df.sharpe.min(), (df.sharpe > 0).mean() * 100))
    print("買い持ちとの差(ネットSharpe - 買い持ちSharpe)平均 %.3f、買い持ちを上回った設定 %d/%d" % (
        (df.sharpe - df.bh_sharpe).mean(), (df.sharpe > df.bh_sharpe).sum(), len(df)))
    print("\n=== モデル別・資産別の平均ネットSharpe ===")
    print(df.groupby("model").sharpe.mean().round(3).to_string()); print(df.groupby("h").sharpe.mean().round(3).to_string())
    print(df.groupby("asset").agg(sharpe=("sharpe", "mean"), bh=("bh_sharpe", "mean")).round(3).to_string())

    # 帰無分布
    if a.perms > 0:
        rng = np.random.default_rng(0)
        fr = rng.uniform(0.1, 0.9, a.perms)
        t1 = time.time()
        nulls = Parallel(n_jobs=4)(delayed(null_job)(data, f) for f in fr)
        print(f"\n帰無(ラベル循環シフト){a.perms}回、{time.time()-t1:.0f}s")
        keys = list(real.keys())
        obs = {k: sharpe(real[k]["pnl"]) for k in keys}
        nm = np.array([[nl.get(k, 0.0) for k in keys] for nl in nulls])  # perms x configs
        obs_v = np.array([obs[k] for k in keys])
        # 個別p(片側)・最大統計による補正p
        p_raw = (1 + (nm >= obs_v).sum(0)) / (1 + a.perms)
        maxnull = nm.max(1)
        p_fwer = np.array([(1 + (maxnull >= o).sum()) / (1 + a.perms) for o in obs_v])
        bi = int(np.argmax(obs_v))
        print("最良設定:", keys[bi], "Sharpe %.2f  個別p=%.3f  最大統計補正p=%.3f" % (obs_v[bi], p_raw[bi], p_fwer[bi]))
        print("帰無の『最大Sharpe』分布: 中央値 %.2f / 95%%点 %.2f / 最大 %.2f ; 帰無の平均Sharpe %.3f(実際の平均 %.3f)" % (
            np.median(maxnull), np.quantile(maxnull, 0.95), maxnull.max(), nm.mean(), obs_v.mean()))
        print("個別p<0.05 の設定数: %d / %d(偶然の期待 %.1f)、補正p<0.05: %d" % ((p_raw < 0.05).sum(), len(keys), 0.05 * len(keys), (p_fwer < 0.05).sum()))
        # 全体: 設定平均Sharpe vs 帰無の設定平均
        pm = (1 + (nm.mean(1) >= obs_v.mean()).sum()) / (1 + a.perms)
        print("全設定の平均Sharpeに対する順列p = %.3f" % pm)
        top = np.argsort(-obs_v)[:5]
        print("上位5設定の個別p / 補正p:", [(keys[i], round(float(p_raw[i]), 3), round(float(p_fwer[i]), 3)) for i in top])

    # 特徴量重要度(ロジスティック h=1、全資産・全フォールドの標準化係数)
    print("\n=== 特徴量重要度(logit, h=1): 平均|標準化係数|, 符号一貫率(フォールド) 上位 ===")
    agg = {}
    for (s, m, h), v in real.items():
        if m == "logit" and h == 1 and v["imps"]:
            names = data[s]["names"]
            M = np.array(v["imps"])
            for j, nme in enumerate(names):
                key = nme if nme.startswith(("own_", "dow", "month", "m_")) else "x_" + nme.split("_")[-1]  # 他資産は1d/5dで集約
                agg.setdefault(key, []).append((np.abs(M[:, j]).mean(), max((M[:, j] > 0).mean(), (M[:, j] < 0).mean())))
    imp = pd.DataFrame({k: np.mean(v, 0) for k, v in agg.items()}, index=["mean_abs_coef", "sign_consistency"]).T.sort_values("mean_abs_coef", ascending=False)
    print(imp.head(10).round(3).to_string())
    # 年別分布(上位3設定+全設定の平均)
    print("\n=== 年別ネット損益(%)  上位3設定と全設定平均 ===")
    top3 = df.head(3)
    yrs = list(range(2009, 2027))
    tab = {}
    for _, r_ in top3.iterrows():
        v = real[(r_.asset, r_.model, int(r_.h))]
        tab[f"{r_.asset}/{r_.model}/h{int(r_.h)}"] = {y: v["pnl"][v["dates"].year == y].sum() * 100 for y in yrs}
    allm = {y: np.mean([v["pnl"][v["dates"].year == y].sum() * 100 for v in real.values() if (v["dates"].year == y).any()]) for y in yrs}
    tab["全72平均"] = allm
    print(pd.DataFrame(tab).round(1).to_string())
    # 集約ポートフォリオ(全設定等ウェイト): 実際に『使える』かの上限的見積もり
    ser = pd.concat([pd.Series(v["pnl"], index=v["dates"]) for v in real.values()], axis=1).fillna(0).mean(1)
    print("\n全72設定を等ウェイトで束ねた場合: ネットSharpe %.2f, 年率 %.2f%%" % (sharpe(ser.values), ser.mean() * 252 * 100))
    print(f"総実行時間 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
