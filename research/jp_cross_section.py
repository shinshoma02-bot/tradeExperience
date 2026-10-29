"""日本株クロスセクション検証(価格と出来高だけ)。

使い方: python research/jp_cross_section.py
- JPX上場銘柄一覧(Excel)から、市場・規模区分で層別に約600銘柄を無作為抽出(seed固定。リターンは見ずに抽出)
- Yahoo Financeの日足(終値・出来高)を research/data/jp_xs/ にキャッシュ(初回のみ取得、約0.25秒間隔、429は待って再試行)
- 月末終値でシグナル → 翌営業日の終値で買い → 翌月の同じ日に手仕舞い(ルックアヘッド無し)
- 等ウェイト、上位10分位のロングのみ(ユニバース等ウェイト超過)と、上位−下位(L-S)
- 生存者バイアス: 今も上場している銘柄だけ。上場廃止銘柄は含まれない(特に小型株で過去が良く見える)
- 配当は含まない(価格リターンのみ)。分割等は日次±50%超を除外
"""
import json
import math
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "jp_xs"
DATA.mkdir(parents=True, exist_ok=True)
XLSX = DATA / "data_j.xlsx"
XLSX_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xlsx"
UA = {"User-Agent": "Mozilla/5.0"}  # 長いUAは429になる
SEED = 20261029
N_PRIME_BY_SIZE = {"TOPIX Core30": 10, "TOPIX Large70": 30, "TOPIX Mid400": 80,
                   "TOPIX Small 1": 60, "TOPIX Small 2": 60, "-": 10}
N_STANDARD, N_GROWTH = 200, 160
SPLIT_H = pd.Timestamp("2019-01-01")
START = pd.Timestamp("2005-01-01")
LIQ_MIN = 5e7          # 平均売買代金(円) 流動性フィルタ
COMMISSION = 0.001     # 片道0.1%(手数料ありの場合)


def http(url, tries=6):
    err = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except Exception as e:  # noqa
            err = e
            time.sleep(2 * (k + 1) if "429" in str(e) else 1)  # 429はバックオフ
    raise err


def load_universe():
    if not XLSX.exists():
        XLSX.write_bytes(http(XLSX_URL))
    d = pd.read_excel(XLSX)
    d["コード"] = d["コード"].astype(str).str.strip()
    d = d[d["コード"].str.fullmatch(r"\d{4}")]
    mk = d["市場・商品区分"]
    rng = np.random.RandomState(SEED)
    picks = []
    pr = d[mk.str.contains("プライム（内国")]
    for sz, n in N_PRIME_BY_SIZE.items():
        g = pr[pr["規模区分"] == sz]
        picks.append(g.sample(min(n, len(g)), random_state=rng).assign(grp="prime/" + sz))
    for key, n, name in (("スタンダード（内国", N_STANDARD, "standard"), ("グロース（内国", N_GROWTH, "growth")):
        g = d[mk.str.contains(key)]
        picks.append(g.sample(min(n, len(g)), random_state=rng).assign(grp=name))
    return pd.concat(picks)[["コード", "銘柄名", "grp"]].reset_index(drop=True)


def fetch(code):
    f = DATA / f"{code}.json"
    if f.exists():
        return json.loads(f.read_text())
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{code}.T"
           f"?interval=1d&period1=0&period2={int(time.time())}")
    try:
        j = json.loads(http(url))
        r = j["chart"]["result"][0]
        q = r["indicators"]["quote"][0]
        out = {"t": r["timestamp"], "c": q["close"], "v": q["volume"]}
    except Exception as e:  # noqa
        return {"err": str(e)[:80]}  # 失敗はキャッシュしない(次回再取得)
    f.write_text(json.dumps(out))
    time.sleep(0.25)
    return out


def build_panels(uni):
    closes, vols, bad = {}, {}, []
    for code in list(uni["コード"]) + ["1348"]:
        o = fetch(code)
        if "err" in o or not o.get("t"):
            bad.append(code)
            continue
        idx = pd.to_datetime(o["t"], unit="s", utc=True).tz_convert("Asia/Tokyo").tz_localize(None).normalize()
        s = pd.Series(o["c"], index=idx, dtype=float)
        v = pd.Series(o["v"], index=idx, dtype=float)
        s = s[~s.index.duplicated(keep="last")]
        v = v[~v.index.duplicated(keep="last")]
        if s.notna().sum() < 300:
            bad.append(code)
            continue
        closes[code], vols[code] = s, v
    return pd.DataFrame(closes).sort_index(), pd.DataFrame(vols).sort_index(), bad


def tick(p):
    """東証の標準的な呼値(TOPIX100以外の標準テーブル)。円。"""
    bounds = [(3000, 1), (5000, 5), (30000, 10), (50000, 50), (300000, 100), (500000, 500),
              (3e6, 1000), (5e6, 5000), (3e7, 10000)]
    t = np.full(np.shape(p), 50000.0)
    for ub, tk in reversed(bounds):
        t = np.where(p <= ub, tk, t)
    return t


def tstat(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 12 or x.std(ddof=1) == 0:
        return np.nan
    return x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))


def ann(x):
    return np.nanmean(x) * 12 * 100  # 年率%(算術)


def main():
    t0 = time.time()
    uni = load_universe()
    C, V, bad = build_panels(uni)
    topix = C["1348"] if "1348" in C else None
    C, V = C.drop(columns=["1348"], errors="ignore"), V.drop(columns=["1348"], errors="ignore")
    print(f"取得: {C.shape[1]}銘柄(失敗/短すぎ {len(bad)})  期間 {C.index[0].date()}〜{C.index[-1].date()}  {time.time()-t0:.0f}s")
    grp = uni.set_index("コード")["grp"].reindex(C.columns)
    print(grp.str.split("/").str[0].value_counts().to_dict())

    # --- 異常値: 日次±50%超は除外(その日の変化を0にして指数化)
    R = C.pct_change(fill_method=None)
    nbad = int((R.abs() > 0.5).sum().sum())
    R = R.where(R.abs() <= 0.5)
    P = (1 + R.fillna(0)).cumprod().where(C.notna())
    P = P.ffill(limit=5)
    print(f"±50%超の日次変化 {nbad}件を除外")

    cal = C.index[C.notna().sum(axis=1) >= 20]
    me = pd.Series(cal, index=cal).groupby([cal.year, cal.month]).last().values
    me = pd.DatetimeIndex(me)
    pos = {d: i for i, d in enumerate(C.index)}
    TV = (C * V)
    avgTV = TV.rolling(60, min_periods=40).mean()
    traded = (V > 0).rolling(60, min_periods=40).sum()
    r1m = P / P.shift(21) - 1
    r1w = P / P.shift(5) - 1
    mom = P.shift(21) / P.shift(252) - 1
    vol60 = R.rolling(60, min_periods=40).std()
    mx = R.rolling(21, min_periods=15).max()
    surge = V.rolling(20, min_periods=15).mean() / V.shift(20).rolling(240, min_periods=120).mean()
    score = {
        "反転1か月(下落銘柄を買う)": -r1m,
        "反転1週間(下落銘柄を買う)": -r1w,
        "モメンタム12-1": mom,
        "低ボラ(60日)": -vol60,
        "宝くじ回避(MAX小)": -mx,
        "出来高急増(20日/過去平均)": surge,
        "低位株(株価が安い)": -C,
        "小型(売買代金が小さい)": -avgTV,
    }
    sig_names = list(score)

    # 月次リターン: 月末の翌営業日の終値 → 翌月末の翌営業日の終値
    ent_idx = [pos[me[k]] + 1 for k in range(len(me))]
    months = []
    for k in range(len(me) - 1):
        a, b = ent_idx[k], ent_idx[k + 1]
        if b >= len(C.index) or me[k] < START:
            continue
        months.append((k, a, b))

    mret, meta = [], []
    Pv = P.values
    Cv = C.values
    for k, a, b in months:
        i = pos[me[k]]
        ok = (traded.values[i] >= 40) & ~np.isnan(Pv[a]) & ~np.isnan(Pv[b]) & ~np.isnan(Cv[i])
        mret.append(Pv[b] / Pv[a] - 1)
        meta.append((me[k], ok, avgTV.values[i], Cv[i]))
    dates = [m[0] for m in meta]
    M = len(dates)
    print(f"月数 {M}  ({dates[0].date()}〜{dates[-1].date()})")

    tp = topix.reindex(C.index).ffill()
    tpx = np.array([tp.values[b] / tp.values[a] - 1 for _, a, b in months])

    def run(sig_df, liq_only, side_pct=0.1, mask_extra=None):
        S = sig_df.values
        out = {"top": [], "bot": [], "uni": [], "ntop": [], "to_top": [], "to_ls": [],
               "c_top": [], "c_ls": [], "c_top_w": [], "c_ls_w": [], "price_top": []}
        prev_top, prev_bot = set(), set()
        for m, (k, a, b) in enumerate(months):
            _, ok, atv, c0 = meta[m]
            i = pos[me[k]]
            elig = ok & ~np.isnan(S[i])
            if liq_only:
                elig &= atv >= LIQ_MIN
            if mask_extra is not None:
                elig &= mask_extra[m]
            idx = np.where(elig)[0]
            r = mret[m]
            idx = idx[~np.isnan(r[idx])]
            if len(idx) < 60:
                for kk in out:
                    out[kk].append(np.nan if kk != "price_top" else None)
                prev_top, prev_bot = set(), set()
                continue
            s = S[i][idx]
            rk = stats.rankdata(s, method="ordinal") / len(s)
            top = idx[rk > 1 - side_pct]
            bot = idx[rk <= side_pct]
            out["top"].append(np.mean(r[top]))
            out["bot"].append(np.mean(r[bot]))
            out["uni"].append(np.mean(r[idx]))
            out["ntop"].append(len(top))
            ts, bs = set(top.tolist()), set(bot.tolist())

            def side_cost(newset, oldset, wide=1.0):
                # 片側コスト = 0.5呼値/株価 × 売買した銘柄のウェイト。回転 = 入れ替え割合(片道)
                ch, gone = newset - oldset, oldset - newset
                n = max(len(newset), 1)
                cs = 0.0
                for j in list(ch) + list(gone):
                    if not np.isnan(c0[j]):
                        cs += 0.5 * wide * float(tick(c0[j])) / c0[j] / n
                return cs, (len(ch) + len(gone)) / (2 * n)
            ct, tt = side_cost(ts, prev_top)
            cb, tb = side_cost(bs, prev_bot)
            ct2, _ = side_cost(ts, prev_top, 2.0)
            cb2, _ = side_cost(bs, prev_bot, 2.0)
            out["c_top"].append(ct)
            out["c_ls"].append(ct + cb)
            out["c_top_w"].append(ct2)
            out["c_ls_w"].append(ct2 + cb2)
            out["to_top"].append(tt)
            out["to_ls"].append((tt + tb) / 2)
            out["price_top"].append(c0[top])
            prev_top, prev_bot = ts, bs
        return {k: (np.array(v, float) if k != "price_top" else v) for k, v in out.items()}

    dts = pd.DatetimeIndex(dates)
    first = np.asarray(dts < SPLIT_H)
    ntests = [0]

    def summarise(o):
        top, bot, uni = o["top"], o["bot"], o["uni"]
        ex = top - uni
        ls = top - bot
        d = {}
        for nm, x, cc, ccw, to in (("long超過", ex, o["c_top"], o["c_top_w"], o["to_top"]),
                                   ("L-S", ls, o["c_ls"], o["c_ls_w"], o["to_ls"])):
            ntests[0] += 1
            xn = x - np.nan_to_num(cc)
            xw = x - np.nan_to_num(ccw)
            xc = xn - COMMISSION * (2 * np.nan_to_num(to) if nm == "long超過" else 4 * np.nan_to_num(to))
            d[nm] = dict(gross=ann(x), t=tstat(x), h1=ann(x[first]), t1=tstat(x[first]), h2=ann(x[~first]),
                         t2=tstat(x[~first]), net_spread=ann(xn), t_net=tstat(xn), net_wide=ann(xw),
                         net_fee=ann(xc), to=np.nanmean(to) * 100)
        d["vsTOPIX"] = ann(top - tpx)
        d["n"] = int(np.nanmean(o["ntop"]))
        return d

    allres = {}
    for liq in (False, True):
        u = "流動性5千万円以上" if liq else "全銘柄"
        for sg in sig_names:
            o = run(score[sg], liq)
            allres[(u, sg)] = (o, summarise(o))
    uni_all = allres[("全銘柄", sig_names[0])][0]["uni"]
    uni_liq = allres[("流動性5千万円以上", sig_names[0])][0]["uni"]
    print("\n=== ベンチマーク(価格のみ、年率%、算術) ===")
    print(f"ユニバース等ウェイト(全) {ann(uni_all):.1f}% / (流動性) {ann(uni_liq):.1f}% / TOPIX(1348) {ann(tpx):.1f}%")
    print(f"  前半(〜2018) 全:{ann(uni_all[first]):.1f}% 流動性:{ann(uni_liq[first]):.1f}% TOPIX:{ann(tpx[first]):.1f}% / 後半 全:{ann(uni_all[~first]):.1f}% 流動性:{ann(uni_liq[~first]):.1f}% TOPIX:{ann(tpx[~first]):.1f}%")
    print(f"  生存者バイアスの目安(等ウェイト−TOPIX): 全 前半 {ann(uni_all[first]-tpx[first]):+.1f}%/年, 後半 {ann(uni_all[~first]-tpx[~first]):+.1f}%/年")

    def show(u):
        print(f"\n--- {u} (年率%。long超過=上位10%−ユニバース等ウェイト。スプ後=1呼値スプレッド控除、広スプ=2呼値、手数料後=スプ後−片道0.1%) ---")
        print(f"{'シグナル':<20}{'型':<8}{'年率':>6}{'t':>6}{'前半':>7}{'t':>6}{'後半':>7}{'t':>6}{'スプ後':>7}{'t':>6}{'広スプ':>7}{'手数料後':>8}{'回転%':>6}{'vsTPX':>7}{'n':>4}")
        for sg in sig_names:
            d = allres[(u, sg)][1]
            for nm in ("long超過", "L-S"):
                x = d[nm]
                vt = d["vsTOPIX"] if nm == "long超過" else float("nan")
                print(f"{sg:<18}{nm:<8}{x['gross']:>6.1f}{x['t']:>6.2f}{x['h1']:>7.1f}{x['t1']:>6.2f}{x['h2']:>7.1f}{x['t2']:>6.2f}{x['net_spread']:>7.1f}{x['t_net']:>6.2f}{x['net_wide']:>7.1f}{x['net_fee']:>8.1f}{x['to']:>6.0f}{vt:>7.1f}{d['n']:>4}")

    show("全銘柄")
    show("流動性5千万円以上")

    # --- 流動性層別
    print("\n=== 流動性3分位(各月の売買代金で分割)ごと: シグナル上位30%−下位30%(L-S)。long超過=上位30%−その層の等ウェイト ===")
    tv_rank = []
    for m in range(M):
        _, ok, atv, c0 = meta[m]
        r = np.full(atv.shape, np.nan)
        idx = np.where(ok & ~np.isnan(atv))[0]
        if len(idx) >= 60:
            r[idx] = stats.rankdata(atv[idx]) / len(idx)
        tv_rank.append(r)
    buckets = {"小(売買代金下位1/3)": lambda r: (r <= 1 / 3), "中": lambda r: (r > 1 / 3) & (r <= 2 / 3),
               "大(上位1/3)": lambda r: r > 2 / 3}
    for bn, fn in buckets.items():
        mask = [fn(r) for r in tv_rank]
        med = np.nanmedian([np.nanmedian(meta[m][2][mask[m]]) for m in range(M) if mask[m].any()]) / 1e6
        print(f"[{bn}] 売買代金の中央値 約{med:.0f}百万円/日")
        print(f"{'シグナル':<20}{'L-S年率':>8}{'t':>6}{'前半':>7}{'後半':>7}{'スプ後':>8}{'t':>6}{'広スプ':>8}{'long超過':>9}{'t':>6}{'回転%':>6}")
        for sg in sig_names:
            o = run(score[sg], False, 0.3, mask)
            ex, ls = o["top"] - o["uni"], o["top"] - o["bot"]
            ntests[0] += 1
            lsn = ls - np.nan_to_num(o["c_ls"])
            lsw = ls - np.nan_to_num(o["c_ls_w"])
            print(f"{sg:<18}{ann(ls):>8.1f}{tstat(ls):>6.2f}{ann(ls[first]):>7.1f}{ann(ls[~first]):>7.1f}{ann(lsn):>8.1f}{tstat(lsn):>6.2f}{ann(lsw):>8.1f}{ann(ex):>9.1f}{tstat(ex):>6.2f}{np.nanmean(o['to_ls'])*100:>6.0f}")

    # --- 1月効果
    print("\n=== 1月効果(小型−大型: 売買代金下位1/3 − 上位1/3 の等ウェイト月次リターン差) ===")
    sm, lg = [], []
    for m in range(M):
        r = mret[m]
        s_ = buckets["小(売買代金下位1/3)"](tv_rank[m]) & ~np.isnan(r)
        l_ = buckets["大(上位1/3)"](tv_rank[m]) & ~np.isnan(r)
        sm.append(np.mean(r[s_]) if s_.sum() > 10 else np.nan)
        lg.append(np.mean(r[l_]) if l_.sum() > 10 else np.nan)
    sm, lg = np.array(sm), np.array(lg)
    diff = sm - lg
    mon = np.array([d.month for d in dts])
    for nm, mk in (("1月(12月末→1月末)", mon == 12), ("1月以外の月", mon != 12)):
        x = diff[mk]
        print(f"{nm}: 小型−大型 平均 {np.nanmean(x)*100:+.2f}%/月 (t={tstat(x):.2f}, n={np.sum(~np.isnan(x))})")
    x, f12 = diff[mon == 12], first[mon == 12]
    print(f"  1月のみ 前半 {np.nanmean(x[f12])*100:+.2f}% (t={tstat(x[f12]):.2f}) / 後半 {np.nanmean(x[~f12])*100:+.2f}% (t={tstat(x[~f12]):.2f})")
    ntests[0] += 1
    print(f"  参考: 小型の1月平均 {np.nanmean(sm[mon==12])*100:+.2f}%, 大型の1月平均 {np.nanmean(lg[mon==12])*100:+.2f}%")

    # --- 多重比較
    N = ntests[0]
    alpha = 0.05 / N
    thr = stats.norm.isf(alpha / 2)
    print(f"\n=== 多重比較 ===\n検証した主要パターン数 N={N}(8シグナル×2ユニバース×2構成=32 + 流動性層別8×3=24 + 1月効果1)。"
          f"Bonferroni 5%: |t| > {thr:.2f}(1検定あたりp<{alpha:.4f})。補正なしの目安は 1.96")
    print("パラメータ(21/5/252/60日、10分位、5千万円、0.5呼値)は事前に固定。探索で選んだ変種はなし。")
    best = []
    for key, (o, d) in allres.items():
        for nm in ("long超過", "L-S"):
            best.append((abs(d[nm]["t"]), key, nm, d[nm]))
    best.sort(key=lambda z: -z[0])
    print("tの絶対値上位(グロス):")
    for a_, key, nm, x in best[:8]:
        print(f"  {key[0]}/{key[1]} {nm}: 年{x['gross']:+.1f}% t={x['t']:.2f} 前半t={x['t1']:.2f} 後半t={x['t2']:.2f} スプレッド後 年{x['net_spread']:+.1f}% t={x['t_net']:.2f} {'<補正後も有意>' if a_ > thr else ''}")

    # --- 必要資金
    print("\n=== 必要資金(10銘柄を100株ずつ) ===")
    p_last = meta[-1][3]
    p_last = p_last[~np.isnan(p_last)]
    q = lambda z: np.percentile(p_last * 100, z)
    print(f"直近(標本{len(p_last)}銘柄)の100株代金(円): 10%点 {q(10):,.0f} / 25% {q(25):,.0f} / 中央値 {q(50):,.0f} / 75% {q(75):,.0f} / 90% {q(90):,.0f}")
    print(f"  10銘柄×100株: 中央値の銘柄なら {q(50)*10:,.0f}円、安い方25%点なら {q(25)*10:,.0f}円")
    for lim in (10000, 30000, 50000, 100000):
        print(f"  100株が{lim:,}円以下の銘柄: {np.mean(p_last*100<=lim)*100:.1f}% ({int((p_last*100<=lim).sum())}銘柄)")
    cheap = p_last[p_last * 100 <= 10000]
    if len(cheap):
        tk = tick(cheap) / cheap * 100
        print(f"  1万円で100株買える低位株(株価100円以下)の呼値/株価: 中央値 {np.median(tk):.2f}%, 最大 {tk.max():.2f}%")
    print("シグナル別: 上位10分位(全銘柄)の直近36か月の100株代金 → 10銘柄合計")
    for sg in sig_names:
        o = allres[("全銘柄", sg)][0]
        ps = [p for p in o["price_top"][-36:] if p is not None]
        if not ps:
            continue
        pp = np.concatenate([np.asarray(p) for p in ps]) * 100
        pp = pp[~np.isnan(pp)]
        print(f"  {sg:<24} 中央値×10 {np.median(pp)*10:>10,.0f}円 / 25%点×10 {np.percentile(pp,25)*10:>10,.0f}円 / 100株1万円以内の割合 {np.mean(pp<=10000)*100:.1f}%")
    print(f"\n実行時間 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
