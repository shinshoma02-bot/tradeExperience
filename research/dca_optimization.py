"""積立(DCA)の方法そのものの最適化検証。新NISA想定(売却益・配当は非課税、買付手数料0円。--fee 0.001 で0.1%を試せる)。

  python research/dca_optimization.py            # research/data/dca_raw.json のキャッシュを使う
  python research/dca_optimization.py --refresh  # Yahoo から取り直す
  python research/dca_optimization.py --fee 0.001 --cash 0.0

設計(全ルール共通の土俵):
- 毎月の初営業日の終値で買う。シグナルは「前営業日の終値」までの情報だけ(ルックアヘッドなし)。
- 全ルール同じ現金口座を持つ: 開始時に予備資金 R=12×月額 を入れ、毎月 B を足し、買付額=min(現金, 倍率×B)。
  使わなかった現金は年利 --cash(既定0.5%)で増える。DCAも予備資金を使わず寝かせる=『持つコスト』は全ルールに公平に入る。
- 結果は月額Bに比例する(1万円でも3万円でも倍率・勝率・IRR差は同じ)。円額は B を掛けるだけ。
- 配当: ^GSPC は価格に年2%を上乗せ(近似)、^N225 は配当なし(価格のみ=やや不利に出る)、VT/VFITX は調整後終値。GC=F は価格のみ。
- 円ベース: S&P(ドル)× USDJPY(1日ラグ)。シグナルはドル建て価格から。
- 窓は開始月を1か月ずつずらした全ローリング窓(重なるので独立ではない)。信頼区間は窓長を1ブロックにした循環ブロックブートストラップ。
- 試したルール数は最後に表示(多重比較: ブロック数が少ないので |z|>3.1 を目安にする)。前半=開始2000年前、後半=開始2000年以降。
"""
import argparse, json, os, sys, time, urllib.request
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "dca_raw.json")
SYMS = ["^GSPC", "^N225", "VT", "USDJPY=X", "VFITX", "GC=F"]
RES_MONTHS = 12          # 予備資金(月額の何か月分)
VA_GROWTH = 0.05         # バリューアベレージングの目標成長率(定番5%)


def fetch(refresh=False):
    if os.path.exists(CACHE) and not refresh:
        return json.load(open(CACHE))
    out, now = {}, int(time.time())
    for s in SYMS:
        u = f"https://query1.finance.yahoo.com/v8/finance/chart/{s}?period1=0&period2={now}&interval=1d"
        r = json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=40))["chart"]["result"][0]
        a = r["indicators"].get("adjclose", [{}])[0].get("adjclose") or r["indicators"]["quote"][0]["close"]
        out[s] = list(zip(r["timestamp"], a))
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    json.dump(out, open(CACHE, "w"))
    return out


def series(raw, s):
    x = pd.Series({pd.Timestamp(t, unit="s").normalize(): v for t, v in raw[s] if v is not None}).sort_index().astype(float)
    x = x[~x.index.duplicated(keep="last")]
    r = x.pct_change().abs()
    bad = r[r > 0.15]
    if len(bad):
        print(f"  [{s}] 日次±15%超 {len(bad)}件(確認): " + ", ".join(f"{d.date()}({x.pct_change()[d]:+.0%})" for d in bad.index[:6]))
    return x


def monthly_frame(px_sig, px_val):
    """px_sig: シグナル用の日次価格, px_val: 評価用の日次価格(同じ日付)。月初の営業日ごとに、前営業日までのシグナルを付ける。"""
    d = pd.DataFrame({"sig": px_sig, "val": px_val}).dropna()
    d["dd"] = d.sig / d.sig.cummax() - 1
    d["below"] = d.sig < d.sig.rolling(200).mean()
    d["ma_ok"] = d.sig.rolling(200).mean().notna()
    prev = d[["dd", "below", "ma_ok"]].shift(1)
    d["dd"], d["below"], d["ma_ok"] = prev.dd, prev.below, prev.ma_ok
    first = d.groupby([d.index.year, d.index.month]).head(1).iloc[1:]   # 前日情報がない最初の月は捨てる
    return first


def multipliers(m):
    dd, below, ok = m.dd.values, m.below.values.astype(bool), m.ma_ok.values.astype(bool)
    hi = dd >= -1e-12
    return {
        "dd10:-10%で1.5倍": np.where(dd <= -0.10, 1.5, 1.0),
        "dd20:-20%で2倍": np.where(dd <= -0.20, 2.0, 1.0),
        "ladder:-10%1.5倍/-20%2倍": np.where(dd <= -0.20, 2.0, np.where(dd <= -0.10, 1.5, 1.0)),
        "MA200下で1.5倍": np.where(ok & below, 1.5, 1.0),
        "高値更新中は0.5倍": np.where(hi, 0.5, 1.0),
        "高値更新中は停止": np.where(hi, 0.0, 1.0),
    }


def sim_window(P, mult, s, H, cr, fee, va=False, ls=False, inv0=False):
    """開始 s から H か月。戻り値: (最終資産, キャッシュフロー配列[月0..H])。月額B=1。"""
    cash, sh = float(RES_MONTHS), 0.0
    flows = np.zeros(H + 1)
    flows[0] -= RES_MONTHS
    gm = (1 + VA_GROWTH) ** (1 / 12) - 1
    if inv0:   # 対照: 予備資金も最初に全額投資(=現金を持たないDCA)
        sh = RES_MONTHS * (1 - fee) / P[s]; cash = 0.0
    if ls:
        sh = H * (1 - fee) / P[s]
        flows[0] -= H
        W = sh * P[s + H] + RES_MONTHS * (1 + cr) ** H
        flows[H] = W
        return W, flows
    for k in range(H):
        t = s + k
        cash = cash * (1 + cr) + 1.0
        flows[k] -= 1.0
        if va:
            target = ((1 + gm) ** (k + 1) - 1) / gm
            want = min(max(target - sh * P[t], 0.0), 2.0)
        else:
            want = mult[t]
        a = min(cash, want)
        sh += a * (1 - fee) / P[t]
        cash -= a
    W = sh * P[s + H] + cash * (1 + cr)   # 最終月の現金利息は片側のみ=誤差
    flows[H] = W
    return W, flows


def irr_monthly(F):
    """F: (n, H+1) のキャッシュフロー。月次IRRを二分法で解き年率に。"""
    lo, hi = np.full(len(F), -0.5), np.full(len(F), 0.5)
    k = np.arange(F.shape[1])
    for _ in range(60):
        mid = (lo + hi) / 2
        npv = (F / (1 + mid)[:, None] ** k).sum(1)
        up = npv > 0          # NPV>0 → 利回りはもっと高い
        lo = np.where(up, mid, lo); hi = np.where(up, hi, mid)
    return (1 + (lo + hi) / 2) ** 12 - 1


def block_boot(d, L, n_rep=3000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(d); nb = max(1, int(np.ceil(n / L)))
    dd = np.concatenate([d, d])  # 循環
    means = np.empty(n_rep)
    for i in range(n_rep):
        st = rng.integers(0, n, nb)
        means[i] = np.concatenate([dd[s:s + L] for s in st])[:n].mean()
    return np.percentile(means, [2.5, 97.5]), means.std()


def run_rules(name, m_sig, P, H, args, starts_dates):
    mults = multipliers(m_sig)
    n = len(P) - H
    if n < 12:
        print(f"  {name} {H//12}年: 窓が{n}個で不足"); return {}
    rules = {"DCA": None}
    rules.update(mults); rules["VA(5%目標,上限2倍)"] = "va"; rules["一括(最初に全額)"] = "ls"; rules["対照:予備を即投資のDCA"] = "inv0"
    res = {}; rand_ctrl = {}
    for rn, mu in rules.items():
        Ws, Fs = [], []
        for s in range(n):
            if rn == "VA(5%目標,上限2倍)": W, F = sim_window(P, None, s, H, args.cash / 12, args.fee, va=True)
            elif rn.startswith("対照"): W, F = sim_window(P, np.ones(len(P)), s, H, args.cash / 12, args.fee, inv0=True)
            elif rn.startswith("一括"): W, F = sim_window(P, None, s, H, args.cash / 12, args.fee, ls=True)
            else: W, F = sim_window(P, mu if mu is not None else np.ones(len(P)), s, H, args.cash / 12, args.fee)
            Ws.append(W); Fs.append(F)
        res[rn] = (np.array(Ws), irr_monthly(np.array(Fs)))
        if mu is not None and not isinstance(mu, str) and rn in mults:   # 乱数対照: 同じ頻度・同じ倍率をランダムな月に
            rng = np.random.default_rng(1); pr = np.mean(mu != 1.0); v = mu[mu != 1.0].mean() if pr > 0 else 1.0
            acc = np.zeros(n)
            for _ in range(6):
                rm = np.where(rng.random(len(P)) < pr, v, 1.0)
                acc += np.array([sim_window(P, rm, s, H, args.cash / 12, args.fee)[0] for s in range(n)])
            rand_ctrl[rn] = acc / 6
    base_W, base_I = res["DCA"]
    contrib = RES_MONTHS + H
    yrs = np.array(starts_dates[:n])
    half1, half2 = yrs < 2000, yrs >= 2000
    dca_ret = base_W / contrib
    print(f"\n### {name} / {H//12}年 窓{n}個(開始 {starts_dates[0]}〜{starts_dates[n-1]}), ブロック≈{n/H:.1f}個")
    print(f"DCA 最終資産/拠出(予備資金込): 中央値 {np.median(dca_ret):.2f}倍, 最悪 {dca_ret.min():.2f}倍, 最良 {dca_ret.max():.2f}倍; "
          f"IRR(年) 中央値 {np.median(base_I)*100:.1f}% 最悪 {base_I.min()*100:.1f}% ; 拠出を割った窓 {np.mean(dca_ret<1)*100:.0f}% ; "
          f"月1万円なら中央 {np.median(base_W)*1e4/1e4:.0f}万円(拠出{contrib}万), 3万円なら {np.median(base_W)*3:.0f}万円")
    print(f"{'ルール':26s}{'平均比':>7s}{'中央比':>7s}{'勝率':>6s}{'最悪比':>7s}{'最良比':>7s}{'logCI95':>16s}{'z':>6s}{'IRR差pt':>8s}{'乱数同頻度比':>10s}{'前半比/勝':>12s}{'後半比/勝':>12s}")
    out = {}
    for rn, (W, I) in res.items():
        if rn == "DCA": continue
        ratio = W / base_W; d = np.log(ratio)
        ci, se = block_boot(d, H)
        few = n / H < 2.5      # 独立ブロックが少なすぎると区間は当てにならない
        z = d.mean() / se if (se > 0 and not few) else np.nan
        cs = f"[{ci[0]:+.3f},{ci[1]:+.3f}]" if not few else "[ブロック不足]"
        rc = f"{(rand_ctrl[rn]/base_W).mean():10.3f}" if rn in rand_ctrl else f"{'-':>10s}"
        def hf(mask): return (f"{ratio[mask].mean():.3f}/{(ratio[mask] > 1).mean()*100:.0f}%" if mask.sum() > 5 else "-")
        print(f"{rn:26s}{ratio.mean():7.3f}{np.median(ratio):7.3f}{(ratio>1).mean()*100:5.0f}%{ratio.min():7.3f}{ratio.max():7.3f}"
              f"  {cs:>14s}{z:6.1f}{(I-base_I).mean()*100:8.2f}{rc}{hf(half1):>12s}{hf(half2):>12s}")
        out[rn] = (ratio.mean(), (ratio > 1).mean(), z)
    return out


def rebalance_study(raw, args):
    """株70/債30(VFITX 1991〜)または株70/金30(GC=F 2000〜)。新規積立のみのリバランス vs 年1回の売買リバランス vs 放置。月額B=1、予備資金なし。"""
    sp = series(raw, "^GSPC"); sp = sp * np.exp(0.02 * (sp.index - sp.index[0]).days / 365.25)
    print("\n## 5. リバランス(株は^GSPC+配当2%近似、ドル建て)  ※売買リバランスは売買額の0.1%を手数料として計上(NISA内は非課税)")
    for alt_name, alt in (("債 VFITX", series(raw, "VFITX")), ("金 GC=F", series(raw, "GC=F"))):
        d = pd.DataFrame({"s": sp, "a": alt}).dropna()
        m = d.groupby([d.index.year, d.index.month]).head(1)
        S, A = m.s.values, m.a.values
        yrs = m.index.year.values
        for H in (120, 240):
            n = len(S) - H
            if n < 12: continue
            rows = {k: [] for k in ("株100%DCA", "70/30放置(配分通り積立)", "70/30新規資金で調整", "70/30年1回売買調整")}
            under = {k: [] for k in rows}
            for s0 in range(n):
                for k in rows:
                    hs, ha, contrib, mn = 0.0, 0.0, 0.0, 9.0
                    for t in range(s0, s0 + H):
                        if k == "株100%DCA": hs += 1 / S[t]
                        elif k == "70/30放置(配分通り積立)": hs += 0.7 / S[t]; ha += 0.3 / A[t]
                        else:
                            vs, va_ = hs * S[t], ha * A[t]
                            if k == "70/30新規資金で調整":
                                tot = vs + va_ + 1.0
                                need_s = max(0.7 * tot - vs, 0.0); need_a = max(0.3 * tot - va_, 0.0)
                                f = 1.0 / (need_s + need_a); hs += need_s * f / S[t]; ha += need_a * f / A[t]
                            else:
                                hs += 0.7 / S[t]; ha += 0.3 / A[t]
                                if (t - s0) % 12 == 11:   # 年1回
                                    vs, va_ = hs * S[t], ha * A[t]; tot = vs + va_
                                    traded = abs(vs - 0.7 * tot)
                                    tot -= traded * args.fee if args.fee else traded * 0.001
                                    hs, ha = 0.7 * tot / S[t], 0.3 * tot / A[t]
                        contrib += 1
                        mn = min(mn, (hs * S[t] + ha * A[t]) / contrib)
                    rows[k].append(hs * S[s0 + H] + ha * A[s0 + H]); under[k].append(mn)
            base = np.array(rows["70/30放置(配分通り積立)"])
            print(f"\n### 株70/{alt_name[0]}30 / {H//12}年 窓{n}個(開始{yrs[0]}〜{yrs[n-1]}, ブロック≈{n/H:.1f}個)")
            print(f"{'方式':26s}{'拠出比(中央)':>12s}{'放置比(平均)':>12s}{'勝率':>6s}{'最悪窓比':>9s}{'logCI95(放置比)':>20s}{'z':>6s}{'最悪の含み(値/拠出)':>20s}")
            for k in rows:
                W = np.array(rows[k]); r = W / base; dl = np.log(r)
                ci, se = block_boot(dl, H) if k != "70/30放置(配分通り積立)" else (np.array([0, 0]), 1)
                z = dl.mean() / se if k != "70/30放置(配分通り積立)" else 0
                print(f"{k:26s}{np.median(W)/H:12.2f}{r.mean():12.3f}{(r>1).mean()*100:5.0f}%{r.min():9.3f}  [{ci[0]:+.3f},{ci[1]:+.3f}]{z:6.1f}{np.min(under[k]):14.2f}(中央{np.median(under[k]):.2f})")


def lost_decades(raw, args):
    print("\n## 6. 日経225『失われた30年』: 月額1(予備資金なし)の素のDCA。配当なし(価格のみ)で、実際より不利な点に注意")
    n225 = series(raw, "^N225"); sp = series(raw, "^GSPC"); sp = sp * np.exp(0.02 * (sp.index - sp.index[0]).days / 365.25)
    vt = series(raw, "VT")
    for nm, x in (("日経225(価格)", n225), ("日経225(配当年2%を仮に上乗せ)", n225 * np.exp(0.02 * (n225.index - n225.index[0]).days / 365.25)), ("S&P500(配当2%近似)", sp), ("VT 世界株", vt)):
        m = x.groupby([x.index.year, x.index.month]).head(1); P = m.values; yrs = m.index.year.values
        line = f"{nm:30s}"
        for H in (120, 240, 360):
            n = len(P) - H
            if n < 12: line += f" | {H//12}年: 窓不足"; continue
            res = []
            for s in range(n):
                sh = np.sum(1 / P[s:s + H]); W = sh * P[s + H]
                F = np.full(H + 1, -1.0); F[H] = W; res.append((W / H, F))
            ratio = np.array([r[0] for r in res]); I = irr_monthly(np.array([r[1] for r in res]))
            line += f" | {H//12}年 窓{n} 最終/拠出 中央{np.median(ratio):.2f} 最悪{ratio.min():.2f} 元本割れ{np.mean(ratio<1)*100:.0f}% IRR中央{np.median(I)*100:.1f}%"
        print(line)
    # 1990年開始のDCAの軌跡
    x = n225; m = x.groupby([x.index.year, x.index.month]).head(1)
    for start in (1989, 1990, 1995, 2000):
        P = m[m.index.year >= start].values
        if len(P) < 60: continue
        sh = np.cumsum(1 / P); val = sh * P; con = np.arange(1, len(P) + 1)
        yr_idx = [i for i in range(len(P)) if (m[m.index.year >= start].index[i].month == 12)]
        pts = ", ".join(f"{m[m.index.year >= start].index[i].year}:{val[i]/con[i]:.2f}" for i in yr_idx[4::5])
        print(f"  日経 {start}年から毎月DCA(価格のみ)の 時価/拠出 の推移: {pts}")
    # 円建て S&P (1996〜) の10年窓
    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--fee", type=float, default=0.0)
    ap.add_argument("--cash", type=float, default=0.005, help="現金の年利")
    args = ap.parse_args()
    raw = fetch(args.refresh)
    print(f"設定: 買付手数料 {args.fee*100:.2f}%, 現金利回り {args.cash*100:.1f}%, 予備資金 {RES_MONTHS}か月分, 税: NISA想定で非課税")
    spx = series(raw, "^GSPC"); spx = spx * np.exp(0.02 * (spx.index - spx.index[0]).days / 365.25)
    n225 = series(raw, "^N225"); vt = series(raw, "VT"); fx = series(raw, "USDJPY=X")
    fxl = fx.shift(1)   # 1日ラグ
    spx_jpy = (spx * fxl.reindex(spx.index).ffill()).dropna()
    spx_jpy = spx_jpy[spx_jpy.index >= fx.index[0] + pd.Timedelta(days=5)]
    sets = [("S&P500 ドル建て(1950〜)", spx, spx), ("日経225 価格のみ(1965〜)", n225, n225),
            ("S&P500 円建て(1996〜)", spx, spx_jpy), ("VT 世界株 ドル建て(2008〜)", vt, vt)]
    summary = {}
    for name, sig, val in sets:
        m = monthly_frame(sig, val.reindex(sig.index).ffill() if len(val) != len(sig) else val)
        P = m.val.values; years = m.index.year.values
        for H in (120, 240):
            summary[(name, H)] = run_rules(name, m, P, H, args, list(years))
    rebalance_study(raw, args)
    lost_decades(raw, args)
    nrules = 6 + 1 + 1 + 0
    print(f"\n試したルール数: 積立ルール{nrules}(閾値は定番値のみ固定)×データ4×期間2 = {nrules*8}検定 + リバランス3方式×2資産×2期間 = 12検定。"
          "窓は重なるのでブロック数(≈窓数/窓長)が実質の標本数。|z|>3.1(≈Bonferroni)を超えない差は『DCAと区別できない』と読む。")


if __name__ == "__main__":
    main()
