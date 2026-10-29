"""短期FX(時間帯・仲値・ギャップ・ロンドンフィックス)の検証。

実行: python research/fx_intraday.py   (Yahoo Finance から取得。キャッシュは /tmp 配下 or FX_CACHE)
方針:
- 全てカレンダー/時刻で決まるルール(ルックアヘッド無し)。ギャップ逆張りのみ、ギャップを見てから
  「次の足の始値」で入る。約定は足の始値(スプレッドは片道c bpを往復で控除して感度分析)。
- 当日中に手仕舞い(スワップ無し)。時刻は JST、米英の夏時間は tz で厳密に処理。
- 試したパターン数を数え、Bonferroni で補正した閾値と比べる。
- 前半(期間の前半)で選び、後半で確かめる。
"""
import json
import os
import subprocess
import sys
from statistics import NormalDist

import numpy as np
import pandas as pd

CACHE = os.environ.get("FX_CACHE", "/tmp/claude-0/-home-user-tradeExperience/ce681a55-ab87-515d-97f7-3cd9e02c3837/scratchpad/fxcache")
PAIRS = ["USDJPY=X", "EURJPY=X", "AUDJPY=X", "GBPJPY=X", "EURUSD=X"]
COSTS = [0.3, 0.5, 1.0]           # 片道スプレッド(bp)。往復はその2倍
N_TRIALS = 0                      # 試したパターン数(随時加算)


def fetch(sym, iv, rg):
    os.makedirs(CACHE, exist_ok=True)
    f = f"{CACHE}/{sym}_{iv}_{rg}.json"
    if not os.path.exists(f):
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval={iv}&range={rg}"
        out = subprocess.check_output(["curl", "-s", "-A", "Mozilla/5.0", url])
        open(f, "wb").write(out)
    r = json.load(open(f))["chart"]["result"][0]
    d = pd.DataFrame(r["indicators"]["quote"][0], index=pd.to_datetime(r["timestamp"], unit="s"))
    d = d[["open", "high", "low", "close"]].dropna()
    return d[~d.index.duplicated()]


def clean_intraday(d, maxjump=0.02):
    """1時間(or 15分)足の異常値除去: 足内/足間で2%超の動きは欠損扱いで落とす。"""
    bad = ((d.close / d.open - 1).abs() > maxjump) | ((d.open / d.close.shift() - 1).abs() > 0.05)
    return d[~bad], int(bad.sum())


def load_hourly():
    out = {}
    for s in PAIRS:
        d, nbad = clean_intraday(fetch(s, "1h", "730d"))
        d = d.copy()
        d["jst"] = d.index + pd.Timedelta(hours=9)
        utc = d.index.tz_localize("UTC")
        d["lon"] = utc.tz_convert("Europe/London").tz_localize(None)
        d["ny"] = utc.tz_convert("America/New_York").tz_localize(None)
        d["bp"] = (d.open.shift(-1) / d.open - 1) * 1e4        # 始値→次の始値
        gap = d.index.to_series().diff().shift(-1) != pd.Timedelta(hours=1)
        d.loc[gap.values, "bp"] = np.nan                         # 次の足が連続でない(週末/欠損)ものは除外
        out[s] = d
        print(f"{s}: {len(d)}本 {d.index[0]}〜{d.index[-1]} UTC, 異常値除去 {nbad}本")
    return out


# ---------- 統計 ----------
def tstat(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 5 or x.std(ddof=1) == 0:
        return np.nan
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))


def maxdd(x):
    c = np.cumsum(np.nan_to_num(np.asarray(x, float)))
    return float((c - np.maximum.accumulate(c)).min()) if len(c) else np.nan


def bonf_z(n, alpha=0.05):
    return NormalDist().inv_cdf(1 - alpha / n / 2)


def summarize(x, dates, label, sign=1):
    """x: 1取引あたりbp(グロス)。sign=-1 なら売り。dates: 取引日(前半/後半の分割用)。"""
    x = pd.Series(np.asarray(x, float) * sign, index=pd.to_datetime(dates)).dropna().sort_index()
    mid = x.index[0] + (x.index[-1] - x.index[0]) / 2
    a, b = x[x.index <= mid], x[x.index > mid]
    row = dict(label=label, n=len(x), gross=x.mean(), t=tstat(x), h1=a.mean(), t1=tstat(a), h2=b.mean(), t2=tstat(b))
    for c in COSTS:
        row[f"net{c}"] = x.mean() - 2 * c
    row["dd_net0.5"] = maxdd(x.values - 1.0)
    return row


def show(rows, cols=None):
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    print(df.round(2).to_string(index=False))


# ---------- 日本の祝日(2010年以降の法令に準拠した簡易実装) ----------
def nth_weekday(y, m, wd, n):
    d = pd.Timestamp(y, m, 1)
    d += pd.Timedelta(days=(wd - d.dayofweek) % 7 + 7 * (n - 1))
    return d


def jp_holidays(y):
    h = {pd.Timestamp(y, 1, 1), nth_weekday(y, 1, 0, 2), pd.Timestamp(y, 2, 11), pd.Timestamp(y, 4, 29),
         pd.Timestamp(y, 5, 3), pd.Timestamp(y, 5, 4), pd.Timestamp(y, 5, 5), pd.Timestamp(y, 11, 3), pd.Timestamp(y, 11, 23)}
    if y >= 2020 and y != 2019:
        h.add(pd.Timestamp(y, 2, 23))
    if y <= 2018:
        h.add(pd.Timestamp(y, 12, 23))
    k = y - 1980
    h.add(pd.Timestamp(y, 3, int(20.8431 + 0.242194 * k - k // 4)))
    h.add(pd.Timestamp(y, 9, int(23.2488 + 0.242194 * k - k // 4)))
    h.add(nth_weekday(y, 9, 0, 3))
    if y == 2020: h |= {pd.Timestamp(y, 7, 23), pd.Timestamp(y, 7, 24), pd.Timestamp(y, 8, 10)}
    elif y == 2021: h |= {pd.Timestamp(y, 7, 22), pd.Timestamp(y, 7, 23), pd.Timestamp(y, 8, 9)}
    else:
        h.add(nth_weekday(y, 7, 0, 3)); h.add(nth_weekday(y, 10, 0, 2))
        if y >= 2016: h.add(pd.Timestamp(y, 8, 11))
    if y == 2019: h |= {pd.Timestamp(y, 4, 30), pd.Timestamp(y, 5, 1), pd.Timestamp(y, 5, 2), pd.Timestamp(y, 10, 22)}
    for d in sorted(h):                                   # 振替休日
        if d.dayofweek == 6:
            x = d + pd.Timedelta(days=1)
            while x in h: x += pd.Timedelta(days=1)
            h.add(x)
    for d in sorted(h):                                   # 国民の休日(祝日に挟まれた平日)
        x = d + pd.Timedelta(days=1)
        if x not in h and x + pd.Timedelta(days=1) in h and x.dayofweek < 6:
            h.add(x)
    return h


def is_bizday(d, HOL):
    if d.dayofweek >= 5 or d in HOL: return False
    if (d.month, d.day) in [(1, 2), (1, 3), (12, 31)]: return False       # 銀行休業日
    return True


def gotobi_set(y0, y1):
    HOL = set()
    for y in range(y0 - 1, y1 + 2): HOL |= jp_holidays(y)
    out = set()
    for y in range(y0, y1 + 1):
        for m in range(1, 13):
            for dd in (5, 10, 15, 20, 25, 30):
                try: d = pd.Timestamp(y, m, dd)
                except ValueError: continue          # 2月30日は無し(月末にはずらさない)
                while not is_bizday(d, HOL): d -= pd.Timedelta(days=1)
                out.add(d)
    return out, HOL


# ---------- 1. 時間帯別 ----------
def hour_of_day(H):
    global N_TRIALS
    rows = []
    for tz in ("jst", "lon"):
        for s, d in H.items():
            x = d.dropna(subset=["bp"]).copy()
            x["hr"] = x[tz].dt.hour; x["date"] = x[tz].dt.normalize()
            for hr in range(24):
                g = x[x.hr == hr]
                if len(g) < 100: continue
                N_TRIALS += 1
                r = summarize(g.bp.values, g.date.values, f"{tz}|{s}|{hr:02d}h")
                r["tz"] = tz; r["pair"] = s; r["hr"] = hr; r["_g"] = g
                rows.append(r)
    return rows


def part1(H):
    print("\n===== 1. 時間帯別(1時間保有、始値→次の始値、当日中に手仕舞い) =====")
    rows = hour_of_day(H)
    zb = bonf_z(len(rows))
    print(f"試した(時間帯×通貨ペア×時刻基準 jst/ロンドン現地)= {len(rows)}通り。Bonferroni(5%)のt閾値 = {zb:.2f}")
    full = sorted(rows, key=lambda r: -abs(r["t"]))
    print("全期間 |t| 上位6(符号は買い基準):")
    show([{k: v for k, v in r.items() if not k.startswith("_") and k not in ("tz", "pair", "hr")} for r in full[:6]])
    nz = sum(abs(r["t"]) > 1.96 for r in rows)
    print(f"|t|>1.96 は {nz}/{len(rows)} 通り(偶然の期待値 {0.05*len(rows):.1f}通り), Bonferroni通過 {sum(abs(r['t'])>zb for r in rows)}通り")
    # 前半で選び後半で確認
    print("前半|t|上位8を、符号(買い/売り)を前半で決めて後半に適用:")
    sel = sorted(rows, key=lambda r: -abs(r["t1"]))[:8]
    out = []
    for r in sel:
        sg = 1 if r["h1"] > 0 else -1
        g = r["_g"]; x = pd.Series(g.bp.values * sg, index=pd.to_datetime(g.date.values)).sort_index()
        mid = x.index[0] + (x.index[-1] - x.index[0]) / 2
        b = x[x.index > mid]
        out.append(dict(label=r["label"], dir="買" if sg > 0 else "売", h1_bp=r["h1"] * sg, t1=r["t1"] * sg,
                        h2_bp=b.mean(), t2=tstat(b), h2_net0_5=b.mean() - 1.0, n2=len(b)))
    show(out)
    keep = sum(1 for o in out if o["h2_bp"] > 0 and o["t2"] > 1.96)
    print(f"→ 後半でも正かつ|t|>1.96 は {keep}/8 通り")
    # DST別(JST基準の上位1つ)
    return rows


def sessions(H):
    global N_TRIALS
    print("\n--- セッション別(現地時刻の固定窓、保有=窓の長さ、始値→終了時刻の始値) ---")
    wins = {"東京 9-15 JST": ("jst", 9, 15), "ロンドン 8-16 現地": ("lon", 8, 16), "NY 8-16 現地": ("ny", 8, 16)}
    rows = []
    for nm, (tz, a, b) in wins.items():
        for s, d in H.items():
            x = d.copy(); x["hr"] = x[tz].dt.hour; x["date"] = x[tz].dt.normalize()
            ent = x[x.hr == a]; ex = x[x.hr == b].set_index("date").open
            e = ent.set_index("date").open
            r = ((ex.reindex(e.index) / e - 1) * 1e4).dropna()
            r = r[r.index.dayofweek < 5]
            N_TRIALS += 1
            rows.append(summarize(r.values, r.index, f"{nm}|{s}"))
    show(rows)
    print(f"(15通り, Bonferroni t閾値 {bonf_z(15):.2f})")


# ---------- 2. 仲値 ----------
def part2(H):
    global N_TRIALS
    print("\n===== 2. 仲値(五十日)効果 =====")
    gs, HOL = gotobi_set(2024, 2026)
    rows = []; keep = {}
    for s in ["USDJPY=X", "EURJPY=X", "AUDJPY=X", "GBPJPY=X"]:
        d = H[s]
        for a, b, nm in [(8, 10, "8:00→10:00"), (9, 10, "9:00→10:00")]:
            x = d.copy(); x["hr"] = x.jst.dt.hour; x["date"] = x.jst.dt.normalize()
            e = x[x.hr == a].set_index("date").open
            ex = x[x.hr == b].set_index("date").open
            r = ((ex.reindex(e.index) / e - 1) * 1e4).dropna()
            r = r[[is_bizday(i, HOL) for i in r.index]]
            isg = np.array([i in gs for i in r.index])
            g, o = r[isg], r[~isg]
            N_TRIALS += 1
            diff = g.mean() - o.mean()
            se = np.sqrt(g.var(ddof=1) / len(g) + o.var(ddof=1) / len(o))
            sg = summarize(g.values, g.index, f"{s}|{nm}|五十日 買い")
            sg["other_mean"] = o.mean(); sg["diff"] = diff; sg["t_diff"] = diff / se
            rows.append(sg); keep[(s, nm)] = (g, o)
    zb = bonf_z(8 + 2)
    show(rows)
    print(f"(JST 8-10時。五十日{len(keep[('USDJPY=X','8:00→10:00')][0])}日 / 通常日{len(keep[('USDJPY=X','8:00→10:00')][1])}日。"
          f"仲値系8通り+日足2通り=10通りでBonferroni t閾値 {zb:.2f}。他の日のコスト控除後は net列を参照)")
    g, o = keep[("USDJPY=X", "8:00→10:00")]
    print(f"USDJPY 8→10: 通常日(買い)平均 {o.mean():.2f}bp (t={tstat(o):.2f})。五十日平均 {g.mean():.2f}bp")
    # 15分足で9:55に近い出口(60日のみ・参考)
    d15, _ = clean_intraday(fetch("USDJPY=X", "15m", "60d"))
    d15 = d15.copy(); d15["jst"] = d15.index + pd.Timedelta(hours=9)
    d15["date"] = d15.jst.dt.normalize(); d15["hm"] = d15.jst.dt.hour * 100 + d15.jst.dt.minute
    e = d15[d15.hm == 800].set_index("date").open; ex = d15[d15.hm == 950].set_index("date").open   # 9:50足の始値≒9:50
    r = ((ex.reindex(e.index) / e - 1) * 1e4).dropna(); r = r[[is_bizday(i, HOL) for i in r.index]]
    isg = np.array([i in gs for i in r.index])
    print(f"[参考・15分足60日] USDJPY 8:00→9:50: 五十日 n={isg.sum()} 平均 {r[isg].mean():.2f}bp / 通常日 n={(~isg).sum()} 平均 {r[~isg].mean():.2f}bp(標本小さく結論不可)")
    # 日足15年
    dd = fetch("USDJPY=X", "1d", "max")
    dd.index = dd.index.normalize()
    ret = dd.close.pct_change() * 1e4
    big = ret.abs() > 500
    print(f"日足 USDJPY {dd.index[0].date()}〜{dd.index[-1].date()}: ±5%超 {int(big.sum())}本を除外")
    ret = ret[~big].dropna(); ret = ret[ret.index >= "2010-01-01"]
    gs2, HOL2 = gotobi_set(2010, 2026)
    drows = []
    for nm, rr in [("当日の日足(JST9:00起点の24h)", ret), ("前日の日足(五十日の9:00 JSTで終わる24h)", ret.shift(-1).reindex(ret.index))]:
        pass
    # 五十日が属する足: 当日足(その日)と、仲値直前に終わる足(=前の取引日の足)
    idx = ret.index
    isg_today = np.array([i in gs2 for i in idx])
    prev_is_g = np.zeros(len(idx), bool)
    pos = {d: k for k, d in enumerate(idx)}
    gl = sorted(gs2)
    for gd in gl:
        k = idx.searchsorted(gd, side="left")
        if k > 0 and k < len(idx) and idx[k] == gd: prev_is_g[k - 1] = True
        elif k > 0 and k < len(idx): prev_is_g[k - 1] = True
    for nm, mask in [("五十日当日の足", isg_today), ("五十日の仲値で終わる足(前営業日)", prev_is_g)]:
        N_TRIALS += 1
        for per, sl in [("全期間2010-", slice(None)), ("2010-2017", ret.index < "2018-01-01"), ("2018-", ret.index >= "2018-01-01")]:
            m = np.ones(len(idx), bool) if isinstance(sl, slice) else sl
            a, b = ret[mask & m], ret[~mask & m]
            diff = a.mean() - b.mean(); se = np.sqrt(a.var() / len(a) + b.var() / len(b))
            drows.append(dict(定義=nm, 期間=per, n_五十日=len(a), 五十日bp=a.mean(), 通常bp=b.mean(), 差bp=diff, t=diff / se))
    show(drows)
    print("(日足Yahooの日付境界は厳密に東京時間ではなく、日足は粗い診断。1時間足の結果を主とする)")
    return keep


# ---------- 3. 週明け・金曜・ロンドンフィックス ----------
def part3(H):
    global N_TRIALS
    print("\n===== 3. 週明けギャップ・月曜寄り・金曜引け・ロンドンフィックス =====")
    rows = []
    for s, d in H.items():
        x = d.copy()
        newwk = (x.index.to_series().diff() > pd.Timedelta(hours=40)).values
        ix = np.where(newwk)[0]
        recs = []
        for i in ix:
            if i + 1 >= len(x) or i == 0: continue
            gap = (x.open.iloc[i] / x.close.iloc[i - 1] - 1) * 1e4
            # シグナル(ギャップ)は最初の足の始値で判明 → 約定は2本目の始値
            e = x.open.iloc[i + 1]
            jst = x.jst.iloc[i + 1]
            tgt = jst.normalize() + pd.Timedelta(hours=15)             # 月曜 15:00 JST(東京引け)で手仕舞い
            if tgt <= jst: continue
            hit = x[(x.jst == tgt)]
            if hit.empty: continue
            xo = hit.open.iloc[0]
            recs.append((x.index[i].normalize(), gap, (xo / e - 1) * 1e4, (xo / x.open.iloc[i] - 1) * 1e4))
        r = pd.DataFrame(recs, columns=["date", "gap", "ret_after", "ret_from_open"]).set_index("date")
        # (a) ギャップ逆張り: -sign(gap)×(2本目始値→月曜15時JST)
        N_TRIALS += 1
        sgn = -np.sign(r.gap)
        rows.append(summarize((r.ret_after * sgn).values, r.index, f"ギャップ逆張り|{s}"))
        # (b) 月曜の寄り(最初の足の始値)で買い、15:00 JSTで手仕舞い(寄りで約定できると仮定=甘め)
        N_TRIALS += 1
        rows.append(summarize(r.ret_from_open.values, r.index, f"月曜寄り買い→15時JST|{s}"))
        # (c) 金曜: NY現地12:00始値で買い→その週の最終足の終値(NY17時)
        x["nyh"] = x.ny.dt.hour; x["nyd"] = x.ny.dt.normalize()
        fr = x[x.ny.dt.dayofweek == 4]
        recs2 = []
        for dte, g in fr.groupby("nyd"):
            e = g[g.nyh == 12]
            if e.empty: continue
            recs2.append((dte, (g.close.iloc[-1] / e.open.iloc[0] - 1) * 1e4))
        r2 = pd.DataFrame(recs2, columns=["date", "ret"]).set_index("date")
        N_TRIALS += 1
        rows.append(summarize(r2.ret.values, r2.index, f"金曜NY12時→引け買い|{s}"))
        # (d) ロンドンフィックス(16:00 London): 直前1時間(15→16)、直後1時間(16→17)
        x["lh"] = x.lon.dt.hour; x["ld"] = x.lon.dt.normalize()
        wd = x[(x.lon.dt.dayofweek < 5) & x.bp.notna()]
        for hh, nm in [(15, "フィックス前1h(15→16)"), (16, "フィックス後1h(16→17)")]:
            g = wd[wd.lh == hh]
            N_TRIALS += 1
            rows.append(summarize(g.bp.values, g.ld.values, f"{nm}買い|{s}"))
    show(rows)
    print(f"(25通り, Bonferroni t閾値 {bonf_z(25):.2f}。ロンドンは現地時刻基準=冬JST24時/夏23時が16:00に相当)")
    return rows


# ---------- 円換算 ----------
def yen(bp_per_trade, trades_per_year, price=150.0, capital=10000, lev=25):
    notional = capital * lev
    units = int(notional // price)            # 1通貨単位ずつ
    per_trade_yen = units * price * bp_per_trade / 1e4
    return units, per_trade_yen, per_trade_yen * trades_per_year


def main():
    H = load_hourly()
    r1 = part1(H)
    sessions(H)
    part2(H)
    part3(H)
    print(f"\n===== 試したパターン総数(多重比較の母数)= {N_TRIALS}  → Bonferroni t閾値 {bonf_z(N_TRIALS):.2f}(5%) =====")
    print("円換算の目安: 元手1万円×25倍=約25万円分(≒USDJPY 1,600通貨)。1bp=約25円/取引。")
    for bp in (0.5, 1, 2, 5):
        u, p, yr = yen(bp, 250)
        print(f"  純利益 {bp}bp/日 ×250日 → 1日 約{p:.0f}円 / 年 約{yr:,.0f}円 (保有{u}通貨)")


if __name__ == "__main__":
    main()
