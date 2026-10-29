"""株式分割の『発表日』データの入手可否の確認 + 既存データでの追加の切り分け(発表日なしの検証)。
実行: python research/jp_split_announce.py   (先に python research/jp_cross_section.py でキャッシュ research/data/jp_xs/ を作る)

■ 1. 発表日データの取得を『試す』部分(robots.txt の確認のみ。本文の大量取得はしない)
  - TDnet(release.tdnet.info)は robots.txt が Disallow: / → 自動取得はしない。閲覧は直近31日分のみで過去数年は取れない。
    履歴は有料のTDnet API(JPX、約款に同意が必要)。
  - 株探(kabutan.jp)は curl の既定UAでは robots.txt が403、ブラウザ風UAでは200(空)。UAを変えて回避する類なので、規約も未確認のため使わない。
  - Yahoo!ファイナンス(日本)の個別ページのスクレイピングは、規約が禁止していると考えられるが未確認 → 使わない。
  - EDINETは分割の発表日一覧を提供する制度ではない(分割の取締役会決議は通常、臨時報告書の対象外)。未確認。
  → 結論: 合法・無料で過去数年分の『発表日』つき分割一覧は取れなかった。以降は発表日なしの分析。

■ 2. 発表日なしの分析(重要な限界)
  - L が大きいほど『発表前』の期間を含む。どの L が発表後かはこのデータでは確認できない。
  - 生存者バイアス(今も上場する609銘柄のみ)、試した窓は L=5/10/15/20 と日次プロファイル(多重比較)
  - Yahooの終値・出来高は分割調整済み。当時の株価は『調整後 × 以降の分割比率の積』で復元(概算)。
"""
import json
import os
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data", "jp_xs") + os.sep
UA = {"User-Agent": "Mozilla/5.0"}


def probe():
    print("=== 1. 発表日データの取得の試み(robots.txt の確認のみ) ===")
    for name, url in [("TDnet", "https://www.release.tdnet.info/robots.txt"),
                      ("JPX", "https://www.jpx.co.jp/robots.txt"),
                      ("株探", "https://kabutan.jp/robots.txt"),
                      ("YahooJP", "https://finance.yahoo.co.jp/robots.txt")]:
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=15) as r:
                txt = r.read().decode("utf-8", "ignore")
            head = " | ".join(line.strip() for line in txt.splitlines()[:3])
            print(f"  {name:8s} HTTP {r.status}: {head[:90]}")
        except Exception as e:  # noqa  (403等は回避しない)
            print(f"  {name:8s} 取得不可: {str(e)[:60]}")
    print("  → TDnetは Disallow: / かつ閲覧は直近31日のみ。株探はUA次第で403(規約未確認のため不使用)。履歴の発表日は無料・合法では取れず。\n")


def load(code):
    d = json.load(open(D + f"{code}.json"))
    idx = pd.to_datetime(d["t"], unit="s").normalize()
    c = pd.Series(d["c"], index=idx, dtype=float)
    v = pd.Series(d["v"], index=idx, dtype=float)
    keep = ~c.index.duplicated()
    c, v = c[keep], v[keep]
    ok = c.notna()
    return c[ok], v[ok]


def tstat(x, key):
    m = x.groupby(key).mean()
    return m.mean() / (m.std() / np.sqrt(len(m))) if len(m) > 2 else np.nan


def main():
    probe()
    if not os.path.exists(D + "splits.json"):
        print("キャッシュ無し: 先に python research/jp_cross_section.py を実行してください。")
        return
    sp = json.load(open(D + "splits.json"))
    meta = pd.read_excel(D + "data_j.xlsx")
    meta["コード"] = meta["コード"].astype(str).str.strip()
    size = dict(zip(meta["コード"], meta["規模区分"]))
    B, _ = load("1348")
    B = B[B.index >= "2009-01-01"]

    def ar(px, i0, i1):
        j0, j1 = B.index.searchsorted(px.index[i0]), B.index.searchsorted(px.index[i1])
        if j1 >= len(B):
            return np.nan
        return (px.iloc[i1] / px.iloc[i0] - 1) - (B.iloc[j1] / B.iloc[j0] - 1)

    Ls = [5, 10, 15, 20]
    PRE = 40
    rows, prof = [], []
    for code, evl in sp.items():
        if not os.path.exists(D + f"{code}.json"):
            continue
        px, vol = load(code)
        if len(px) < 300:
            continue
        evs = [(pd.Timestamp(ts, unit="s").normalize(), r) for ts, r in evl]
        for d0, r in evs:
            if r < 1.5 or d0 < pd.Timestamp("2010-06-01"):
                continue
            i = px.index.searchsorted(d0)
            if i < 140 or i + 3 >= len(px):
                continue
            later = np.prod([rr for dd, rr in evs if dd >= d0 and rr > 0])  # 当時の価格 = 調整後 × 以降の分割比の積
            e = i - 1 - 20
            row = dict(code=code, d=d0, y=d0.year, ratio=r, size=size.get(code, "-"),
                       turn=float((px.iloc[e - 20:e] * vol.iloc[e - 20:e]).mean() * later),  # 概算(出来高の調整有無は未確認)
                       raw_px=float(px.iloc[i - 1 - 10] * later))
            for L in Ls:
                row[f"w{L}"] = ar(px, i - 1 - L, i - 1)
            rows.append(row)
            if i - 1 - PRE >= 0:
                prof.append([ar(px, i - 1 - PRE, i - 1 + k) if i - 1 + k < len(px) else np.nan
                             for k in range(-PRE, 16)])
    E = pd.DataFrame(rows).dropna(subset=[f"w{L}" for L in Ls])
    mon = E.d.dt.to_period("M")
    print(f"=== 2. 発表日なしの分析: 分割 {len(E)}件(2010-06〜、比率1.5倍以上、現存609銘柄の標本) ===")
    print("試した窓: L=5/10/15/20 の4つ × コスト2通り × 層別(年/比率/規模/売買代金)。多重比較に注意。")
    print("注: 『発表前を含む』= L が大きいほど未来の分割を知っている前提。取引可能とは言えない。\n")

    def block(title, groups):
        print(f"[{title}]  平均/中央(コスト前) と 0.4%・1%控除後の 平均/中央")
        for name, g in groups:
            if len(g) < 8:
                continue
            s = f"  {name:14s} n={len(g):3d}"
            for L in (10, 20):
                x = g[f"w{L}"]
                s += (f" | L{L}: {x.mean():+.1%}/{x.median():+.1%} 0.4%:{x.mean()-.004:+.1%}/{x.median()-.004:+.1%}"
                      f" 1%:{x.mean()-.01:+.1%}/{x.median()-.01:+.1%}")
            print(s)
        print()

    print("[全体]  L  平均  中央  月クラスタt  0.4%後(平均/中央)  1%後(平均/中央)")
    for L in Ls:
        x = E[f"w{L}"]
        print(f"   {L:2d} {x.mean():+.2%} {x.median():+.2%}  t={tstat(x, mon):.2f}  "
              f"{x.mean()-.004:+.2%}/{x.median()-.004:+.2%}  {x.mean()-.01:+.2%}/{x.median()-.01:+.2%}")
    print()
    block("年代別", [(f"{a}-{b}", E[(E.y >= a) & (E.y <= b)]) for a, b in
                     [(2010, 2014), (2015, 2019), (2020, 2023), (2024, 2026)]])
    block("分割比率別", [("1.5〜2倍未満", E[E.ratio < 2]), ("2〜3倍未満", E[(E.ratio >= 2) & (E.ratio < 3)]),
                       ("3倍以上", E[E.ratio >= 3])])
    block("規模区分別(現在の区分)", [(s, E[E["size"] == s]) for s in sorted(E["size"].unique())])
    block("売買代金別(20営業日平均,概算)", [("5千万円未満", E[E.turn < 5e7]),
                                    ("5千万〜5億円", E[(E.turn >= 5e7) & (E.turn < 5e8)]),
                                    ("5億円以上", E[E.turn >= 5e8])])
    block("流動性フィルタ(売買代金5千万円以上)", [("5千万円以上", E[E.turn >= 5e7])])

    P = pd.DataFrame(prof, columns=range(-PRE, 16)).dropna()
    m = P.mean()
    print(f"[日次プロファイル] 権利付き最終日=0。-40営業日起点の累積超過(n={len(P)})")
    print("  日      : " + " ".join(f"{k:+4d}" for k in range(-40, 16, 5)))
    print("  平均累積: " + " ".join(f"{m[k]:+5.1%}" for k in range(-40, 16, 5)))
    print("  5営業日ごとの増分: " + " ".join(f"{m[k]-m[k-5]:+5.1%}" for k in range(-35, 16, 5)))
    print("  → 段差が1か所に集中していれば発表日に近い可能性。均等なら事前の上昇(選別)。単独では発表日の証明にならない。\n")

    print("=== 3. 少額での実行可能性(概算) ===")
    px10 = E.raw_px
    print(f"分割前の株価(L=10営業日前、概算): 中央 {px10.median():,.0f}円, 25%点 {px10.quantile(.25):,.0f}, 75%点 {px10.quantile(.75):,.0f}")
    yrs = max(1, E.y.max() - E.y.min() + 1)
    per_year = len(E) / yrs
    print(f"標本の年間件数 約{per_year:.0f}件(上場約3,900社のうち609社の層別抽出。東証全体は粗く標本の約6倍と推定、未確認)")
    for cap in (1e4, 1e5, 1e6):
        print(f"  資金{cap:>9,.0f}円: 1株が買える {(px10 <= cap).mean():5.1%} / 100株単元が買える {(px10 * 100 <= cap).mean():5.1%}")
    print("\n期待収益(円/回) = 投資額 × (超過リターン − コスト)。L=10(発表後かは未確認)。S株の実際のコストは証券会社の公式ページで要確認。")
    x = E["w10"]
    for cap in (1e4, 1e5, 1e6):
        for cost in (0.004, 0.01):
            print(f"  資金{cap:>9,.0f}円 コスト{cost:.1%}: 平均 {cap*(x.mean()-cost):+,.0f}円 / 中央 {cap*(x.median()-cost):+,.0f}円 (1回)")
        n_aff = per_year * (px10 <= cap).mean()
        print(f"    1株で買える件数: 標本内 年{n_aff:.1f}件、東証全体への粗い換算 年{n_aff*6:.0f}件(全部は取れない)")
    print("\n注意: 超過リターンはTOPIX比。TOPIXをヘッジしない限り実損益は絶対リターン。")
    print("税: 売却益は申告分離20.315%(要確認)。NISA成長投資枠なら非課税だが損失の損益通算不可(要確認)。")


if __name__ == "__main__":
    main()
