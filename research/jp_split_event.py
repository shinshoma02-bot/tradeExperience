"""日本株の株式分割の前後の値動き(イベントスタディ)。K(日本株クロスセクション)のキャッシュ research/data/jp_xs/ を使う。
実行: python research/jp_split_event.py   (先に python research/jp_cross_section.py でキャッシュを作る)

分かること: 『権利付き最終日(=ex日の前営業日)の L営業日前に買い、権利付き最終日に売る』の対TOPIX(1348)超過リターン。
重要な限界(必ず読むこと):
 - 発表日のデータが無い。L が大きいほど「発表前」の期間を含み、未来の分割を知っている前提(ルックアヘッド)になる。
   発表の後だと確実に言える短い窓(L≦10程度)だけが取引可能な形に近い
 - 今も上場している609銘柄・分割405件(2010年以降、比率1.5倍以上)のみ。生存者バイアス
 - 試した窓は L=5/10/15/20 の4つ + 事前の窓(多重比較に注意)。コストは往復0.4%を仮定
"""
import json
import os

import numpy as np
import pandas as pd

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "jp_xs") + os.sep
sp = json.load(open(D + "splits.json"))


def load(code):
    d = json.load(open(D + f"{code}.json"))
    s = pd.Series(d["c"], index=pd.to_datetime(d["t"], unit="s").normalize(), dtype=float).dropna()
    return s[~s.index.duplicated()]


B = load("1348")
B = B[B.index >= "2009-01-01"]


def ar(px, i0, i1):
    ta, tb = px.index[i0], px.index[i1]
    j0, j1 = B.index.searchsorted(ta), B.index.searchsorted(tb)
    if j1 >= len(B):
        return np.nan
    return (px.iloc[i1] / px.iloc[i0] - 1) - (B.iloc[j1] / B.iloc[j0] - 1)


LS = [5, 10, 15, 20]
events, controls = [], []
for code in [c for c in sp if os.path.exists(D + f"{c}.json")]:
    px = load(code)
    if len(px) < 300:
        continue
    evs = [(pd.Timestamp(ts, unit="s").normalize(), r) for ts, r in sp[code] if r >= 1.5]
    idx = [px.index.searchsorted(d) for d, _ in evs]
    for (d0, r), i in zip(evs, idx):
        if d0 < pd.Timestamp("2010-06-01") or i < 140 or i + 3 >= len(px):
            continue
        row = dict(code=code, d=d0, ratio=r, y=d0.year, prior=ar(px, i - 130, i - 21))
        row.update({f"w{L}": ar(px, i - 1 - L, i - 1) for L in LS})
        events.append(row)
    bad = np.zeros(len(px), bool)
    for i in idx:
        bad[max(0, i - 200): i + 70] = True
    for k in range(140, len(px) - 25, 5):
        if bad[k] or px.index[k] < pd.Timestamp("2010-06-01"):
            continue
        row = dict(code=code, d=px.index[k], y=px.index[k].year, prior=ar(px, k - 110, k))
        row.update({f"w{L}": ar(px, k, k + L) for L in LS})
        controls.append(row)

E, C = pd.DataFrame(events).dropna(), pd.DataFrame(controls).dropna()
lo, hi = E.prior.quantile(0.25), E.prior.quantile(0.75)
Cm = C[(C.prior >= lo) & (C.prior <= hi)]  # 事前の上昇が分割銘柄と同程度の、分割しなかった銘柄
print(f"分割 {len(E)}件 / 対照 {len(C)}点")
print(f"{'L':>3} {'平均':>8} {'中央':>8} {'月クラスタt':>9} {'〜2018':>8} {'2019〜':>8} {'勝率':>5} | 対照平均 | コスト0.4%後 平均/中央")
for L in LS:
    x = E[f"w{L}"]
    mm = E.groupby(E.d.dt.to_period("M"))[f"w{L}"].mean()
    t = mm.mean() / (mm.std() / np.sqrt(len(mm)))
    print(f"{L:3d} {x.mean():+8.2%} {x.median():+8.2%} {t:9.2f} {E[E.y < 2019][f'w{L}'].mean():+8.2%} "
          f"{E[E.y >= 2019][f'w{L}'].mean():+8.2%} {(x > 0).mean():5.0%} | {Cm[f'w{L}'].mean():+8.2%} | "
          f"{x.mean() - 0.004:+.2%} / {x.median() - 0.004:+.2%}")
