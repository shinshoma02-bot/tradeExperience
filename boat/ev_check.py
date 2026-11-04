"""過去の締切時オッズを使った検証: 「期待値が高い買い目」は得か / モデルは市場に何か足せているか。

- 予想の確率は、これまでと同じ「過去だけで学習」したウォークフォワードのもの(未来を見ていない)
- 払戻は公式の確定払戻(競走成績)。オッズは締切時オッズ(確定払戻とほぼ同じ)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from boat import model as M
from boat import odds_hist


def run(P1, T, meta, races) -> dict:
    odds = odds_hist.load()
    m = meta.merge(races, on=["date", "race_no"], how="left")
    ok = (~np.isnan(P1[:, 0]) & (m["w1"] >= 0).values & m["win_pay"].notna().values & m["3連単_combo"].notna().values)
    rows = [i for i in np.where(ok)[0] if (str(m.at[i, "date"].date()), int(m.at[i, "race_no"])) in odds
            and len(odds[(str(m.at[i, "date"].date()), int(m.at[i, "race_no"]))]["o3"]) == 120
            and len(odds[(str(m.at[i, "date"].date()), int(m.at[i, "race_no"]))]["win"]) >= 5]
    out: dict = {"n_races": len(rows)}
    if len(rows) < 200:
        return out
    out["from"] = str(m.loc[rows, "date"].min().date())
    out["to"] = str(m.loc[rows, "date"].max().date())

    strat: dict[str, list] = {}

    def add(label, ret, cost=1.0):
        strat.setdefault(label, []).append((ret, cost))

    PM = np.zeros((len(rows), 6))
    for n, i in enumerate(rows):
        key = (str(m.at[i, "date"].date()), int(m.at[i, "race_no"]))
        od = odds[key]
        wl = int(m.at[i, "win_lane"])
        wp = m.at[i, "win_pay"] / 100.0
        p = P1[i]
        inv = np.array([1 / od["win"].get(l, np.inf) for l in range(1, 7)])
        PM[n] = inv / inv.sum()
        # 単勝: 期待値 >= th の艇すべて
        for th in (1.0, 1.1, 1.2, 1.5):
            for l in range(1, 7):
                o = od["win"].get(l)
                if o and p[l - 1] * o >= th:
                    add(f"単勝: 期待値{th}以上の艇すべて", wp if wl == l else 0.0)
        o_fav = min(od["win"], key=lambda k: od["win"][k])
        add("単勝: 一番人気(基準)", wp if wl == o_fav else 0.0)
        # 3連単: 確率1%以上で期待値が基準以上 / 期待値上位k点
        c3, p3 = m.at[i, "3連単_combo"], m.at[i, "3連単_pay"] / 100.0
        ev = []
        for c, (a, b, d) in enumerate(M.COMBOS):
            k3 = f"{a + 1}-{b + 1}-{d + 1}"
            if T[i, c] >= 0.01 and k3 in od["o3"]:
                ev.append((T[i, c] * od["o3"][k3], k3))
        for th in (1.0, 1.2, 1.5, 2.0):
            for e, k3 in ev:
                if e >= th:
                    add(f"3連単: 期待値{th}以上(確率1%以上)の組合せすべて", p3 if c3 == k3 else 0.0)
        ev.sort(reverse=True)
        for k in (1, 3, 5):
            sel = [k3 for _, k3 in ev[:k]]
            if len(sel) == k:
                add(f"3連単: 期待値の高い上位{k}点", p3 if c3 in sel else 0.0, cost=k)
        # 基準: 人気上位5点(オッズの低い順)
        fav5 = [k for k, _ in sorted(od["o3"].items(), key=lambda x: x[1])[:5]]
        add("3連単: 人気上位5点(基準)", p3 if c3 in fav5 else 0.0, cost=5)

    res = []
    for label, lst in strat.items():
        ret = np.array([r for r, _ in lst])
        cost = np.array([c for _, c in lst])
        n = len(ret)
        # レース単位ではなく買い目単位でならして出す(買い目数が多いので標準誤差は大きめに見る)
        roi = float(ret.sum() / cost.sum())
        se = float(ret.std(ddof=1) / np.sqrt(n) / cost.mean()) if n > 2 else None
        res.append({"label": label, "bets": int(cost.sum()), "hits": int((ret > 0).sum()), "roi": roi, "se": se})
    out["roi"] = res

    # 市場+モデルの混合: log p_市場 と log p_モデル を使った条件付きロジットを前半で学習→後半で評価
    y = m["w1"].values[rows].astype(int)
    Pm = np.clip(P1[rows], 1e-6, 1)
    F = np.stack([np.log(np.clip(PM, 1e-6, 1)), np.log(Pm)], axis=2)
    mask = np.ones((len(rows), 6), bool)
    order = np.argsort(m.loc[rows, "date"].values)
    half = len(rows) // 2
    tr, te = order[:half], order[half:]
    w = M.fit(F[tr], mask[tr], y[tr], lam=0.0)
    def ll(p, idx):
        return float(-np.mean(np.log(np.clip(p[idx, y[idx]], 1e-9, 1))))
    pb = M.softmax(F @ w, mask)
    out["blend"] = {"coef_market": float(w[0]), "coef_model": float(w[1]),
                    "logloss_market": ll(PM, te), "logloss_model": ll(Pm / Pm.sum(1, keepdims=True), te),
                    "logloss_blend": ll(pb, te), "n_test": int(len(te))}
    return out
