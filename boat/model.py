"""丸亀ボート: 着順予測モデル(ウォークフォワード検証つき)。

- 1レース6艇。「どの艇が勝つか」を条件付きロジット(各艇のスコアをsoftmax)で予測する。
  1着→2着→3着の3段階を別々に学習し、3連単120通りの確率を出す。
- 特徴量は「そのレースの前に分かること」だけ(番組表の勝率/級別/モーター/ボート + 選手の丸亀での過去成績)。
  選手の過去成績は、前の開催日までの分で計算(同日は含めない)。
- 検証は期間を3か月ずつ進める: 「その期間より前」だけで学習 → その期間を予想。
使い方: python -m boat.model  → boat/site/data.json
"""
from __future__ import annotations

import itertools
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from boat import fan, parse

HERE = Path(__file__).parent
OUT = HERE / "site" / "data.json"
START = date(2024, 1, 1)  # これより前は断片しか取っていない
LAMBDA = 3.0  # L2正則化(過剰適合の抑え。2024年後半だけで選んだ。検証期間の2025年以降は見ていない)
PRIOR_N = 6  # 選手の過去成績は平均に引き寄せる(少数サンプルの過大評価を防ぐ)
# f_* = 公式の期別成績(全国・半年ごと)からそのコースの1着率/3連対率/平均STなど、s_* = 今節(この開催の前日まで)の成績。
# どちらも2024年後半だけを見て採用を決めた(logloss 1.238→1.214)。検証期間(2025年〜)でも 1.212→1.201
NUM = ["nat_win", "nat_2r", "loc_win", "loc_2r", "motor_2r", "boat_2r", "cls_n", "age", "weight",
       "h_fin", "h_win", "h_st"] + fan.FEATS + ["s_fin", "s_st"]
SERIES_K = 3  # 今節の成績は平均(3.5着)に引き寄せる
COMBOS = list(itertools.permutations(range(6), 3))  # 120通り


# ───────── 特徴量 ─────────
def add_history(ent: pd.DataFrame) -> pd.DataFrame:
    """選手ごとの丸亀での過去成績(その日より前だけ)を付ける。"""
    ent = ent.copy()
    ent["date"] = pd.to_datetime(ent["date"])
    done = ent[ent["finish"].notna()].copy()
    done["fin6"] = np.where(done["finish"] > 0, done["finish"], 6)
    done["win"] = (done["finish"] == 1).astype(float)
    done["st_ok"] = done["st"].notna().astype(float)
    done["st0"] = done["st"].fillna(0.0)
    daily = done.groupby(["reg", "date"]).agg(
        n=("fin6", "size"), sf=("fin6", "sum"), w=("win", "sum"), sst=("st0", "sum"), nst=("st_ok", "sum")
    ).reset_index().sort_values(["reg", "date"])
    for c in ["n", "sf", "w", "sst", "nst"]:
        daily[c] = daily.groupby("reg")[c].cumsum()
    daily = daily.rename(columns={"date": "asof"}).sort_values("asof")
    ent = ent.sort_values("date")
    ent = pd.merge_asof(ent, daily, left_on="date", right_on="asof", by="reg",
                        direction="backward", allow_exact_matches=False)
    for c in ["n", "sf", "w", "sst", "nst"]:
        ent[c] = ent[c].fillna(0.0)
    ent["h_fin"] = (ent["sf"] + 3.5 * PRIOR_N) / (ent["n"] + PRIOR_N)
    ent["h_win"] = (ent["w"] + (1 / 6) * PRIOR_N) / (ent["n"] + PRIOR_N)
    ent["h_st"] = (ent["sst"] + 0.16 * PRIOR_N) / (ent["nst"] + PRIOR_N)
    return ent.drop(columns=["asof", "sf", "w", "sst", "nst"])


def add_series(ent: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    """今節(同じ開催の初日〜前日)の平均着順と平均ST。調子とモーターの出方が表れる。当日のレースは含めない。"""
    ent = ent.copy()
    dn = races.drop_duplicates("date").set_index("date")["day_no"]
    ent["day_no"] = ent["date"].map(dn).fillna(1).astype(int)
    ent = ent.sort_values(["reg", "date", "race_no"]).reset_index(drop=True)
    s_fin = np.full(len(ent), 3.5)
    s_st = np.full(len(ent), 0.16)
    s_n = np.zeros(len(ent))
    for _, g in ent.groupby("reg", sort=False):
        idx, d, k = g.index.values, g["date"].values, g["day_no"].values
        has = g["finish"].notna().values
        fin = np.where(g["finish"].values > 0, g["finish"].values, 6.0)
        st = g["st"].values
        for a in range(len(idx)):
            if k[a] <= 1:
                continue
            sel = (d < d[a]) & (d >= d[a] - np.timedelta64(int(k[a]) - 1, "D")) & has
            n = int(sel.sum())
            if n:
                s_n[idx[a]] = n
                s_fin[idx[a]] = fin[sel].mean()
                v = st[sel]
                v = v[~np.isnan(v)]
                if len(v):
                    s_st[idx[a]] = v.mean()
    ent["s_n"] = s_n
    ent["s_fin"] = (s_fin * s_n + 3.5 * SERIES_K) / (s_n + SERIES_K)
    ent["s_st"] = (s_st * s_n + 0.16 * SERIES_K) / (s_n + SERIES_K)
    return ent


def build(ent: pd.DataFrame, races: pd.DataFrame, with_ex: bool = False):
    """X: (レース数, 6, 特徴数), mask: (レース数, 6)=その艇が出走, meta: レースごとの情報。"""
    ent = add_history(ent)
    ent = fan.attach(ent)
    ent = add_series(ent, races)
    ent["cls_n"] = ent["cls"].map(parse.CLASS).astype(float)
    ent = ent.sort_values(["date", "race_no", "lane"]).reset_index(drop=True)
    keys = ent[["date", "race_no"]].drop_duplicates().reset_index(drop=True)
    keys["ri"] = np.arange(len(keys))
    ent = ent.merge(keys, on=["date", "race_no"])
    R = len(keys)
    # 数値特徴は「そのレースの平均からの差」にする(強い相手が多いレースかどうかを反映)
    num = NUM + (["ex_time"] if with_ex else [])  # 展示タイム(小さいほど速い)は直前にしか分からない
    for c in num:
        ent[c + "_d"] = (ent[c] - ent.groupby("ri")[c].transform("mean")).fillna(0.0)
    names = ["lane2", "lane3", "lane4", "lane5", "lane6"]
    rows = []
    L = ent["lane"].values
    cols = [(L == k).astype(float) for k in range(2, 7)]
    for c in num:
        d = ent[c + "_d"].values
        names += [c, c + "×1コース", c + "×4-6コース"]
        cols += [d, d * (L == 1), d * (L >= 4)]
    names += ["準優・優勝戦×1コース", "進入固定×1コース"]
    rk = ent.merge(races[["date", "race_no", "kind", "fixed"]], on=["date", "race_no"], how="left")
    final = rk["kind"].fillna("").str.contains("優勝").values.astype(float)
    fixed = rk["fixed"].fillna(False).astype(bool).values.astype(float)
    cols.append(final * (L == 1))
    cols.append(fixed * (L == 1))
    F = np.stack(cols, axis=1)
    X = np.zeros((R, 6, F.shape[1]))
    mask = np.zeros((R, 6), bool)
    ri = ent["ri"].values
    X[ri, L - 1] = F
    mask[ri, L - 1] = True
    meta = keys.copy()
    meta["date"] = pd.to_datetime(meta["date"])
    # 結果: 着順の艇(0始まりの枠)
    fin = np.full((R, 3), -1, int)
    e2 = ent[ent["finish"].isin([1, 2, 3])]
    fin[e2["ri"].values, e2["finish"].astype(int).values - 1] = e2["lane"].values - 1
    meta["w1"], meta["w2"], meta["w3"] = fin[:, 0], fin[:, 1], fin[:, 2]
    # 欠場や事故で6艇走っていないレースは、完走した艇だけを2・3着の候補にする(単純化)
    return X, mask, meta, names, ent


# ───────── 条件付きロジット ─────────
def fit(X: np.ndarray, mask: np.ndarray, y: np.ndarray, lam: float = LAMBDA, iters: int = 25) -> np.ndarray:
    """X:(R,6,F) mask:(R,6) y:(R,) 正解の枠(0始まり)。ニュートン法。"""
    R, _, F = X.shape
    w = np.zeros(F)
    Y = np.zeros((R, 6))
    Y[np.arange(R), y] = 1
    for _ in range(iters):
        p = softmax(X @ w, mask)
        g = np.einsum("rif,ri->f", X, Y - p) - lam * w
        xb = np.einsum("rif,ri->rf", X, p)
        H = -np.einsum("ri,rif,rig->fg", p, X, X) + xb.T @ xb - lam * np.eye(F)
        step = np.linalg.solve(H, g)
        w = w - step
        if np.abs(step).max() < 1e-7:
            break
    return w


def softmax(s: np.ndarray, mask: np.ndarray) -> np.ndarray:
    s = np.where(mask, s, -1e9)
    s = s - s.max(axis=1, keepdims=True)
    e = np.exp(s) * mask
    return e / e.sum(axis=1, keepdims=True)


def fit3(X, mask, meta) -> list[np.ndarray]:
    """1着・2着・3着の3段階。2着は1着を除いた5艇から、3着は上位2艇を除いた4艇から選ぶ。"""
    ws = []
    w1 = meta["w1"].values
    w2 = meta["w2"].values
    w3 = meta["w3"].values
    ok1 = w1 >= 0
    ws.append(fit(X[ok1], mask[ok1], w1[ok1]))
    ok2 = ok1 & (w2 >= 0)
    m2 = mask[ok2].copy()
    m2[np.arange(ok2.sum()), w1[ok2]] = False
    ws.append(fit(X[ok2], m2, w2[ok2]))
    ok3 = ok2 & (w3 >= 0)
    m3 = mask[ok3].copy()
    m3[np.arange(ok3.sum()), w1[ok3]] = False
    m3[np.arange(ok3.sum()), w2[ok3]] = False
    ws.append(fit(X[ok3], m3, w3[ok3]))
    return ws


def trifecta(X, mask, ws) -> np.ndarray:
    """(R,120) 3連単の確率。COMBOSの並び。"""
    R = len(X)
    p1 = softmax(X @ ws[0], mask)
    s2, s3 = X @ ws[1], X @ ws[2]
    out = np.zeros((R, len(COMBOS)))
    for c, (i, j, k) in enumerate(COMBOS):
        m = mask.copy()
        m[:, i] = False
        q2 = softmax(s2, m)
        m[:, j] = False
        q3 = softmax(s3, m)
        out[:, c] = p1[:, i] * q2[:, j] * q3[:, k]
    ok = mask.sum(axis=1) >= 3
    out[~ok] = 0
    return out


def win_probs(X, mask, ws) -> np.ndarray:
    return softmax(X @ ws[0], mask)


# ───────── 検証 ─────────
def walk_forward(X, mask, meta, first_test: str, step_months: int = 3, cols=None):
    """期間を進めながら「過去だけで学習 → 次の期間を予想」。全期間ぶんのテスト予想を返す。"""
    if cols is not None:
        X = X[:, :, cols]
    dates = meta["date"]
    start = pd.Timestamp(first_test)
    end = dates.max()
    P1 = np.full(mask.shape, np.nan)
    T = np.full((len(meta), len(COMBOS)), np.nan)
    while start <= end:
        nxt = start + pd.DateOffset(months=step_months)
        tr = (dates < start).values & (meta["w1"] >= 0).values
        te = ((dates >= start) & (dates < nxt)).values
        if te.any() and tr.sum() > 500:
            ws = fit3(X[tr], mask[tr], meta[tr].reset_index(drop=True))
            P1[te] = win_probs(X[te], mask[te], ws)
            T[te] = trifecta(X[te], mask[te], ws)
        start = nxt
    return P1, T


# ───────── 成績の集計(検証) ─────────
def _combo_idx(s: str) -> int:
    i, j, k = (int(x) - 1 for x in s.split("-"))
    return COMBOS.index((i, j, k))


def evaluate(meta: pd.DataFrame, races: pd.DataFrame, P1, T, base: dict[str, np.ndarray]) -> dict:
    m = meta.merge(races, on=["date", "race_no"], how="left")
    te = (~np.isnan(P1[:, 0]) & (m["w1"] >= 0).values & m["win_pay"].notna().values
          & m["3連単_combo"].notna().values & m["2連単_combo"].notna().values)
    N = int(te.sum())
    out: dict = {"n_races": N, "from": str(m.loc[te, "date"].min().date()), "to": str(m.loc[te, "date"].max().date())}
    y = m["w1"].values[te]
    ar = np.arange(N)

    def ll(P):
        return float(-np.mean(np.log(np.clip(P[te][ar, y], 1e-9, 1))))

    out["logloss"] = {"モデル": ll(P1), **{k: ll(v) for k, v in base.items()}}
    out["top1_acc"] = {"モデル": float(np.mean(P1[te].argmax(1) == y)),
                       "1号艇": float(np.mean(y == 0))}
    # キャリブレーション: 予想した最大確率の帯ごとに、実際の的中率
    pt = P1[te].max(1)
    hit = P1[te].argmax(1) == y
    cal = []
    for lo, hi in [(0, .25), (.25, .35), (.35, .45), (.45, .55), (.55, .65), (.65, .75), (.75, 1.01)]:
        k = (pt >= lo) & (pt < hi)
        if k.sum() >= 20:
            cal.append({"lo": lo, "hi": min(hi, 1.0), "n": int(k.sum()), "pred": float(pt[k].mean()),
                        "actual": float(hit[k].mean())})
    out["calibration"] = cal

    # ── 回収率(100円ずつ買ったとき。払戻は公式の確定払戻) ──
    def roi(ret: np.ndarray, label: str, cost: float = 1.0):
        r = ret / cost
        return {"label": label, "n": int(len(r)), "hit": float(np.mean(ret > 0)),
                "roi": float(r.mean()), "se": float(r.std(ddof=1) / np.sqrt(len(r)))}

    wp = m["win_pay"].values[te] / 100.0
    wl = m["win_lane"].values[te].astype(int) - 1
    rois = []
    rois.append(roi(np.where(wl == 0, wp, 0.0), "単勝: 毎レース1号艇"))
    top = P1[te].argmax(1)
    rois.append(roi(np.where(wl == top, wp, 0.0), "単勝: モデルの本命(毎レース)"))
    for th in (0.4, 0.5, 0.6, 0.7):
        k = pt >= th
        if k.sum() >= 30:
            rois.append(roi(np.where(wl[k] == top[k], wp[k], 0.0), f"単勝: モデル本命(確率{int(th*100)}%以上のレースだけ)"))
    # 2連単・3連単
    Tt = T[te]
    order = np.argsort(-Tt, axis=1)
    c3 = np.array([_combo_idx(s) for s in m["3連単_combo"].values[te]])
    pay3 = m["3連単_pay"].values[te] / 100.0
    rank3 = np.argmax(order == c3[:, None], axis=1)
    for kk in (1, 3, 5, 10, 20):
        ret = np.where(rank3 < kk, pay3, 0.0)
        rois.append(roi(ret, f"3連単: モデル上位{kk}点(毎レース)", cost=kk))
    # 2連単: 3連単確率を3着で足し合わせる
    ex = np.zeros((N, 36))
    for c, (i, j, k) in enumerate(COMBOS):
        ex[:, i * 6 + j] += Tt[:, c]
    c2 = np.array([int(s.split("-")[0]) - 1 for s in m["2連単_combo"].values[te]]) * 6 + \
        np.array([int(s.split("-")[1]) - 1 for s in m["2連単_combo"].values[te]])
    pay2 = m["2連単_pay"].values[te] / 100.0
    order2 = np.argsort(-ex, axis=1)
    rank2 = np.argmax(order2 == c2[:, None], axis=1)
    for kk in (1, 3, 5):
        rois.append(roi(np.where(rank2 < kk, pay2, 0.0), f"2連単: モデル上位{kk}点(毎レース)", cost=kk))
    out["roi"] = rois
    out["trifecta_rank_hist"] = {"top1": float(np.mean(rank3 < 1)), "top5": float(np.mean(rank3 < 5)),
                                 "top10": float(np.mean(rank3 < 10)), "top20": float(np.mean(rank3 < 20))}
    return out


def alt_methods(e: pd.DataFrame, meta: pd.DataFrame, races: pd.DataFrame, P1, P1x) -> list[dict]:
    """予想のしかた別に「本命が1着になった割合」と単勝の回収率を比べる(検証期間、同じレース群ではなく各指標が使えるレース)。"""
    m = meta.merge(races[["date", "race_no", "win_lane", "win_pay"]], on=["date", "race_no"], how="left")
    te = ~np.isnan(P1[:, 0]) & (m["w1"] >= 0).values & m["win_pay"].notna().values
    R = len(meta)
    ri, ln = e["ri"].values, e["lane"].values - 1

    def mat(v):
        M = np.full((R, 6), np.nan)
        M[ri, ln] = v
        return M

    inds = [
        ("1号艇をいつも本命にする", mat(-e["lane"].values.astype(float)), 6),
        ("全国勝率が一番高い艇", mat(e["nat_win"].values), 6),
        ("モーター2連率が一番高い艇", mat(e["motor_2r"].values), 6),
        ("そのコースの1着率が一番高い艇(全国・直近半年)", mat(e["f_cw"].values), 5),
        ("今節の平均着順が一番良い艇(2日目以降)", mat(np.where(e["s_n"].values > 0, -e["s_fin"].values, np.nan)), 5),
        ("展示タイムが一番速い艇", mat(-e["ex_time"].values) if "ex_time" in e else np.full((R, 6), np.nan), 5),
        ("この予想(全部まとめたモデル)", P1, 3),
        ("この予想+展示タイム", P1x, 3),
    ]
    wl = m["win_lane"].values
    wp = m["win_pay"].values / 100.0
    out = []
    for label, M, need in inds:
        ok = te & ((~np.isnan(M)).sum(axis=1) >= need)
        if ok.sum() < 100:
            continue
        top = np.nanargmax(np.where(np.isnan(M[ok]), -1e18, M[ok]), axis=1) + 1
        hit = top == wl[ok]
        ret = np.where(hit, wp[ok], 0.0)
        out.append({"label": label, "n": int(ok.sum()), "hit": float(hit.mean()), "roi": float(ret.mean()),
                    "se": float(ret.std(ddof=1) / np.sqrt(len(ret)))})
    return out


def describe(ent: pd.DataFrame, races: pd.DataFrame) -> dict:
    """丸亀の全期間の傾向(コース別・レース番号別・風別・級別・決まり手)。"""
    res = ent[ent["finish"].notna()].copy()
    d: dict = {}
    nrace = int(res.groupby(["date", "race_no"]).ngroups)
    d["n_races"] = nrace
    d["from"] = str(pd.to_datetime(res["date"]).min().date())
    d["to"] = str(pd.to_datetime(res["date"]).max().date())
    lane = []
    for l in range(1, 7):
        r = res[res["lane"] == l]
        lane.append({"lane": l, "n": len(r), "w1": float((r["finish"] == 1).mean()),
                     "w2": float(r["finish"].isin([1, 2]).mean()), "w3": float(r["finish"].isin([1, 2, 3]).mean())})
    d["lane"] = lane
    rr = races.dropna(subset=["win_lane"]).copy()
    rr["l1"] = rr["win_lane"] == 1
    d["by_race_no"] = [{"race_no": int(k), "n": int(len(g)), "lane1": float(g["l1"].mean()),
                        "pay3_median": float(g["3連単_pay"].median()),
                        "high": float((g["3連単_pay"] >= 10000).mean())}
                       for k, g in rr.groupby("race_no")]
    rr["wbin"] = pd.cut(rr["wind"], [-1, 1, 3, 5, 99], labels=["0-1m", "2-3m", "4-5m", "6m以上"])
    d["by_wind"] = [{"bin": str(k), "n": int(len(g)), "lane1": float(g["l1"].mean())}
                    for k, g in rr.groupby("wbin", observed=True)]
    d["by_wind_dir"] = [{"dir": k, "n": int(len(g)), "lane1": float(g["l1"].mean())}
                        for k, g in rr.groupby("wind_dir") if len(g) >= 40]
    d["by_wave"] = [{"bin": str(k), "n": int(len(g)), "lane1": float(g["l1"].mean())}
                    for k, g in rr.assign(wv=pd.cut(rr["wave"], [-1, 2, 5, 9, 99], labels=["0-2cm", "3-5cm", "6-9cm", "10cm以上"]))
                    .groupby("wv", observed=True)]
    kim = rr["kimarite"].value_counts()
    d["kimarite"] = [{"name": k, "n": int(v), "rate": float(v / len(rr))} for k, v in kim.items() if k]
    cl = []
    for c in ["A1", "A2", "B1", "B2"]:
        r = res[res["cls"] == c]
        if len(r) >= 50:
            cl.append({"cls": c, "n": len(r), "w1": float((r["finish"] == 1).mean()),
                       "w3": float(r["finish"].isin([1, 2, 3]).mean())})
    d["by_class"] = cl
    # 1号艇が負けた時の勝者(どこから来るか)
    lost = rr[~rr["l1"]]
    d["lane1_lost_winner"] = [{"lane": int(k), "rate": float(v / len(lost))}
                              for k, v in lost["win_lane"].value_counts().sort_index().items()]
    return d


def racer_table(ent: pd.DataFrame) -> list[dict]:
    res = ent[ent["finish"].notna()].copy()
    res["date"] = pd.to_datetime(res["date"])
    out = []
    last = ent.sort_values("date").groupby("reg").tail(1).set_index("reg")
    for reg, g in res.groupby("reg"):
        if len(g) < 18:
            continue
        stv = g["st"].dropna()
        l1 = g[g["lane"] == 1]
        out.append({"reg": int(reg), "name": last.loc[reg, "name"], "cls": last.loc[reg, "cls"],
                    "branch": last.loc[reg, "branch"], "n": len(g), "w1": float((g["finish"] == 1).mean()),
                    "w3": float(g["finish"].isin([1, 2, 3]).mean()), "st": float(stv.mean()) if len(stv) else None,
                    "l1n": len(l1), "l1w": float((l1["finish"] == 1).mean()) if len(l1) >= 5 else None,
                    "nat_win": float(last.loc[reg, "nat_win"])})
    return sorted(out, key=lambda x: -x["w1"])


# ───────── サイト用データ ─────────
def _top_combos(t: np.ndarray, k: int = 10) -> list[dict]:
    idx = np.argsort(-t)[:k]
    return [{"c": "-".join(str(x + 1) for x in COMBOS[i]), "p": round(float(t[i]), 5)} for i in idx]


def _top_exacta(t: np.ndarray, k: int = 6) -> list[dict]:
    """3連単の確率を3着で足し合わせて、2連単(1着-2着)の確率にする。"""
    ex = np.zeros(36)
    for c, (i, j, _) in enumerate(COMBOS):
        ex[i * 6 + j] += t[c]
    idx = np.argsort(-ex)[:k]
    return [{"c": f"{i // 6 + 1}-{i % 6 + 1}", "p": round(float(ex[i]), 5)} for i in idx]


def race_cards(ent, races, meta, mask, P1, T, rows: np.ndarray, live=None, P1x=None, Tx=None) -> list[dict]:
    """レース1件ずつの表示用データ(予想 + あれば結果)。"""
    rmap = races.set_index(["date", "race_no"])
    out = []
    byri = {ri: g.sort_values("lane") for ri, g in ent.groupby("ri")}
    for r in rows:
        mr = meta.iloc[r]
        g = byri[int(mr["ri"])]
        key = (mr["date"], int(mr["race_no"]))
        rc = rmap.loc[key] if key in rmap.index else None
        boats = []
        for _, b in g.iterrows():
            l = int(b["lane"])
            boats.append({"lane": l, "name": b["name"], "reg": int(b["reg"]), "cls": b["cls"],
                          "nat_win": float(b["nat_win"]), "loc_win": float(b["loc_win"]),
                          "motor": int(b["motor"]), "motor_2r": float(b["motor_2r"]),
                          "boat_2r": float(b["boat_2r"]), "age": int(b["age"]),
                          "n_hist": int(b["n"]), "h_fin": round(float(b["h_fin"]), 2),
                          "cw": round(float(b["f_cw"]), 3) if pd.notna(b.get("f_cw")) else None,
                          "s_n": int(b["s_n"]), "s_fin": round(float(b["s_fin"]), 2) if b["s_n"] > 0 else None,
                          "p": round(float(P1[r, l - 1]), 4),
                          "finish": int(b["finish"]) if pd.notna(b.get("finish")) else None,
                          "ex_time": float(b["ex_time"]) if pd.notna(b.get("ex_time")) else None,
                          "st": float(b["st"]) if pd.notna(b.get("st")) else None})
        c = {"date": str(mr["date"].date()), "race_no": int(mr["race_no"]), "boats": boats,
             "top": _top_combos(T[r], 10), "top2": _top_exacta(T[r])}
        lv = (live or {}).get(key)
        if lv and lv.get("result"):  # 終わったレース: 競走成績(K)の公開前でも結果ページから結果を出す
            rs = lv["result"]
            for b in boats:
                if b["lane"] in rs.get("order", []):
                    b["finish"] = rs["order"].index(b["lane"]) + 1
                if rs.get("st", {}).get(str(b["lane"])) is not None:
                    b["st"] = rs["st"][str(b["lane"])]
        if lv:
            c["live"] = {k: lv[k] for k in ("weather", "course_order") if k in lv}
            for b in boats:
                lb = (lv.get("boats") or {}).get(str(b["lane"])) or (lv.get("boats") or {}).get(b["lane"])
                if lb:
                    b["ex_time"], b["tilt"] = lb["ex_time"], lb.get("tilt")
                if P1x is not None and lb:
                    b["px"] = round(float(P1x[r, b["lane"] - 1]), 4)
            if Tx is not None and lv.get("boats"):
                c["top_x"] = _top_combos(Tx[r], 10)
                c["top2_x"] = _top_exacta(Tx[r])
            rs = lv.get("result")
            if rs:
                c["3連単_combo"], c["3連単_pay"] = rs["tri"], rs["tri_pay"]
                if rs.get("win_pay"):
                    c["win_pay"], c["win_lane"] = rs["win_pay"], rs.get("win_lane")
                if rs.get("kimarite"):
                    c["kimarite"] = rs["kimarite"]
            wo = lv.get("win_odds") or {}
            if wo:
                c["win_odds"] = {int(k): v for k, v in wo.items()}
            o3 = lv.get("odds3t") or {}
            if o3:
                tt = Tx[r] if (Tx is not None and lv.get("boats")) else T[r]
                evs = []
                for i, cmb in enumerate(COMBOS):
                    k3 = "-".join(str(x + 1) for x in cmb)
                    if tt[i] >= 0.01 and k3 in o3:
                        evs.append({"c": k3, "p": round(float(tt[i]), 4), "odds": o3[k3],
                                    "ev": round(float(tt[i] * o3[k3]), 3)})
                c["ev_all"] = evs  # 前向き記録用(確率1%以上の全組合せ)
                c["ev3"] = sorted(evs, key=lambda x: -x["ev"])[:8]
                c["odds_top"] = [{"c": k, "odds": v} for k, v in sorted(o3.items(), key=lambda x: x[1])[:5]]
        if rc is not None:
            for k in ("kind", "close", "weather", "wind_dir", "wind", "wave", "kimarite", "title", "day_no",
                      "3連単_combo", "3連単_pay", "win_pay", "win_lane"):
                v = rc.get(k)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    c[k] = v.item() if hasattr(v, "item") else v
        out.append(c)
    return out


def run(first_test: str | None = None) -> dict:
    ent, races = parse.load(START)
    live = load_live()
    if live:  # 開催前レースの展示タイムを差し込む(展示込みモデル用)
        ent["ex_time"] = ent.apply(lambda r: _live_ex(live, r), axis=1) if "ex_time" not in ent else \
            ent["ex_time"].where(ent["ex_time"].notna(), ent.apply(lambda r: _live_ex(live, r), axis=1))
    X, mask, meta, names, e = build(ent, races)
    Xe, _, _, names_e, _ = build(ent, races, with_ex=True)
    done = (meta["w1"] >= 0).values
    t0 = pd.Timestamp(first_test) if first_test else pd.Timestamp("2025-01-01")
    P1, T = walk_forward(X, mask, meta, str(t0.date()), 3)
    lane_cols = list(range(5))
    off_cols = lane_cols + [names.index(n) for n in
                            ["nat_win", "nat_win×1コース", "nat_win×4-6コース", "cls_n", "cls_n×1コース", "cls_n×4-6コース"]]
    base = {"枠番だけ": walk_forward(X, mask, meta, str(t0.date()), 3, cols=lane_cols)[0],
            "枠番+公式の全国勝率・級別": walk_forward(X, mask, meta, str(t0.date()), 3, cols=off_cols)[0]}
    ev = evaluate(meta, races, P1, T, base)
    P1x, Tx = walk_forward(Xe, mask, meta, str(t0.date()), 3)
    evx = evaluate(meta, races, P1x, Tx, {})
    from boat import ev_check, odds_hist
    odds_hist.unpack()  # 過去の締切時オッズ(boat/data/odds)があれば、市場との比較・期待値の検証もする
    ev["odds_check"] = ev_check.run(P1, T, meta, races)
    ev["methods"] = alt_methods(e, meta, races, P1, P1x)
    ev["with_exhibition"] = {"logloss": evx["logloss"]["モデル"], "top1_acc": evx["top1_acc"]["モデル"],
                             "roi": evx["roi"]}

    # 最終モデル(結果が出ている全レースで学習)→ 開催前のレースを予想
    ws = fit3(X[done], mask[done], meta[done].reset_index(drop=True))
    last_done = meta.loc[done, "date"].max()
    wse = fit3(Xe[done], mask[done], meta[done].reset_index(drop=True))
    up_rows = np.where(~done & (meta["date"] > last_done).values)[0]
    P1u = np.zeros(mask.shape)
    Tu = np.zeros((len(meta), len(COMBOS)))
    P1ux = np.zeros(mask.shape)
    Tux = np.zeros((len(meta), len(COMBOS)))
    if len(up_rows):
        P1u[up_rows] = win_probs(X[up_rows], mask[up_rows], ws)
        Tu[up_rows] = trifecta(X[up_rows], mask[up_rows], ws)
        P1ux[up_rows] = win_probs(Xe[up_rows], mask[up_rows], wse)
        Tux[up_rows] = trifecta(Xe[up_rows], mask[up_rows], wse)
    livemap = {}
    for dd in (live or {}).get("days", []):
        for r in dd["races"]:
            livemap[(pd.Timestamp(dd["date"]), r["race_no"])] = r
    upcoming = race_cards(e, races, meta, mask, P1u, Tu, up_rows, livemap, P1ux, Tux)

    # 答え合わせ用: 検証期間(過去だけで学習した予想)の直近レース
    te_rows = np.where(~np.isnan(P1[:, 0]) & done)[0]
    recent = te_rows[-240:][::-1]
    browse = race_cards(e, races, meta, mask, P1, T, recent)

    from boat import forward
    n_rec = forward.record(upcoming)  # オッズが取れているレースだけ記録
    for c in upcoming:
        c.pop("ev_all", None)
    fwd = forward.evaluate(races)
    fwd["recorded_now"] = n_rec

    last_b = e["date"].max()
    cur = e[e["date"] == last_b].drop_duplicates("motor").sort_values("motor_2r", ascending=False)
    motors = [{"motor": int(r.motor), "m2r": float(r.motor_2r)} for r in cur.itertuples()]

    nxt = next_series_info(ent)
    data = {
        "next": nxt,
        "built": str(date.today()), "live_asof": (live or {}).get("asof"), "last_result": str(last_done.date()),
        "validation": ev, "forward": fwd, "describe": describe(ent, races), "racers": racer_table(ent),
        "upcoming": upcoming, "browse": browse,
        "motors": {"asof": str(last_b.date()), "list": motors},
        "weights": {n: round(float(w), 3) for n, w in zip(names, ws[0])},
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, default=_js), encoding="utf-8")
    return data


def next_series_info(ent: pd.DataFrame) -> dict | None:
    """次の開催(公式の出場予定選手)に、期別成績と丸亀での過去成績を付ける。番組表が出る前の「下見」用。"""
    f = HERE / "site" / "next.json"
    if not f.exists():
        return None
    nx = json.loads(f.read_text(encoding="utf-8"))
    if not nx or not nx.get("racers"):
        return nx or None
    tab = fan.latest_table()
    res = ent[ent["finish"].notna()]
    hist = res.groupby("reg").agg(n=("finish", "size"), w1=("finish", lambda x: float((x == 1).mean())),
                                  w3=("finish", lambda x: float(x.isin([1, 2, 3]).mean())))
    for r in nx["racers"]:
        t = tab.get(r["reg"])
        if t:
            r.update(t)
        if r["reg"] in hist.index and hist.loc[r["reg"], "n"] >= 6:
            h = hist.loc[r["reg"]]
            r.update({"m_n": int(h["n"]), "m_w1": round(float(h["w1"]), 3), "m_w3": round(float(h["w3"]), 3)})
    nx["racers"].sort(key=lambda r: -(r.get("win") or 0))
    return nx


def load_live() -> dict | None:
    f = HERE / "site" / "live.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def _live_ex(live: dict, r) -> float:
    for dd in live.get("days", []):
        if dd["date"] == str(r["date"].date()):
            for x in dd["races"]:
                if x["race_no"] == r["race_no"]:
                    b = (x.get("boats") or {}).get(str(r["lane"]))
                    return b["ex_time"] if b else np.nan
    return np.nan


def _js(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (pd.Timestamp, date)):
        return str(o)[:10]
    raise TypeError(type(o))


if __name__ == "__main__":
    d = run()
    v = d["validation"]
    print(f"検証 {v['from']}〜{v['to']} {v['n_races']}レース")
    print("logloss", {k: round(x, 4) for k, x in v["logloss"].items()})
    print("本命的中率", {k: round(x, 3) for k, x in v["top1_acc"].items()})
    for r in v["roi"]:
        print(f"{r['label']:<36} n={r['n']:<5} 的中{r['hit']*100:5.1f}% 回収{r['roi']*100:5.1f}% (±{2*r['se']*100:.0f})")
    print("予想対象:", len(d["upcoming"]), "レース")
