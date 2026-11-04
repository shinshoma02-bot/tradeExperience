"""JRA全レースの着順予測モデル(ウォークフォワード検証つき)と、今週の予想。

- 特徴量は全部「そのレースの前に分かること」だけ。騎手・調教師・馬の成績は
  そのレースより前の分(同日は含めない)で計算する。
- 学習は「予測する年より前」だけ。2020年は2016-19で学習、2021年は2016-20で…と進める。
- データは keiba/data/all/*.csv.gz(全レース)。無ければ重賞だけの旧CSVを使う。
- 出力: keiba/site/data.json(検証結果+閲覧用レース)、keiba/site/upcoming.json(今週の予想)
使い方: python -m keiba.model           # 検証 + data.json
        python -m keiba.model upcoming  # 今週の出馬表を取って予想 + upcoming.json
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
ALL = HERE / "data" / "all"
OLD_CSV = HERE / "data" / "jra_graded_results.csv"
OUT = HERE / "site" / "data.json"
OUT_UP = HERE / "site" / "upcoming.json"
PRIOR_N = 20  # 騎手/調教師の勝率は平均に引き寄せる(少数サンプルの過大評価を防ぐ)
VIEW_DAYS = 45  # サイトで閲覧できる重賞以外のレースは直近これだけ(JSONを軽くする)

KLASS = {"新馬": 0, "未勝利": 1, "500万下": 2, "1勝クラス": 2, "1000万下": 3, "2勝クラス": 3,
         "1600万下": 4, "3勝クラス": 4, "オープン": 5}
GRADE_LV = {"G3": 6, "G2": 7, "G1": 8}


def load() -> pd.DataFrame:
    files = sorted(glob.glob(str(ALL / "*.csv.gz")))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.read_csv(OLD_CSV)
    df["grade"] = df["grade"].fillna("")
    df["klass"] = df.get("klass", pd.Series("", index=df.index)).fillna("")
    df["venue"] = df.get("venue", pd.Series("", index=df.index)).fillna("")
    df["race_no"] = df.get("race_no", pd.Series(0, index=df.index)).fillna(0).astype(int)
    df = df[df["place"] > 0].copy()  # 競走中止・除外は予測の対象外
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df = df.sort_values(["date", "race_id", "num"]).reset_index(drop=True)
    return prep(df)


def prep(df: pd.DataFrame) -> pd.DataFrame:
    df["field"] = df.groupby("race_id")["num"].transform("count")
    df["win"] = (df["place"] == 1).astype(int)
    df["top3"] = (df["place"] <= 3).astype(int)
    df["klass_lv"] = [GRADE_LV.get(g, KLASS.get(k, 2)) for g, k in zip(df["grade"], df["klass"])]
    return df


def add_history(df: pd.DataFrame) -> pd.DataFrame:
    """騎手・調教師・馬の「過去成績」を、時系列順に一日ずつ更新しながら付ける。"""
    stat = {k: defaultdict(lambda: [0, 0, 0]) for k in ("jockey", "trainer")}  # 出走, 勝, 3着内
    hist = defaultdict(list)  # 馬 -> [(date, place, pop, field)]
    cols = defaultdict(list)
    n_all = w_all = t_all = 0
    cur = ["jockey", "trainer", "horse", "win", "top3", "place", "pop", "field"]
    for date, day in df.groupby("date", sort=True):
        rate_w = (w_all + 1) / (n_all + 10)
        rate_t = (t_all + 3) / (n_all + 10)
        recs = list(day[cur].itertuples(index=False, name=None))
        for jk, tr, horse, *_ in recs:
            for k, v in (("jockey", jk), ("trainer", tr)):
                n, w, t = stat[k][v]
                cols[f"{k}_win"].append((w + PRIOR_N * rate_w) / (n + PRIOR_N))
                cols[f"{k}_top3"].append((t + PRIOR_N * rate_t) / (n + PRIOR_N))
            h = hist[horse]
            cols["n_prev"].append(len(h))
            if h:
                last = h[-1]
                cols["prev_place"].append(min(last[1], 10))
                cols["prev_pop"].append(min(last[2], 10))
                cols["rest_days"].append(min((date - last[0]).days, 365))
                last3 = h[-3:]
                cols["prev_over"].append(sum(min(p[2], 10) - min(p[1], 10) for p in last3) / len(last3))
                cols["prev_pct"].append(sum(p[1] / max(p[3], 1) for p in last3) / len(last3))
                cols["horse_wr"].append(sum(1 for p in h if p[1] == 1) / len(h))
            else:
                cols["prev_place"].append(5.5)
                cols["prev_pop"].append(5.5)
                cols["rest_days"].append(90)
                cols["prev_over"].append(0.0)
                cols["prev_pct"].append(0.5)
                cols["horse_wr"].append(0.0)
        for jk, tr, horse, win, top3, place, pop, field in recs:  # 当日の結果は翌日以降にだけ反映
            if place <= 0:  # 今週の出走予定(結果未確定)は統計に入れない
                continue
            for k, v in (("jockey", jk), ("trainer", tr)):
                s = stat[k][v]
                s[0] += 1
                s[1] += win
                s[2] += top3
            hist[horse].append((date, place, pop, field))
            n_all += 1
            w_all += win
            t_all += top3
    for k, v in cols.items():
        df[k] = v
    return df


def features(df: pd.DataFrame, use_pop: bool) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["age"] = df["age"]
    X["female"] = (df["sex"] == "牝").astype(int)
    X["gelding"] = (df["sex"] == "セ").astype(int)
    X["weight"] = df["weight"].fillna(55)
    X["bw"] = df["bw"].fillna(df["bw"].median())  # 馬体重はレース当日の発表(出走前に分かる)
    X["bw_diff"] = df["bw_diff"].fillna(0).clip(-20, 20)
    X["field"] = df["field"]
    X["num_rel"] = df["num"] / df["field"]
    X["dist"] = df["dist"] / 1000
    X["turf"] = (df["ground"] == "芝").astype(int)
    X["klass_lv"] = df["klass_lv"]
    for k in ("jockey_win", "jockey_top3", "trainer_win", "trainer_top3", "n_prev", "prev_place",
              "prev_pop", "rest_days", "prev_over", "prev_pct", "horse_wr"):
        X[k] = df[k]
    # 同じレース内での相対値(同じ条件の馬どうしで比べる)
    g = df["race_id"]
    for k in ("weight", "bw", "jockey_win", "trainer_win", "prev_place", "prev_over", "prev_pct", "horse_wr"):
        X[k + "_rel"] = X[k] - X.groupby(g)[k].transform("mean")
    if use_pop:
        X["log_pop"] = np.log(df["pop"].clip(1, 18))
        X["is_fav"] = (df["pop"] == 1).astype(int)
    return X


def race_normalize(p: np.ndarray, race_ids: pd.Series, total: float) -> np.ndarray:
    s = pd.Series(p).groupby(np.asarray(race_ids)).transform("sum").values
    return p / s * total


def fit_predict(Xtr, ytr, Xte, kind: str) -> np.ndarray:
    if kind == "logit":
        sc = StandardScaler().fit(Xtr)
        m = LogisticRegression(C=0.3, max_iter=2000).fit(sc.transform(Xtr), ytr)
        return m.predict_proba(sc.transform(Xte))[:, 1]
    m = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=150,
                                       l2_regularization=1.0, random_state=0).fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


MODELS = {  # 名前 -> (人気を使う?, 学習器)
    "人気だけ": ("pop_only", None),
    "独自(人気なし)": (False, "logit"),
    "独自+人気(線形)": (True, "logit"),
    "独自+人気(木)": (True, "gbm"),
}
BEST = "独自+人気(木)"


def evaluate(T: pd.DataFrame, key: str = "pw", asc: bool = False) -> dict:
    """各レースで『モデルの1位の馬』に100円賭け続けたら、の成績(同点は馬番の小さい方)。"""
    S = T.sort_values(["race_id", key, "num"], ascending=[True, asc, True])
    top = S.groupby("race_id", sort=False).head(1)
    n = len(top)
    w = S[S["win"] == 1].drop_duplicates("race_id")
    return dict(
        races=int(n), win_rate=float(top["win"].mean()), top3_rate=float(top["top3"].mean()),
        win_roi=float(top["win_pay"].sum() / (100 * n)), place_roi=float(top["place_pay"].sum() / (100 * n)),
        win_roi_se=float((top["win_pay"] / 100.0).std(ddof=1) / np.sqrt(n)),  # 回収率の標準誤差(±のぶれ)
        win_logloss=None if asc else float(-np.log(w[key].clip(lower=1e-6)).mean()),
    )


def run() -> dict:
    df = add_history(load())
    years = sorted(int(y) for y in df["year"].unique())
    first_test = years[0] + 4 if len(years) > 5 else years[0] + 3
    test_years = [y for y in years if y >= first_test]
    pred = {k: pd.Series(np.nan, index=df.index) for k in MODELS}
    pred3 = {k: pd.Series(np.nan, index=df.index) for k in MODELS}
    Xs = {True: features(df, True), False: features(df, False)}
    for y in test_years:
        tr, te = df["year"] < y, df["year"] == y
        for name, (use_pop, kind) in MODELS.items():
            if use_pop == "pop_only":
                # 人気順位ごとの過去の勝率/3着内率をそのまま使う(市場の見立てそのもの)
                pw = df.loc[te, "pop"].map(df[tr].groupby("pop")["win"].mean()).fillna(0.003)
                pt = df.loc[te, "pop"].map(df[tr].groupby("pop")["top3"].mean()).fillna(0.02)
            else:
                X = Xs[use_pop]
                pw = pd.Series(fit_predict(X[tr], df.loc[tr, "win"], X[te], kind), index=df.index[te])
                pt = pd.Series(fit_predict(X[tr], df.loc[tr, "top3"], X[te], kind), index=df.index[te])
            pred[name][te] = race_normalize(pw.values, df.loc[te, "race_id"], 1.0)
            pred3[name][te] = race_normalize(pt.values, df.loc[te, "race_id"], 3.0).clip(0, 1)
        print("検証済み", y, flush=True)
    te_all = df["year"] >= first_test
    T = df[te_all].copy()
    summary, by_class = {}, {}
    for name in MODELS:
        T["pw"] = pred[name][te_all]
        summary[name] = evaluate(T)
    summary["_baseline_fav"] = evaluate(T.assign(pw=T["pop"].astype(float)), asc=True)  # 人気1位=1番人気を買う
    # クラス別(最良モデル vs 1番人気)。区分が多いほど偶然のぶれも増える(多重比較)
    T["pw"] = pred[BEST][te_all]
    T["cls"] = np.where(T["grade"] != "", "重賞", T["klass"].replace(
        {"500万下": "1勝クラス", "1000万下": "2勝クラス", "1600万下": "3勝クラス"}))
    for c, g in T.groupby("cls"):
        if g["race_id"].nunique() >= 200:
            by_class[c] = dict(model=evaluate(g), fav=evaluate(g.assign(pw=g["pop"].astype(float)), asc=True))
    # 「モデルが人気(=みんな)より高く見ている馬」だけを全部買ったら? 払戻は勝った馬の単勝払戻で正確に出せる
    V = df[te_all].copy()
    V["ratio"] = pred[BEST][te_all] / pred["人気だけ"][te_all]
    value = []
    for k in (1.1, 1.25, 1.5, 2.0, 3.0):
        b = V[V["ratio"] >= k]
        r = b["win_pay"] / 100.0
        value.append(dict(ratio=k, bets=int(len(b)), hit=float(b["win"].mean()), roi=float(r.mean()),
                          se=float(r.std(ddof=1) / np.sqrt(len(b))),
                          place_roi=float((b["place_pay"] / 100.0).mean())))
    return dict(df=df, pred=pred, pred3=pred3, summary=summary, by_class=by_class, value=value,
                test_years=test_years)


def horse_rows(g: pd.DataFrame, **extra) -> list[dict]:
    return [dict(
        num=int(h["num"]), name=h["horse"], sex=h["sex"], age=int(h["age"]), jockey=h["jockey"],
        pop=int(h["pop"]), **{k: f(h) for k, f in extra.items()},
    ) for _, h in g.iterrows()]


def build_site(res: dict) -> None:
    df = res["df"]
    last = df["date"].max()
    view = df[df["year"].isin(res["test_years"]) &
              ((df["grade"] != "") | (df["date"] > last - pd.Timedelta(days=VIEW_DAYS)))].copy()
    view["pw"] = res["pred"][BEST][view.index]
    view["pt"] = res["pred3"][BEST][view.index]
    races = []
    for rid, g in view.groupby("race_id", sort=False):
        g = g.sort_values("place")
        r0 = g.iloc[0]
        races.append(dict(
            id=rid, date=r0["date"].strftime("%Y-%m-%d"), venue=r0["venue"], no=int(r0["race_no"]),
            name=r0["race"], grade=r0["grade"], klass=r0["klass"],
            ground=r0["ground"], dist=int(r0["dist"]), going=r0["going"],
            horses=horse_rows(g, place=lambda h: int(h["place"]), pw=lambda h: round(float(h["pw"]), 4),
                              pt=lambda h: round(float(h["pt"]), 4)),
            win_pay=int(g["win_pay"].max()),
        ))
    races.sort(key=lambda r: (r["date"], r["venue"], r["no"]), reverse=True)
    payload = dict(
        generated=pd.Timestamp.today().strftime("%Y-%m-%d"),
        data_range=[df["date"].min().strftime("%Y-%m-%d"), last.strftime("%Y-%m-%d")],
        n_rows=int(len(df)), n_races=int(df["race_id"].nunique()),
        test_years=res["test_years"], best=BEST, summary=res["summary"], by_class=res["by_class"], value=res["value"], races=races,
    )
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def predict_upcoming() -> None:
    """今週の出馬表を取得し、過去の全データで学習したモデルで勝率を出す。"""
    from keiba.upcoming import fetch
    up = pd.DataFrame(fetch())
    if up.empty:
        print("出馬表が取れませんでした(木曜16時ごろまでは空。予想は更新しません)")
        return
    hist = load()
    up["date"] = pd.to_datetime(up["date"])
    up["year"] = up["date"].dt.year
    up["place"] = 0
    up["win_pay"] = up["place_pay"] = 0
    # 履歴データと同じ列にそろえる(今週ぶんは結果が無いので place=0、統計には入れない)
    both = pd.concat([hist.assign(_up=0), up.assign(_up=1)], ignore_index=True)
    both = both.sort_values(["date", "race_id", "num"]).reset_index(drop=True)
    for c in ("win", "top3"):
        both[c] = both[c].fillna(0).astype(int)
    both["place"] = both["place"].fillna(0)
    both["grade"] = both["grade"].fillna("")
    both["klass"] = both["klass"].fillna("")
    both = add_history(prep(both))
    tr, te = both["_up"] == 0, both["_up"] == 1
    out = both[te].copy()
    out["has_pop"] = out.groupby("race_id")["pop"].transform(lambda s: bool((s > 0).all()))
    Xp, Xo = features(both, True), features(both, False)
    ytr_w, ytr_t = both.loc[tr, "win"], both.loc[tr, "top3"]
    pw_pop = fit_predict(Xp[tr], ytr_w, Xp[te], "gbm")
    pt_pop = fit_predict(Xp[tr], ytr_t, Xp[te], "gbm")
    pw_own = fit_predict(Xo[tr], ytr_w, Xo[te], "logit")
    pt_own = fit_predict(Xo[tr], ytr_t, Xo[te], "logit")
    out["pw"] = np.where(out["has_pop"], pw_pop, pw_own)
    out["pt"] = np.where(out["has_pop"], pt_pop, pt_own)
    out["pw_own"] = race_normalize(pw_own, out["race_id"], 1.0)
    out["pw"] = race_normalize(out["pw"].values, out["race_id"], 1.0)
    out["pt"] = race_normalize(out["pt"].values, out["race_id"], 3.0).clip(0, 1)
    races = []
    for rid, g in out.groupby("race_id", sort=False):
        g = g.sort_values("pw", ascending=False)
        r0 = g.iloc[0]
        races.append(dict(
            id=rid, date=r0["date"].strftime("%Y-%m-%d"), venue=r0["venue"], no=int(r0["race_no"]),
            name=r0["race"], grade=r0["grade"], klass=r0["klass"], ground=r0["ground"], dist=int(r0["dist"]),
            start=r0["start"] if isinstance(r0["start"], str) else "", with_odds=bool(r0["has_pop"]),
            horses=horse_rows(
                g, odds=lambda h: None if pd.isna(h["odds"]) else float(h["odds"]),
                pw=lambda h: round(float(h["pw"]), 4), pt=lambda h: round(float(h["pt"]), 4),
                pw_own=lambda h: round(float(h["pw_own"]), 4), n_prev=lambda h: int(h["n_prev"])),
        ))
    races.sort(key=lambda r: (r["date"], r["venue"], r["no"]))
    OUT_UP.write_text(json.dumps(dict(
        generated=pd.Timestamp.now(tz="Asia/Tokyo").strftime("%Y-%m-%d %H:%M"), races=races,
    ), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    from keiba import forward
    n = forward.record(races)  # 発走前のレースだけ、予想とオッズを記録(後で答え合わせ)
    print("今週の予想:", len(races), "レース", len(out), "頭 →", OUT_UP, f"/ 前向き記録 {n}レース")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "upcoming":
        return predict_upcoming()
    res = run()
    n = res["summary"]["_baseline_fav"]["races"]
    print(f"検証対象: {res['test_years']} / {n}レース")
    print(f"{'モデル':<16}{'1位的中':>8}{'3着内':>8}{'単勝回収':>9}{'±':>6}{'複勝回収':>9}{'logloss':>9}")
    for k, s in res["summary"].items():
        name = "1番人気を買う" if k == "_baseline_fav" else k
        ll = "-" if s["win_logloss"] is None else format(s["win_logloss"], ".3f")
        print(f"{name:<16}{s['win_rate']:>8.1%}{s['top3_rate']:>8.1%}{s['win_roi']:>9.1%}{s['win_roi_se']:>6.1%}"
              f"{s['place_roi']:>9.1%}{ll:>9}")
    print("\nクラス別(最良モデル vs 1番人気)")
    for c, v in res["by_class"].items():
        m, f = v["model"], v["fav"]
        print(f"{c:<10}{m['races']:>6}R  モデル 的中{m['win_rate']:.1%} 単勝{m['win_roi']:.1%} 複勝{m['place_roi']:.1%}"
              f" | 1番人気 的中{f['win_rate']:.1%} 単勝{f['win_roi']:.1%} 複勝{f['place_roi']:.1%}")
    print("\nモデルが人気より高く見た馬を全部買う(単勝100円ずつ)")
    for v in res["value"]:
        print(f"  モデル/人気 >= {v['ratio']:<4} {v['bets']:>6}点 的中{v['hit']:.1%} 単勝回収{v['roi']:.1%}(±{v['se']:.1%}) 複勝回収{v['place_roi']:.1%}")
    build_site(res)
    print("書き出し", OUT, OUT.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
