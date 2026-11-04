"""重賞レースの着順予測モデル(ウォークフォワード検証つき)。

- 特徴量は全部「そのレースの前に分かること」だけ。騎手・調教師・馬の成績は
  そのレースより前の分(同日は含めない)で計算する。
- 学習は「予測する年より前」だけ。2022年は2019-21で学習、2023年は2019-22で…と進める。
- 出力: keiba/site/data.json(サイトが読む) と標準出力のサマリ
使い方: python -m keiba.model
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
CSV = HERE / "data" / "jra_graded_results.csv"
OUT = HERE / "site" / "data.json"
FIRST_TEST_YEAR = 2022
PRIOR_N = 20  # 騎手/調教師の勝率は「平均に引き寄せる」ための仮想サンプル数(少数の過大評価を防ぐ)


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df = df[df["place"] > 0].copy()  # 競走中止・除外は予測の対象外
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df = df.sort_values(["date", "race_id", "num"]).reset_index(drop=True)
    df["field"] = df.groupby("race_id")["num"].transform("count")
    df["win"] = (df["place"] == 1).astype(int)
    df["top3"] = (df["place"] <= 3).astype(int)
    return df


def add_history(df: pd.DataFrame) -> pd.DataFrame:
    """騎手・調教師・馬の「過去成績」を、時系列順に一日ずつ更新しながら付ける。"""
    stat = {k: defaultdict(lambda: [0, 0, 0]) for k in ("jockey", "trainer")}  # 出走, 勝, 3着内
    hist = defaultdict(list)  # 馬 -> [(date, place, pop)]
    cols = defaultdict(list)
    base_w = base_t = 0.0
    n_all = w_all = t_all = 0
    for date, day in df.groupby("date", sort=True):
        rate_w = (w_all + 1) / (n_all + 10)
        rate_t = (t_all + 3) / (n_all + 10)
        for _, r in day.iterrows():
            for k in ("jockey", "trainer"):
                n, w, t = stat[k][r[k]]
                cols[f"{k}_win"].append((w + PRIOR_N * rate_w) / (n + PRIOR_N))
                cols[f"{k}_top3"].append((t + PRIOR_N * rate_t) / (n + PRIOR_N))
                cols[f"{k}_n"].append(n)
            h = hist[r["horse"]]
            cols["n_prev"].append(len(h))
            if h:
                last = h[-1]
                cols["prev_place"].append(min(last[1], 10))
                cols["prev_pop"].append(min(last[2], 10))
                cols["rest_days"].append(min((date - last[0]).days, 365))
                cols["prev_over"].append(np.mean([min(p[2], 10) - min(p[1], 10) for p in h[-3:]]))
            else:
                cols["prev_place"].append(5.5)
                cols["prev_pop"].append(5.5)
                cols["rest_days"].append(90)
                cols["prev_over"].append(0.0)
        for _, r in day.iterrows():  # 当日の結果は翌日以降にだけ反映
            for k in ("jockey", "trainer"):
                s = stat[k][r[k]]
                s[0] += 1
                s[1] += r["win"]
                s[2] += r["top3"]
            hist[r["horse"]].append((date, r["place"], r["pop"]))
            n_all += 1
            w_all += r["win"]
            t_all += r["top3"]
    for k, v in cols.items():
        df[k] = v
    return df


def features(df: pd.DataFrame, use_pop: bool) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["age"] = df["age"]
    X["female"] = (df["sex"] == "牝").astype(int)
    X["gelding"] = (df["sex"] == "セ").astype(int)
    X["weight"] = df["weight"].fillna(df["weight"].median())
    # 馬体重はレース当日の発表(出走前に分かる)。欠損は中央値
    X["bw"] = df["bw"].fillna(df["bw"].median())
    X["bw_diff"] = df["bw_diff"].fillna(0).clip(-20, 20)
    X["field"] = df["field"]
    X["num_rel"] = df["num"] / df["field"]
    X["dist"] = df["dist"] / 1000
    X["turf"] = (df["ground"] == "芝").astype(int)
    for k in ("jockey_win", "jockey_top3", "trainer_win", "trainer_top3", "n_prev",
              "prev_place", "prev_pop", "rest_days", "prev_over"):
        X[k] = df[k]
    # 同じレース内での相対値(同じ条件の馬どうしで比べる)
    for k in ("weight", "bw", "jockey_win", "trainer_win", "prev_place", "prev_over"):
        X[k + "_rel"] = X[k] - X.groupby(df["race_id"])[k].transform("mean")
    if use_pop:
        X["log_pop"] = np.log(df["pop"].clip(1, 18))
        X["is_fav"] = (df["pop"] == 1).astype(int)
    return X


def race_normalize(p: np.ndarray, race_ids: pd.Series, total: float) -> np.ndarray:
    s = pd.Series(p).groupby(race_ids.values).transform("sum").values
    return p / s * total


def fit_predict(Xtr, ytr, Xte, kind: str) -> np.ndarray:
    if kind == "logit":
        sc = StandardScaler().fit(Xtr)
        m = LogisticRegression(C=0.3, max_iter=2000).fit(sc.transform(Xtr), ytr)
        return m.predict_proba(sc.transform(Xte))[:, 1]
    m = GradientBoostingClassifier(n_estimators=120, max_depth=2, learning_rate=0.05,
                                   subsample=0.8, random_state=0).fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def run() -> dict:
    df = add_history(load())
    test_years = [y for y in sorted(df["year"].unique()) if y >= FIRST_TEST_YEAR]
    models = {  # 名前 -> (使う特徴量, 学習器)
        "人気だけ": ("pop_only", None),
        "独自(人気なし)": (False, "logit"),
        "独自+人気(線形)": (True, "logit"),
        "独自+人気(木)": (True, "gbm"),
    }
    pred = {k: pd.Series(np.nan, index=df.index) for k in models}
    pred3 = {k: pd.Series(np.nan, index=df.index) for k in models}
    for y in test_years:
        tr, te = df["year"] < y, df["year"] == y
        for name, (use_pop, kind) in models.items():
            if use_pop == "pop_only":
                # 人気順位ごとの過去の勝率/3着内率をそのまま使う(市場の見立てそのもの)
                tab_w = df[tr].groupby("pop")["win"].mean()
                tab_t = df[tr].groupby("pop")["top3"].mean()
                pw = df.loc[te, "pop"].map(tab_w).fillna(0.005)
                pt = df.loc[te, "pop"].map(tab_t).fillna(0.03)
            else:
                X = features(df, use_pop)
                pw = pd.Series(fit_predict(X[tr], df.loc[tr, "win"], X[te], kind), index=df.index[te])
                pt = pd.Series(fit_predict(X[tr], df.loc[tr, "top3"], X[te], kind), index=df.index[te])
            pred[name][te] = race_normalize(pw.values, df.loc[te, "race_id"], 1.0)
            pred3[name][te] = race_normalize(pt.values, df.loc[te, "race_id"], 3.0).clip(0, 1)
    te_all = df["year"] >= FIRST_TEST_YEAR
    T = df[te_all].copy()
    summary = {}
    for name in models:
        T["pw"], T["pt"] = pred[name][te_all], pred3[name][te_all]
        summary[name] = evaluate(T)
    summary["_baseline_fav"] = evaluate(T.assign(pw=1 / T["pop"].clip(1), pt=1 / T["pop"].clip(1)),
                                        rank_key="pop")
    summary["_baseline_fav"]["win_logloss"] = None  # 確率になっていないので比較対象外(JSONではnull)
    return dict(df=df, T=T, pred=pred, pred3=pred3, summary=summary, models=list(models), test_years=test_years)


def evaluate(T: pd.DataFrame, rank_key: str | None = None) -> dict:
    """各レースで『モデルの1位の馬』に100円賭け続けたら、の成績。"""
    races = []
    for _, g in T.groupby("race_id"):
        if rank_key:
            top = g.sort_values(rank_key).iloc[0]
        else:
            top = g.sort_values("pw", ascending=False).iloc[0]
        races.append(dict(
            hit1=int(top["place"] == 1), hit3=int(top["place"] <= 3),
            win_pay=top["win_pay"], place_pay=top["place_pay"],
            logp=np.log(max(g.loc[g["win"] == 1, "pw"].iloc[0], 1e-6)) if g["win"].sum() else np.nan,
        ))
    R = pd.DataFrame(races)
    n = len(R)
    return dict(
        races=n,
        win_rate=R["hit1"].mean(), top3_rate=R["hit3"].mean(),
        win_roi=R["win_pay"].sum() / (100 * n), place_roi=R["place_pay"].sum() / (100 * n),
        win_logloss=-R["logp"].mean(),
    )


def build_site(res: dict) -> None:
    df, T = res["df"], res["T"]
    best = "独自+人気(木)"
    T = T.copy()
    for m in res["models"]:
        T["pw_" + m] = res["pred"][m][T.index]
        T["pt_" + m] = res["pred3"][m][T.index]
    races = []
    for rid, g in T.groupby("race_id", sort=False):
        g = g.sort_values("place")
        r0 = g.iloc[0]
        races.append(dict(
            id=rid, date=r0["date"].strftime("%Y-%m-%d"), name=r0["race"], grade=r0["grade"],
            ground=r0["ground"], dist=int(r0["dist"]), going=r0["going"],
            horses=[dict(
                num=int(h["num"]), name=h["horse"], sex=h["sex"], age=int(h["age"]), jockey=h["jockey"],
                pop=int(h["pop"]), place=int(h["place"]),
                pw=round(float(h["pw_" + best]), 4), pt=round(float(h["pt_" + best]), 4),
                pw_own=round(float(h["pw_独自(人気なし)"]), 4),
                pw_mkt=round(float(h["pw_人気だけ"]), 4),
            ) for _, h in g.iterrows()],
            win_pay=int(g["win_pay"].max()),
        ))
    races.sort(key=lambda r: r["date"], reverse=True)
    payload = dict(
        generated=pd.Timestamp.today().strftime("%Y-%m-%d"),
        data_range=[df["date"].min().strftime("%Y-%m-%d"), df["date"].max().strftime("%Y-%m-%d")],
        n_train_rows=int(len(df)), test_years=[int(y) for y in res["test_years"]],
        best=best, summary=res["summary"], races=races,
    )
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def main() -> None:
    res = run()
    print(f"検証対象: {[int(y) for y in res['test_years']]} / {res['summary']['_baseline_fav']['races']}レース")
    print(f"{'モデル':<16}{'1位的中':>8}{'3着内':>8}{'単勝回収':>9}{'複勝回収':>9}{'logloss':>9}")
    for k, s in res["summary"].items():
        name = "1番人気を買う" if k == "_baseline_fav" else k
        print(f"{name:<16}{s['win_rate']:>8.1%}{s['top3_rate']:>8.1%}{s['win_roi']:>9.1%}{s['place_roi']:>9.1%}{('-' if s['win_logloss'] is None else format(s['win_logloss'], '.3f')):>9}")
    build_site(res)
    print("書き出し", OUT, OUT.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
