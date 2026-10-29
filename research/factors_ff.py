import os
import re

import numpy as np
import pandas as pd

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data") + os.sep

def load(fn):
    rows, cols = [], None
    for line in open(fn, encoding="latin-1"):
        p = [x.strip() for x in line.strip().split(",")]
        if cols is None and p and p[0] == "" and len(p) >= 2 and p[1]: cols = p[1:]; continue
        if cols and re.fullmatch(r"\d{6}", p[0]):
            rows.append([pd.Timestamp(p[0] + "01")] + [float(x) for x in p[1:]])
        elif cols and rows and not re.fullmatch(r"\d{6}", p[0]): break
    df = pd.DataFrame(rows, columns=["d"] + cols).set_index("d")
    return df.replace(-99.99, np.nan) / 100.0

us = load(D + "ff_F-F_Research_Data_5_Factors_2x3_CSV.csv")
mom = load(D + "ff_F-F_Momentum_Factor_CSV.csv"); mom.columns = [c.strip() for c in mom.columns]
us["MOM"] = mom.iloc[:, 0]
jp = load(D + "ff_Japan_5_Factors_CSV.csv"); jm = load(D + "ff_Japan_MOM_Factor_CSV.csv"); jp["MOM"] = jm.iloc[:, 0]

def desc(df, cols, a, b):
    x = df.loc[a:b, cols].dropna()
    n = len(x); m = x.mean() * 12 * 100; s = x.std() * np.sqrt(12) * 100
    t = x.mean() / (x.std() / np.sqrt(n))
    eq = (1 + x).cumprod(); dd = ((eq / eq.cummax()) - 1).min() * 100
    return pd.DataFrame({"年率%": m.round(1), "シャープ": (m / s).round(2), "t値": t.round(1), "最大下落%": dd.round(0)})

names = {"Mkt-RF": "市場(株-安全資産)", "SMB": "小型株-大型株", "HML": "割安-割高(バリュー)", "RMW": "高収益-低収益(クオリティ)", "CMA": "保守的投資-積極投資", "MOM": "モメンタム"}
if __name__ == "__main__":
    cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]
    for label, df, per in (("米国", us, [("1963-07", "1990-12"), ("1991-01", "2009-12"), ("2010-01", "2026-09"), ("2019-01", "2026-09")]),
                           ("日本", jp, [("1990-07", "2009-12"), ("2010-01", "2026-09"), ("2019-01", "2026-09")])):
        print("\n####", label, "(ロング・ショート因子の月次リターン。t値が2を超えると『偶然とは言いにくい』目安)")
        for a, b in per:
            d = desc(df, cols, a, b); d.index = [names[c] for c in d.index]
            print(f"\n[{a} 〜 {b}]"); print(d.to_string())
