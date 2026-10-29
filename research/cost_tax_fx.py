"""コスト・税・為替の最適化の試算(日本の個人、NISA中心)。投資助言ではなく計算の整理。
実行: python research/cost_tax_fx.py   (Yahoo と BIS に接続できる環境。取得結果は ./_cache に保存)
1) 信託報酬の差の複利計算  2) 円建て SPY のヘッジ有無比較  3) 配当の税(特定口座 vs NISA)
4) 売買・為替コストの少額での影響  5) クレカ積立ポイントの金額換算
税率・手数料・還元率は前提値(変わる。要確認)。
"""
import io, json, os, subprocess
import numpy as np
import pandas as pd

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cache")
os.makedirs(CACHE, exist_ok=True)
UA = "Mozilla/5.0"


def curl(url, name):
    p = os.path.join(CACHE, name)
    if not os.path.exists(p) or os.path.getsize(p) < 1000:
        subprocess.run(["curl", "-sS", "-A", UA, "-o", p, url], check=True)
    return p


# ---------- 1. 信託報酬の差 ----------
def fv(monthly, years, annual, cost):
    r = (1 + annual - cost) ** (1 / 12) - 1   # 月利(コストは年率で差し引く近似)
    n = years * 12
    return monthly * (((1 + r) ** n - 1) / r) * (1 + r)  # 月初積立


def section_cost():
    print("=== 1. 信託報酬 0.05% vs 0.5% (年率コストを差し引く近似。税・為替は無視) ===")
    print("月額  年数  年率  元本     0.05%後     0.5%後    差(円)   差/元本")
    for m in (10000, 30000):
        for y in (20, 30):
            for a in (0.05, 0.07):
                lo, hi = fv(m, y, a, 0.0005), fv(m, y, a, 0.005)
                print(f"{m:>6} {y:>3}年 {a:.0%} {m*12*y:>9,} {lo:>11,.0f} {hi:>11,.0f} {lo-hi:>9,.0f} {(lo-hi)/(m*12*y):6.1%}")
    # 1800万円を一括保有した場合の年間コスト額
    print("参考: 1,000万円保有時の年間コスト 0.05%=5,000円 / 0.5%=50,000円")


# ---------- 2. 為替ヘッジ ----------
def chart_df(path, adj=False):
    j = json.load(open(path))["chart"]["result"][0]
    ts = pd.to_datetime(j["timestamp"], unit="s").normalize()
    ind = j["indicators"]
    c = (ind["adjclose"][0]["adjclose"] if adj else ind["quote"][0]["close"])
    s = pd.Series(c, index=ts, dtype=float).dropna()
    return s[~s.index.duplicated()]


def stats(r, label):
    r = r.dropna()
    eq = (1 + r).cumprod()
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    mdd = (eq / eq.cummax() - 1).min()
    sh = r.mean() / r.std() * np.sqrt(252)
    vol = r.std() * np.sqrt(252)
    print(f"{label:<22} 年率{cagr:6.2%} 年vol{vol:6.1%} 最大下落{mdd:7.1%} シャープ(超過無し){sh:5.2f}")
    return eq


def year_ret(r, y):
    x = r[str(y)]
    return (1 + x).prod() - 1


def section_fx(hedge_spread=0.001):
    print("\n=== 2. 円建て SPY: ヘッジなし / ヘッジあり(金利差を差し引く) ===")
    spy = chart_df(curl("https://query1.finance.yahoo.com/v8/finance/chart/SPY?period1=1104537600&period2=2000000000&interval=1d&events=div,split", "spy.json"), adj=True)
    fx = chart_df(curl("https://query1.finance.yahoo.com/v8/finance/chart/USDJPY%3DX?period1=1104537600&period2=2000000000&interval=1d", "fx.json"))
    bis = pd.read_csv(curl("https://stats.bis.org/api/v2/data/dataflow/BIS/WS_CBPOL/1.0/M.JP+US?format=csv&startPeriod=2004-12", "bis.csv"))
    bis["d"] = pd.to_datetime(bis.TIME_PERIOD + "-01")
    rt = bis.pivot(index="d", columns="REF_AREA", values="OBS_VALUE").sort_index() / 100
    idx = spy.index
    rt = rt.reindex(rt.index.union(idx)).ffill().reindex(idx)
    # Yahooの為替の日付tの終値は実質0:00UTC頃 → 米国株の同日終値(21:00UTC)に合わせるため、翌日ラベルの値を使う(1日ラグ補正)
    fxa = fx.shift(-1).reindex(idx)
    df = pd.DataFrame({"spy": spy, "fx": fxa, "us": rt["US"], "jp": rt["JP"]}).dropna()
    df = df[df.index >= "2005-01-03"]
    rs, rf = df.spy.pct_change(), df.fx.pct_change()
    unhedged = (1 + rs) * (1 + rf) - 1
    # ヘッジコスト≒(米金利-日金利)+上乗せ。米金利が高いほどヘッジは不利(円買い/ドル売りの先物は金利差分ディスカウント)
    hc = (df.us - df.jp + hedge_spread) / 365.0 * df.index.to_series().diff().dt.days.fillna(1)
    hedged = rs - hc
    usd = rs
    print(f"期間 {df.index[0].date()} 〜 {df.index[-1].date()} / 平均 米金利{df.us.mean():.2%} 日金利{df.jp.mean():.2%} / ヘッジ上乗せ{hedge_spread:.2%}(仮定)")
    stats(usd, "SPY ドル建て")
    stats(unhedged, "円建て ヘッジなし")
    stats(hedged, "円建て ヘッジあり")
    print("年別(%)  ドル建て / ヘッジなし / ヘッジあり")
    for y in (2008, 2009, 2013, 2020, 2022, 2024):
        if str(y) in unhedged.index.year.astype(str):
            print(f"  {y}: {year_ret(usd,y):7.1%} {year_ret(unhedged,y):7.1%} {year_ret(hedged,y):7.1%}")
    for a, b, lab in (("2005", "2012", "2005-2012"), ("2013", "2026", "2013-2026"), ("2022", "2026", "2022-2026")):
        u, h = unhedged[a:b], hedged[a:b]
        yrs = (u.index[-1] - u.index[0]).days / 365.25
        print(f"  {lab}: ヘッジなし年率{(1+u).prod()**(1/yrs)-1:6.2%}  ヘッジあり年率{(1+h).prod()**(1/yrs)-1:6.2%}")
    # 2008年ピークからの最大下落(金融危機中)
    for lab, s_, e_ in (("2007-10〜2009-03", "2007-10-09", "2009-03-09"), ("2022 年初来安値", "2022-01-03", "2022-10-14")):
        print(f"  {lab}: ドル建て{(1+usd[s_:e_]).prod()-1:6.1%} ヘッジなし{(1+unhedged[s_:e_]).prod()-1:6.1%} ヘッジあり{(1+hedged[s_:e_]).prod()-1:6.1%}")
    print("注: ヘッジコストは政策金利差の概算(実際のヘッジ付ファンドは先物・コール金利差+運用費用で、数字は異なる)。要確認")


# ---------- 3. 配当の税 ----------
def section_tax(years=20, price_ret=0.05, div_yield=0.03, monthly=30000):
    print("\n=== 3. 配当の税: 高配当(利回り3%) vs 無配当(成長)、年平均総リターン同一(price_ret+div)の比較 ===")
    US_WH, JP = 0.10, 0.20315
    print(f"米国源泉{US_WH:.0%}, 日本{JP:.3%}。外国税額控除ありの実質: 配当 {JP:.2%}(控除フル)、NISA: 配当 {US_WH:.0%}(米国分のみ・戻らない)")
    print(f"二重課税(控除なし)の配当実質税率 {US_WH + (1-US_WH)*JP:.2%}")

    def run(div, growth, acct):
        v = 0.0; basis = 0.0
        for m in range(years * 12):
            v += monthly; basis += monthly
            v *= (1 + growth) ** (1 / 12)
            d = v * div / 12
            if acct == "nisa": tax = d * US_WH           # 米国のみ戻らない
            elif acct == "tokutei_ctrl": tax = d * JP      # 外国税額控除で米国分を取り戻す(上限内)
            elif acct == "tokutei_noctrl": tax = d * (US_WH + (1 - US_WH) * JP)
            v += d - tax                                   # 税引後を再投資
            basis += d - tax
        gain = v - basis
        if acct == "nisa": exit_tax = 0
        else: exit_tax = max(gain, 0) * JP                 # 最後に全売却
        return v - exit_tax, basis
    tot = price_ret + div_yield
    print(f"総リターン年{tot:.0%}同一、毎月{monthly:,}円×{years}年。最終手取り(円)")
    for name, div, g in (("高配当(配当3%+値上がり2%)", 0.03, 0.02), ("成長(配当0%+値上がり5%)", 0.0, 0.05)):
        row = []
        for acct in ("nisa", "tokutei_ctrl", "tokutei_noctrl"):
            v, b = run(div, g, acct); row.append(v)
        print(f"  {name}: NISA {row[0]:>12,.0f} / 特定(控除あり) {row[1]:>12,.0f} / 特定(控除なし) {row[2]:>12,.0f}")
    # 日本株/国内ファンドは源泉なしで NISA が全額非課税
    print("参考: 国内の配当(日本株・国内上場)は NISA で完全非課税。米国配当は NISA でも10%引かれる(要確認)")


# ---------- 4. 売買・為替コスト ----------
def section_trade():
    print("\n=== 4. 少額での売買・為替コスト(前提値。要確認) ===")
    for amt in (10000, 30000, 100000):
        # NISA買付手数料0 / 為替スプレッド 0〜25銭(1ドル150円想定) / 非NISA手数料0.495%
        fx25 = 0.25 / 150
        print(f"  {amt:>7,}円分を買う: 為替25銭なら {amt*fx25:,.0f}円 往復{amt*fx25*2:,.0f}円 / 非NISA約定0.495%(上限22ドル) {amt*0.00495:,.0f}円 / 為替0銭+NISA 0円")
    print("  投信: ノーロード(購入時手数料0)前提。購入時3%なら1万円で300円=信託報酬0.5%の6年分")


# ---------- 5. クレカ積立 ----------
def section_card():
    print("\n=== 5. クレカ積立ポイント(還元率は前提値。要確認) ===")
    for rate in (0.005, 0.01, 0.02):
        for m in (10000, 50000, 100000):
            print(f"  還元{rate:.1%} 月{m:>7,}円積立 → 年{m*12*rate:>7,.0f}円 ({m*12*rate/(m*12):.1%}) ; 信託報酬0.5%→0.05%の節約(積立元本の半分が平均残高の想定)")
            break
    # 還元率と信託報酬の同じ土俵: 年間コスト額との比較
    bal = 3_000_000
    print(f"  参考: 残高{bal:,}円の年間信託報酬 0.05%={bal*0.0005:,.0f}円 0.5%={bal*0.005:,.0f}円。月3万円(年36万円)の還元0.5%={360000*0.005:,.0f}円/1%={360000*0.01:,.0f}円")


if __name__ == "__main__":
    section_cost()
    section_fx()
    section_tax()
    section_trade()
    section_card()
