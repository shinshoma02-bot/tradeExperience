"""放置運用のための定期処理: 成績の日次記録、週次レポート、月次リマインダー、節目の通知。

どれもスマホ通知(ntfy)と記録だけで、売買には一切関わらない。通知は『いつ・何回出たか』を記録に残し、
再起動しても二重に送らない。通知が多いと見なくなるので、週1・月1・節目だけにしている。
"""
from datetime import datetime


def iso_week_key(now: datetime) -> str:
    y, w, _ = now.isocalendar()
    return f"weekly:{y}-W{w:02d}"


def month_key(now: datetime) -> str:
    return f"monthly:{now.year}-{now.month:02d}"


def max_drawdown_pct(equities: list[float]) -> float:
    """資産の推移(円)から、直前の最高値からの最大の落ち込み(%、0以下)。"""
    peak, worst = None, 0.0
    for e in equities:
        peak = e if peak is None else max(peak, e)
        if peak:
            worst = min(worst, (e / peak - 1) * 100)
    return round(worst, 2)


def weekly_message(st: dict, dd: float, ai_days: int | None) -> tuple[str, str]:
    bench = st.get("benchmark_spy_return_pct")
    lines = [
        f"資産 {st['equity_jpy']:,.0f}円(元手{st['capital_jpy']:,.0f}円、{st['return_pct']:+.2f}%)",
        f"最大の落ち込み {dd:.1f}%、想定手数料 {st['fees_jpy']:,.0f}円",
        f"運用{st['days_running']}日目、決済{st['closed_trades']}回",
    ]
    if bench is not None:
        lines.append(f"同期間のSPY持ちっぱなし {bench:+.1f}%")
    g = st["real_money_gates"]
    lines.append("合格ライン: " + " / ".join(f"{k}{'○' if v else '×'}" for k, v in g.items()))
    if ai_days is not None:
        lines.append(f"AIの予測の記録 {ai_days}日分(60日分たまるまで的中率は偶然と区別できない)")
    lines.append("相場が荒れていても、積立は止めないのが検証の結論です")
    return "週次レポート(模擬売買)", "\n".join(lines)


def monthly_message() -> tuple[str, str]:
    return "今月のチェック(1分)", (
        "① 積立額は予定どおり入っているか\n"
        "② 通信費・保険などの固定費の請求額に変化はないか(年に数万円の差が出る)\n"
        "③ NISAの年間枠の残り(つみたて120万円・成長240万円)\n"
        "運用の細かい値動きは見なくて大丈夫です。")


def milestone_messages(st: dict) -> list[tuple[str, str, str]]:
    """(記録キー, タイトル, 本文)。条件を満たした節目だけ、1回ずつ。"""
    out = []
    if st["days_running"] >= 90:
        out.append(("milestone:days90", "模擬売買が90日たちました", "合格ラインの1つ目に到達。成績を見て、続けるか判断する時期です。"))
    if st["closed_trades"] >= 30:
        out.append(("milestone:trades30", "決済した取引が30回に到達", "標本が少したまりました。偶然との区別はまだ難しいので、数字だけで判断しないでください。"))
    if st["ready_for_real_money"]:
        out.append(("milestone:ready", "合格ラインを全部満たしました",
                    "実資金の成功を保証するものではありません。使うなら失ってもよい額から。SPY持ちっぱなしとの比較も見てください。"))
    return out


def drawdown_from_high(closes_with_ts: list[tuple[int, float]]) -> tuple[float, int] | None:
    """(現在の高値からの下落率(%、0以下), 高値を付けた日のタイムスタンプ)。データが無ければ None。"""
    if not closes_with_ts:
        return None
    ts_high, high = max(closes_with_ts, key=lambda x: x[1])
    return round((closes_with_ts[-1][1] / high - 1) * 100, 2), ts_high


CRASH_LEVELS = (15, 25)


def crash_messages(dd_pct: float, ts_high: int) -> list[tuple[str, str, str]]:
    """市場(SPY)が高値から-15%/-25%に達したとき、積立を続ける理由を1通だけ送る(下落の局面ごとに各1回)。
    根拠: 積立を止める/売る行動は、実データで最終資産を中央値で約10〜40%削った(例外あり。research/behavior_gap.py)。"""
    out = []
    for lv in CRASH_LEVELS:
        if dd_pct <= -lv:
            out.append((f"crash{lv}:{ts_high}", f"相場が高値から-{lv}%です(積立は続けましょう)",
                        "過去のデータでは、下落で積立を止める・売ると、最終資産が中央値で約10〜40%減りました(例外もあります)。\n"
                        "確認すること: ①生活防衛資金は足りているか ②積立額に無理はないか ③10年以内に取り崩す予定はないか\n"
                        "『売る・止める』を押したくなったら、翌日まで待ってください。値動きを毎日見る必要はありません。"))
    return out
