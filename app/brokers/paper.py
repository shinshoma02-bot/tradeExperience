from app.brokers.base import Broker
from app.models import Fill, Signal


class PaperBroker(Broker):
    """お金を使わない模擬ブローカー。シグナルの価格でそのまま約定させる。

    ポジションは (戦略, 銘柄) ごとの正味数量(買い=プラス、売り=マイナス)と平均取得価格で持つ。
    戦略ごとに分けているのは、複数の戦略の成績を比較するため。
    """

    def __init__(self) -> None:
        self._pos: dict[tuple[str, str], tuple[float, float]] = {}  # key -> (qty, avg_price)

    def execute(self, signal: Signal) -> Fill | None:
        key = (signal.strategy, signal.symbol)
        qty, avg = self._pos.get(key, (0.0, 0.0))

        if signal.action == "close":
            if qty == 0:
                return None
            side = "sell" if qty > 0 else "buy"
            trade_qty = abs(qty)
        else:
            side = signal.action
            trade_qty = signal.qty
            if trade_qty <= 0:
                return None

        signed = trade_qty if side == "buy" else -trade_qty
        pnl = 0.0
        new_qty = qty + signed

        if qty == 0 or (qty > 0) == (signed > 0):
            # 新規 or 買い増し/売り増し: 平均取得価格を更新
            total = abs(qty) + trade_qty
            avg = (abs(qty) * avg + trade_qty * signal.price) / total
        else:
            # 逆方向 = 決済(一部 or 全部、場合によってはドテン)
            closing = min(abs(qty), trade_qty)
            direction = 1 if qty > 0 else -1
            pnl = (signal.price - avg) * closing * direction
            if abs(new_qty) < 1e-12:
                new_qty, avg = 0.0, 0.0
            elif (new_qty > 0) != (qty > 0):
                avg = signal.price  # ドテンした分の新しい取得価格

        if new_qty == 0:
            self._pos.pop(key, None)
        else:
            self._pos[key] = (new_qty, avg)

        return Fill(
            market=signal.market,
            strategy=signal.strategy,
            symbol=signal.symbol,
            side=side,
            qty=trade_qty,
            price=signal.price,
            realized_pnl=round(pnl, 8),
        )

    def positions(self) -> dict:
        return {f"{s}:{sym}": {"qty": q, "avg_price": a} for (s, sym), (q, a) in self._pos.items()}
