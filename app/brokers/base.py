from abc import ABC, abstractmethod

from app.models import Fill, Signal


class Broker(ABC):
    """市場ごとの発注窓口。実ブローカーはこれを継承して作る。"""

    @abstractmethod
    def execute(self, signal: Signal) -> Fill | None:
        """注文を出して約定結果を返す。何もしなかったら None。"""

    @abstractmethod
    def positions(self) -> dict:
        """現在のポジション一覧。"""
