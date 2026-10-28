from typing import Literal, Optional

from pydantic import BaseModel, Field

Market = Literal["crypto", "fx", "us_stock", "jp_stock", "futures"]


class Signal(BaseModel):
    """TradingViewのアラートから届く売買シグナル。"""

    secret: str
    market: Market
    strategy: str = Field(default="default", description="どの戦略のシグナルか(成績比較用)")
    symbol: str
    action: Literal["buy", "sell", "close"]
    price: float = Field(gt=0, description="TradingViewの {{close}} を渡す")
    qty: float = Field(default=0, ge=0, description="close のときは省略可")
    quote: str = Field(default="USD", description="価格の通貨。JPY なら円建て")


class Fill(BaseModel):
    """約定結果。"""

    market: str
    strategy: str
    symbol: str
    side: Literal["buy", "sell"]
    qty: float
    price: float
    realized_pnl: float = 0.0
    quote: str = "USD"
