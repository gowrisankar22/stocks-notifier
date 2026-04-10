from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel


class SignalAction(str, Enum):
    STRONG_BUY = "STRONG BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG SELL"


class TimeframeSignal(BaseModel):
    timeframe: str
    label: str
    action: SignalAction
    score: float
    confidence: float
    details: dict[str, float]


class TechnicalIndicators(BaseModel):
    rsi: dict[str, Any] | None = None
    macd: dict[str, Any] | None = None
    bollinger_bands: dict[str, Any] | None = None
    sma: dict[str, float] | None = None
    ema: dict[str, float] | None = None
    stochastic: dict[str, Any] | None = None
    adx: dict[str, Any] | None = None
    atr: dict[str, Any] | None = None
    obv: dict[str, Any] | None = None
    fibonacci: dict[str, float] | None = None
    volume_analysis: dict[str, Any] | None = None
    trend_strength: dict[str, Any] | None = None


class PredictionResult(BaseModel):
    direction: str
    confidence: float
    model_accuracy: float | None = None
    features_used: int = 0


class StockAnalysis(BaseModel):
    ticker: str
    name: str
    price: float
    change: float
    change_percent: float
    volume: int
    market_cap: float | None = None
    fifty_two_week_high: float | None = None
    fifty_two_week_low: float | None = None
    currency: str = "USD"
    currency_symbol: str = "$"
    indicators: TechnicalIndicators
    signals: dict[str, TimeframeSignal]
    prediction: dict[str, PredictionResult] | None = None
    news: list[dict[str, Any]] = []
    chart_data: dict[str, list] | None = None
    timestamp: str


class WatchlistItem(BaseModel):
    ticker: str
    name: str
    price: float
    change: float
    change_percent: float
    overall_signal: SignalAction


class SearchResult(BaseModel):
    ticker: str
    name: str
    exchange: str
    type: str
