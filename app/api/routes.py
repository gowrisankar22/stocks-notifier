from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from app.config import DEFAULT_MARKET, MARKETS, settings
from app.models import SignalAction, StockAnalysis, TechnicalIndicators
from app.services.data_fetcher import fetcher
from app.services.signal_engine import prediction_engine, signal_engine
from app.services.technical_analysis import TechnicalAnalyzer

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

# Session state
_current_market: str = DEFAULT_MARKET
_watchlist: set[str] = set(MARKETS[DEFAULT_MARKET].default_tickers)


def _market_info() -> dict[str, Any]:
    m = MARKETS[_current_market]
    return {
        "id": m.id,
        "name": m.name,
        "flag": m.flag,
        "currency": m.currency,
        "currency_symbol": m.currency_symbol,
    }


# ======================================================================
# Market endpoints
# ======================================================================


@router.get("/markets")
async def list_markets() -> dict[str, Any]:
    markets = [
        {
            "id": m.id,
            "name": m.name,
            "flag": m.flag,
            "currency": m.currency,
            "currency_symbol": m.currency_symbol,
        }
        for m in MARKETS.values()
    ]
    return {"markets": markets, "active": _current_market}


@router.post("/markets/{market_id}")
async def switch_market(market_id: str) -> dict[str, Any]:
    global _current_market, _watchlist

    market_id = market_id.lower()
    if market_id not in MARKETS:
        return {"error": f"Unknown market: {market_id}", "available": list(MARKETS.keys())}

    _current_market = market_id
    _watchlist = set(MARKETS[market_id].default_tickers)
    logger.info("Switched to market: %s", market_id)

    return {"status": "switched", **_market_info()}


# ======================================================================
# Stock analysis endpoints
# ======================================================================


@router.get("/analyze/{ticker}")
async def analyze_stock(ticker: str) -> StockAnalysis:
    """Full analysis for a single ticker."""
    return await asyncio.to_thread(_run_analysis, ticker.upper())


@router.get("/quick/{ticker}")
async def quick_quote(ticker: str) -> dict[str, Any]:
    """Lightweight quote + overall signal."""
    ticker = ticker.upper()
    quote = await asyncio.to_thread(fetcher.get_quote, ticker)

    try:
        df = await asyncio.to_thread(fetcher.get_history, ticker, period="6mo")
        analyzer = TechnicalAnalyzer(df)
        indicators = analyzer.calculate_all()
        signals = signal_engine.generate_all_signals(indicators)
        overall = signal_engine.overall_action(signals)
    except Exception:
        overall = SignalAction.HOLD

    return {**quote, "overall_signal": overall.value}


@router.get("/search")
async def search_stocks(q: str = Query(..., min_length=1)) -> list[dict]:
    results = await asyncio.to_thread(fetcher.search, q)
    return results


# ======================================================================
# Watchlist endpoints
# ======================================================================


@router.get("/watchlist")
async def get_watchlist() -> dict[str, Any]:
    tasks = [asyncio.to_thread(_quick_watchlist_item, t) for t in sorted(_watchlist)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    items = [r for r in results if isinstance(r, dict)]
    return {"items": items, "market": _market_info()}


@router.post("/watchlist/{ticker}")
async def add_to_watchlist(ticker: str) -> dict[str, str]:
    ticker = ticker.upper()
    _watchlist.add(ticker)
    return {"status": "added", "ticker": ticker}


@router.delete("/watchlist/{ticker}")
async def remove_from_watchlist(ticker: str) -> dict[str, str]:
    ticker = ticker.upper()
    _watchlist.discard(ticker)
    return {"status": "removed", "ticker": ticker}


# ======================================================================
# WebSocket
# ======================================================================

_ws_clients: set[WebSocket] = set()


@router.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    _ws_clients.add(websocket)
    logger.info("WebSocket client connected (%d total)", len(_ws_clients))
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                action = msg.get("action")
                if action == "analyze":
                    ticker = msg.get("ticker", "").upper()
                    if ticker:
                        result = await asyncio.to_thread(_run_analysis, ticker)
                        await websocket.send_text(
                            json.dumps({"type": "analysis", "data": result.model_dump()})
                        )
                elif action == "subscribe":
                    tickers = [t.upper() for t in msg.get("tickers", [])]
                    for t in tickers:
                        _watchlist.add(t)
                    await websocket.send_text(
                        json.dumps({"type": "subscribed", "tickers": tickers})
                    )
            except json.JSONDecodeError:
                await websocket.send_text(
                    json.dumps({"type": "error", "message": "Invalid JSON"})
                )
    except WebSocketDisconnect:
        pass
    finally:
        _ws_clients.discard(websocket)
        logger.info("WebSocket client disconnected (%d remaining)", len(_ws_clients))


async def broadcast_updates() -> None:
    """Background task that periodically pushes watchlist updates."""
    while True:
        await asyncio.sleep(settings.refresh_interval_seconds)
        if not _ws_clients or not _watchlist:
            continue

        for ticker in list(_watchlist):
            try:
                item = await asyncio.to_thread(_quick_watchlist_item, ticker)
                payload = json.dumps({
                    "type": "quote_update",
                    "data": item,
                    "market": _market_info(),
                })
                dead: list[WebSocket] = []
                for ws in _ws_clients:
                    try:
                        await ws.send_text(payload)
                    except Exception:
                        dead.append(ws)
                for ws in dead:
                    _ws_clients.discard(ws)
            except Exception as e:
                logger.warning("Broadcast failed for %s: %s", ticker, e)


# ======================================================================
# Internal helpers
# ======================================================================


def _run_analysis(ticker: str) -> StockAnalysis:
    quote = fetcher.get_quote(ticker)

    df_long = fetcher.get_history(ticker, period="2y")
    analyzer = TechnicalAnalyzer(df_long)
    indicators = analyzer.calculate_all()
    chart_data = analyzer.get_chart_data(periods=120)

    signals = signal_engine.generate_all_signals(indicators)

    prediction = None
    if settings.ml_prediction_enabled:
        try:
            prediction = prediction_engine.predict_all(df_long)
            prediction = {k: v.model_dump() for k, v in prediction.items()} if prediction else None
        except Exception as e:
            logger.warning("Prediction failed for %s: %s", ticker, e)

    news = fetcher.get_news(ticker)

    m = MARKETS.get(_current_market, MARKETS[DEFAULT_MARKET])
    stock_currency = quote.get("currency") or m.currency
    stock_symbol = quote.get("currency_symbol") or m.currency_symbol

    return StockAnalysis(
        ticker=ticker,
        name=quote["name"],
        price=quote["price"],
        change=quote["change"],
        change_percent=quote["change_percent"],
        volume=quote["volume"] or 0,
        market_cap=quote.get("market_cap"),
        fifty_two_week_high=quote.get("fifty_two_week_high"),
        fifty_two_week_low=quote.get("fifty_two_week_low"),
        currency=stock_currency,
        currency_symbol=stock_symbol,
        indicators=TechnicalIndicators(**{
            k: v for k, v in indicators.items() if k in TechnicalIndicators.model_fields
        }),
        signals={k: v.model_dump() for k, v in signals.items()},
        prediction=prediction,
        news=news,
        chart_data=chart_data,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _quick_watchlist_item(ticker: str) -> dict[str, Any]:
    quote = fetcher.get_quote(ticker)
    try:
        df = fetcher.get_history(ticker, period="6mo")
        analyzer = TechnicalAnalyzer(df)
        indicators = analyzer.calculate_all()
        signals = signal_engine.generate_all_signals(indicators)
        overall = signal_engine.overall_action(signals)
    except Exception:
        overall = SignalAction.HOLD

    return {
        "ticker": ticker,
        "name": quote["name"],
        "price": quote["price"],
        "change": quote["change"],
        "change_percent": quote["change_percent"],
        "overall_signal": overall.value,
        "currency": quote.get("currency", "USD"),
        "currency_symbol": quote.get("currency_symbol", "$"),
    }
