from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class DataCache:
    """Simple TTL cache for stock data."""

    def __init__(self, ttl: int = 60):
        self._store: dict[str, tuple[Any, float]] = {}
        self._ttl = ttl

    def get(self, key: str) -> Any | None:
        if key in self._store:
            value, ts = self._store[key]
            if time.time() - ts < self._ttl:
                return value
            del self._store[key]
        return None

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (value, time.time())

    def invalidate(self, key: str) -> None:
        self._store.pop(key, None)


class StockDataFetcher:
    """Fetches and caches stock data from Yahoo Finance."""

    def __init__(self, cache_ttl: int = 60):
        self._cache = DataCache(ttl=cache_ttl)

    def get_history(
        self, ticker: str, period: str = "1y", interval: str = "1d"
    ) -> pd.DataFrame:
        cache_key = f"hist_{ticker}_{period}_{interval}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)

        if df.empty:
            raise ValueError(f"No data found for ticker: {ticker}")

        self._cache.set(cache_key, df)
        return df

    def get_info(self, ticker: str) -> dict[str, Any]:
        cache_key = f"info_{ticker}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        stock = yf.Ticker(ticker)
        info = stock.info

        if not info or info.get("trailingPegRatio") is None and not info.get("regularMarketPrice"):
            fast = stock.fast_info
            info = {
                "regularMarketPrice": getattr(fast, "last_price", 0),
                "regularMarketVolume": getattr(fast, "last_volume", 0),
                "marketCap": getattr(fast, "market_cap", 0),
                "previousClose": getattr(fast, "previous_close", 0),
                "longName": ticker,
                "shortName": ticker,
            }

        self._cache.set(cache_key, info)
        return info

    CURRENCY_SYMBOLS: dict[str, str] = {
        "USD": "$", "EUR": "\u20ac", "GBP": "\u00a3", "CHF": "CHF\u00a0",
        "SEK": "kr", "DKK": "kr", "NOK": "kr", "JPY": "\u00a5",
        "CNY": "\u00a5", "INR": "\u20b9", "GBp": "\u00a3",
    }

    def get_quote(self, ticker: str) -> dict[str, Any]:
        info = self.get_info(ticker)

        price = info.get("regularMarketPrice") or info.get("currentPrice", 0)
        prev_close = info.get("regularMarketPreviousClose") or info.get("previousClose", 0)
        change = price - prev_close if prev_close else 0
        change_pct = (change / prev_close * 100) if prev_close else 0

        raw_currency = info.get("currency") or info.get("financialCurrency") or "USD"
        # LSE reports in pence (GBp); convert to pounds for display
        if raw_currency == "GBp":
            price = price / 100
            change = change / 100
            prev_close = prev_close / 100 if prev_close else 0
            raw_currency = "GBP"

        currency_symbol = self.CURRENCY_SYMBOLS.get(raw_currency, raw_currency + " ")

        return {
            "ticker": ticker.upper(),
            "name": info.get("longName") or info.get("shortName", ticker),
            "price": round(price, 2),
            "change": round(change, 2),
            "change_percent": round(change_pct, 2),
            "volume": info.get("regularMarketVolume") or info.get("volume", 0),
            "market_cap": info.get("marketCap"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            "pe_ratio": info.get("trailingPE"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "currency": raw_currency,
            "currency_symbol": currency_symbol,
        }

    def get_news(self, ticker: str) -> list[dict[str, Any]]:
        cache_key = f"news_{ticker}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            stock = yf.Ticker(ticker)
            raw_news = stock.news or []

            news_items = []
            for item in raw_news[:10]:
                content = item.get("content", {}) if isinstance(item, dict) else {}
                news_items.append({
                    "title": content.get("title") or item.get("title", ""),
                    "publisher": content.get("provider", {}).get("displayName", "")
                    if isinstance(content.get("provider"), dict)
                    else item.get("publisher", ""),
                    "link": content.get("canonicalUrl", {}).get("url", "")
                    if isinstance(content.get("canonicalUrl"), dict)
                    else item.get("link", ""),
                    "published": content.get("pubDate", "") or item.get("providerPublishTime", ""),
                })

            self._cache.set(cache_key, news_items)
            return news_items
        except Exception as e:
            logger.warning("Failed to fetch news for %s: %s", ticker, e)
            return []

    def search(self, query: str) -> list[dict[str, str]]:
        try:
            results = yf.Search(query)
            quotes = getattr(results, "quotes", []) or []
            return [
                {
                    "ticker": q.get("symbol", ""),
                    "name": q.get("longname") or q.get("shortname", ""),
                    "exchange": q.get("exchange", ""),
                    "type": q.get("quoteType", ""),
                }
                for q in quotes[:10]
                if q.get("symbol")
            ]
        except Exception:
            try:
                stock = yf.Ticker(query.upper())
                info = stock.info
                if info and (info.get("regularMarketPrice") or info.get("currentPrice")):
                    return [{
                        "ticker": query.upper(),
                        "name": info.get("longName") or info.get("shortName", query),
                        "exchange": info.get("exchange", ""),
                        "type": info.get("quoteType", "EQUITY"),
                    }]
            except Exception:
                pass
            return []


fetcher = StockDataFetcher(cache_ttl=60)
