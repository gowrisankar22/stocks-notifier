from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import ta

logger = logging.getLogger(__name__)


class TechnicalAnalyzer:
    """Comprehensive technical-indicator calculator over OHLCV data."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._standardize_columns()

    def _standardize_columns(self) -> None:
        self.df.columns = [c.lower().replace(" ", "_") for c in self.df.columns]
        if "adj_close" in self.df.columns:
            self.df.rename(columns={"adj_close": "close"}, inplace=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_all(self) -> dict[str, Any]:
        results: dict[str, Any] = {}
        calculators = [
            ("rsi", self.rsi),
            ("macd", self.macd),
            ("bollinger_bands", self.bollinger_bands),
            ("sma", self.sma),
            ("ema", self.ema),
            ("stochastic", self.stochastic),
            ("adx", self.adx),
            ("atr", self.atr),
            ("obv", self.obv),
            ("fibonacci", self.fibonacci_levels),
            ("volume_analysis", self.volume_analysis),
            ("trend_strength", self.trend_strength),
        ]
        for name, fn in calculators:
            try:
                results[name] = fn()
            except Exception as e:
                logger.warning("Indicator %s failed: %s", name, e)
                results[name] = None
        return results

    # ------------------------------------------------------------------
    # Individual indicators
    # ------------------------------------------------------------------

    def rsi(self, period: int = 14) -> dict[str, Any]:
        indicator = ta.momentum.RSIIndicator(self.df["close"], window=period)
        values = indicator.rsi().dropna()
        if values.empty:
            return {"value": None}

        current = float(values.iloc[-1])
        previous = float(values.iloc[-2]) if len(values) > 1 else current

        if current < 30:
            zone = "oversold"
        elif current > 70:
            zone = "overbought"
        else:
            zone = "neutral"

        return {
            "value": round(current, 2),
            "previous": round(previous, 2),
            "trend": "rising" if current > previous else "falling",
            "zone": zone,
        }

    def macd(
        self, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> dict[str, Any]:
        indicator = ta.trend.MACD(
            self.df["close"],
            window_slow=slow,
            window_fast=fast,
            window_sign=signal,
        )
        macd_line = indicator.macd().dropna()
        signal_line = indicator.macd_signal().dropna()
        histogram = indicator.macd_diff().dropna()

        if macd_line.empty:
            return {"macd": None}

        crossover = self._detect_crossover(macd_line, signal_line)

        hist_values = histogram.tail(5).tolist()
        if len(hist_values) >= 2:
            hist_trend = "expanding" if abs(hist_values[-1]) > abs(hist_values[-2]) else "contracting"
        else:
            hist_trend = "unknown"

        return {
            "macd": round(float(macd_line.iloc[-1]), 4),
            "signal": round(float(signal_line.iloc[-1]), 4),
            "histogram": round(float(histogram.iloc[-1]), 4),
            "crossover": crossover,
            "histogram_trend": hist_trend,
        }

    def bollinger_bands(self, period: int = 20, std_dev: int = 2) -> dict[str, Any]:
        indicator = ta.volatility.BollingerBands(
            self.df["close"], window=period, window_dev=std_dev
        )
        upper = indicator.bollinger_hband()
        middle = indicator.bollinger_mavg()
        lower = indicator.bollinger_lband()
        pband = indicator.bollinger_pband()
        wband = indicator.bollinger_wband()

        price = float(self.df["close"].iloc[-1])
        upper_val = float(upper.iloc[-1])
        lower_val = float(lower.iloc[-1])

        if price > upper_val:
            position = "above_upper"
        elif price < lower_val:
            position = "below_lower"
        else:
            position = "inside"

        return {
            "upper": round(upper_val, 2),
            "middle": round(float(middle.iloc[-1]), 2),
            "lower": round(lower_val, 2),
            "percent_b": round(float(pband.iloc[-1]), 4),
            "bandwidth": round(float(wband.iloc[-1]), 4),
            "position": position,
        }

    def sma(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        price = float(self.df["close"].iloc[-1])

        for period in [10, 20, 50, 100, 200]:
            values = self.df["close"].rolling(window=period).mean()
            valid = values.dropna()
            if not valid.empty:
                val = float(valid.iloc[-1])
                result[f"sma_{period}"] = round(val, 2)
                result[f"price_vs_sma_{period}"] = round(
                    ((price - val) / val) * 100, 2
                )

        sma_50 = result.get("sma_50")
        sma_200 = result.get("sma_200")
        if sma_50 is not None and sma_200 is not None:
            result["golden_cross"] = sma_50 > sma_200

        return result

    def ema(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for period in [9, 12, 21, 26, 50, 200]:
            indicator = ta.trend.EMAIndicator(self.df["close"], window=period)
            values = indicator.ema_indicator().dropna()
            if not values.empty:
                result[f"ema_{period}"] = round(float(values.iloc[-1]), 2)
        return result

    def stochastic(self, k_period: int = 14, d_period: int = 3) -> dict[str, Any]:
        indicator = ta.momentum.StochasticOscillator(
            self.df["high"],
            self.df["low"],
            self.df["close"],
            window=k_period,
            smooth_window=d_period,
        )
        k = indicator.stoch().dropna()
        d = indicator.stoch_signal().dropna()

        if k.empty:
            return {"k": None}

        k_val = float(k.iloc[-1])
        d_val = float(d.iloc[-1]) if not d.empty else None

        if k_val > 80:
            zone = "overbought"
        elif k_val < 20:
            zone = "oversold"
        else:
            zone = "neutral"

        crossover = "none"
        if d_val is not None and len(k) > 1 and len(d) > 1:
            crossover = self._detect_crossover(k, d)

        return {
            "k": round(k_val, 2),
            "d": round(d_val, 2) if d_val is not None else None,
            "zone": zone,
            "crossover": crossover,
        }

    def adx(self, period: int = 14) -> dict[str, Any]:
        indicator = ta.trend.ADXIndicator(
            self.df["high"], self.df["low"], self.df["close"], window=period
        )
        adx_values = indicator.adx().dropna()
        plus_di = indicator.adx_pos().dropna()
        minus_di = indicator.adx_neg().dropna()

        if adx_values.empty:
            return {"adx": None}

        adx_val = float(adx_values.iloc[-1])

        if adx_val > 50:
            strength = "very_strong"
        elif adx_val > 25:
            strength = "strong"
        elif adx_val > 20:
            strength = "developing"
        else:
            strength = "weak"

        return {
            "adx": round(adx_val, 2),
            "plus_di": round(float(plus_di.iloc[-1]), 2) if not plus_di.empty else None,
            "minus_di": round(float(minus_di.iloc[-1]), 2) if not minus_di.empty else None,
            "trend_strength": strength,
            "trend_direction": "bullish"
            if not plus_di.empty
            and not minus_di.empty
            and float(plus_di.iloc[-1]) > float(minus_di.iloc[-1])
            else "bearish",
        }

    def atr(self, period: int = 14) -> dict[str, Any]:
        indicator = ta.volatility.AverageTrueRange(
            self.df["high"], self.df["low"], self.df["close"], window=period
        )
        values = indicator.average_true_range().dropna()
        if values.empty:
            return {"value": None}

        atr_val = float(values.iloc[-1])
        price = float(self.df["close"].iloc[-1])
        atr_pct = (atr_val / price) * 100

        if atr_pct > 4:
            volatility = "very_high"
        elif atr_pct > 2.5:
            volatility = "high"
        elif atr_pct > 1.5:
            volatility = "medium"
        else:
            volatility = "low"

        return {
            "value": round(atr_val, 2),
            "percent": round(atr_pct, 2),
            "volatility": volatility,
        }

    def obv(self) -> dict[str, Any]:
        indicator = ta.volume.OnBalanceVolumeIndicator(
            self.df["close"], self.df["volume"]
        )
        values = indicator.on_balance_volume()
        if values.empty:
            return {"value": None}

        obv_sma = values.rolling(window=20).mean()
        current = float(values.iloc[-1])
        sma_val = float(obv_sma.iloc[-1]) if not obv_sma.dropna().empty else current

        return {
            "value": int(current),
            "sma_20": int(sma_val),
            "trend": "bullish" if current > sma_val else "bearish",
        }

    def fibonacci_levels(self) -> dict[str, float]:
        recent = self.df.tail(60)
        high = float(recent["high"].max())
        low = float(recent["low"].min())
        diff = high - low

        return {
            "high": round(high, 2),
            "low": round(low, 2),
            "level_236": round(high - 0.236 * diff, 2),
            "level_382": round(high - 0.382 * diff, 2),
            "level_500": round(high - 0.500 * diff, 2),
            "level_618": round(high - 0.618 * diff, 2),
            "level_786": round(high - 0.786 * diff, 2),
        }

    def volume_analysis(self) -> dict[str, Any]:
        vol = self.df["volume"]
        current = float(vol.iloc[-1])
        avg_20 = float(vol.rolling(20).mean().iloc[-1])
        avg_50 = float(vol.rolling(50).mean().iloc[-1]) if len(vol) >= 50 else avg_20

        return {
            "current": int(current),
            "avg_20": int(avg_20),
            "avg_50": int(avg_50),
            "ratio": round(current / avg_20, 2) if avg_20 > 0 else 0,
            "trend": "increasing" if avg_20 > avg_50 else "decreasing",
            "spike": current > avg_20 * 2,
        }

    def trend_strength(self) -> dict[str, Any]:
        close = self.df["close"]
        price = float(close.iloc[-1])

        checks: list[tuple[str, bool]] = []
        sma_values: dict[int, float] = {}

        for period in [20, 50, 200]:
            sma = close.rolling(period).mean()
            valid = sma.dropna()
            if not valid.empty:
                val = float(valid.iloc[-1])
                sma_values[period] = val
                checks.append((f"Price > SMA{period}", price > val))

        if 20 in sma_values and 50 in sma_values:
            checks.append(("SMA20 > SMA50", sma_values[20] > sma_values[50]))
        if 50 in sma_values and 200 in sma_values:
            checks.append(("SMA50 > SMA200", sma_values[50] > sma_values[200]))

        ret_5 = float((close.iloc[-1] / close.iloc[-5] - 1) * 100) if len(close) > 5 else 0
        ret_20 = float((close.iloc[-1] / close.iloc[-20] - 1) * 100) if len(close) > 20 else 0
        checks.append(("5d return positive", ret_5 > 0))
        checks.append(("20d return positive", ret_20 > 0))

        bullish = sum(1 for _, v in checks if v)
        total = len(checks) or 1
        score = bullish / total

        if score >= 0.7:
            direction = "bullish"
        elif score <= 0.3:
            direction = "bearish"
        else:
            direction = "neutral"

        return {
            "score": round(score, 2),
            "direction": direction,
            "checks": {name: val for name, val in checks},
            "returns": {
                "5d": round(ret_5, 2),
                "20d": round(ret_20, 2),
            },
        }

    # ------------------------------------------------------------------
    # Chart data
    # ------------------------------------------------------------------

    def get_chart_data(self, periods: int = 90) -> dict[str, list]:
        df = self.df.tail(periods).copy()

        sma_20 = df["close"].rolling(20).mean()
        sma_50 = df["close"].rolling(50).mean()

        bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)

        return {
            "dates": df.index.strftime("%Y-%m-%d").tolist(),
            "open": [round(float(v), 2) for v in df["open"]],
            "high": [round(float(v), 2) for v in df["high"]],
            "low": [round(float(v), 2) for v in df["low"]],
            "close": [round(float(v), 2) for v in df["close"]],
            "volume": [int(v) for v in df["volume"]],
            "sma_20": [round(float(v), 2) if not pd.isna(v) else None for v in sma_20],
            "sma_50": [round(float(v), 2) if not pd.isna(v) else None for v in sma_50],
            "bb_upper": [round(float(v), 2) if not pd.isna(v) else None for v in bb.bollinger_hband()],
            "bb_lower": [round(float(v), 2) if not pd.isna(v) else None for v in bb.bollinger_lband()],
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_crossover(fast: pd.Series, slow: pd.Series) -> str:
        if len(fast) < 2 or len(slow) < 2:
            return "none"

        idx = fast.index.intersection(slow.index)
        if len(idx) < 2:
            return "none"

        f = fast.loc[idx]
        s = slow.loc[idx]

        prev_diff = float(f.iloc[-2]) - float(s.iloc[-2])
        curr_diff = float(f.iloc[-1]) - float(s.iloc[-1])

        if prev_diff <= 0 < curr_diff:
            return "bullish"
        if prev_diff >= 0 > curr_diff:
            return "bearish"
        return "none"
