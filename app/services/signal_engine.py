from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import ta
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

from app.models import PredictionResult, SignalAction, TimeframeSignal

logger = logging.getLogger(__name__)

# Indicator weights per timeframe — tuned so each row sums to 1.0.
TIMEFRAME_CONFIG: dict[str, dict[str, Any]] = {
    "1_day": {
        "label": "Intraday / 1 Day",
        "weights": {
            "rsi": 0.20,
            "macd": 0.15,
            "stochastic": 0.20,
            "bollinger": 0.15,
            "volume": 0.15,
            "price_action": 0.15,
        },
    },
    "short_term": {
        "label": "Short Term (1-2 wk)",
        "weights": {
            "rsi": 0.15,
            "macd": 0.20,
            "bollinger": 0.15,
            "sma_crossover": 0.15,
            "volume": 0.15,
            "adx": 0.10,
            "trend": 0.10,
        },
    },
    "mid_term": {
        "label": "Mid Term (1-3 mo)",
        "weights": {
            "sma_crossover": 0.20,
            "macd": 0.15,
            "adx": 0.15,
            "rsi": 0.10,
            "volume": 0.15,
            "trend": 0.15,
            "fibonacci": 0.10,
        },
    },
    "long_term": {
        "label": "Long Term (6-12 mo)",
        "weights": {
            "sma_200": 0.20,
            "trend": 0.25,
            "macd": 0.15,
            "volume": 0.15,
            "adx": 0.15,
            "momentum": 0.10,
        },
    },
}


class SignalEngine:
    """Generates buy/sell/hold signals from technical indicators."""

    # ------------------------------------------------------------------
    # Indicator scoring  (each returns -1 … +1)
    # ------------------------------------------------------------------

    @staticmethod
    def _score_rsi(data: dict | None) -> float:
        if not data or data.get("value") is None:
            return 0.0
        v = data["value"]
        if v < 20:
            return 0.9
        if v < 30:
            return 0.6
        if v < 40:
            return 0.25
        if v < 45:
            return 0.1
        if v < 55:
            return 0.0
        if v < 60:
            return -0.1
        if v < 70:
            return -0.25
        if v < 80:
            return -0.6
        return -0.9

    @staticmethod
    def _score_macd(data: dict | None) -> float:
        if not data or data.get("macd") is None:
            return 0.0

        score = 0.0
        crossover = data.get("crossover", "none")
        if crossover == "bullish":
            score += 0.6
        elif crossover == "bearish":
            score -= 0.6

        hist = data.get("histogram", 0) or 0
        if hist > 0:
            score += min(0.4, abs(hist) * 5)
        else:
            score -= min(0.4, abs(hist) * 5)

        if data.get("histogram_trend") == "expanding" and hist > 0:
            score += 0.1
        elif data.get("histogram_trend") == "expanding" and hist < 0:
            score -= 0.1

        return max(-1.0, min(1.0, score))

    @staticmethod
    def _score_stochastic(data: dict | None) -> float:
        if not data or data.get("k") is None:
            return 0.0

        k = data["k"]
        crossover = data.get("crossover", "none")

        score = 0.0
        if k < 20:
            score = 0.7
        elif k < 30:
            score = 0.4
        elif k < 50:
            score = 0.1
        elif k < 70:
            score = -0.1
        elif k < 80:
            score = -0.4
        else:
            score = -0.7

        if crossover == "bullish":
            score += 0.2
        elif crossover == "bearish":
            score -= 0.2

        return max(-1.0, min(1.0, score))

    @staticmethod
    def _score_bollinger(data: dict | None) -> float:
        if not data or data.get("percent_b") is None:
            return 0.0

        pb = data["percent_b"]
        if pb < 0:
            return 0.8
        if pb < 0.2:
            return 0.5
        if pb < 0.4:
            return 0.2
        if pb < 0.6:
            return 0.0
        if pb < 0.8:
            return -0.2
        if pb < 1.0:
            return -0.5
        return -0.8

    @staticmethod
    def _score_volume(data: dict | None) -> float:
        if not data:
            return 0.0

        ratio = data.get("ratio", 1.0)
        trend = data.get("trend", "stable")

        score = 0.0
        if ratio > 2.0:
            score = 0.3
        elif ratio > 1.5:
            score = 0.15
        elif ratio < 0.5:
            score = -0.2

        if trend == "increasing":
            score += 0.1
        elif trend == "decreasing":
            score -= 0.1

        return max(-1.0, min(1.0, score))

    @staticmethod
    def _score_price_action(indicators: dict) -> float:
        ts = indicators.get("trend_strength") or {}
        ret = ts.get("returns", {})
        r5 = ret.get("5d", 0)

        score = 0.0
        if r5 > 3:
            score = 0.5
        elif r5 > 1:
            score = 0.25
        elif r5 > 0:
            score = 0.1
        elif r5 > -1:
            score = -0.1
        elif r5 > -3:
            score = -0.25
        else:
            score = -0.5

        return score

    @staticmethod
    def _score_sma_crossover(indicators: dict) -> float:
        sma = indicators.get("sma") or {}
        if not sma:
            return 0.0

        score = 0.0
        golden = sma.get("golden_cross")
        if golden is True:
            score += 0.5
        elif golden is False:
            score -= 0.5

        pv20 = sma.get("price_vs_sma_20", 0)
        pv50 = sma.get("price_vs_sma_50", 0)

        if pv20 > 0:
            score += 0.15
        else:
            score -= 0.15

        if pv50 > 0:
            score += 0.15
        else:
            score -= 0.15

        return max(-1.0, min(1.0, score))

    @staticmethod
    def _score_adx(data: dict | None) -> float:
        if not data or data.get("adx") is None:
            return 0.0

        adx_val = data["adx"]
        direction = data.get("trend_direction", "neutral")

        if adx_val < 20:
            return 0.0

        if direction == "bullish":
            return min(0.8, adx_val / 60)
        return -min(0.8, adx_val / 60)

    @staticmethod
    def _score_trend(indicators: dict) -> float:
        ts = indicators.get("trend_strength") or {}
        score = ts.get("score", 0.5)
        return (score - 0.5) * 2  # map 0..1 → -1..+1

    @staticmethod
    def _score_fibonacci(indicators: dict) -> float:
        fib = indicators.get("fibonacci")
        ts = indicators.get("trend_strength") or {}
        if not fib:
            return 0.0

        direction = ts.get("direction", "neutral")
        if direction == "bullish":
            return 0.2
        if direction == "bearish":
            return -0.2
        return 0.0

    @staticmethod
    def _score_sma_200(indicators: dict) -> float:
        sma = indicators.get("sma") or {}
        pv200 = sma.get("price_vs_sma_200")
        if pv200 is None:
            return 0.0

        if pv200 > 10:
            return 0.6
        if pv200 > 5:
            return 0.4
        if pv200 > 0:
            return 0.2
        if pv200 > -5:
            return -0.2
        if pv200 > -10:
            return -0.4
        return -0.6

    @staticmethod
    def _score_momentum(indicators: dict) -> float:
        ts = indicators.get("trend_strength") or {}
        ret = ts.get("returns", {})
        r20 = ret.get("20d", 0)

        if r20 > 10:
            return 0.7
        if r20 > 5:
            return 0.4
        if r20 > 0:
            return 0.15
        if r20 > -5:
            return -0.15
        if r20 > -10:
            return -0.4
        return -0.7

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    SCORE_FUNCS: dict[str, str] = {
        "rsi": "_score_rsi",
        "macd": "_score_macd",
        "stochastic": "_score_stochastic",
        "bollinger": "_score_bollinger",
        "volume": "_score_volume",
        "price_action": "_score_price_action",
        "sma_crossover": "_score_sma_crossover",
        "adx": "_score_adx",
        "trend": "_score_trend",
        "fibonacci": "_score_fibonacci",
        "sma_200": "_score_sma_200",
        "momentum": "_score_momentum",
    }

    def _get_score(self, name: str, indicators: dict) -> float:
        func_name = self.SCORE_FUNCS.get(name)
        if not func_name:
            return 0.0
        func = getattr(self, func_name)

        whole_indicator_funcs = {
            "_score_price_action",
            "_score_sma_crossover",
            "_score_trend",
            "_score_fibonacci",
            "_score_sma_200",
            "_score_momentum",
        }
        if func_name in whole_indicator_funcs:
            return func(indicators)
        return func(indicators.get(name.replace("_score_", "")))

    def generate_signal(
        self, indicators: dict, timeframe: str
    ) -> TimeframeSignal:
        cfg = TIMEFRAME_CONFIG[timeframe]
        weights = cfg["weights"]

        scores: dict[str, float] = {}
        for name, weight in weights.items():
            scores[name] = self._get_score(name, indicators)

        composite = sum(scores[k] * weights[k] for k in weights)

        if composite >= 0.55:
            action = SignalAction.STRONG_BUY
        elif composite >= 0.20:
            action = SignalAction.BUY
        elif composite > -0.20:
            action = SignalAction.HOLD
        elif composite > -0.55:
            action = SignalAction.SELL
        else:
            action = SignalAction.STRONG_SELL

        positive = sum(1 for s in scores.values() if s > 0.05)
        negative = sum(1 for s in scores.values() if s < -0.05)
        total = len(scores) or 1
        agreement = max(positive, negative) / total
        confidence = agreement * min(abs(composite) * 2.5, 1.0)

        return TimeframeSignal(
            timeframe=timeframe,
            label=cfg["label"],
            action=action,
            score=round(composite, 3),
            confidence=round(confidence * 100, 1),
            details={k: round(v, 3) for k, v in scores.items()},
        )

    def generate_all_signals(
        self, indicators: dict
    ) -> dict[str, TimeframeSignal]:
        return {
            tf: self.generate_signal(indicators, tf)
            for tf in TIMEFRAME_CONFIG
        }

    @staticmethod
    def overall_action(signals: dict[str, TimeframeSignal]) -> SignalAction:
        score_map = {
            SignalAction.STRONG_BUY: 2,
            SignalAction.BUY: 1,
            SignalAction.HOLD: 0,
            SignalAction.SELL: -1,
            SignalAction.STRONG_SELL: -2,
        }
        tf_weight = {"1_day": 0.15, "short_term": 0.25, "mid_term": 0.35, "long_term": 0.25}
        total = sum(
            score_map[sig.action] * tf_weight.get(tf, 0.25)
            for tf, sig in signals.items()
        )
        if total >= 1.2:
            return SignalAction.STRONG_BUY
        if total >= 0.4:
            return SignalAction.BUY
        if total > -0.4:
            return SignalAction.HOLD
        if total > -1.2:
            return SignalAction.SELL
        return SignalAction.STRONG_SELL


# ======================================================================
# ML Prediction Engine
# ======================================================================


class PredictionEngine:
    """Lightweight ML predictor trained on the stock's own history."""

    HORIZONS = {
        "1_day": 1,
        "short_term": 5,
        "mid_term": 20,
        "long_term": 60,
    }

    def predict_all(self, df: pd.DataFrame) -> dict[str, PredictionResult]:
        features_df = self._build_features(df)
        if features_df is None or len(features_df) < 80:
            return {}

        results: dict[str, PredictionResult] = {}
        for name, horizon in self.HORIZONS.items():
            try:
                result = self._predict_horizon(features_df, df, horizon)
                if result:
                    results[name] = result
            except Exception as e:
                logger.warning("ML prediction failed for horizon %s: %s", name, e)
        return results

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame | None:
        try:
            close = df["close"] if "close" in df.columns else df["Close"]
            high = df["high"] if "high" in df.columns else df["High"]
            low = df["low"] if "low" in df.columns else df["Low"]
            volume = df["volume"] if "volume" in df.columns else df["Volume"]
        except KeyError:
            return None

        feat = pd.DataFrame(index=df.index)

        feat["rsi_14"] = ta.momentum.RSIIndicator(close, 14).rsi()
        feat["rsi_7"] = ta.momentum.RSIIndicator(close, 7).rsi()

        macd = ta.trend.MACD(close)
        feat["macd"] = macd.macd()
        feat["macd_signal"] = macd.macd_signal()
        feat["macd_hist"] = macd.macd_diff()

        for p in [10, 20, 50]:
            sma = close.rolling(p).mean()
            feat[f"price_sma_{p}_ratio"] = close / sma

        bb = ta.volatility.BollingerBands(close)
        feat["bb_pband"] = bb.bollinger_pband()
        feat["bb_wband"] = bb.bollinger_wband()

        stoch = ta.momentum.StochasticOscillator(high, low, close)
        feat["stoch_k"] = stoch.stoch()
        feat["stoch_d"] = stoch.stoch_signal()

        feat["adx"] = ta.trend.ADXIndicator(high, low, close).adx()

        feat["atr_pct"] = (
            ta.volatility.AverageTrueRange(high, low, close).average_true_range()
            / close
            * 100
        )

        for p in [1, 5, 10, 20]:
            feat[f"return_{p}d"] = close.pct_change(p) * 100

        avg_vol = volume.rolling(20).mean()
        feat["volume_ratio"] = volume / avg_vol.replace(0, np.nan)

        feat["volatility_20d"] = close.pct_change().rolling(20).std() * 100

        return feat.dropna()

    def _predict_horizon(
        self, features_df: pd.DataFrame, raw_df: pd.DataFrame, horizon: int
    ) -> PredictionResult | None:
        close = (
            raw_df["close"] if "close" in raw_df.columns else raw_df["Close"]
        )

        future_return = close.shift(-horizon) / close - 1
        target = (future_return > 0).astype(int)
        target = target.reindex(features_df.index)

        valid_mask = target.notna()
        X = features_df.loc[valid_mask]
        y = target.loc[valid_mask]

        if len(X) < 60:
            return None

        train_size = len(X) - max(horizon, 10)
        if train_size < 50:
            return None

        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train = y.iloc[:train_size]

        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
        model.fit(X_train, y_train)

        accuracy = None
        try:
            scores = cross_val_score(
                model, X.iloc[:train_size], y.iloc[:train_size], cv=3, scoring="accuracy"
            )
            accuracy = round(float(scores.mean()) * 100, 1)
        except Exception:
            pass

        last_features = features_df.iloc[[-1]]
        proba = model.predict_proba(last_features)[0]

        up_prob = float(proba[1]) if len(proba) > 1 else 0.5

        if up_prob >= 0.6:
            direction = "UP"
        elif up_prob <= 0.4:
            direction = "DOWN"
        else:
            direction = "SIDEWAYS"

        return PredictionResult(
            direction=direction,
            confidence=round(up_prob * 100, 1),
            model_accuracy=accuracy,
            features_used=len(features_df.columns),
        )


signal_engine = SignalEngine()
prediction_engine = PredictionEngine()
