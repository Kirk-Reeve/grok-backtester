"""Enhanced Ichimoku Cloud Trading Strategy Implementation."""

from typing import Any, Dict

import numpy as np
from pandas import DataFrame, Series
from talib import ATR, MACD, MAX, MIN  # pylint: disable=no-name-in-module

from .base import BaseStrategy


class EnhancedIchimokuStrategy(BaseStrategy):
    """Enhanced Ichimoku Cloud Strategy.

    Based on high-performing Ichimoku strategy (top return rate 1.77 in 2025 study):
    Buy when price > Senkou Span B (cloud base), sell when price < Senkou Span A (cloud top).
    Enhanced with ATR-based target price (take profit) and stop loss for risk management.
    Now includes Chikou Span confirmation for signals, Kumo thickness filter to avoid thin clouds,
    and MACD confirmation for momentum alignment.

    Args:
        params (Dict[str, Any] | None): Dictionary of strategy parameters.
            - conversion_period: int - Tenkan-sen (conversion line) period (default: 9)
            - base_period: int - Kijun-sen (base line) period (default: 26)
            - leading_span_b_period: int - Senkou Span B period (default: 52)
            - lagging_span_period: int - Chikou Span period (default: 26)
            - use_atr_sl_tp: bool - Enable ATR-based stop loss and target price (default: True)
            - atr_period: int - ATR lookback period for SL/TP (default: 14)
            - sl_multiplier: float - Multiplier for stop loss (default: 1.5, e.g., SL = entry - 1.5*ATR)
            - tp_multiplier: float - Multiplier for take profit (default: 2.0, e.g., TP = entry + 2.0*ATR for buys)
            - use_kumo_filter: bool - Filter signals if Kumo thickness < threshold (default: True)
            - kumo_threshold_multiplier: float - Min thickness as multiplier of average price (default: 0.01, e.g., 1% of price)
            - use_macd_confirm: bool - Require MACD crossover confirmation (default: True)
            - macd_fast_period: int - MACD fast EMA period (default: 12)
            - macd_slow_period: int - MACD slow EMA period (default: 26)
            - macd_signal_period: int - MACD signal line period (default: 9)
            - price_column: str - DataFrame column for price data (default: "Adj Close")
    """

    default_params = {
        "conversion_period": 9,
        "base_period": 26,
        "leading_span_b_period": 52,
        "lagging_span_period": 26,
        "use_atr_sl_tp": True,
        "atr_period": 14,
        "sl_multiplier": 1.5,
        "tp_multiplier": 2.0,
        "use_kumo_filter": True,
        "kumo_threshold_multiplier": 0.01,  # e.g., thickness > 1% of avg price
        "use_macd_confirm": True,
        "macd_fast_period": 12,
        "macd_slow_period": 26,
        "macd_signal_period": 9,
        "price_column": "Adj Close",
    }

    required_columns = ["High", "Low"]  # For Ichimoku, ATR, and MACD

    def _validate_params(self, params: Dict[str, Any]) -> None:
        """Validate strategy parameters.

        Args:
            params (Dict[str, Any]): The parameters to validate.

        Raises:
            ValueError: If parameters are invalid.
        """
        periods = [
            params["conversion_period"],
            params["base_period"],
            params["leading_span_b_period"],
            params["lagging_span_period"],
            params["atr_period"],
            params["macd_fast_period"],
            params["macd_slow_period"],
            params["macd_signal_period"],
        ]
        if any(p <= 0 for p in periods):
            raise ValueError("All periods must be positive.")
        if params["sl_multiplier"] <= 0 or params["tp_multiplier"] <= 0:
            raise ValueError("SL/TP multipliers must be positive.")
        if params["kumo_threshold_multiplier"] < 0:
            raise ValueError("Kumo threshold multiplier must be non-negative.")
        if params["macd_fast_period"] >= params["macd_slow_period"]:
            raise ValueError("MACD fast period must be less than slow period.")

    def __init__(self, params: Dict[str, Any] | None = None):
        """Initialize the strategy with parameters.

        Args:
            params (Dict[str, Any] | None): Dictionary of strategy parameters.
        """
        super().__init__(params)
        # No need for manual assignments; use self.params directly where needed

    def generate_signals(self, data: DataFrame) -> Series:
        """Generate trading signals based on enhanced Ichimoku Cloud.

        Uses precise detection: buy (1) when price crosses above Senkou Span B,
        sell (-1) when crosses below Senkou Span A. Includes Chikou confirmation,
        Kumo thickness filter (avoid thin clouds), and MACD confirmation.
        Applies ATR-based SL/TP if enabled (adjusts signals to exit at SL/TP hits).
        Handles NaNs naturally (signals 0 for early bars).
        Prefers 'Adj Close' but falls back to 'Close' if unavailable.

        Args:
            data (DataFrame): DataFrame with 'High', 'Low', 'Close' (or adjusted) columns.

        Returns:
            Series: Series of signals (1: buy, -1: sell, 0: hold).

        Raises:
            ValueError: If required columns are missing or data insufficient.
        """
        super().generate_signals(data)  # Calls base validation for required_columns

        # Determine the actual price column to use, with fallback
        actual_column = self.params["price_column"]
        if actual_column not in data.columns:
            if "Close" in data.columns:
                actual_column = "Close"
                self.logger.warning(
                    "Preferred price column '%s' not found; falling back to 'Close'. "
                    "For accurate backtesting, use adjusted prices if available.",
                    self.params["price_column"],
                )
            else:
                raise ValueError(
                    f"DataFrame must contain '{self.params['price_column']}' or 'Close' column."
                )

        max_period = max(
            self.params["conversion_period"],
            self.params["base_period"],
            self.params["leading_span_b_period"],
            self.params["lagging_span_period"],
            self.params["atr_period"],
            self.params["macd_slow_period"],
        )
        if len(data) < max_period + 1:
            self.logger.warning(
                "Insufficient data for Ichimoku calculation (need at least %d rows); "
                "returning neutral signals.",
                max_period + 1,
            )
            return Series(0, index=data.index, dtype="int8")

        # Extract as NumPy arrays for TA-Lib (optimized C computation)
        high_arr = data["High"].to_numpy(dtype="float64")
        low_arr = data["Low"].to_numpy(dtype="float64")
        close_arr = data[actual_column].to_numpy(dtype="float64")

        # Manual Ichimoku calculation using TA-Lib functions
        # Tenkan-sen: (max high + min low) / 2 over conversion_period
        tenkan_sen = (
            MAX(high_arr, timeperiod=self.params["conversion_period"])
            + MIN(low_arr, timeperiod=self.params["conversion_period"])
        ) / 2

        # Kijun-sen: (max high + min low) / 2 over base_period
        kijun_sen = (
            MAX(high_arr, timeperiod=self.params["base_period"])
            + MIN(low_arr, timeperiod=self.params["base_period"])
        ) / 2

        # Senkou Span A: (tenkan + kijun) / 2, shifted forward by base_period
        senkou_a = (tenkan_sen + kijun_sen) / 2
        senkou_a_series = (
            Series(senkou_a, index=data.index)
            .shift(self.params["base_period"])
            .fillna(0)
        )  # Fill NaN shifts with 0 for safe comparisons

        # Senkou Span B: (max high + min low) / 2 over leading_span_b_period, shifted forward by base_period
        senkou_b = (
            MAX(high_arr, timeperiod=self.params["leading_span_b_period"])
            + MIN(low_arr, timeperiod=self.params["leading_span_b_period"])
        ) / 2
        senkou_b_series = (
            Series(senkou_b, index=data.index)
            .shift(self.params["base_period"])
            .fillna(0)
        )  # Fill NaN shifts with 0 for safe comparisons

        # Chikou Span: current close (not shifted; value at t is close_t for comparison)
        chikou_span = data[actual_column]

        # Price series for crossings and Chikou confirmation
        price_series = data[actual_column]

        # Historical price for Chikou comparison: price from lagging periods ago
        historical_price = price_series.shift(
            self.params["lagging_span_period"]
        ).fillna(0)

        # Detect signals precisely
        # Buy: previous price <= senkou_b and current price > senkou_b (break above cloud base)
        cross_above_b = (price_series > senkou_b_series) & (
            price_series.shift(1) <= senkou_b_series.shift(1)
        )
        # Sell: previous price >= senkou_a and current price < senkou_a (break below cloud top)
        cross_below_a = (price_series < senkou_a_series) & (
            price_series.shift(1) >= senkou_a_series.shift(1)
        )

        # Apply Chikou confirmation
        # For buys: Chikou (close_t) > historical price (price_{t - lagging})
        chikou_bullish = chikou_span > historical_price
        # For sells: Chikou (close_t) < historical price (price_{t - lagging})
        chikou_bearish = chikou_span < historical_price

        cross_above_b &= chikou_bullish
        cross_below_a &= chikou_bearish

        # Apply Kumo thickness filter if enabled
        if self.params["use_kumo_filter"]:
            # Kumo thickness: abs(senkou_a - senkou_b); filter if < threshold * avg price
            kumo_thickness = np.abs(senkou_a_series - senkou_b_series)
            avg_price = (
                price_series.rolling(window=self.params["base_period"]).mean().fillna(0)
            )
            min_thickness = self.params["kumo_threshold_multiplier"] * avg_price

            # Require thickness > min at signal bar
            cross_above_b &= kumo_thickness > min_thickness
            cross_below_a &= kumo_thickness > min_thickness

        # Apply MACD confirmation if enabled
        if self.params["use_macd_confirm"]:
            # Calculate MACD
            macd_line, signal_line, _ = MACD(
                close_arr,
                fastperiod=self.params["macd_fast_period"],
                slowperiod=self.params["macd_slow_period"],
                signalperiod=self.params["macd_signal_period"],
            )
            macd_series = Series(macd_line, index=data.index).fillna(0)
            signal_series = Series(signal_line, index=data.index).fillna(0)

            # Bullish MACD: macd > signal
            macd_bullish = macd_series > signal_series
            # Bearish MACD: macd < signal
            macd_bearish = macd_series < signal_series

            cross_above_b &= macd_bullish
            cross_below_a &= macd_bearish

        # Generate base signals: 1 for buy, -1 for sell, 0 otherwise
        signals = Series(
            0, index=data.index, dtype="int8"
        )  # int8 for memory efficiency
        signals[cross_above_b] = 1
        signals[cross_below_a] = -1

        # Optional: Apply ATR-based SL/TP (simulate exits in signals)
        if self.params["use_atr_sl_tp"]:
            # Calculate ATR
            atr = ATR(
                high_arr, low_arr, close_arr, timeperiod=self.params["atr_period"]
            )
            atr_series = Series(atr, index=data.index).fillna(
                0
            )  # Fill early NaNs for safety

            # Simulate positions and check SL/TP (vectorized approximation; full sim in backtester)
            position = signals.cumsum().clip(
                -1, 1
            )  # Cumulative position (-1 short, 1 long, 0 flat)
            entry_price = price_series.where(
                signals != 0
            ).ffill()  # Forward fill entry prices

            # For longs: SL = entry - sl_mult * ATR, TP = entry + tp_mult * ATR
            sl_long = entry_price - self.params["sl_multiplier"] * atr_series
            tp_long = entry_price + self.params["tp_multiplier"] * atr_series
            # For shorts: SL = entry + sl_mult * ATR, TP = entry - tp_mult * ATR
            sl_short = entry_price + self.params["sl_multiplier"] * atr_series
            tp_short = entry_price - self.params["tp_multiplier"] * atr_series

            # Hit checks (approximate; assumes no gaps)
            hit_sl_long = (position == 1) & (price_series < sl_long)
            hit_tp_long = (position == 1) & (price_series > tp_long)
            hit_sl_short = (position == -1) & (price_series > sl_short)
            hit_tp_short = (position == -1) & (price_series < tp_short)

            # Adjust signals to exit on hits (-signal to close)
            signals[hit_sl_long | hit_tp_long] = -1
            signals[hit_sl_short | hit_tp_short] = 1

        # Log summary (avoid logging full series for large data)
        self.logger.debug(
            "Generated %d signals (%d buys, %d sells) for period %s to %s using column '%s'",
            len(signals),
            (signals == 1).sum(),
            (signals == -1).sum(),
            data.index[0],
            data.index[-1],
            actual_column,
        )
        return signals
