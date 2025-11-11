"""Enhanced RSI Mean Reversion Trading Strategy Implementation."""

from typing import Any, Dict

from pandas import DataFrame, Series
from talib import RSI  # pylint: disable=no-name-in-module

from .base import BaseStrategy


class EnhancedRSIStrategy(BaseStrategy):
    """Enhanced RSI Mean Reversion Strategy.

    Based on high-win-rate RSI strategy (91% reported): Buy when RSI crosses below oversold threshold,
    sell when crosses above overbought. Enhanced with optional trend filter (e.g., price above long MA)
    and volume confirmation for reduced false signals. Uses TA-Lib for optimized RSI computation.

    Args:
        params (Dict[str, Any] | None): Dictionary of strategy parameters.
            - rsi_period: int - RSI lookback period (default: 2, short for sensitivity)
            - overbought: float - Overbought threshold (default: 85.0)
            - oversold: float - Oversold threshold (default: 15.0)
            - use_trend_filter: bool - Require price > long_ma_period MA for buys (default: True)
            - long_ma_period: int - Long moving average period for trend filter (default: 200)
            - use_volume_confirmation: bool - Require volume > vol_ma_period MA for signals (default: True)
            - vol_ma_period: int - Volume moving average period (default: 50)
            - price_column: str - DataFrame column for price data (default: "Adj Close")
            - volume_column: str - DataFrame column for volume data (default: "Volume")
    """

    default_params = {
        "rsi_period": 2,
        "overbought": 85.0,
        "oversold": 15.0,
        "use_trend_filter": True,
        "long_ma_period": 200,
        "use_volume_confirmation": True,
        "vol_ma_period": 50,
        "price_column": "Adj Close",
        "volume_column": "Volume",
    }

    required_columns = ["Volume"]  # price_column handled with fallback

    def _validate_params(self, params: Dict[str, Any]) -> None:
        """Validate strategy parameters.

        Args:
            params (Dict[str, Any]): The parameters to validate.

        Raises:
            ValueError: If parameters are invalid.
        """
        if params["rsi_period"] <= 0:
            raise ValueError("RSI period must be positive.")
        if params["oversold"] >= params["overbought"]:
            raise ValueError("Oversold threshold must be less than overbought.")
        if not (0 <= params["oversold"] <= 100) or not (
            0 <= params["overbought"] <= 100
        ):
            raise ValueError("Thresholds must be between 0 and 100.")
        if params["long_ma_period"] <= 0:
            raise ValueError("Long MA period must be positive.")
        if params["vol_ma_period"] <= 0:
            raise ValueError("Volume MA period must be positive.")

    def __init__(self, params: Dict[str, Any] | None = None):
        """Initialize the strategy with parameters.

        Args:
            params (Dict[str, Any] | None): Dictionary of strategy parameters.
        """
        super().__init__(params)
        # No need for manual assignments; use self.params directly where needed

    def generate_signals(self, data: DataFrame) -> Series:
        """Generate trading signals based on enhanced RSI mean reversion.

        Uses precise crossings: buy (1) when RSI crosses below oversold,
        sell (-1) when crosses above overbought. Optional filters: trend (price > long MA)
        and volume (> vol MA). Handles NaNs naturally (signals 0 for early bars).
        Prefers 'Adj Close' but falls back to 'Close' if unavailable.

        Args:
            data (DataFrame): DataFrame with price, volume columns (e.g., 'Adj Close', 'Volume').

        Returns:
            Series: Series of signals (1: buy, -1: sell, 0: hold).

        Raises:
            ValueError: If required columns are missing or data insufficient.
        """
        super().generate_signals(data)  # Calls base validation for required_columns

        # Determine the actual price column to use, with fallback
        actual_price_col = self.params["price_column"]
        if actual_price_col not in data.columns:
            if "Close" in data.columns:
                actual_price_col = "Close"
                self.logger.warning(
                    "Preferred price column '%s' not found; falling back to 'Close'. "
                    "For accurate backtesting, use adjusted prices if available.",
                    self.params["price_column"],
                )
            else:
                raise ValueError(
                    f"DataFrame must contain '{self.params['price_column']}' or 'Close' column."
                )

        if len(data) < self.params["rsi_period"] + 1:
            self.logger.warning(
                "Insufficient data for RSI calculation (need at least %d rows); "
                "returning neutral signals.",
                self.params["rsi_period"] + 1,
            )
            return Series(0, index=data.index, dtype="int8")

        # Extract prices and volume as NumPy arrays for TA-Lib/efficiency
        prices = data[actual_price_col].to_numpy(dtype="float64")

        # Calculate RSI using TA-Lib
        rsi = RSI(prices, timeperiod=self.params["rsi_period"])

        # Convert to Series for alignment and shifting
        rsi_series = Series(rsi, index=data.index)
        price_series = data[actual_price_col]
        volume_series = data[self.params["volume_column"]]

        # Detect crossings precisely
        # Buy: previous RSI >= oversold and current RSI < oversold (enters oversold)
        enter_oversold = (rsi_series < self.params["oversold"]) & (
            rsi_series.shift(1) >= self.params["oversold"]
        )
        # Sell: previous RSI <= overbought and current RSI > overbought (enters overbought)
        enter_overbought = (rsi_series > self.params["overbought"]) & (
            rsi_series.shift(1) <= self.params["overbought"]
        )

        # Apply optional trend filter: price > long MA for buys (bullish bias)
        if self.params["use_trend_filter"]:
            long_ma = price_series.rolling(window=self.params["long_ma_period"]).mean()
            enter_oversold &= price_series > long_ma

        # Apply optional volume confirmation: volume > vol MA for signals
        if self.params["use_volume_confirmation"]:
            vol_ma = volume_series.rolling(window=self.params["vol_ma_period"]).mean()
            enter_oversold &= volume_series > vol_ma
            enter_overbought &= volume_series > vol_ma

        # Generate signals: 1 for buy, -1 for sell, 0 otherwise
        signals = Series(
            0, index=data.index, dtype="int8"
        )  # int8 for memory efficiency
        signals[enter_oversold] = 1
        signals[enter_overbought] = -1

        # Log summary (avoid logging full series for large data)
        self.logger.debug(
            "Generated %d signals (%d buys, %d sells) for period %s to %s using column '%s'",
            len(signals),
            (signals == 1).sum(),
            (signals == -1).sum(),
            data.index[0],
            data.index[-1],
            actual_price_col,
        )
        return signals
