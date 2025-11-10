"""Relative Strength Index (RSI) trading strategy implementation."""

from typing import Any, Dict

from pandas import DataFrame, Series
from talib import RSI  # pylint: disable=no-name-in-module

from .base import BaseStrategy


class RSIStrategy(BaseStrategy):
    """Relative Strength Index (RSI) Strategy.

    Generates buy signals when RSI crosses below oversold threshold (enters oversold),
    and sell signals when crosses above overbought threshold (enters overbought).
    Uses TA-Lib for optimized computation.

    Args:
        params (Optional[Dict[str, Any]]): Dictionary of strategy parameters.
            - period: int - RSI period (default: 14)
            - overbought: float - Overbought threshold (default: 70.0)
            - oversold: float - Oversold threshold (default: 30.0)
            - price_column: str - DataFrame column for price data (default: "Adj Close")
    """

    default_params = {
        "period": 14,
        "overbought": 70.0,
        "oversold": 30.0,
        "price_column": "Adj Close",
    }

    required_columns = []  # Custom column check in generate_signals due to fallback

    def _validate_params(self, params: Dict[str, Any]) -> None:
        """Validate strategy parameters.

        Args:
            params (Dict[str, Any]): The parameters to validate.

        Raises:
            ValueError: If parameters are invalid.
        """
        if params["period"] <= 0:
            raise ValueError("RSI period must be positive.")
        if params["oversold"] >= params["overbought"]:
            raise ValueError("Oversold threshold must be less than overbought.")
        if not (0 <= params["oversold"] <= 100) or not (
            0 <= params["overbought"] <= 100
        ):
            raise ValueError("RSI thresholds must be between 0 and 100.")

    def __init__(self, params: Dict[str, Any] | None = None):
        """Initialize the strategy with parameters.

        Args:
            params (Dict[str, Any] | None): Dictionary of strategy parameters.
        """
        super().__init__(params)
        # No need for manual assignments; use self.params directly where needed

    def generate_signals(self, data: DataFrame) -> Series:
        """Generate trading signals based on RSI threshold crossings.

        Uses precise detection: buy (1) when RSI crosses below oversold,
        sell (-1) when crosses above overbought. Handles NaNs naturally (signals 0 for early bars).
        Prefers 'Adj Close' but falls back to 'Close' if unavailable.

        Args:
            data (DataFrame): DataFrame with price column (e.g., 'Adj Close' or 'Close').

        Returns:
            Series: Series of signals (1: buy, -1: sell, 0: hold).

        Raises:
            ValueError: If required price column is missing or data insufficient.
        """
        super().generate_signals(
            data
        )  # Calls base validation (no-op since required_columns=[])

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

        if len(data) < self.params["period"] + 1:
            self.logger.warning(
                "Insufficient data for RSI calculation (need at least %d rows); "
                "returning neutral signals.",
                self.params["period"] + 1,
            )
            return Series(0, index=data.index, dtype="int8")

        # Extract prices as NumPy array for TA-Lib (optimized C computation)
        prices = data[actual_column].to_numpy(dtype="float64")  # float64 for precision

        # Calculate RSI using TA-Lib
        rsi = RSI(prices, timeperiod=self.params["period"])

        # Convert to Series for alignment and shifting
        rsi_series = Series(rsi, index=data.index)

        # Detect crossings precisely
        # Buy: previous RSI >= oversold and current RSI < oversold (enters oversold)
        enter_oversold = (rsi_series < self.params["oversold"]) & (
            rsi_series.shift(1) >= self.params["oversold"]
        )
        # Sell: previous RSI <= overbought and current RSI > overbought (enters overbought)
        enter_overbought = (rsi_series > self.params["overbought"]) & (
            rsi_series.shift(1) <= self.params["overbought"]
        )

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
            actual_column,
        )
        return signals
