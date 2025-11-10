"""MACD trading strategy implementation."""

from typing import Any, Dict

from pandas import DataFrame, Series
from talib import MACD  # pylint: disable=no-name-in-module

from .base import BaseStrategy


class MACDStrategy(BaseStrategy):
    """MACD (Moving Average Convergence Divergence) Strategy.

    Generates buy signals on MACD line crossover above signal line,
    and sell on crossover below. Optionally uses histogram for confirmation
    (not enabled by default).

    Args:
        params (Dict[str, Any] | None): Dictionary of strategy parameters.
            - fast_period: int - Fast EMA period (default: 12)
            - slow_period: int - Slow EMA period (default: 26)
            - signal_period: int - Signal line period (default: 9)
            - use_histogram_confirmation: bool - Require positive histogram for buys,
              negative for sells (default: False)
            - price_column: str - DataFrame column for price data (default: "Adj Close")
    """

    default_params = {
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9,
        "use_histogram_confirmation": False,
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
        if params["fast_period"] >= params["slow_period"]:
            raise ValueError("Fast period must be less than slow period.")

    def __init__(self, params: Dict[str, Any] | None = None):
        """Initialize the strategy with parameters.

        Args:
            params (Dict[str, Any] | None): Dictionary of strategy parameters.
        """
        super().__init__(params)
        # No need for manual assignments; use self.params directly where needed

    def generate_signals(self, data: DataFrame) -> Series:
        """Generate trading signals based on MACD crossovers.

        Uses precise detection: buy (1) when MACD crosses above signal,
        sell (-1) when below. Optionally confirms with histogram direction.
        Prefers 'Adj Close' but falls back to 'Close' if unavailable.

        Args:
            data (DataFrame): DataFrame with price column (e.g., 'Adj Close' or 'Close').

        Returns:
            Series: Series of signals (1: buy, -1: sell, 0: hold).

        Raises:
            ValueError: If required price column is missing.
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

        # Extract prices as NumPy array for TA-Lib (optimized C computation)
        prices = data[actual_column].to_numpy(dtype="float64")  # float64 for precision

        # Calculate MACD using TA-Lib
        macd_line, signal_line, histogram = MACD(
            prices,
            fastperiod=self.params["fast_period"],
            slowperiod=self.params["slow_period"],
            signalperiod=self.params["signal_period"],
        )

        # Convert to Series for alignment and shifting (index preserved)
        macd_series = Series(macd_line, index=data.index)
        signal_series = Series(signal_line, index=data.index)
        hist_series = Series(histogram, index=data.index)

        # Detect crossovers precisely
        # Buy: previous MACD <= signal and current MACD > signal
        cross_up = (macd_series > signal_series) & (
            macd_series.shift(1) <= signal_series.shift(1)
        )
        # Sell: previous MACD >= signal and current MACD < signal
        cross_down = (macd_series < signal_series) & (
            macd_series.shift(1) >= signal_series.shift(1)
        )

        # Apply optional histogram confirmation
        if self.params["use_histogram_confirmation"]:
            cross_up &= hist_series > 0  # Increasing momentum for buy
            cross_down &= hist_series < 0  # Decreasing for sell

        # Generate signals: 1 for buy, -1 for sell, 0 otherwise
        signals = Series(
            0, index=data.index, dtype="int8"
        )  # int8 for memory efficiency
        signals[cross_up] = 1
        signals[cross_down] = -1

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
