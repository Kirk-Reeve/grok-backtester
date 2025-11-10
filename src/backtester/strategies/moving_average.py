"""Moving Average Crossover trading strategy implementation."""

from typing import Any, Dict

from pandas import DataFrame, Series

from .base import BaseStrategy


class MovingAverageStrategy(BaseStrategy):
    """Moving Average Crossover Strategy.

    Generates buy signals on short SMA crossover above long SMA,
    and sell on crossover below. Vectorized with rolling operations.

    Args:
        params (Dict[str, Any] | None): Dictionary of strategy parameters.
            - short_window: int - Short SMA period (default: 50)
            - long_window: int - Long SMA period (default: 200)
            - price_column: str - DataFrame column for price data (default: "Adj Close")
    """

    default_params = {
        "short_window": 50,
        "long_window": 200,
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
        if params["short_window"] >= params["long_window"]:
            raise ValueError("Short window must be less than long window.")

    def __init__(self, params: Dict[str, Any] | None = None):
        """Initialize the strategy with parameters.

        Args:
            params (Dict[str, Any] | None): Dictionary of strategy parameters.
        """
        super().__init__(params)
        # No need for manual assignments; use self.params directly where needed

    def generate_signals(self, data: DataFrame) -> Series:
        """Generate trading signals based on SMA crossovers.

        Uses precise detection: buy (1) when short SMA crosses above long SMA,
        sell (-1) when below. Handles NaNs naturally (signals remain 0 for early bars).
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

        # Calculate rolling SMAs vectorized
        prices = data[actual_column]
        short_sma = prices.rolling(window=self.params["short_window"]).mean()
        long_sma = prices.rolling(window=self.params["long_window"]).mean()

        # Detect crossovers precisely
        # Buy: previous short <= long and current short > long
        cross_up = (short_sma > long_sma) & (short_sma.shift(1) <= long_sma.shift(1))
        # Sell: previous short >= long and current short < long
        cross_down = (short_sma < long_sma) & (short_sma.shift(1) >= long_sma.shift(1))

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
