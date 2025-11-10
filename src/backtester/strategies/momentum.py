"""Momentum trading strategy implementation."""

from typing import Any, Dict

from pandas import DataFrame, Series

from .base import BaseStrategy


class MomentumStrategy(BaseStrategy):
    """Momentum Strategy.

    This strategy calculates the rate of change (momentum) over a lookback period.
    It generates buy signals when momentum > buy_threshold, sell when < sell_threshold,
    and hold otherwise. Fully vectorized for efficiency.

    Args:
        params (Dict[str, Any] | None): Dictionary of strategy parameters.
            - lookback: int - Period for momentum calculation (default: 20)
            - buy_threshold: float - Positive threshold for buy (default: 0.05)
            - sell_threshold: float - Negative threshold for sell (default: -0.05)
            - price_column: str - DataFrame column for price data (default: "Adj Close")
    """

    default_params = {
        "lookback": 20,
        "buy_threshold": 0.05,
        "sell_threshold": -0.05,
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
        if params["lookback"] <= 0:
            raise ValueError("Lookback period must be positive.")
        if params["sell_threshold"] >= params["buy_threshold"]:
            raise ValueError("Sell threshold must be less than buy threshold.")

    def __init__(self, params: Dict[str, Any] | None = None):
        """Initialize the strategy with parameters.

        Args:
            params (Dict[str, Any] | None): Dictionary of strategy parameters.
        """
        super().__init__(params)
        # No need for manual assignments; use self.params directly where needed

    def generate_signals(self, data: DataFrame) -> Series:
        """Generate trading signals based on momentum thresholds.

        Uses precise vectorized operations: buy (1) when momentum > buy_threshold,
        sell (-1) when < sell_threshold. Handles NaNs naturally (signals remain 0 for early bars).
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

        # Calculate momentum using vectorized pct_change (NaNs for first lookback-1 bars)
        prices = data[actual_column]
        momentum = prices.pct_change(periods=self.params["lookback"])

        # Detect conditions with boolean masks (NaNs evaluate to False)
        buy_condition = momentum > self.params["buy_threshold"]
        sell_condition = momentum < self.params["sell_threshold"]

        # Generate signals: 1 for buy, -1 for sell, 0 otherwise
        signals = Series(
            0, index=data.index, dtype="int8"
        )  # int8 for memory efficiency
        signals[buy_condition] = 1
        signals[sell_condition] = -1

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
