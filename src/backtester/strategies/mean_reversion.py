"""Mean-Reversion trading strategy implementation."""

from typing import Any, Dict

from numpy import nan
from pandas import DataFrame, Series

from .base import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    """Mean-Reversion Strategy.

    This strategy uses z-score on rolling mean and std to detect deviations.
    Buys when z-score < -threshold (oversold), sells when > threshold (overbought),
    holds otherwise. Vectorized with rolling operations.

    Args:
        params (Dict[str, Any] | None): Dictionary of strategy parameters.
            - window: int - Rolling window for mean and std (default: 20)
            - threshold: float - Z-score deviation threshold (default: 2.0)
            - price_column: str - DataFrame column for price data (default: "Adj Close")
    """

    default_params = {
        "window": 20,
        "threshold": 2.0,
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
        if params["window"] <= 1:
            raise ValueError("Window must be greater than 1.")
        if params["threshold"] <= 0:
            raise ValueError("Threshold must be positive.")

    def __init__(self, params: Dict[str, Any] | None = None):
        """Initialize the strategy with parameters.

        Args:
            params (Dict[str, Any] | None): Dictionary of strategy parameters.
        """
        super().__init__(params)
        # No need for manual assignments; use self.params directly where needed

    def generate_signals(self, data: DataFrame) -> Series:
        """Generate trading signals based on z-score thresholds.

        Uses precise vectorized operations: buy (1) when z-score < -threshold,
        sell (-1) when > threshold. Handles NaNs naturally (signals remain 0 for early bars).
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

        # Calculate rolling mean and std vectorized
        prices = data[actual_column]
        rolling_mean = prices.rolling(window=self.params["window"]).mean()
        rolling_std = prices.rolling(window=self.params["window"]).std()

        # Z-score calculation (NaNs propagate where std is NaN or 0)
        z_score = (prices - rolling_mean) / rolling_std.replace(0, nan)

        # Detect conditions with boolean masks (NaNs evaluate to False)
        oversold = z_score < -self.params["threshold"]
        overbought = z_score > self.params["threshold"]

        # Generate signals: 1 for buy, -1 for sell, 0 otherwise
        signals = Series(
            0, index=data.index, dtype="int8"
        )  # int8 for memory efficiency
        signals[oversold] = 1
        signals[overbought] = -1

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
