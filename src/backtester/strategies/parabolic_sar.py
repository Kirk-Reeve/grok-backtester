"""Parabolic SAR trading strategy implementation."""

from typing import Any, Dict

from pandas import DataFrame, Series
from talib import SAR  # pylint: disable=no-name-in-module

from .base import BaseStrategy


class ParabolicSARStrategy(BaseStrategy):
    """Parabolic SAR Strategy.

    Generates buy signals when price crosses above SAR (trend reversal up),
    and sell when price crosses below SAR (trend reversal down). Uses TA-Lib
    for optimized computation.

    Args:
        params (Dict[str, Any] | None): Dictionary of strategy parameters.
            - acceleration: float - Starting acceleration factor (default: 0.015)
            - maximum: float - Maximum acceleration factor (default: 0.25)
            - price_column: str - DataFrame column for price data (default: "Adj Close")
    """

    default_params = {
        "acceleration": 0.015,
        "maximum": 0.25,
        "price_column": "Adj Close",
    }

    required_columns = ["High", "Low"]  # price_column handled with fallback

    def _validate_params(self, params: Dict[str, Any]) -> None:
        """Validate strategy parameters.

        Args:
            params (Dict[str, Any]): The parameters to validate.

        Raises:
            ValueError: If parameters are invalid.
        """
        if params["acceleration"] <= 0 or params["maximum"] <= 0:
            raise ValueError("Acceleration factors must be positive.")
        if params["acceleration"] > params["maximum"]:
            raise ValueError("Starting acceleration must be <= maximum.")

    def __init__(self, params: Dict[str, Any] | None = None):
        """Initialize the strategy with parameters.

        Args:
            params (Dict[str, Any] | None): Dictionary of strategy parameters.
        """
        super().__init__(params)
        # No need for manual assignments; use self.params directly where needed

    def generate_signals(self, data: DataFrame) -> Series:
        """Generate trading signals based on Parabolic SAR reversals.

        Uses precise detection: buy (1) when price crosses above SAR,
        sell (-1) when price crosses below SAR. Handles NaNs naturally.

        Args:
            data (DataFrame): DataFrame with 'High', 'Low', 'Close' (or adjusted) columns.

        Returns:
            Series: Series of signals (1: buy, -1: sell, 0: hold).

        Raises:
            ValueError: If required columns are missing.
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

        # Extract as NumPy arrays for TA-Lib
        high = data["High"].to_numpy(dtype="float64")
        low = data["Low"].to_numpy(dtype="float64")

        # Calculate SAR using TA-Lib
        sar = SAR(
            high,
            low,
            acceleration=self.params["acceleration"],
            maximum=self.params["maximum"],
        )

        # Convert to Series
        sar_series = Series(sar, index=data.index)
        price_series = data[actual_column]

        # Detect reversals
        # Buy: previous price <= SAR and current price > SAR
        cross_up = (price_series > sar_series) & (
            price_series.shift(1) <= sar_series.shift(1)
        )
        # Sell: previous price >= SAR and current price < SAR
        cross_down = (price_series < sar_series) & (
            price_series.shift(1) >= sar_series.shift(1)
        )

        # Generate signals
        signals = Series(0, index=data.index, dtype="int8")
        signals[cross_up] = 1
        signals[cross_down] = -1

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
