"""Commodity Channel Index (CCI) trading strategy implementation."""

from typing import Any, Dict

from pandas import DataFrame, Series
from talib import CCI  # pylint: disable=no-name-in-module

from .base import BaseStrategy


class CCIStrategy(BaseStrategy):
    """Commodity Channel Index (CCI) Strategy.

    Generates buy signals when CCI crosses below oversold threshold,
    and sell when crosses above overbought threshold. Uses TA-Lib
    for optimized computation.

    Args:
        params (Dict[str, Any] | None): Dictionary of strategy parameters.
            - period: int - CCI period (default: 20)
            - overbought: float - Overbought threshold (default: 100.0)
            - oversold: float - Oversold threshold (default: -100.0)
            - price_column: str - DataFrame column for price data (default: "Adj Close")
    """

    default_params = {
        "period": 20,
        "overbought": 100.0,
        "oversold": -100.0,
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
        if params["period"] <= 0:
            raise ValueError("CCI period must be positive.")
        if params["oversold"] >= params["overbought"]:
            raise ValueError("Oversold threshold must be less than overbought.")

    def __init__(self, params: Dict[str, Any] | None = None):
        """Initialize the strategy with parameters.

        Args:
            params (Dict[str, Any] | None): Dictionary of strategy parameters.
        """
        super().__init__(params)
        # No need for manual assignments; use self.params directly where needed

    def generate_signals(self, data: DataFrame) -> Series:
        """Generate trading signals based on CCI threshold crossings.

        Uses precise detection: buy (1) when CCI crosses below oversold,
        sell (-1) when crosses above overbought. Handles NaNs naturally.

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

        if len(data) < self.params["period"] + 1:
            self.logger.warning(
                "Insufficient data for CCI calculation; returning neutral signals."
            )
            return Series(0, index=data.index, dtype="int8")

        # Extract as NumPy arrays for TA-Lib
        high = data["High"].to_numpy(dtype="float64")
        low = data["Low"].to_numpy(dtype="float64")
        close = data[actual_column].to_numpy(dtype="float64")

        # Calculate CCI using TA-Lib
        cci = CCI(high, low, close, timeperiod=self.params["period"])

        # Convert to Series
        cci_series = Series(cci, index=data.index)

        # Detect crossings
        # Buy: previous CCI >= oversold and current CCI < oversold
        enter_oversold = (cci_series < self.params["oversold"]) & (
            cci_series.shift(1) >= self.params["oversold"]
        )
        # Sell: previous CCI <= overbought and current CCI > overbought
        enter_overbought = (cci_series > self.params["overbought"]) & (
            cci_series.shift(1) <= self.params["overbought"]
        )

        # Generate signals
        signals = Series(0, index=data.index, dtype="int8")
        signals[enter_oversold] = 1
        signals[enter_overbought] = -1

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
