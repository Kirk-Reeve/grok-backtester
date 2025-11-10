"""Stochastic Oscillator trading strategy implementation."""

from typing import Any, Dict

from pandas import DataFrame, Series
from talib import STOCH, MA_Type  # pylint: disable=no-name-in-module

from .base import BaseStrategy


class StochasticStrategy(BaseStrategy):
    """Stochastic Oscillator Strategy.

    Generates buy signals when %K crosses above %D in oversold region,
    and sell when %K crosses below %D in overbought region. Uses TA-Lib
    for optimized computation.

    Args:
        params (Dict[str, Any] | None): Dictionary of strategy parameters.
            - fastk_period: int - Fast %K period (default: 5)
            - slowk_period: int - Slow %K period (default: 3)
            - slowd_period: int - Slow %D period (default: 3)
            - overbought: float - Overbought threshold (default: 80.0)
            - oversold: float - Oversold threshold (default: 20.0)
            - price_column: str - DataFrame column for price data (default: "Adj Close")
    """

    default_params = {
        "fastk_period": 5,
        "slowk_period": 3,
        "slowd_period": 3,
        "overbought": 80.0,
        "oversold": 20.0,
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
        if any(
            p <= 0
            for p in [
                params["fastk_period"],
                params["slowk_period"],
                params["slowd_period"],
            ]
        ):
            raise ValueError("All periods must be positive.")
        if params["oversold"] >= params["overbought"]:
            raise ValueError("Oversold threshold must be less than overbought.")
        if not (0 <= params["oversold"] <= 100) or not (
            0 <= params["overbought"] <= 100
        ):
            raise ValueError("Thresholds must be between 0 and 100.")

    def __init__(self, params: Dict[str, Any] | None = None):
        """Initialize the strategy with parameters.

        Args:
            params (Dict[str, Any] | None): Dictionary of strategy parameters.
        """
        super().__init__(params)
        # No need for manual assignments; use self.params directly where needed

    def generate_signals(self, data: DataFrame) -> Series:
        """Generate trading signals based on Stochastic crossovers in overbought/oversold regions.

        Uses precise detection: buy (1) when %K crosses above %D below oversold,
        sell (-1) when %K crosses below %D above overbought. Handles NaNs naturally.

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

        if (
            len(data)
            < max(
                self.params["fastk_period"],
                self.params["slowk_period"],
                self.params["slowd_period"],
            )
            + 1
        ):
            self.logger.warning(
                "Insufficient data for Stochastic calculation; returning neutral signals."
            )
            return Series(0, index=data.index, dtype="int8")

        # Extract as NumPy arrays for TA-Lib
        high = data["High"].to_numpy(dtype="float64")
        low = data["Low"].to_numpy(dtype="float64")
        close = data[actual_column].to_numpy(dtype="float64")

        # Calculate Stochastic using TA-Lib
        slowk, slowd = STOCH(
            high,
            low,
            close,
            fastk_period=self.params["fastk_period"],
            slowk_period=self.params["slowk_period"],
            slowk_matype=MA_Type.SMA,
            slowd_period=self.params["slowd_period"],
            slowd_matype=MA_Type.SMA,
        )

        # Convert to Series
        slowk_series = Series(slowk, index=data.index)
        slowd_series = Series(slowd, index=data.index)

        # Detect crossovers in regions
        # Buy: %K crosses above %D and both < oversold (or at crossover point)
        cross_up = (slowk_series > slowd_series) & (
            slowk_series.shift(1) <= slowd_series.shift(1)
        )
        in_oversold = (slowk_series < self.params["oversold"]) & (
            slowd_series < self.params["oversold"]
        )
        buy_condition = cross_up & in_oversold

        # Sell: %K crosses below %D and both > overbought
        cross_down = (slowk_series < slowd_series) & (
            slowk_series.shift(1) >= slowd_series.shift(1)
        )
        in_overbought = (slowk_series > self.params["overbought"]) & (
            slowd_series > self.params["overbought"]
        )
        sell_condition = cross_down & in_overbought

        # Generate signals
        signals = Series(0, index=data.index, dtype="int8")
        signals[buy_condition] = 1
        signals[sell_condition] = -1

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
