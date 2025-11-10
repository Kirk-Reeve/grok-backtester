"""Bollinger Bands trading strategy implementation."""

from typing import Any, Dict

from pandas import DataFrame, Series
from talib import BBANDS, MA_Type  # pylint: disable=no-name-in-module

from .base import BaseStrategy


class BollingerBandsStrategy(BaseStrategy):
    """Bollinger Bands Strategy.

    Generates buy signals when price crosses below lower band (oversold),
    and sell when crosses above upper band (overbought). Uses TA-Lib for
    optimized computation.

    Args:
        params (Dict[str, Any] | None): Dictionary of strategy parameters.
            - window: int - Rolling window for SMA and std (default: 20)
            - std_multiplier: float - Multiplier for std deviation (default: 2.0)
            - price_column: str - DataFrame column for price data (default: "Adj Close")
    """

    default_params = {
        "window": 20,
        "std_multiplier": 2.0,
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
        if params["std_multiplier"] <= 0:
            raise ValueError("Std multiplier must be positive.")

    def __init__(self, params: Dict[str, Any] | None = None):
        """Initialize the strategy with parameters.

        Args:
            params (Dict[str, Any] | None): Dictionary of strategy parameters.
        """
        super().__init__(params)
        # No need for manual assignments; use self.params directly where needed

    def generate_signals(self, data: DataFrame) -> Series:
        """Generate trading signals based on Bollinger Band crossings.

        Uses precise detection: buy (1) when price crosses below lower band,
        sell (-1) when crosses above upper band. Handles NaNs naturally (signals 0 for early bars).
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

        # Calculate Bollinger Bands using TA-Lib
        upper_band, _, lower_band = BBANDS(
            prices,
            timeperiod=self.params["window"],
            nbdevup=self.params["std_multiplier"],
            nbdevdn=self.params["std_multiplier"],
            matype=MA_Type.SMA,
        )

        # Convert to Series for alignment and shifting
        price_series = data[actual_column]
        upper_series = Series(upper_band, index=data.index)
        lower_series = Series(lower_band, index=data.index)

        # Detect crossings precisely
        # Buy: previous price >= lower and current price < lower (crosses below lower)
        cross_below_lower = (price_series < lower_series) & (
            price_series.shift(1) >= lower_series.shift(1)
        )
        # Sell: previous price <= upper and current price > upper (crosses above upper)
        cross_above_upper = (price_series > upper_series) & (
            price_series.shift(1) <= upper_series.shift(1)
        )

        # Generate signals: 1 for buy, -1 for sell, 0 otherwise
        signals = Series(
            0, index=data.index, dtype="int8"
        )  # int8 for memory efficiency
        signals[cross_below_lower] = 1
        signals[cross_above_upper] = -1

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
