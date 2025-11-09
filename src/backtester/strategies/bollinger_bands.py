"""Bollinger Bands trading strategy implementation."""

from typing import Any, Dict, Optional

from numpy import where
from pandas import DataFrame, Series

from ..utils.logger import setup_logger
from .base import BaseStrategy

logger = setup_logger(__name__)


class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands Strategy.

    This mean-reversion strategy calculates Bollinger Bands using a rolling SMA (middle band)
    and standard deviation for upper/lower bands. It generates buy signals when price crosses
    below the lower band, sell when crosses above the upper band, and hold otherwise.
    Uses vectorized operations for high performance and scalability.

    Parameters (via config):
    - window: int - Rolling window for SMA and std (default: 20)
    - std_multiplier: float - Multiplier for std deviation (default: 2.0)
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the strategy with parameters.

        :param params: Dictionary of strategy parameters.
        """
        super().__init__(params or {})
        self.window: int = self.params.get("window", 20)
        self.std_multiplier: float = self.params.get("std_multiplier", 2.0)
        if self.window <= 1:
            raise ValueError("Window must be greater than 1.")
        if self.std_multiplier <= 0:
            raise ValueError("Std multiplier must be positive.")
        logger.info(
            "Initialized BollingerBands with window=%s, std_multiplier=%s",
            self.window,
            self.std_multiplier,
        )

    def generate_signals(self, data: DataFrame) -> Series:
        """
        Generate trading signals based on Bollinger Band crossovers.

        :param data: DataFrame with 'Close' column for prices.
        :return: Series of signals (1: buy, -1: sell, 0: hold).
        """
        if "Close" not in data.columns:
            raise ValueError("DataFrame must contain 'Close' column.")

        # Calculate middle band (SMA) and std deviation vectorized
        middle_band = data["Close"].rolling(window=self.window).mean()
        rolling_std = data["Close"].rolling(window=self.window).std()

        # Upper and lower bands
        upper_band = middle_band + (rolling_std * self.std_multiplier)
        lower_band = middle_band - (rolling_std * self.std_multiplier)

        # Generate position indicators: 1 if below lower, -1 if above upper, 0 otherwise
        positions = where(data["Close"] < lower_band, 1, 0)
        positions = where(data["Close"] > upper_band, -1, positions)

        # Convert to Series and diff to detect crossovers (signals only on changes)
        positions_series = Series(positions, index=data.index)
        signals_series = positions_series.diff().fillna(0).clip(lower=-1, upper=1)

        logger.debug(
            "Generated %s signals for %s to %s",
            len(signals_series),
            data.index[0],
            data.index[-1],
        )
        return signals_series
