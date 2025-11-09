"""MACD trading strategy implementation."""

from typing import Any, Dict, Optional

from pandas import DataFrame, Series
from talib import MACD  # pylint: disable=no-name-in-module

from ..utils.logger import setup_logger
from .base import BaseStrategy

logger = setup_logger(__name__)


class MACDStrategy(BaseStrategy):
    """
    MACD (Moving Average Convergence Divergence) Strategy.

    Generates buy signals on MACD line crossover above signal line,
    and sell on crossover below. Includes histogram for confirmation.

    Parameters (via config):
    - fast_period: int - Fast EMA period (default: 12)
    - slow_period: int - Slow EMA period (default: 26)
    - signal_period: int - Signal line period (default: 9)
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the strategy with parameters.

        :param params: Dictionary of strategy parameters.
        """
        super().__init__(params or {})
        self.fast_period: int = self.params.get("fast_period", 12)
        self.slow_period: int = self.params.get("slow_period", 26)
        self.signal_period: int = self.params.get("signal_period", 9)
        if self.fast_period >= self.slow_period:
            raise ValueError("Fast period must be less than slow period.")
        logger.info(
            "Initialized MACD with fast=%s, slow=%s, signal=%s",
            self.fast_period,
            self.slow_period,
            self.signal_period,
        )

    def generate_signals(self, data: DataFrame) -> Series:
        """
        Generate trading signals based on MACD crossovers.

        :param data: DataFrame with 'Close' column for prices.
        :return: Series of signals (1: buy, -1: sell, 0: hold).
        """
        if "Close" not in data.columns:
            raise ValueError("DataFrame must contain 'Close' column.")

        # Calculate MACD using TA-Lib for optimized C-based computation
        macd_line, signal_line, _ = MACD(
            data["Close"].to_numpy(dtype=float),
            fastperiod=self.fast_period,
            slowperiod=self.slow_period,
            signalperiod=self.signal_period,
        )

        # Generate signals: 1 where MACD > signal (crossover up), -1 opposite
        signals: Series[float] = Series(0, index=data.index)
        signals[macd_line > signal_line] = 1
        signals[macd_line < signal_line] = -1

        # Diff to detect actual crossovers
        signals = signals.diff().fillna(0).clip(lower=-1, upper=1)

        logger.debug(
            "Generated %s signals for %s to %s",
            len(signals),
            data.index[0],
            data.index[-1],
        )
        return signals
